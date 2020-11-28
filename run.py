"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img
/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.applications import VGG19, EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from helpers import LogEvents as lev

from data_loader import DataLoader
import data_loader as dl
import numpy as np
import os
from tqdm import trange, tqdm


class SRGAN:
    def __init__(self, profile):
        # Input shape
        self.channels = 3
        self.scale = 4
        self.lr_height = 64  # Low resolution height
        self.lr_width = 64  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * self.scale  # High resolution height
        self.hr_width = self.lr_width * self.scale  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.profile = profile
        self.log_dir = "./logs"

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

#        self.strategy = tf.distribute.MirroredStrategy()
#        nb_gpu = self.strategy.num_replicas_in_sync
#        print("* Found {} GPU".format(nb_gpu))
        # self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync

        # with self.strategy.scope():
        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_feature_extractor()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # with self.strategy.scope():
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_feature_extractor(self):
        """
        Builds a pre-trained Feature Extractor model that outputs image features extracted at the
        third block of the model
        """
        extractor = EfficientNetB7(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        # extractor = VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)

        # Tried:
        # - no change: no shape
        # - block6m_add: shapes in only yellow and red
        # - block5j_add: shapes and blue
        # - block5j_project_conv: blue
        # - block4j_add: Less shapes but colorful
        # - block2g_add: BEST but yellow
        # - block2g_project_conv:
        # - block1d_add: No shape - no color
        output_layer = extractor.get_layer("block2f_add")

        return Model(inputs=extractor.input, outputs=output_layer.output)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        fake = np.zeros((batch_size,) + self.disc_patch)

        #options = tf.data.Options()
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        ds = dl.get_data(hr_size=(self.hr_height, self.hr_width), batch_size=batch_size)
        ds_size = tf.data.experimental.cardinality(ds)
        #ds_hr = ds_hr.with_options(options)
        #ds_lr = ds_lr.with_options(options)

        # ds_hr = self.strategy.experimental_distribute_dataset(ds_hr)
        # ds_lr = self.strategy.experimental_distribute_dataset(ds_lr)
        print("Total iteration per epochs: {}".format(ds_size))

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        logevents = lev.LogEvents(log_dir=self.log_dir)

        for epoch in trange(epochs):
            disc_turn = True
            pbar = tqdm(total=ds_size.numpy())

            step = 0
            if self.profile:
                tf.profiler.experimental.start(self.log_dir)
            g_loss = None
            for imgs_hr in ds:

                imgs_lr = tf.image.resize(imgs_hr, (self.lr_height, self.lr_width))
                # Create valid images based on current batch size to avoid remainder batch error
                valid = np.ones((imgs_hr.shape[0],) + self.disc_patch)
                if disc_turn:
                    # ----------------------
                    #  Train Discriminator
                    # ----------------------
                    fake_hr = self.generator.predict(imgs_lr)

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                    d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                    #d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)
                    # pbar.set_description("Loss Discriminator: {}".format(d_loss))
                    disc_turn = False
                else:
                    # ------------------
                    #  Train Generator
                    # ------------------
                    # The generators want the discriminators to label the generated images as real

                    # Extract ground truth image features using pre-trained VGG19 model
                    image_features = self.vgg.predict(imgs_hr)

                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])
                    # print("Loss: ", g_loss)
                    train_loss.update_state(g_loss)
                    logevents.log_train(step, train_loss.result())
                    disc_turn = True
                pbar.set_description("[Step {}/{}] Loss Generator: {}".format(step, ds_size, g_loss))
                pbar.update(1)
                step += 1
                # If at save interval => save generated image samples
                if step % sample_interval == 0:
                    self.sample_images(epoch, step)
            if self.profile:
                tf.profiler.experimental.stop()
            self.sample_images(epoch, step)
            self.discriminator.save_weights("./checkpoints_disc")
            self.generator.save_weights("./checkpoints_gen")
            self.combined.save_weights("./checkpoints_comb")
            pbar.close()
            #logevents.reset(train_loss)

    def sample_images(self, epoch, step):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['LR', 'HR', 'Generated']
        r, c = 2, 3
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([imgs_lr, imgs_hr, fake_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/epoch-%d_step-%d.png" % (self.dataset_name, epoch, step))
        plt.close()

        # Save low resolution images for comparison
        #for i in range(r):
        #    fig = plt.figure()
        #    plt.imshow(imgs_lr[i])
        #    fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
        #    plt.close()



if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 32
    print("Batch Size:", batch_size)
    gan = SRGAN(profile=False)
    gan.train(epochs=3000, batch_size=batch_size, sample_interval=50)
