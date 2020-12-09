"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img
/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import copy
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.applications import VGG19  # , EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks as callbacks_module
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt
import argparse
import helpers
import logging
from data_loader import DataLoader
import data_loader as dl
import numpy as np
from os import path
import datetime


class SRGAN:
    def __init__(self, config, experimentation_name):
        self.local_config = config
        # Input shape
        self.channels = 3
        self.lr_height = self.local_config.training.low_res  # Low resolution height
        self.lr_width = self.local_config.training.low_res  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        scale = self.local_config.training.scale
        self.hr_height = self.lr_height * scale  # High resolution height
        self.hr_width = self.lr_width * scale  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.batch_size = self.local_config.training.batch_size
        self.experimentation_name = experimentation_name

        # Number of residual blocks in the generator
        # self.n_residual_blocks = config.model.n_residual_blocks

        if helpers.is_multi_gpus():
            self.strategy = tf.distribute.MirroredStrategy()
            nb_gpu = self.strategy.num_replicas_in_sync
            print("* Found {} GPU".format(nb_gpu))
            self.batch_size = self.batch_size * self.strategy.num_replicas_in_sync
            print("* New batch size: {}".format(self.batch_size))
        else:
            self.strategy = None

        with helpers.get_strategy_scope(self.strategy):
            optimizer = Adam(config.optimizer.learning_rate, config.optimizer.momentum)

            # We use a pre-trained VGG19 model to extract image features from the high resolution
            # and the generated high resolution images and minimize the mse between them
            self.extractor = self.build_feature_extractor()
            self.extractor.trainable = False
            self.extractor.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

            # Configure data loader
            self.data_dir = config.data.base_dir
            self.dataset_name = config.data.dataset
            self.data_path = "{}/{}".format(self.data_dir, self.dataset_name)
            self.data_loader = DataLoader(dataset_name="{}/{}".format(self.data_path, config.data.datasubset),
                                          img_res=(self.hr_height, self.hr_width))

            self.ds = dl.get_data(hr_size=(self.hr_height, self.hr_width), batch_size=self.batch_size,
                                  data_dir=self.data_path)
            self.ds_size = tf.data.experimental.cardinality(self.ds)
            log.info("Total iteration per epochs: {}".format(self.ds_size))

            # Calculate output shape of D (PatchGAN)
            patch = int(self.hr_height / 2 ** 4)
            self.disc_patch = (patch, patch, 1)

            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])
            # log.info("Discriminator Model Metrics: {}".format(self.discriminator.metrics_names))

            # Build the generator
            self.generator = self.build_generator()
            # log.info("Generator Model Metrics: {}".format(self.generator.metrics_names))

            # High res. and low res. images
            img_hr = Input(shape=self.hr_shape, name="HR")
            img_lr = Input(shape=self.lr_shape, name="LR")

            # Generate high res. version from low res.
            fake_hr = self.generator(img_lr)

            # Extract image features of the generated img
            fake_features = self.extractor(fake_hr)

            # For the combined model we will only train the generator
            self.discriminator.trainable = False

            # Discriminator determines validity of generated high res. images
            validity = self.discriminator(fake_hr)

            self.combined = Model([img_lr, img_hr], [validity, fake_features], name="GAN")
            self.combined.compile(loss=['binary_crossentropy', 'mse'],
                                  loss_weights=[1e-3, 1],
                                  optimizer=optimizer)
            # log.info("GAN Model Metrics: {}".format(self.combined.metrics_names))
            log.info("GAN Model Summary:")
            self.combined.summary()

            # Callbacks
            callbacks = self.set_callbacks(config, experimentation_name)

            self.cbm = callbacks_module.CallbackList(callbacks, model=self.combined,
                                                     add_history=True,
                                                     add_progbar=True,
                                                     verbose=1,
                                                     epochs=self.local_config.training.epochs,
                                                     steps=int(self.ds_size))

    def set_callbacks(self, local_config, experimentation_name):
        # Tensorboard
        callbacks = []

        if config.mode == "train":

            callbacks.append(helpers.callback_tensorboard(local_config, experimentation_name))

            # Checkpoints
            checkpoint_cb, checkpoint_filepath = helpers.callback_checkpoints(local_config, experimentation_name)
            callbacks.append(checkpoint_cb)
            if path.exists(checkpoint_filepath):
                # TODO Load and save weights of other models
                self.combined.load_weights(checkpoint_filepath)

            # Sample image
            image_sample_cb = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.sample_images(epoch),
                on_batch_end=lambda batch, logs:
                self.sample_images(batch) if batch % local_config.training.sample_interval == 0 else None
            )
            callbacks.append(image_sample_cb)

            # TODO Push Checkpoint to S3
            # S3.upload(filename)
        return callbacks

    def build_feature_extractor(self):
        """
        Builds a pre-trained Feature Extractor model that outputs image features extracted at the
        third block of the model
        """
        # extractor = EfficientNetB7(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        # Tried:
        # - no change: no shape
        # - block6m_add: shapes in only yellow and red
        # - block5j_add: shapes and blue
        # - block5j_project_conv: blue
        # - block4j_add: Less shapes but colorful
        # - block2g_add: BEST but yellow
        # - block2g_project_conv:
        # - block2f_add: lightly yellow
        # - block1d_add: No shape - no color
        # - block3b_add: Yellow, and noise
        # - block3b_drop: blue
        # output_layer = extractor.get_layer("block4b_add")

        extractor = VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        output_layer = extractor.layers[9]

        return Model(inputs=extractor.input, outputs=output_layer.output, name="VGG19")

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = PReLU()(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(layer_input)
            u = UpSampling2D(size=2)(u)
            u = PReLU()(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        nb_filters = self.local_config.model.g_filters

        # Pre-residual block
        c1 = Conv2D(nb_filters, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = PReLU()(c1)

        # Propagate through residual blocks
        r = residual_block(c1, nb_filters)
        for _ in range(self.local_config.model.n_residual_blocks - 1):
            r = residual_block(r, nb_filters)

        # Post-residual block
        c2 = Conv2D(nb_filters, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr, name="Generator")

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        nb_filters = self.local_config.model.d_filters

        d1 = d_block(d0, nb_filters, bn=False)
        d2 = d_block(d1, nb_filters, strides=2)
        d3 = d_block(d2, nb_filters * 2)
        d4 = d_block(d3, nb_filters * 2, strides=2)
        d5 = d_block(d4, nb_filters * 4)
        d6 = d_block(d5, nb_filters * 4, strides=2)
        d7 = d_block(d6, nb_filters * 8)
        d8 = d_block(d7, nb_filters * 8, strides=2)

        d9 = Dense(nb_filters * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity, name="Discriminator")

    def train(self, epochs):

        # Callbacks
        self.cbm.on_train_begin()
        training_logs = None

        for epoch in range(epochs):
            # Callbacks
            self.cbm.on_epoch_begin(epoch)
            self.combined.reset_metrics()

            step = 0
            logs = None
            ds_iter = iter(self.ds)
            while (disc_imgs_hr := next(ds_iter, None)) is not None and \
                    (gen_imgs_hr := next(ds_iter, None)) is not None:
                # Callbacks
                self.cbm.on_train_batch_begin(step)

                disc_imgs_lr = tf.image.resize(disc_imgs_hr, (self.lr_height, self.lr_width))

                # Create valid images based on current batch size to avoid remainder batch error
                current_batch_size = disc_imgs_lr.shape[0]
                valid = np.ones((current_batch_size,) + self.disc_patch)
                fake = np.zeros((current_batch_size,) + self.disc_patch)

                # Train Discriminator

                fake_hr = self.generator.predict(disc_imgs_lr)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(disc_imgs_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss, d_accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train Generator
                gen_imgs_lr = tf.image.resize(gen_imgs_hr, (self.lr_height, self.lr_width))
                # Extract ground truth image features using pre-trained VGG19 model
                image_features = self.extractor.predict(gen_imgs_hr)

                # Train the generators
                loss = self.combined.train_on_batch([gen_imgs_lr, gen_imgs_hr], [valid, image_features])

                logs = {"loss": loss[0]} # , "discriminator_loss": d_loss, "discriminator_accuracy": d_accuracy}

                step += 2
                # Callbacks
                self.cbm.on_train_batch_end(step, logs)

            # Callbacks
            epoch_logs = copy.copy(logs)
            self.cbm.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs

        # Callbacks
        self.cbm.on_train_end(logs=training_logs)

        return training_logs

    def sample_images(self, batch):
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
        fig.savefig("images/{}/step-{}.png".format(self.dataset_name, batch))
        plt.close()


def grid_search(local_config, experiment):
    HP_RES_BLOCKS = hp.HParam('n_residual_blocks', hp.Discrete(local_config.model.n_residual_blocks))
    HP_G_FILTERS = hp.HParam('g_filters', hp.Discrete(local_config.model.g_filters))
    HP_D_FILTERS = hp.HParam('d_filters', hp.Discrete(local_config.model.d_filters))
    METRIC_LOSS = 'loss'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    grid_search_logdir = '{}/gridsearch/{}-{}'.format(local_config.tensorboard.log_dir, current_time, experiment)
    with tf.summary.create_file_writer(grid_search_logdir).as_default():
        hp.hparams_config(
            hparams=[HP_RES_BLOCKS, HP_G_FILTERS, HP_D_FILTERS],
            metrics=[hp.Metric(METRIC_LOSS, display_name='Loss')],
        )

    session_num = 0

    for n_residual_blocks in HP_RES_BLOCKS.domain.values:
        for g_filters in HP_G_FILTERS.domain.values:
            for d_filters in HP_D_FILTERS.domain.values:
                grid_config = local_config
                grid_config.model.n_residual_blocks = n_residual_blocks
                grid_config.model.g_filters = g_filters
                grid_config.model.d_filters = d_filters
                log.info("Training with")
                log.info("- n_residual_blocks: {}".format(n_residual_blocks))
                log.info("- g_filters: {}".format(g_filters))
                log.info("- d_filters: {}".format(d_filters))
                hparams = {
                    HP_RES_BLOCKS: n_residual_blocks,
                    HP_G_FILTERS: g_filters,
                    HP_D_FILTERS: d_filters,
                }
                run_dir = "{}/run-{}".format(grid_search_logdir, session_num)
                with tf.summary.create_file_writer(run_dir).as_default():
                    hp.hparams(hparams)  # record the values used in this trial
                    try:
                        gan_grid = SRGAN(grid_config, experiment)
                        logs = gan_grid.train(epochs=grid_config.training.epochs)
                        tf.summary.scalar(METRIC_LOSS, logs["loss"], step=1)
                    except tf.errors.ResourceExhaustedError as e:
                        log.error("Not enough GPU Memory")
                session_num += 1


if __name__ == '__main__':
    """
    Start of the program to train/predict/evaluate
    """

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="original", help="Name of the configuration file in the config folder")
    parser.add_argument("--mode", default="train", help="train, tune, predict")
    parser.add_argument("--image", default="", help="path to an image (test mode)")
    args = parser.parse_args()

    # Set memory growth on GPU
    helpers.memory_growth()

    # Read config
    config = helpers.get_config(args.conf)
    experimentation_name = "{}-res_block{}_batch{}_lr20-4_filters{}" \
        .format(args.mode, config.model.n_residual_blocks, config.training.batch_size, config.model.d_filters)

    if args.mode == "train":
        gan = SRGAN(config, experimentation_name)
        gan.train(epochs=config.training.epochs)
    elif args.mode == "tune":
        grid_search(config, experimentation_name)
    elif args.mode == "predict":
        if args.image is not None:
            # TODO
            # predict(config, args.image)
            pass
        else:
            ValueError("'image' argument needs to be specified")
