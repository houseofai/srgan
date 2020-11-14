from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res + (3,)

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./data/%s/*' % self.dataset_name)

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = Image.open(img_path).convert('RGB')

            h, w, c = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = img.resize((h, w))
            img_lr = img.resize((low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


def get_data(data_dir="./data/", img_res=(128, 128), batch_size=32):
    low_h, low_w = int(img_res[0] / 4), int(img_res[0] / 4)
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5)

    ds_hr = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                # validation_split=0.2,
                                                                # subset="training",
                                                                seed=123,
                                                                image_size=img_res,
                                                                batch_size=batch_size
                                                                )

    ds_lr = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                # validation_split=0.2,
                                                                # subset="training",
                                                                seed=123,
                                                                image_size=(low_h, low_w),
                                                                batch_size=batch_size
                                                                )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_hr = ds_hr.map(lambda x, y: (normalization_layer(x), y)) \
        .map(lambda x, y: (tf.image.random_flip_left_right(x), y)) \
        .cache().prefetch(buffer_size=AUTOTUNE)
    ds_lr = ds_lr.map(lambda x, y: (normalization_layer(x), y)) \
        .map(lambda x, y: (tf.image.random_flip_left_right(x), y)) \
        .cache().prefetch(buffer_size=AUTOTUNE)

    return ds_hr, ds_lr
