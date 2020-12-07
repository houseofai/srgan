from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res + (3,)

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('%s/*' % self.dataset_name)

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = Image.open(img_path).convert('RGB')

            h, w, c = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = np.array(img.resize((h, w)))
            img_lr = np.array(img.resize((low_h, low_w)))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


def normalize(image):
    return image / 127.5 - 1.


def flip(image):
    return tf.image.random_flip_left_right(image)


def get_data(hr_size, batch_size, data_dir="./data/"):
    # low_h, low_w = int(hr_size[0] / scale_down), int(hr_size[0] / scale_down)

    print("Preprocessing data for HR{}".format(hr_size))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, image_size=hr_size, label_mode=None,
                                                             batch_size=batch_size) \
        .map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .prefetch(buffer_size=AUTOTUNE)
    # ds_hr = ds
    # ds_lr = ds.map(lambda x: (tf.image.resize(x, (low_h, low_w))), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    #    .prefetch(buffer_size=AUTOTUNE)

    return ds  # , ds_lr
