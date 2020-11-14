from glob import glob
import numpy as np
from PIL import Image


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res+(3,)

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('./data/%s/*' % self.dataset_name)

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, c = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = np.resize(img, self.img_res)
            img_lr = np.resize(img, (low_h, low_w, 3))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.divide(imgs_hr, 127.5) - 1.
        imgs_lr = np.divide(imgs_lr, 127.5) - 1.

        return imgs_hr, imgs_lr

    def imread(self, path):
        im = Image.open(path).convert('RGB')
        im = np.asarray(im, dtype=np.float)
        return im
