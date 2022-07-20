import os
import random
import numpy as np
import cv2

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def create_mask(width, height, mask_width=None, mask_height=None, x=None, y=None):
    if not mask_width:
        mask_width = width // 2
    if not mask_height:
        mask_height = height // 2
    mask = np.zeros((height, width, 1))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y: mask_y + mask_height, mask_x: mask_x + mask_width, :] = 1
    return mask


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None
    if dataroot is not None:
        assert os.path.isdir(dataroot), '{:s} is not a valid directory'.format(dataroot)
        paths = []
        for dirpath, _, fnames in sorted(os.walk(dataroot)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    paths.append(img_path)
        assert paths, '{:s} has no valid image file'.format(dataroot)
        paths = sorted(paths)
    return paths


def read_img(path, mode=1):
    img = cv2.imread(path, mode)
    # img = img.astype(np.float32) / 255.

    return img


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]