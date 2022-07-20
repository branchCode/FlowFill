import torch
import random
import numpy as np
import cv2
import torch.utils.data as data
import torchvision.transforms.functional as F
import data.util as util

from .degradation import prior_degradation

class InpaintDataset(data.Dataset):
    def __init__(self, opt):
        super(InpaintDataset, self).__init__()

        self.opt = opt
        self.phase = opt['phase']
        self.img_paths = util.get_image_paths(opt['img_dataroot'])
        self.mask_paths = util.get_image_paths(opt['mask_dataroot'])

        self.input_size = opt['img_size']

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = util.read_img(img_path)
        img = self.resize(img, 256, 256)
        img = cv2.blur(img, (self.opt['ksize'], self.opt['ksize']))
        img = self.resize(img, self.input_size, self.input_size)
        
        if self.mask_paths:
            # external mask
            if self.phase == 'train':
                if np.random.binomial(1, 0.5) == 1:
                    mask_index = random.randint(0, len(self.mask_paths) - 1)
                    mask = util.read_img(self.mask_paths[mask_index], mode=-1)
                    mask = self.resize(mask, self.input_size, self.input_size)
                    mask = (mask > 0) * 1
                else:
                    # mask_width = random.randint(self.input_size//8, self.input_size//8*7)
                    # mask_height = random.randint(self.input_size//8, self.input_size//8*7)
                    mask = util.create_mask(self.input_size, self.input_size)
            else:
                mask_type = self.opt['mask_type'] if self.opt['mask_type'] else 3
                mask_index = random.randint((mask_type - 1) * 2000, mask_type * 2000 - 1)
                # mask_index = index + 4000
                mask = util.read_img(self.mask_paths[mask_index], mode=-1)
                mask = self.resize(mask, self.input_size, self.input_size)
                mask = (mask > 0) * 1
        else:
            # regular mask
            mask_type = self.opt['mask_type'] if self.opt['mask_type'] else 0
            if mask_type == 0:
                # random  position
                mask = util.create_mask(self.input_size, self.input_size)
            else:
                # center hole
                mask = util.create_mask(self.input_size, self.input_size, x=self.input_size//4, y=self.input_size//4)

        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]

        if self.phase == 'train':
            img = util.augment([img], self.opt['use_flip'], self.opt['use_rot'])[0]
            mask = util.augment([mask], self.opt['use_flip'], self.opt['use_rot'])[0]

        # img = prior_degradation(img, self.clusters, self.input_size) / 255.
        # img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        # img = torch.from_numpy(np.array(img)).view(-1, 3).float()
        # img = ((img[:, None, :] - self.C[None, :, :]) ** 2).sum(-1).argmin(1)
        # img = self.C[img].view(self.input_size, self.input_size, 3).numpy()
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() / 255.
 
        mask = torch.from_numpy(np.ascontiguousarray(np.transpose(mask, (2, 0, 1)))).float()

        return {'img': img, 'mask': mask, 'path': img_path}


    def resize(self, img, h, w, center_crop=True):
        img_h, img_w = img.shape[:2]

        if img_h != img_w:
            side = np.minimum(img_h, img_w)
            j = random.randrange(0, img_h - side + 1)
            i = random.randrange(0, img_w - side + 1)
            img = img[j: j + side, i: i + side, ...]

        img = cv2.resize(img, [h, w])

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]

        return img