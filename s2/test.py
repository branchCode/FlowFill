import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import cv2
from glob import glob
from PIL import Image
from tqdm import tqdm
import random
import argparse
from models.bat_model import GPT, GPTConfig
from models.networks.generator import UpsamplerGenerator
from models.modules.flow_network import ConditionalFlow

os.environ['CUDA_VISIABLE_DEVICE'] = '1'

parser = argparse.ArgumentParser(description='PyTorch Template')

parser.add_argument('--num_sample', type=int, default=3, help='input batch size')
parser.add_argument('--up_model', type=str, default='celebahq_up', help='name of upsampler model')
parser.add_argument('--input_dir', type=str,
                    help='dir of input images, png foramt is hardcoded in line 121, please modify if needed.')
parser.add_argument('--mask_dir', type=str, help='dir of masks, filename should match with input images')
parser.add_argument('--save_dir', type=str, default='results/', help='dir for saving results')

parser.add_argument('--flow_path', type=str, default="/home/zym/FlowFill/s1/experiments/Inpaint_celeba_hq_32_L4K10_bk10_d50000/models/latest_G.pth", help='flow path')
parser.add_argument('--L', type=int, default=4, help='L flow blocks of flow model.')
parser.add_argument('--K', type=int, default=10, help='K flow steps of each flow blocks.')
parser.add_argument('--heat', type=float, default=1.0, help='sampling heat')
parser.add_argument('--mask_id', type=int, default=4, help='mask type')
parser.add_argument('--ksize', type=int, default=10, help='the kernel size of blur operation')

args = parser.parse_args()


def imread_torch(path, mask_dir, size=256, mask_index=None):
    img = Image.open(path)
    if size == 32:
        img = Image.fromarray(cv2.blur(np.asarray(img), (args.ksize, args.ksize)))
    img = torch.from_numpy(np.array(img.resize([size, size], Image.BICUBIC)))
    img = img.permute(2, 0, 1)
    if size == 32:
        img = img / 255.
    else:
        img = img / 127.5 - 1.
    
    if mask_index is None:
        if args.mask_id > 4:
            mask_index = random.randint(2 * 2000, 6 * 2000 - 1)
        else:
            mask_index = random.randint(args.mask_id * 2000, (args.mask_id + 2) * 2000 - 1)
    mask = np.array(Image.open(mask_dir[mask_index]).resize([size, size], Image.NEAREST))[:, :, None]
    mask = torch.from_numpy(mask)
    mask = (mask > 0).float()
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    img, mask = img.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0)
    masked_img = img * (1 - mask) + mask
    return img.cuda(), mask.cuda(), masked_img.cuda(), mask_index


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gaussian_batch(dims):
    return torch.randn(tuple(dims)).cuda()
    
cluster = torch.from_numpy(np.load('./kmeans_centers.npy')).cuda()

def color_quantize(x):
        # x (3,32,32)
        xpt = x.float().permute(1,2,0).contiguous().view(-1, 3)
        ix = ((xpt[:, None, :] - cluster[None, :, :])**2).sum(-1).argmin(1)
        return ix
        
def dequantize(ix, size=32):
    return cluster[ix].view(size, size, 3).permute(2, 0, 1).contiguous()


if __name__ == "__main__":
    
    num_sample = args.num_sample
    up_model = args.up_model
    input_dir = args.input_dir
    mask_dir = args.mask_dir
    save_dir = args.save_dir

    flow = ConditionalFlow(in_channels=3, L=args.L, K=args.K, n_trans=0, heat=args.heat, size=32)
    zshape = flow.out_shapes()
    weights = torch.load(args.flow_path, map_location='cpu')
    flow.load_state_dict(weights)
    flow = flow.cuda()
    flow.eval()


    class Options():
        netG = 'Upsampler'
        ngf = 64
        norm_G = 'spectralspadeposition3x3'
        resnet_n_blocks = 6
        use_attention = True
        input_nc = 4
        gpu_ids = [0]
        semantic_nc = 4


    save_path = './checkpoints/{}/latest_net_G.pth'.format(up_model)
    opt = Options()
    netG = UpsamplerGenerator(opt)
    weights = torch.load(save_path, map_location='cpu')
    netG.load_state_dict(weights)
    netG = netG.cuda()
    netG.eval()
    create_dir(save_dir)
    img_paths = sorted(glob(input_dir + '/*/*.jpg'))
    mask_paths = sorted(glob(mask_dir + '/*.png'))
    # test 100 for quick eval
    for p in tqdm(img_paths):
        img, mask, masked_img, mask_index = imread_torch(p, mask_paths, 256)
        _, mask_32, img_32, _ = imread_torch(p, mask_paths, 32, mask_index)
        
        Image.fromarray((((masked_img[0] + 1.) / 2.).cpu().permute(1, 2, 0).detach().numpy() * 255.).astype(np.uint8)).save(
                os.path.join(save_dir, p.split('/')[-1].replace('.jpg', '_masked.png')))

        masked_img = torch.cat([masked_img, mask], 1)
        for i in range(num_sample):
        
            z = gaussian_batch([1] + zshape) * args.heat
            img_prior, _ = flow(z, imgs=img_32, masks=mask_32, rev=True)
            img_prior = torch.clamp(img_prior, 0, 1.)
            img_prior = img_32 * (1 - mask_32) + img_prior * mask_32
            sample_tensor = img_prior * 2. - 1.
            sample_tensor = dequantize(color_quantize(sample_tensor[0])).unsqueeze(0)

            _, sample_up = netG([masked_img, sample_tensor])
            sample_up = sample_up * mask + masked_img[:, :3] * (1 - mask)
            sample_up = sample_up[0].cpu().permute(1, 2, 0).detach().numpy()
            sample_up = ((sample_up + 1) * 127.5).astype(np.uint8)
            Image.fromarray((((sample_tensor[0] + 1.) / 2.).cpu().permute(1, 2, 0).detach().numpy() * 255.).astype(np.uint8)).save(
                os.path.join(save_dir, p.split('/')[-1].replace('.jpg', '_{}_prior.png'.format(i))))
            Image.fromarray(sample_up).save(
                os.path.join(save_dir, p.split('/')[-1].replace('.jpg', '_{}.png'.format(i))))