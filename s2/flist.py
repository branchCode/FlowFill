
import argparse
import os
import glob
from random import shuffle


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default="/data/zym/celeba", type=str, help='The data folder path')
parser.add_argument('--save_root', default='datasets/flist/celeba_hq', type=str, help='The save path')


if __name__ == '__main__':
    args = parser.parse_args()

    train_flist = glob.glob(args.data_root + '/train/*/*.jpg')
    shuffle(train_flist)
    val_flist = glob.glob(args.data_root + '/val/*/*.jpg')
    test_flist = glob.glob(args.data_root + '/test/*/*.jpg')

    train_save_path = args.save_root + '/train.flist'
    val_save_path = args.save_root + '/valid.flist'
    test_save_path = args.save_root + '/test.flist'

    if not os.path.exists(train_save_path):
        os.mknod(train_save_path)

    if not os.path.exists(val_save_path):
        os.mknod(val_save_path)

    if not os.path.exists(test_save_path):
        os.mknod(test_save_path)

    fo = open(train_save_path, "w")
    fo.write("\n".join(train_flist))
    fo.close()

    fo = open(val_save_path, "w")
    fo.write("\n".join(val_flist))
    fo.close()

    fo = open(test_save_path, "w")
    fo.write("\n".join(test_flist))
    fo.close()