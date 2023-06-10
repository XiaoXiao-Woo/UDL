# GPLv3 License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Author  : Xiao Wu
# @reference:

import torch
from torch import nn
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import numpy as np
import psutil
import os
import h5py
from UDL.Basis.auxiliary import show_memory_info
from UDL.derain.common.derain_dataset import derainSession as DataSession


def make_H5Dataset_unfold(file, loader, shape, group=128, mode="none", suffix='.h5'):

    batch_size = loader.batch_size
    height, width, channel = shape[1:]

    x = np.ones(shape=[1, channel, height, width]) * -1
    print("init:", x.shape)
    y = np.ones(shape=[1, channel, height, width]) * -1
    print("init:", y.shape)
    with h5py.File(file + suffix, 'w') as f:
        f.create_dataset('img', data=np.array(x, dtype=np.float32), maxshape=(None, channel, height, width),
                         # chunks=(100*batch_size, 1, 28, 28),
                         dtype='float32')
        f.create_dataset('gt',  data=np.array(y, dtype=np.float32), maxshape=(None, channel, height, width),
                         # chunks=(100*batch_size,),
                        )
    show_memory_info('prepare OK')
    h5f = h5py.File(file+suffix, 'a')
    show_memory_info('h5f')
    count = 0
    n_loader = len(loader)
    for j in range(1):
        for i, batch in enumerate(loader, 1):
            # B, C, H, W -> B, C*k*k, N
            # B*N, C, k, k
            (O, B) = batch['O'], batch['B']

            O = F.unfold(O, (height, width), stride=(height, width)).view(batch_size, channel, height, width, -1).permute(0, 4, 1, 2, 3).view(-1, channel, height, width)
            B = F.unfold(B, (height, width), stride=(height, width)).view(batch_size, channel, height, width, -1).permute(0, 4, 1, 2, 3).view(-1, channel, height, width)
            N = O.shape[0]
            print("iter: {}, img batch {} N: {}".format(i, O.shape, N))
            count += N
            h5f['img'].resize([count, channel, height, width])
            h5f['gt'].resize([count, channel, height, width])
            show_memory_info(0)
            # h5f['img'][N * (i-1):i * N] = O
            # h5f['gt'][N * (i-1):i * N] = B
            h5f['img'][N * (j * n_loader + (i-1)):(i + j * n_loader) * N] = O
            h5f['gt'][N * (j * n_loader + (i-1)):(i + j * n_loader) * N] = B

    print(h5f['img'].shape)
    h5f.close()

def make_H5Dataset(file, loader, shape, group=128, mode="none", suffix='.h5'):

    batch_size = loader.batch_size
    height, width, channel = shape[1:]
    nums = 0
    x = np.ones(shape=[group * batch_size, channel, height, width]) * -1
    print("init:", x.shape)
    y = np.ones(shape=[group * batch_size, channel, height, width]) * -1
    print("init:", y.shape)
    N = len(loader)
    Z = N // group
    show_memory_info(0)
    with h5py.File(file + suffix, 'w') as f:
        f.create_dataset('img', data=np.array(x, dtype=np.float32), maxshape=(None, channel, height, width),
                         # chunks=(100*batch_size, 1, 28, 28),
                         dtype='float32')
        f.create_dataset('gt',  data=np.array(y, dtype=np.float32), maxshape=(None, channel, height, width),
                         # chunks=(100*batch_size,),
                        )
    show_memory_info('prepare OK')
    h5f = h5py.File(file+suffix, 'a')
    show_memory_info('h5f')
    count = -1
    n_loader = len(loader)
    for i, batch in enumerate(loader, 1):

        # (img, label) = batch
        (O, B) = batch['O'], batch['B']
        print("iter: {}, img batch {}".format(i, O.shape))
        if i < n_loader:# and nums + batch_size < len(val_loader) * batch_size: #不构成一批，会引入空内容
            x[nums:nums + batch_size, ...] = O
            y[nums:nums + batch_size, ...] = B
            nums += batch_size
        if i % group == 0:
            count = i // group
            print(count)
            # to_h5py(x, y, file + str(i // 1000) + suffix)
            # with h5py.File(file+suffix, 'a') as h5f:
            h5f['img'].resize([group * batch_size * count, channel, height, width])
            h5f['gt'].resize([group * batch_size * count, channel, height, width])
            show_memory_info(0)
            h5f['img'][group * batch_size * (count - 1):group * batch_size * count] = x
            h5f['gt'][group * batch_size * (count - 1):group * batch_size * count] = y
            show_memory_info(0)
            nums = 0
            print(h5f['img'].shape)
            continue

        if count == Z:
            if i % group == 0:
                tail_nums = N * batch_size - h5f['img'].shape[0]
                x = np.ones(shape=[tail_nums, channel, height, width]) * -1
                print(x.shape)
                y = np.ones(shape=[tail_nums, channel, height, width]) * -1
                print(y.shape)
            else:
                h5f['img'].resize([N * batch_size, channel, height, width])
                h5f['gt'].resize([N * batch_size, channel, height, width])
            nums = 0



    print(h5f['img'].shape)
    h5f.close()


class Pachify(nn.Module):

    def __init__(self, args):

        self.patch_size = args.patch_size

    def forward(self, x):

        x = F.unfold(x, self.patch_size, stride=self.patch_size)
        print(x.shape)

        return x

if __name__ == "__main__":

    print('==> Preparing data..')

    batch_size = 60

    from UDL.Basis.option import derain_cfg
    args = derain_cfg()
    args.samples_per_gpu = 1
    args.workers_per_gpu = 1
    args.patch_size = 128
    sess = DataSession(args)
    pachify = Pachify(args)
    # train_loader, _ = sess.get_dataloader("Rain200L", False)
    # val_loader, _ = sess.get_eval_dataloader("Rain200L", False)
    test_loader, _ = sess.get_eval_dataloader("test12", False)
    # print("train_set: [{}/{}]".format(len(train_loader), len(train_loader) * batch_size))
    # print("val_set: [{}/{}]".format(len(val_loader), len(val_loader) * batch_size))
    # train_loader, _ = sess.get_eval_dataloader("test12", False)

    show_memory_info(0)
    # make_H5Dataset_unfold("./test12_chunks", test_loader, [1, 64, 64, 3], group=len(test_loader), mode="unfold")
    # make_H5Dataset_unfold("./Rain200L_val_chunks", val_loader, [1,  args.patch_size, args.patch_size, 3], group=len(val_loader), mode="unfold")
    # make_H5Dataset_unfold("./Rain200L_train_chunks", train_loader, [1, args.patch_size, args.patch_size, 3],  group=len(train_loader))


    import scipy.io as sio

    for idx, batch in enumerate(test_loader):

        print(idx, batch['O'].shape, batch['B'].shape)
        sio.savemat("test12_chunks.mat", {'img': batch['O'].numpy().transpose(0, 2, 3, 1), 'gt': batch['B'].numpy().transpose(0, 2, 3, 1)})
        break

    # print(train_loader.batch_size)
    out = sio.loadmat("test12_chunks.mat")
    print(out['img'].shape, np.min(out['img']), np.max(out['img']))
    print(out['gt'].shape)