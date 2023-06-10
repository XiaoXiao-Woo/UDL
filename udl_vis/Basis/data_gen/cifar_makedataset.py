# GPLv3 License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Author  : Xiao Wu
# @reference:
import torch
from torchvision import transforms
import torchvision
import numpy as np
import psutil
import os
import h5py
# from utils.utils import show_memory_info

batch_size = 256



def make_H5Dataset(file, loader,shape, group=128, suffix='.h5'):
    os.makedirs(os.path.dirname(file), exist_ok=True)

    #loader.batch_size
    height, width, channel = shape[1:]

    nums = 0
    x = np.ones(shape=[group * batch_size, channel, height, width]) * -1
    print(x.shape)
    y = np.ones(shape=[group * batch_size, ]) * -1
    print(y.shape)
    N = len(loader)
    Z = N // group
    # show_memory_info(0)
    with h5py.File(file + suffix, 'w') as f:
        f.create_dataset('img', data=np.array(x, dtype=np.float32), maxshape=(None, channel, height, width),
                         # chunks=(100*batch_size, 1, 28, 28),
                         dtype='float32')
        f.create_dataset('gt',  data=np.array(y, dtype=np.int8), maxshape=(None,),
                         # chunks=(100*batch_size,),
                        )
    # show_memory_info('prepare OK')
    h5f = h5py.File(file+suffix, 'a')
    # show_memory_info('h5f')
    count = 0
    for i, batch in enumerate(loader, 1):

        (img, label) = batch
        # print("iter: {}, img batch {}".format(i, img.shape[0]))
        if i < len(loader):# and nums + batch_size < len(val_loader) * batch_size: #不构成一批，会引入空内容
            x[nums:nums + batch_size, ...] = img
            y[nums:nums + batch_size, ...] = label
            nums += batch_size
        if i % group == 0:
            count = i // group
            print(count)
            # to_h5py(x, y, file + str(i // 1000) + suffix)
            # with h5py.File(file+suffix, 'a') as h5f:
            h5f['img'].resize([group * batch_size * count, channel, height, width])
            h5f['gt'].resize([group * batch_size * count, ])
            # show_memory_info(0)
            h5f['img'][group * batch_size * (count - 1):group * batch_size * count] = x
            h5f['gt'][group * batch_size * (count - 1):group * batch_size * count] = y
            # show_memory_info(0)
            nums = 0
            print(h5f['img'].shape)
            continue

        if count == Z:
            if i % group == 0:
                tail_nums = N * batch_size - h5f['img'].shape[0]
                x = np.ones(shape=[tail_nums, channel, height, width]) * -1
                print(x.shape)
                y = np.ones(shape=[tail_nums, ]) * -1
                print(y.shape)
            else:
                h5f['img'].resize([N * batch_size, channel, height, width])
                h5f['gt'].resize([N * batch_size, ])
            nums = 0



    print(h5f['img'].shape)
    h5f.close()





if __name__ == "__main__":

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='../00-raw', train=True, download=True, transform=transform_train)

    val_set = torchvision.datasets.CIFAR10(
        root='../00-raw', train=False, download=True, transform=transform_test)

    # train_set = torchvision.datasets.CIFAR100(
    #     root='../../00-raw', train=True, download=True, transform=transform_train)
    #
    # val_set = torchvision.datasets.CIFAR100(
    #     root='../../00-raw', train=False, download=True, transform=transform_test)

    print(len(train_set), len(val_set), )

    # 50000，10000取128bs扩充为50048 10112
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    print("train_set: [{}/{}]".format(len(train_loader), len(train_loader) * batch_size))
    print("val_set: [{}/{}]".format(len(val_loader), len(val_loader) * batch_size))

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # show_memory_info(0)
    #
    make_H5Dataset("../00-data/cifar10_val_chunks", val_loader, val_set.data.shape, group=32)
    make_H5Dataset("../00-data/cifar10_train_chunks", train_loader, train_set.data.shape, group=32)
    print(train_loader.batch_size)