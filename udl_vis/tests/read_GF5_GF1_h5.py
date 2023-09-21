# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/9/5 23:43
# @Author  : Xiao Wu
# @reference: 
#
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

data = h5py.File("D:/Datasets/hisr/GF5-GF1/test_GF5_GF1.h5")
GT = np.asarray(data['GT']).transpose(0, 2, 3, 1)
HSI_up = np.asarray(data['HSI_up']).transpose(0, 2, 3, 1)
LRHSI = np.asarray(data['LRHSI']).transpose(0, 2, 3, 1)
RGB = np.asarray(data['RGB']).transpose(0, 2, 3, 1)

print(GT.shape, HSI_up.shape, LRHSI.shape, RGB.shape)
print(data.keys(), GT.max(), HSI_up.max(), LRHSI.max(), RGB.max())

# X = np.random.randn(135, 80, 80, 150)
# print(X[0, :, :, [1, 2, 3]].shape)
# X = torch.randn(135, 80, 80, 150)
# print(X[0, :, :, [1, 2, 3]].shape)

num = GT.shape[0]

# for i in range(num):
#     _, axes = plt.subplots(2, 2)
#     axes[0, 0].imshow(GT[i, :, :, [13, 37, 60]].transpose(1, 2, 0))
#     axes[0, 1].imshow(HSI_up[i, :, :, [13, 37, 60]].transpose(1, 2, 0))
#     axes[1, 0].imshow(LRHSI[i, :, :, [13, 37, 60]].transpose(1, 2, 0))
#     axes[1, 1].imshow(RGB[i, :, :, [0, 1, 2]].transpose(1, 2, 0))
#
#     plt.savefig(f"GF5-GF1/{i}.png")
#     plt.show()




