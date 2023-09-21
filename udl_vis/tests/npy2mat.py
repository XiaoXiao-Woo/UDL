# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/9/5 22:00
# @Author  : Xiao Wu
# @reference: 
#
import numpy as np
import scipy.io as sio


C = np.load(r"D:\Datasets\hisr\GF5-GF1\C.npy".replace('\\', '/'))
print(C.shape)  # (5, 5)
R = np.load(r"D:\Datasets\hisr\GF5-GF1\R.npy".replace('\\', '/'))
print(R.shape)  # (150, 4)
PAN = np.load(r"D:\Datasets\hisr\GF5-GF1\reg_pan.npy".replace('\\', '/'))
MSI = np.load(r"D:\Datasets\hisr\GF5-GF1\reg_msi.npy".replace('\\', '/'))
print(PAN.shape, MSI.shape)  # (2200, 2288, 4) (1100, 1144, 150)

PAN = np.load(r"D:\Datasets\hisr\GF5-GF1\pan.npy".replace('\\', '/'))
MSI = np.load(r"D:\Datasets\hisr\GF5-GF1\msi.npy".replace('\\', '/'))
print(PAN.shape, MSI.shape)  # (2200, 2288, 4) (1100, 1144, 150)

# sio.savemat(r"D:\Datasets\hisr\GF5-GF1\C.mat".replace('\\', '/'), {"blur": C})
# sio.savemat(r"D:\Datasets\hisr\GF5-GF1\R.mat".replace('\\', '/'), {"R": R})
# sio.savemat(r"D:\Datasets\hisr\GF5-GF1\reg_pan.mat".replace('\\', '/'), {"pan": PAN})
# sio.savemat(r"D:\Datasets\hisr\GF5-GF1\reg_msi.mat".replace('\\', '/'), {"msi": MSI})