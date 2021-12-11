# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.  
import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

    #####必要函数
    def __getitem__(self, index):
        return {'gt':self.gt[index, :, :, :].float(),
               'lms':self.lms[index, :, :, :].float(),
               'ms':self.ms[index, :, :, :].float(),
               'pan':self.pan[index, :, :, :].float()}

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
