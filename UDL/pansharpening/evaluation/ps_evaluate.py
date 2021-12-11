# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
import os
import datetime
import imageio
import numpy as np
import cv2
import h5py
import torch
import torch.nn.functional as F
from scipy import io as sio
from torch.utils.data import DataLoader, Dataset
from UDL.Basis.auxiliary import MetricLogger, SmoothedValue, set_random_seed, log_string
from UDL.Basis.dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
from UDL.pansharpening.common.evaluate import analysis_accu
from UDL.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
from UDL.Basis.zoom_image_region import show_region_images

# dmd
def load_gt_compared(file_path_gt, file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    try:
        gt = torch.from_numpy(data1['gt'] / 2047.0)
    except KeyError:
        print(data1.keys())
    compared_data = torch.from_numpy(data2['output_dmdnet_newdata6'] * 2047.0)
    return gt, compared_data


def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs


def load_dataset_singlemat_hp(file_path):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy(get_edge(data['ms'] / 2047)).permute(2, 0, 1).unsqueeze(dim=0)  # CxHxW= 8x64x64
    mms_hp = F.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
                        mode="bilinear", align_corners=True)
    pan_hp = torch.from_numpy(get_edge(data['pan'] / 2047))   # HxW = 256x256
    gt = torch.from_numpy(data['gt'] / 2047.0)

    return lms.squeeze().float(), mms_hp.squeeze().float(), ms_hp.squeeze().float(), pan_hp.float(), gt.float()


def load_dataset_singlemat(file_path):
    data = sio.loadmat(file_path)  # HxWxC
    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1).unsqueeze(dim=0)  # CxHxW= 8x64x64
    mms = F.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                        mode="bilinear", align_corners=True)
    pan = torch.from_numpy(data['pan'] / 2047.0)  # HxW = 256x256
    gt = torch.from_numpy(data['gt'] / 2047.0)

    return lms.squeeze().float(), mms.squeeze().float(), ms.squeeze().float(), pan.float(), gt.float()


def load_dataset_H5_hp(file_path, use_cuda=True):
    data = h5py.File(file_path)  # NxHxWxC

    # tensor type: NxCxHxW:

    lms = torch.from_numpy(data['lms'] / 2047)#.permute(0, 3, 1, 2)
    ms_hp = torch.from_numpy(get_edge(data['ms'] / 2047))#.permute(0, 3, 1, 2)  # NxCxHxW:
    mms_hp = torch.nn.functional.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
                                          mode="bilinear", align_corners=True)
    pan = data['pan'][:, :, :, np.newaxis]  # NxHxWxC (C=1)
    pan_hp = torch.from_numpy(get_edge(pan / 2047))#.permute(0, 3, 1, 2)  # Nx1xHxW:
    gt = torch.from_numpy(data['gt'][...]).cuda().float()

    return {'lms': lms,
            'mms:': mms_hp,
            'ms': ms_hp,
            'pan': pan_hp,
            'gt': gt.permute([0, 2, 3, 1])
            }

def load_dataset_H5(file_path, use_cuda=True):
    data = h5py.File(file_path)  # CxHxW
    print(data.keys())
    # tensor type:
    if use_cuda:
        lms = torch.from_numpy(data['lms'][...] / 2047.0).cuda().float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / 2047.0).cuda().float()  # CxHxW= 8x64x64
        mms = torch.nn.functional.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                                              mode="bilinear", align_corners=True)
        pan = torch.from_numpy(data['pan'][...] / 2047.0).cuda().float()  # HxW = 256x256

        gt = torch.from_numpy(data['gt'][...]).cuda().float()

    else:
        lms = torch.from_numpy(data['lms'][...] / 2047.0).float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / 2047.0).float()  # CxHxW= 8x64x64
        mms = torch.nn.functional.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                                              mode="bilinear", align_corners=True)
        pan = torch.from_numpy(data['pan'][...] / 2047.0).float()  # HxW = 256x256

        gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'mms:': mms,
            'ms': ms,
            'pan': pan,
            'gt': gt.permute([0, 2, 3, 1])
            }


class MultiExmTest_h5(Dataset):

    def __init__(self, file_path, dataset_name, suffix='.h5'):
        super(MultiExmTest_h5, self).__init__()

        # 一次性载入到内存
        if 'hp' not in dataset_name:
            data = load_dataset_H5(file_path, False)
        elif 'hp' in dataset_name:
            data = load_dataset_H5_hp(file_path, False)
        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError
        if suffix == '.mat':
            self.lms = data['lms'].permute(0, 3, 1, 2)  # CxHxW = 8x256x256
            self.ms = data['ms'].permute(0, 3, 1, 2)  # CxHxW= 8x64x64
            self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2) * 2, self.ms.size(3) * 2),
                                                       mode="bilinear", align_corners=True)
            self.pan = data['pan'].unsqueeze(1)
            self.gt = data['gt'].permute(0, 3, 1, 2)

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'mms': self.mms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...],
                'gt': self.gt[item, ...]
                }

    def __len__(self):
        return self.gt.shape[0]


class SingleDataset(Dataset):

    dataset = ["new_data10", "new_data11", "new_data12_512",
               "new_data3_wv2", "new_data4_wv2", "new_data5_wv2",
               "new_data6", "new_data7", "new_data8", "new_data9",
               "new_data_OrigScale3", "new_data_OrigScale4"
               ]

    def __init__(self, file_lists, dataset_name):
        self.file_lists = file_lists
        self.file_nums = len(file_lists)
        self.dataset = {}
        self.dataset_name = dataset_name
        if 'hp' not in dataset_name:
            self.dataset = load_dataset_singlemat
        elif 'hp' in dataset_name:
            self.dataset = load_dataset_singlemat_hp
        else:
            print(f"{dataset_name} is not supported in evalution")
            raise NotImplementedError

    def __getitem__(self, idx):
        file_path = self.file_lists[idx % self.file_nums]
        test_lms, test_mms, test_ms, test_pan, gt = self.dataset(file_path)

        if 'hp' not in self.dataset_name:
            return {'gt': (gt * 2047),
                    'lms': test_lms,
                    'mms': test_mms,
                    'ms': test_ms,
                    'pan': test_pan.unsqueeze(dim=0),
                    'filename': file_path}
        else:
            return {'gt': (gt * 2047),
                    'lms': test_lms,
                    'mms_hp': test_mms,
                    'ms_hp': test_ms,
                    'pan_hp': test_pan.unsqueeze(dim=0),
                    'filename': file_path}

    def __len__(self):
        return self.file_nums


def mpl_save_fig(filename):
    plt.savefig(f"{filename}", format='svg', dpi=300, pad_inches=0, bbox_inches='tight')


def save_results(idx, save_model_output, filename, save_fmt, output):
    if filename is None:
        save_name = os.path.join(f"{save_model_output}",
                                 "output_mulExm_{}.mat".format(idx))
        sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})
    else:
        filename = os.path.basename(filename).split('.')[0]
        if save_fmt != 'mat':
            output = showimage8(output)
            filename = '/'.join([save_model_output, filename + ".png"])
            # plt.imsave(filename, output, dpi=300)
            show_region_images(output, xywh=[50, 100, 50, 50], sub_width="20%", sub_height="20%",
                               sub_ax_anchor=(0, 0, 1, 1))
            mpl_save_fig(filename)
        else:
            filename = '/'.join([save_model_output, "output_" + filename + ".mat"])
            sio.savemat(filename, {'sr': output.cpu().detach().numpy()})



def eval_framework(args, eval_loader, model):
    metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ", window_size=len(eval_loader),
                                 eval=args.eval)
    header = 'TestEpoch: [{0}]'.format(args.start_epoch)
    save_dir = args.model_save_dir
    save_fmt = args.save_fmt
    # switch to evaluate mode
    model.eval()
    print_seq = np.ceil(len(eval_loader) / 100)
    for batch, idx in metric_logger.log_every(eval_loader, print_seq, header):
        filename = batch.get('filename')
        if filename is not None:
            filename = filename[0]
            log_string(filename)
        output, gt = model.eval_step(batch)
        result_our = torch.squeeze(output).permute(1, 2, 0)
        result_our = result_our * 2047
        metrics = analysis_accu(gt.cuda().squeeze(0), result_our, 4)
        # [val, index_cut] = find(PanNet2_Q8~=0&PanNet2_ERGAS<=10&PanNet2_SAM<=10);
        if idx not in [220, 231, 236, 469, 766, 914]:
            metric_logger.update_dict(metrics)
            if save_fmt is not None:
                save_results(idx, save_dir, filename, save_fmt, result_our)

    stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
    log_string("Averaged stats: {}".format(metric_logger))

    return stats  # stats
