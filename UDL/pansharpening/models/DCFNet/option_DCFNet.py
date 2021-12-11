# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
import argparse
from UDL.Basis.option import panshaprening_cfg, Config
import warnings
import os

cfg = Config(panshaprening_cfg())

script_path = os.path.dirname(os.path.dirname(__file__))

root_dir = script_path.split(cfg.task)[0]
model_path = f'{root_dir}/Weights/{cfg.task}/DCFNet/857.pth.tar'

parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                    help='path to save model')
# * Training
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr_scheduler', default=True, type=bool)
parser.add_argument('-samples_per_gpu', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--workers_per_gpu', default=0, type=int)
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# * Model and Dataset
parser.add_argument('--arch', '-a', metavar='ARCH', default='DCFNet', type=str,
                    choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet', 'DCFNet'])
parser.add_argument('--dataset', default='wv3_singleMat', type=str,
                    choices=[None, 'wv2', 'wv3', 'wv4', 'qb', 'gf',
                             'wv2_hp', ...,
                             'wv3_fr', 'wv3_singleMat', 'wv3_multi_exm1258', 'wv3_multi_exm78'],
                    help="performing evalution for patch2entire")
parser.add_argument('--eval', default=True, type=bool,
                    help="performing evalution for patch2entire")

args = parser.parse_args()
args.start_epoch = args.best_epoch = 1
args.experimental_desc = "Test"


cfg.merge_args2cfg(args)
print(cfg.pretty_text)


# * Importantly
warning = f"you are using {cfg.dataset}, note that FusionNet, DiCNN, PNN don't have high-pass filter"
warnings.warn(warning)

# print(cfg.auto_argparser()[1].pretty_text)