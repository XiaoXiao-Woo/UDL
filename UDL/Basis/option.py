# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
import argparse
import platform
import warnings
import os
from UDL.UDL.Basis.config import Config

def common_cfg():

    script_path = os.path.dirname(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='PyTorch Training')
    # * Logger
    parser.add_argument('--use-log', default=True
                        , type=bool)
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--use-tb', default=False, type=bool)

    # * DDP
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', default=0, type=int,
                        help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
    parser.add_argument('--backend', default='nccl', type=str,  # gloo
                        help='distributed backend')
    parser.add_argument('--dist-url', default='env://',
                        type=str,  # 'tcp://224.66.41.62:23456'
                        help='url used to set up distributed training')
    # * AMP
    parser.add_argument('--amp', default=None, type=bool,
                        help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')

    # * Training
    parser.add_argument('--accumulated-step', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm')

    # * extra
    parser.add_argument('--seed', default=10, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--metrics', type=str, default='min',
                        choices=['min', 'max'],
                        help='maximum/minimum value of RGB')
    parser.add_argument('--reg', type=bool, default=True,
                        help='loss with l2 reguliarization for nn.Connv2D, '
                             'which is very important for classical panshrapening!!! ')


    parser.add_argument('--crop_batch_size', type=int, default=128,
                        help='input batch size for-'
                             ' training')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')



    args = parser.parse_args()
    args.once_epoch = False
    args.reset_lr = False
    args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
    assert args.accumulated_step > 0
    if args.metrics == 'min':
        args.best_prec1 = 10000
        args.best_prec5 = 10000
    else:
        args.best_prec1 = 0
        args.best_prec5 = 0

    args.load_model_strict = True

    return parser, args

def panshaprening_cfg():

    parser, args = common_cfg()

    args.scale = [1]
    if platform.system() == 'Linux':
        args.data_dir = './dataset/pansharpening'
    if platform.system() == "Windows":
        args.data_dir = 'D:/Datasets/pansharpening'

    args.best_prec1 = 10000
    args.best_prec5 = 10000
    args.metrics = 'min'
    args.task = "pansharpening"
    args.save_fmt = "mat" # fmt is mat or not mat

    return args
