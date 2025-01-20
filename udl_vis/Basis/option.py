# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
import argparse
import platform

# import warnings
import os
from udl_vis.Basis.python_sub_class import TaskDispatcher
from udl_vis.Basis.config import Config
from udl_vis.Basis.logger import print_log
import warnings

import ast
import json


def parse_nested_list(string):
    """解析嵌套列表的字符串形式."""
    try:
        if "[" in string and "]" in string:
            return ast.literal_eval(string)
        elif isinstance(string, str):
            return eval(string)
        else:
            string
    except (ValueError, SyntaxError):
        print(string, type(string))
        raise argparse.ArgumentTypeError(f"Invalid nested list format: {string}")


def parse_dict(string):
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid dictionary format. Use JSON format.")


def parse_bool(value):
    """将字符串转换为布尔值或 None。"""
    if value.lower() == "none":
        return None
    elif value.lower() == "false":
        return False
    elif value.lower() == "true":
        return True
    else:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")


def common_cfg():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # * Logger
    parser.add_argument("--use_log", default=True, type=bool)
    parser.add_argument(
        "--log-dir", metavar="DIR", default="logs", help="path to save log"
    )
    parser.add_argument(
        "--tfb-dir", metavar="DIR", default=None, help="useless in this script."
    )
    parser.add_argument("--use-tfb", default=False, type=bool)

    # * DDP
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "dp"],
        default="none",
        help="job launcher",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank",
    )
    parser.add_argument(
        "--backend", default="nccl", type=str, help="distributed backend"  # gloo
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,  # 'tcp://224.66.41.62:23456'
        help="url used to set up distributed training",
    )

    parser.add_argument(
        "--use_ds", default=False, type=str, help="deepspeed support from accelerate"
    )

    # * AMP
    # parser.add_argument('--amp', default=None, type=bool,
    #                     help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
    # parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
    #                     help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument(
        "--fp16_cfg",
        default=dict(),
        type=dict,
        help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=[None, "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    # * Training
    parser.add_argument("--accumulated-step", default=1, type=int)
    parser.add_argument(
        "--grad_clip_norm", type=float, default=None, help="dataset file extension"
    )
    parser.add_argument(
        "--grad_clip_value", type=float, default=None, help="dataset file extension"
    )

    # * extra
    parser.add_argument(
        "--seed", default=10, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--reg",
        type=bool,
        default=True,
        help="loss with l2 reguliarization for nn.Conv2D, "
        "which is very important for classical panshrapening!!! ",
    )

    parser.add_argument(
        "--crop_batch_size",
        type=int,
        default=128,
        help="input batch size for-" " training",
    )
    parser.add_argument(
        "--rgb_range", type=int, default=255, help="maximum value of RGB"
    )
    parser.add_argument(
        "--model_style",
        type=str,
        default=None,
        help="model_style is used to recursive/cascade or GAN training",
    )
    parser.add_argument("--mode", type=str, default=None, help="dataset file extension")
    parser.add_argument("--task", type=str, default=None, help="dataset file extension")
    parser.add_argument("--arch", type=str, default="", help="dataset file extension")

    args = parser.parse_args(args=[])
    args.global_rank = 0
    args.once_epoch = False
    args.reset_lr = False
    # args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
    args.save_top_k = 5
    args.start_epoch = 1
    assert args.accumulated_step > 0
    args.load_model_strict = True
    args.resume_mode = "best"
    args.validate = False
    args.gpu_ids = [0]
    args.prefix_model = ""
    args.use_colorlog = True
    args.use_save = True
    args.test = ""
    args.code_dir = ""
    args.start_save_epoch = 1
    args.earlyStopping = True
    args.flag_fast_train = True
    args.runner_mode = "epoch"
    args.dist_params = dict(backend="nccl")
    args.img_range = 1.0
    args.metrics = "loss"
    args.save_fmt = "png"
    args.task = "none"
    # args.ouf_of_epochs = 200
    args.start_save_best_epoch = 200
    args.revise_keys = [(r"^module\.", "")]
    args.precision = 5
    # args.workflow = []

    return Config(args)


def nni_cfg(args):
    if args.mode == "nni":
        import nni

        tuner_params = nni.get_next_parameter()
        print_log("launcher: nni is running. \n", tuner_paramsm)
        args.merge_from_dict(tuner_params)
    return args


class get_cfg(TaskDispatcher, name="entrypoint"):
    def __init__(self, task=None, arch=None, **kwargs):
        super(get_cfg, self).__init__()

        args = common_cfg()
        # args.mode = 'nni'
        if arch is not None:
            args.arch = arch
        if args.mode == "nni":
            args = nni_cfg(args)
        # args.__delattr__('workflow')

        if hasattr(args, "task"):
            cfg = TaskDispatcher.new(cfg=args, task=task, arch=args.arch, **kwargs)
            cfg.merge_from_dict(args)
        elif task in TaskDispatcher._task.keys():
            cfg = TaskDispatcher.new(cfg=args, task=task, arch=args.arch, **kwargs)
            cfg.merge_from_dict(args)
        else:
            raise ValueError(
                f"mode starter don't have task={task} but expected"
                f"one of {super()._task.keys()} in TaskDispatcher"
            )
        # cfg.setdefault('workflow', [])
        cfg = data_cfg(cfg)
        # print(cfg.pretty_text)

        self.merge_from_dict(cfg)


def data_cfg(cfg):
    # if cfg.get('config', None) is not None and os.path.isfile(cfg.config):
    #     print_log(f"reading {cfg.config}")
    #     cfg.merge_from_dict(cfg.fromfile(cfg.config))
    # else:
    #     print_log(f"reading {cfg.config} failed")

    if cfg.get("data", None) is not None and callable(cfg.data):
        data_func = cfg.pop("data")
        cfg.merge_from_dict(Config(data_func(cfg.data_dir)))

    cfg.workflow = cfg.get("workflow", [])
    if cfg.get("norm_cfg", None) is not None and cfg.launcher == "none":
        cfg.norm_cfg = "BN"

    # modify loading COCO from extern
    # if hasattr(cfg, 'data'):
    #     cfg.data.train['ann_file'] = cfg.data.train['ann_file'].replace('data', cfg.data_dir)
    #     cfg.data.train['img_prefix'] = cfg.data.train['img_prefix'].replace('data', cfg.data_dir)
    #     cfg.data.val['ann_file'] = cfg.data.val['ann_file'].replace('data', cfg.data_dir)
    #     cfg.data.val['img_prefix'] = cfg.data.val['img_prefix'].replace('data', cfg.data_dir)
    #     cfg.samples_per_gpu = cfg.data.samples_per_gpu
    #     cfg.workers_per_gpu = cfg.data.workers_per_gpu

    return cfg
