# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
import argparse
import copy
import os
import os.path as osp
import shutil
import warnings
import random
import numpy as np
import torch
import torch.distributed as dist
import time

# 1.14s
# from UDL.AutoDL import build_model, getDataSession, ModelDispatcher
from udl_vis.Basis.auxiliary import init_random_seed, set_random_seed
from udl_vis.mmcv.utils.logging import print_log, create_logger

# 1.5s
from udl_vis.mmcv.runner import init_dist, find_latest_checkpoint
from udl_vis.mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from udl_vis.mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    AcceleratorOptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    TextLoggerHook,
    build_runner,
    get_dist_info,
)


# 10s
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)
from torch import distributed as dist
from udl_vis.plugin.base import get_data_loader



def run_mmcv1_engine(
    cfg,
    logger,
    task_model,
    build_model,
    getDataSession,
    runner=None,
    distributed=False,
    meta=None,
    **kwargs,
):
    if hasattr(model, "init_weights"):
        model.init_weights()

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model.model = MMDistributedDataParallel(
            model.model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    elif cfg.launcher == "dp":
        from torch.nn.parallel import DataParallel

        model.model = DataParallel(model.model.cuda())  # , device_ids=cfg.gpu_ids)

    if runner is not None:

        def build_runner(runner_cfg, default_args):
            runner_cfg = copy.deepcopy(runner_cfg)
            runner_cfg.pop("type")
            default_args.update(runner_cfg)
            return runner(**default_args)

    else:
        from udl_vis.mmcv.runner import build_runner

        # build_runner = lambda *args, **kwargs: build_runner(*args, **kwargs)

    # 改到 build_model里，一次性设置，方便查找
    if cfg.get("optimizer", None) is not None:
        optimizer = build_optimizer(model.model.module, cfg.optimizer)

    # 兼容argparser和配置文件的
    if "runner" not in cfg:
        cfg.runner = {"type": "EpochBasedRunner", "max_epochs": cfg.epochs}  # argparser
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "epochs" in cfg and "max_iters" not in cfg.runner:
            cfg.runner["max_epochs"] = cfg.epochs
            # assert cfg.epochs == cfg.runner['max_epochs'], print(cfg.epochs, cfg.runner['max_epochs'])

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            runner_mode=cfg.runner_mode,
            optimizer=optimizer,
            seed=cfg.seed,
            work_dir=cfg.work_dir,  ###
            # tfb_dir = cfg.tfb_dir,
            logger=logger,
            meta=meta,
            # ouf_of_epochs=cfg.ouf_of_epochs,
            opt_cfg={
                "log_iter_interval": cfg.log_iter_interval,
                "log_epoch_interval": cfg.log_epoch_interval,
                "save_interval": cfg.save_interval,
                "accumulated_step": cfg.accumulated_step,
                "mixed_precision": cfg.mixed_precision,
                "grad_clip_norm": cfg.grad_clip_norm,
                "dataset": cfg.dataset,
                "img_range": cfg.img_range,
                "metrics": cfg.metrics,
                "save_fmt": cfg.save_fmt,
                "device": cfg.device,
                # 'mode': cfg.mode,
                "test": cfg.test,
                "eval": cfg.eval,
                # 'val_mode': cfg.valid_or_test, # 在base_runner的resume里用于设置测试最大轮数来评估训练好的模型
            },
        ),
    )

    # fp16 setting
    fp16_cfg = cfg.fp16_cfg
    optimizer_config = cfg.get(
        "optimizer_config",
        dict(grad_clip_norm=cfg.grad_clip_norm, grad_clip_value=cfg.grad_clip_value),
    )
    if cfg.mixed_precision == "fp16" and cfg.launcher != "accelerator":
        fp16_cfg.setdefault("loss_scale", "dynamic")
    assert isinstance(optimizer_config, dict)
    assert isinstance(fp16_cfg, dict)
    if fp16_cfg:
        optimizer_config = Fp16OptimizerHook(
            **optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif optimizer_config:
        optimizer_config = (
            AcceleratorOptimizerHook(**optimizer_config)
            if cfg.launcher == "accelerator"
            else OptimizerHook(**optimizer_config)
        )
    ############################################################
    # register training hooks
    ############################################################
    if cfg.get("config", None) is not None and os.path.isfile(cfg.config):
        """
        optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
        optimizer_config = dict(grad_clip=None)
        lr_config = dict(policy='step', step=[100, 150])
        checkpoint_config = dict(interval=1)
        log_config = dict(
            interval=100,
            hooks=[
                dict(type='TextLoggerHook'),
                # dict(type='TensorboardLoggerHook')
            ])
        """
        runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get("momentum_config", None),
            custom_hooks_config=cfg.get("custom_hooks", None),
        )

    elif (
        cfg.get("log_config", None) is None
        and len(cfg.workflow)
        and cfg.workflow[0][0] != "simple_train"
    ):
        # 提供time, data_time, memory等，并且用于mode里区别IterBasedRunner? 在train模式下提供了有无time的区别
        if cfg.mode == "nni":
            runner.register_custom_hooks({"type": "NNIHook", "priority": "very_low"})
        if scheduler is not None:
            runner.register_lr_hook(
                scheduler
            )  # dict(policy=scheduler.__class__.__name__[:-2]))
        if cfg.use_save:
            runner.register_checkpoint_hook(
                dict(
                    type="ModelCheckpoint",
                    indicator="loss",
                    save_top_k=cfg.save_top_k,
                    save_interval=cfg.save_interval,
                    earlyStopping=cfg.earlyStopping,
                    start_save_epoch=cfg.start_save_epoch,
                    flag_fast_train=cfg.flag_fast_train,
                    start_save_best_epoch=cfg.start_save_best_epoch,
                )
            )
        runner.register_optimizer_hook(optimizer_config)  # ExternOptimizer
        runner.register_timer_hook(dict(type="IterTimerHook"))
        log_config = [dict(type="TextLoggerHook")]
        if cfg.use_tfb and cfg.tfb_dir is not None:
            log_config.append(dict(type="TensorboardLoggerHook", log_dir=cfg.tfb_dir))
        runner.register_logger_hooks(
            dict(
                epoch_interval=cfg.log_epoch_interval,
                iter_interval=cfg.log_iter_interval,
                runner_mode=cfg.runner_mode,
                hooks=log_config,
                precision=cfg.precision,
            )
        )

    else:
        runner.register_checkpoint_hook(dict(type="ModelCheckpoint", indicator="loss"))

    ############################################################
    # 载入模型
    ############################################################
    resume_from = None
    if cfg.get("resume_from", None) is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    # if cfg.get('resume_from', None):
    state_dataloader = runner.resume(
        cfg.resume_from,
        cfg.resume_mode,
        cfg.reset_lr,
        cfg.lr,
        cfg.prefix_model,
        cfg.revise_keys,
    )
    if cfg.get("load_from", None) and cfg.get("resume_from", None) is not None:
        runner.load_checkpoint(cfg.load_from, cfg.resume_mode)


    
    cfg, data_loaders, generator = get_data_loader(cfg, getDataSession, state_dataloader)
    runner.generator = generator

    ############################################################
    # 载入数据，运行模型
    ############################################################
    # print(inspect.getfile(model.model.__class__).split(cfg.arch)[0])
    if not isinstance(cfg.code_dir, list):
        cfg.code_dir = [cfg.code_dir]
    if cfg.work_dir is not None:
        os.makedirs("/".join([cfg.work_dir, "codes"]), exist_ok=True)
        for path in cfg.code_dir:
            if os.path.isfile(path):
                filename = "/".join([cfg.work_dir, "codes", os.path.basename(path)])
                shutil.copyfile(path, filename)
                print_log(f"copied {path} into {filename}", logger=logger)

    rank = int(os.environ.get("RANK", -1))
    if rank == 0 or rank == -1:
        print_log(cfg.pretty_text, logger=logger)

    try:
        runner.accelerator = model.accelerator
    except:
        runner.accelerator = None

    runner.run(data_loaders, cfg.workflow, cfg=cfg)
