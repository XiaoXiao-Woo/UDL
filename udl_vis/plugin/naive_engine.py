if __name__ == "__main__":
    from rich.traceback import install

    install()

import torch
from udl_vis.Basis.dist_utils import master_only
from udl_vis.plugin.mmcv1_engine import (
    set_random_seed,
    get_data_loader,
    print_log,
)
from udl_vis.mmcv.runner import (
    MetricLogger,
    load_checkpoint,
    find_latest_checkpoint,
    save_checkpoint,
)

import time
from functools import partial
import os
import platform
import shutil
import numpy as np
from collections import OrderedDict
import datetime
from pancollection.common.data import DummySession
from udl_vis.plugin.engine_utils import parse_dispatcher
from udl_vis.Basis.optim.optimizer import Optimizer
import ipdb
import inspect
from udl_vis.plugin.base import run_engine


class NaiveEngine:
    def __init__(self, cfg, task_model, build_model, logger):
        super().__init__()

        self.logger = logger
        self.cfg = cfg
        self.device = cfg.device

        try:
            self.model, criterion, self._optimizer, scheduler = build_model(
                device=self.device
            )(cfg)
        except Exception as e:
            parameter = inspect.signature(build_model)
            print(parameter)
            raise e

        self.device = cfg.device

        self.optimizer_wrapper = Optimizer(
            self.model,
            self._optimizer,
            cfg.distributed,
            fp16=cfg.fp16,
            fp_scaler=cfg.fp_scaler,
            grad_clip_norm=cfg.grad_clip_norm,
            grad_clip_value=cfg.grad_clip_value,
            detect_anomalous_params=cfg.detect_anomalous_params,
            accelerator=None,
        )
        self.scheduler = scheduler
        self.task_model = task_model(cfg.device, self.model, criterion)

    def train_step(self, batch, **kwargs):
        with torch.amp.autocast(self.device):
            metrics = self.task_model.train_step(batch, **kwargs)

            self.model.zero_grad(set_to_none=True)
            self._optimizer.zero_grad(set_to_none=True)
            # self.ema_net._zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # print(f"loss is nan, skip {_skip_n} batch(es) in this epoch")

        return metrics

    def val_step(self, batch, **kwargs):
        return self.task_model.val_step(batch, **kwargs)

    def test_step(self, batch, **kwargs):
        return self.task_model.val_step(batch, **kwargs)

    def after_train_epoch(self, loss, epoch):
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step(epoch)

    def before_run(self):
        # EMA
        # load checkpoint
        if self.cfg.resume_from is not None:
            self.state_dataloader = self.resume()

    @master_only
    def save_ckpt(
        self,
        epoch,
        _iter,
        filename_tmpl="model_{}.pth",
        save_optimizer=True,
        meta=None,
        create_symlink=True,
    ):

        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"meta should be a dict or None, but got {type(meta)}")
        if meta is not None:
            meta.update(meta)

        meta.update(model=self.model, epoch=epoch, iter=_iter)

        filename = filename_tmpl.format(epoch)
        filepath = os.path.join(self.cfg.model_dir, filename)
        if save_optimizer:
            optimizer = self.optimizer_wrapper.optimizer
            log_str = "optimizer"
        else:
            optimizer = None
            log_str = ""

        meta.update(optimizer=optimizer)

        save_checkpoint(filepath, meta=meta)

        if create_symlink:
            dst_file = os.path.join(self.cfg.model_dir, "latest.pth")
            if platform.system() != "Windows":
                if os.path.lexists(dst_file):
                    os.remove(dst_file)
                os.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

        # FIXME: loss scaler
        print_log(
            f"Manual saved model, {log_str}, and RNG generator in {self.model_dir}",
            logger=self.logger,
        )

    def resume(
        self,
        resume_filename,
        resume_mode,
        reset_lr,
        lr,
        prefix,
        revise_keys,
        resume_optimizer=True,
        map_location="default",
        strict=False,
    ):
        cfg = self.cfg
        if map_location == "default":
            # if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(
                resume_mode,
                (
                    os.path.dirname(resume_filename)
                    if os.path.isdir(os.path.dirname(resume_filename))
                    else cfg.work_dir
                ),
                self.model,
                resume_filename,
                map_location=lambda storage, loc: storage.cuda(device_id),
                prefix=prefix,
                strict=strict,
                revise_keys=revise_keys,
            )
        # else:
        #     checkpoint = load_checkpoint(
        #         resume, resume_mode, prefix=prefix, revise_keys=revise_keys
        #     )
        else:
            checkpoint = load_checkpoint(
                resume_filename,
                resume_mode,
                self.model,
                map_location=map_location,
                prefix=prefix,
                revise_keys=revise_keys,
            )

        resume_epoch = checkpoint["meta"]["epoch"]
        if cfg.eval:
            cfg.max_epochs = resume_epoch + 1
        resume_iter = checkpoint["meta"]["iter"]

        meta.setdefault("hook_msgs", {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        meta["hook_msgs"].update(checkpoint["meta"].get("hook_msgs", {}))

        # resume meta information meta
        meta = checkpoint["meta"]

        optimizer = self.optimizer_wrapper.optimizer

        if "optimizer" in checkpoint and resume_optimizer:
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(checkpoint["optimizer"])
                if lr > 0 and reset_lr:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
            elif isinstance(optimizer, dict):
                for k in optimizer.keys():
                    optimizer[k].load_state_dict(checkpoint["optimizer"][k])
                if lr > 0 and reset_lr:
                    for param_group in optimizer[k].param_groups:
                        param_group["lr"] = lr
                print_log("loaded checkpoint.optimizer.", logger=self.logger)
            else:
                raise TypeError(
                    "Optimizer should be dict or torch.optim.Optimizer "
                    f"but got {type(optimizer)}"
                )

        print_log(
            f"resumed epoch {resume_epoch}, iter {resume_iter}", logger=self.logger
        )

        return checkpoint["meta"]["state_dataloader"]


def run_naive_engine(cfg, logger, task_model, build_model, getDataSession, **kwargs):

    cfg.model_dir = os.path.join(cfg.work_dir, "checkpoints")
    cfg.results_dir = os.path.join(cfg.work_dir, "results")
    cfg.summaries_dir = os.path.join(cfg.work_dir, "summaries")

    set_random_seed(cfg.seed)

    runner = NaiveEngine(cfg, task_model, build_model, logger)

    state_dataloader = runner.before_run()

    run_engine(cfg, runner, getDataSession, cfg.workflow, logger, state_dataloader)


def test_pansharpening():
    from omegaconf import OmegaConf, DictConfig
    from udl_vis.Basis.option import Config
    from pancollection import getDataSession
    from pancollection.models.FusionNet.model_fusionnet import build_fusionnet
    from pancollection.models.base_model import PanSharpeningModel

    # from udl_vis import trainer
    import hydra
    from hydra.core.hydra_config import HydraConfig
    import os
    from udl_vis.mmcv.utils import create_logger
    from torch import distributed as dist

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "23456"
    # os.environ["WORLD_SIZE"] = "2"
    # os.environ["RANK"] = "0"
    @hydra.main(
        config_path="/Data2/woo/NIPS/PanCollection/pancollection/configs",
        config_name="model",
    )
    def hydra_run(cfg: DictConfig):
        if isinstance(cfg, DictConfig):
            cfg = Config(OmegaConf.to_container(cfg, resolve=True))
            cfg.merge_from_dict(cfg.args)
            cfg.__delattr__("args")
            hydra_cfg = HydraConfig.get()
            cfg.work_dir = hydra_cfg.runtime.output_dir

        cfg.backend = "naive"
        cfg.workflow = [("train", 1), ("test", 1)]
        cfg.dataset_type = "Dummy"
        cfg.distributed = False
        cfg.plugins = []
        print(cfg.pretty_text)

        logger = create_logger(cfg, work_dir=cfg.work_dir)
        run_naive_engine(
            cfg, logger, PanSharpeningModel, build_fusionnet, getDataSession
        )

    return hydra_run()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    test_pansharpening()
