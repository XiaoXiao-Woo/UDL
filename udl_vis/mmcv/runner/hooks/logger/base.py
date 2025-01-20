# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import time
from ..hook import Hook


class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations). Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default True.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        epoch_interval=10,
        iter_interval=10,
        #  show_iter=False, # epoch_based_runner achieve iter_based_runner
        ignore_last=True,
        reset_flag=False,
        runner_mode="epoch",
    ):
        self.epoch_interval = epoch_interval
        self.iter_interval = iter_interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = True if runner_mode == "epoch" else False
        self.epoch = 1  # Note that: keep runner.epoch consistent
        # self.show_iter = show_iter

    @abstractmethod
    def log(self, runner):
        pass

    @staticmethod
    def is_scalar(val, include_np=True, include_torch=True):
        """Tell the input variable is a scalar or not.

        Args:
            val: Input variable.
            include_np (bool): Whether include 0-d np.ndarray as a scalar.
            include_torch (bool): Whether include 0-d torch.Tensor as a scalar.

        Returns:
            bool: True or False.
        """
        if isinstance(val, numbers.Number):
            return True
        elif include_np and isinstance(val, np.ndarray) and val.ndim == 0:
            return True
        elif include_torch and isinstance(val, torch.Tensor) and len(val) == 1:
            return True
        else:
            return False

    def get_mode(self, runner):
        if runner.mode == "train":
            mode = "train"
            # if 'time' in runner.log_buffer.meters: #output
            #     mode = 'train'
            # else:
            #     mode = 'val'
        elif runner.mode == "val":
            mode = "val"
        elif runner.mode == "test":
            mode = "test"
        else:
            raise ValueError(
                f"runner mode should be 'train' or 'val' or 'test', "
                f"but got {runner.mode}"
            )
        return mode

    def get_epoch(self, runner):
        if runner.mode == "train":
            epoch = runner.epoch + 1
        elif runner.mode != "train":
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(
                f"runner mode should be 'train' or 'val', " f"but got {runner.mode}"
            )
        return epoch

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if inner_iter:  # self.by_epoch and
            current_iter = runner.inner_iter + 1
        else:
            if runner.mode == "train":
                current_iter = runner.iter + 1
            else:
                current_iter = runner.iter

        return current_iter

    def get_lr_tags(self, runner):
        tags = {}
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                tags[f"learning_rate/{name}"] = value[0]
        else:
            tags["learning_rate"] = lrs[0]
        return tags

    def get_momentum_tags(self, runner):
        tags = {}
        momentums = self.current_momentum(runner)
        if isinstance(momentums, dict):
            for name, value in momentums.items():
                tags[f"momentum/{name}"] = value[0]
        else:
            tags["momentum"] = momentums[0]
        return tags

    def get_loggable_tags(
        self,
        runner,
        allow_scalar=True,
        allow_text=False,
        add_mode=True,
        tags_to_skip=(
            "time",
            "data_time",
            "learning_rate",
            "pan2ms",
            "grad_norm",
            "lr",
            "memory",
        ),
    ):
        tags = {}
        for var, val in self.metrics.items():  # log_buffer.output
            if var in tags_to_skip:
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            if isinstance(val, np.ndarray):
                print("")
            if add_mode:
                var = f"{self.get_mode(runner)}/{var}"
            tags[var] = val
        tags.update(self.get_lr_tags(runner))
        tags.update(self.get_momentum_tags(runner))
        return tags

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        # if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
        #     runner.log_buffer.average(self.interval)
        # elif not self.by_epoch and self.every_n_iters(runner, self.interval):
        #     runner.log_buffer.average(self.interval)
        # elif self.end_of_epoch(runner) and not self.ignore_last:
        #     # not precise but more stable
        #     runner.log_buffer.average(self.interval)

        # if runner.log_buffer.ready:
        #     self.log(runner)
        #     if self.reset_flag:
        #         runner.log_buffer.clear_output()
        # print(self.end_of_n_inner_iters(runner))
        # self.metrics["epoch_time"]  = runner.epoch_time
        if self.by_epoch:
            if (
                self.every_n_inner_iters(runner, self.iter_interval)
                or self.end_of_n_inner_iters(runner)
            ) and self.every_n_epochs(
                runner, self.epoch_interval
            ):  # \
                if hasattr(runner, "accelerator"):
                    runner.accelerator.wait_for_everyone()
                if self.end_of_n_inner_iters(runner):
                    runner.log_buffer.synchronize_between_processes()
                self.metrics = {
                    k: meter.avg if not hasattr(meter, "image") else meter.image
                    for k, meter in runner.log_buffer.meters.items()
                }
                self.log(runner)

        elif (
            not self.by_epoch
            and self.every_n_iters(runner, self.iter_interval)
            or self.end_of_n_iters(runner)
            or self.end_of_n_inner_iters(runner)
        ):
            if hasattr(self, "accelerator"):
                runner.accelerator.wait_for_everyone()
            runner.log_buffer.synchronize_between_processes()
            if self.every_n_epochs(runner, self.epoch_interval) or self.every_n_iters(
                runner, self.iter_interval
            ):
                self.metrics = {
                    k: meter.avg if not hasattr(meter, "image") else meter.image
                    for k, meter in runner.log_buffer.meters.items()
                }

            self.log(runner)

    # after_train_iter:  -> ckpt after_train_epoch -> text after_train_epoch
    def after_train_epoch(self, runner):
        # if runner.log_buffer.ready:
        if self.every_n_epochs(runner, self.epoch_interval) or self.is_last_epoch(
            runner
        ):
            # self.log(ruznner)
            if self.is_last_epoch(runner):
                runner.eval_flag = True
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        # runner.log_buffer.average()
        if hasattr(runner, "accelerator"):
            runner.accelerator.wait_for_everyone()
        runner.log_buffer.synchronize_between_processes()
        self.metrics = {
            k: meter.avg if not hasattr(meter, "image") else meter.image
            for k, meter in runner.log_buffer.meters.items()
        }
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_val_iter(self, runner):
        self.metrics = {
            k: meter.val if not hasattr(meter, "image") else meter.image
            for k, meter in runner.log_buffer.meters.items()
        }
        self.log(runner)
