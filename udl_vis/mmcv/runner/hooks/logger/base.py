# Copyright (c) OpenMMLab. All rights reserved.
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

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

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = by_epoch

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
        if runner.mode == 'train':
            mode = 'train'
            # if 'time' in runner.log_buffer.meters: #output
            #     mode = 'train'
            # else:
            #     mode = 'val'
        elif runner.mode == 'val':
            mode = 'val'
        elif runner.mode == 'test':
            mode = 'test'
        else:
            raise ValueError(f"runner mode should be 'train' or 'val' or 'test', "
                             f'but got {runner.mode}')
        return mode

    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode != 'train':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            if runner.mode == 'train':
                current_iter = runner.iter + 1
            else:
                current_iter = runner.iter

        return current_iter

    def get_lr_tags(self, runner):
        tags = {}
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                tags[f'learning_rate/{name}'] = value[0]
        else:
            tags['learning_rate'] = lrs[0]
        return tags

    def get_momentum_tags(self, runner):
        tags = {}
        momentums = self.current_momentum(runner)
        if isinstance(momentums, dict):
            for name, value in momentums.items():
                tags[f'momentum/{name}'] = value[0]
        else:
            tags['momentum'] = momentums[0]
        return tags

    def get_loggable_tags(self,
                          runner,
                          allow_scalar=True,
                          allow_text=False,
                          add_mode=True,
                          tags_to_skip=('time', 'data_time', 'learning_rate', 'pan2ms', 'grad_norm', 'lr', 'memory')):
        tags = {}
        for var, val in runner.metrics.items():#log_buffer.output
            if var in tags_to_skip:
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            if isinstance(val, np.ndarray):
                print("")
            if add_mode:
                var = f'{self.get_mode(runner)}/{var}'
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
        if self.by_epoch and (self.every_n_inner_iters(runner, self.interval) or self.end_of_n_inner_iters(runner)):
            # runner.log_buffer.ready = True
            self.log(runner)
            # if self.reset_flag:
            #     runner.log_buffer.clear_output()


    def after_train_epoch(self, runner):
        # if runner.log_buffer.ready:
        if self.every_n_epochs(runner, self.interval) or self.is_last_epoch(runner):
            # self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        # runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_val_iter(self, runner):
        self.log(runner)