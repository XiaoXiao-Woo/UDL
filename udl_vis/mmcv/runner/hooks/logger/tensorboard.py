# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch
from udl_vis.mmcv.utils import TORCH_VERSION, digit_version
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook
import numpy as np


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(self,
                 log_dir=None,
                 epoch_interval=10,
                 iter_interval=10,
                 precisio=5,
                 ignore_last=True,
                 reset_flag=False,
                 runner_mode="epoch"):
        super(TensorboardLoggerHook, self).__init__(epoch_interval, iter_interval, ignore_last,
                                                    reset_flag, runner_mode)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        # self.log_dir = osp.join(runner.work_dir, 'tf_logs')  # runner.tfb_dir #
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif not isinstance(val, np.ndarray):
                self.writer.add_scalar(tag, val, self.get_iter(runner))

            if isinstance(val, np.ndarray):
                if runner.epoch % self.interval == 0:
                    self.writer.add_image(tag, val, dataformats="HWC", global_step=self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()

    def current_momentum(self, runner):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        optimizer = runner.optimizer
        if optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(optimizer)
        elif isinstance(optimizer, dict):
            momentums = dict()
            for name, optim in optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums
