# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod

import torch
import numpy as np
from torch.optim import Optimizer

from udl_vis import mmcv
from ..parallel import is_module_wrapper
from .checkpoint import load_checkpoint, print_log
from .dist_utils import get_dist_info
from .hooks import HOOKS, Hook
from .log_buffer import LogBuffer
from .priority import Priority, get_priority
from .utils import get_time_str
from .record import MetricLogger


class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """

    def __init__(self,
                 model,
                 runner_mode="epoch",
                 batch_processor=None,
                 optimizer=None,
                 seed=None,
                 work_dir=None,
                 tfb_dir = None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None,
                 opt_cfg=None):
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                f'but got {type(batch_processor)}')
            warnings.warn(
                'batch_processor is deprecated, please implement '
                'train_step() and val_step() in the model instead.',
                DeprecationWarning)
            # raise an error is `batch_processor` is not None and
            # `model.train_step()` exists.
            if is_module_wrapper(model):
                _model = model.module
            else:
                _model = model
            if hasattr(_model, 'train_step') or hasattr(_model, 'val_step'):
                raise RuntimeError(
                    'batch_processor and model.train_step()/model.val_step() '
                    'cannot be both available.')
        # else:
        #     assert hasattr(model, 'train_step')

        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')

        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            warnings.warn(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')
        self.test_interval = 1
        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        self.opt_cfg = opt_cfg
        self.earlyStop = False
        self.seed = seed
        self.runner_mode=runner_mode
        self.by_epoch = True if runner_mode == "epoch" else False
        self.end_of_iter = False
        self.amp = torch.amp
        # self.ouf_of_epochs = ouf_of_epochs
        # create work_dir
        # save_dir = opt_cfg['save_dir']
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            self.save_dir = self.work_dir + "/results"
            if os.path.isdir(work_dir):
                mmcv.mkdir_or_exist(self.save_dir)
                # mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
            self.save_dir = None
        else:
            raise TypeError(f'"work_dir: {work_dir}" must be a str or None')
        if tfb_dir is not None:
            self.tfb_dir = tfb_dir



        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self.outputs = {}

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = MetricLogger(logger=logger, delimiter="  ")  # LogBuffer()

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    # @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, **kwargs):
        pass

    # @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def current_momentum(self):
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

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg):
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def get_hook_info(self):
        # Get hooks info in each stage
        stage_hook_map = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name
            except ValueError:
                priority = hook.priority
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def load_checkpoint(self,
                        filename,
                        resume_mode,
                        map_location='cpu',
                        prefix='',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):

        return load_checkpoint(
            resume_mode,
            os.path.dirname(filename) if os.path.isdir(os.path.dirname(filename)) else self.work_dir,
            self.model,
            filename,
            map_location,
            prefix,
            strict,
            self.logger,
            revise_keys=revise_keys)

    def resume(self,
               resume, resume_mode,
               reset_lr, lr, prefix, revise_keys,
               resume_optimizer=True,
               map_location='default'):

        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    resume, resume_mode,
                    map_location=lambda storage, loc: storage.cuda(device_id), prefix=prefix, revise_keys=revise_keys)
            else:
                checkpoint = self.load_checkpoint(resume, resume_mode, prefix=prefix, revise_keys=revise_keys)
        else:
            checkpoint = self.load_checkpoint(
                resume, resume_mode, map_location=map_location, prefix=prefix, revise_keys=revise_keys)

        # checkpoint.setdefault('meta', {'epoch': 0, 'iter': 0,
        #                                'state_dataloader': None})
        self._epoch = checkpoint['meta']['epoch']
        if self.opt_cfg['eval']:
            self._max_epochs = self._epoch + 1
        self._iter = checkpoint['meta']['iter']
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault('hook_msgs', {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        # Re-calculate the number of iterations when resuming
        # models with different number of GPUs
        # if 'config' in checkpoint['meta']:
        #     config = mmcv.Config.fromstring(
        #         checkpoint['meta']['config'], file_format='.py')
        #     previous_gpu_ids = config.get('gpu_ids', None)
        #     if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
        #             previous_gpu_ids) != self.world_size:
        #         self._iter = int(self._iter * len(previous_gpu_ids) /
        #                          self.world_size)
        #         print_log('the iteration number is changed due to '
        #                          'change of GPU number', logger=self.logger)

        # resume meta information meta
        self.meta = checkpoint['meta']
        # if optimizer is not None:
        #     if checkpoint.get('optimizer') is not None:
        #         optimizer.load_state_dict(checkpoint['optimizer'])
        #
        #     if lr > 0 and reset_lr:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr
        #     print_log("loaded checkpoint.optimizer")
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                # self.optimizer.param_groups[0]['params'][0].mean()
                # Out[14]: tensor(0.0024, device='cuda:0', grad_fn=<MeanBackward0>)
                # Out[15]: tensor(-0.0046, device='cuda:0', grad_fn=<MeanBackward0>)
                # checkpoint['optimizer']['param_groups'][0]['params'][0]
                # 0
                # param_sum = np.sum([param.sum().cpu().numpy() for _, v in
                #                     checkpoint['optimizer']['state'].items() for param in list(v.values())])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # new_opt_param_groups = np.sum([param.cpu().detach().numpy().sum() for param in
                #                     self.optimizer.param_groups[0]['params']])
                # new_opt_state = np.sum([param.cpu().detach().numpy().sum() for _, v in
                #                     self.optimizer.state.items()  for param in list(v.values())])

                # self.optimizer.param_groups.checkpoint['optimizer']['param_groups']
                if lr > 0 and reset_lr:
                    for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                # values = [np.mean(v['exp_avg'].cpu().numpy()) for v in self.optimizer.state_dict()['param_groups']] #.items()
                # print_log(f"loaded checkpoint.optimizer, opt={new_opt_param_groups}, {new_opt_state}", logger=self.logger) #
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
                if lr > 0 and reset_lr:
                    for param_group in self.optimizer[k].param_groups:
                            param_group['lr'] = lr
                print_log("loaded checkpoint.optimizer.", logger=self.logger)
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        print_log(f'resumed epoch {self.epoch}, iter {self.iter}', logger=self.logger)

        return checkpoint['meta']['state_dataloader']

    def register_lr_hook(self, lr_config):
        from torch.optim import lr_scheduler

        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `
            # CosineAnnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            # hook = mmcv.build_from_cfg(lr_config, HOOKS)
            # try:
            # hook = mmcv.build_from_cfg(lr_config, HOOKS)
            # except KeyError as e:
            #     # warns = f"{e}, Instead, load torch.optim"
            #     # warnings.warn(warns)
            #     # from udl_vis.Basis.optim import lr_scheduler
            #     # from torch import optim
            #     # hook = lr_scheduler(self.current_lr(), self.epoch)
            #     # scheduler = getattr(optim, policy_type)
            #     # hook.set_optimizer(self.optimizer, scheduler)
            #     return 0
        else:
            class_name = lr_config.__class__.__name__
            # lr_scheduler.StepLR(step_size=, gamma=)
            # lr_config.MultiStepLR(milestones=, gamma=)
            if "MultiStepLR" in class_name:
                lr_config = {"type": 'StepLrUpdaterHook',
                             "by_epoch": lr_config.by_epoch,
                             "step": list(lr_config.milestones),
                             "gamma": lr_config.gamma}
            elif "StepLR" in class_name:
                lr_config = {"type": 'StepLrUpdaterHook',
                             "by_epoch": lr_config.by_epoch if hasattr(lr_config, 'by_epoch') else True,
                             "step": lr_config.step_size,
                             "gamma": lr_config.gamma}
            elif "CosineAnnealingLR" in class_name:
                lr_config = {"type": 'CosineAnnealingLrUpdaterHook',
                             'min_lr': 0}


        hook = mmcv.build_from_cfg(lr_config, HOOKS)

        self.register_hook(hook, priority='VERY_HIGH')

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for
            # `CosineAnnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = mmcv.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook, priority='HIGH')

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        if self.by_epoch:
            self.register_hook(hook, priority='NORMAL')
        else:
            self.register_hook(hook, priority='VERY_LOW')

    def register_logger_hooks(self, log_config):
        if log_config is None:
            return
        # log_interval = log_config['interval']
        for info in log_config.pop('hooks'):
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=log_config)
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_timer_hook(self, timer_config):
        if timer_config is None:
            return
        if isinstance(timer_config, dict):
            timer_config_ = copy.deepcopy(timer_config)
            hook = mmcv.build_from_cfg(timer_config_, HOOKS)
        else:
            hook = timer_config
        self.register_hook(hook, priority='LOW')

    def register_custom_hooks(self, custom_config):
        if custom_config is None:
            return

        if not isinstance(custom_config, list):
            custom_config = [custom_config]

        for item in custom_config:
            if isinstance(item, dict):
                self.register_hook_from_cfg(item)
            else:
                self.register_hook(item, priority='NORMAL')

    def register_profiler_hook(self, profiler_config):
        if profiler_config is None:
            return
        if isinstance(profiler_config, dict):
            profiler_config.setdefault('type', 'ProfilerHook')
            hook = mmcv.build_from_cfg(profiler_config, HOOKS)
        else:
            hook = profiler_config
        self.register_hook(hook)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                timer_config=dict(type='IterTimerHook'),
                                custom_hooks_config=None):
        """Register default and custom hooks for training.

        Default and custom hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_timer_hook(timer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_logger_hooks(log_config)
        self.register_custom_hooks(custom_hooks_config)