# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
import datetime
from abc import ABCMeta, abstractmethod
import os
import time
from os import path as osp
import torch
from udl_vis.mmcv.runner import MetricLogger, get_dist_info, get_time_str, get_host_info, \
    detect_anomalous_parameters, clip_grads, load_checkpoint, get_priority, hooks
from udl_vis.mmcv.utils import print_log, is_str, mkdir_or_exist, Config


class Trainer(BaseRunner, Hook_v2):

    def __init__(self, cfg, logger,
                 model, optimizer, scheduler,
                 hook={},
                 meta=None):
        super(Trainer, self).__init__(cfg, logger, model,
                                      optimizer, scheduler, hook, meta)
        self.detect_anomalous_params = False

    def run_optimizer(self):
        self.grad_clip = 0
        self.optimizer.zero_grad()
        if self.detect_anomalous_params:
            detect_anomalous_parameters(self.model.model, self.outputs['loss'], self.logger)
        self.outputs['loss'].backward()
        if not hasattr(self.model, 'train'):
            grad_norm = clip_grads(self.grad_clip, self.model.model.parameters())
        else:
            grad_norm = clip_grads(self.grad_clip, self.model.parameters())

        self.log_buffer.update_dict({'grad_norm': float(grad_norm)})
        self.optimizer.step()

class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:
    - ``train()``
    - ``val()``

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

    def __init__(self, cfg, logger,
                 model, optimizer, scheduler,
                 hook={},
                 meta=None):

        # TODO: as the enum
        self.seed = cfg.seed
        self.max_epochs = cfg.epochs
        work_dir = cfg.work_dir
        if is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            self.save_dir = cfg.work_dir + "/results"
            if osp.isdir(cfg.work_dir):
                mkdir_or_exist(self.save_dir)
        elif work_dir is None:
            self.work_dir = None
            self.save_dir = None
        else:
            raise TypeError(f'"work_dir: {work_dir}" must be a str or None')

        self.earlyStop = False
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.meta = meta
        self.mode = None
        self.rank, self.world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.test_interval = 1
        self.hooks = []
        self.epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.outputs = {}
        self.logger = logger

        # New
        self.cfg = cfg
        self.grad_clip = cfg.grad_clip

        self.detect_anomalous_params = False

        self.log_buffer = MetricLogger(logger=logger, delimiter="  ")  # LogBuffer()


        hook_dict = dict(
            ModelCheckpoint=dict(type=hooks.ModelCheckpoint, priority='NORMAL',
                                 indicator='loss', save_top_k=cfg.save_top_k,
                                 use_save=cfg.use_save, save_interval=cfg.save_interval, earlyStopping=cfg.earlyStopping,
                                 start_save_epoch=cfg.start_save_epoch, flag_fast_train=cfg.flag_fast_train),
            TextLogger=dict(type=hooks.TextLoggerHook, interval=cfg.log_interval,
                            priority='VERY_LOW'),
            IterTimer=dict(type=hooks.IterTimerHook, priority='LOW')
        )
        if cfg.use_tfb:
            hook_dict['TensorboardLogger'] = dict(type='TensorboardLoggerHook')
        if cfg.mode == 'nni':
            hook_dict['NNIHook'] = dict(type='NNIHook', priority='very_low')
        hook_dict.update(hook)
        self.init_hook(hook_dict)


    def resume(self,
               filename, resume_mode,
               reset_lr, lr, prefix,
               resume_optimizer=True,
               map_location='default',
               strict=False,
               revise_keys=[(r'^module.', '')]
               ):

        work_dir = os.path.dirname(filename) if os.path.isdir(os.path.dirname(filename)) else self.cfg.work_dir
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = "cpu"
        checkpoint = load_checkpoint(resume_mode, work_dir, self.model, filename,
                                     map_location=map_location,
                                     prefix=prefix, strict=strict, logger=self.logger, revise_keys=revise_keys)

        self.epoch = checkpoint['meta']['epoch']
        if self.cfg['eval']:
            self.max_epochs = self.epoch + 1
        self.iter = checkpoint['meta']['iter']
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault('hook_msgs', {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        # Re-calculate the number of iterations when resuming
        # models with different number of GPUs
        if 'config' in checkpoint['meta']:
            config = Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
                    previous_gpu_ids) != self.world_size:
                self.iter = int(self.iter * len(previous_gpu_ids) /
                                self.world_size)
                print_log('the iteration number is changed due to '
                          'change of GPU number', logger=self.logger)

        # resume meta information meta
        self.meta = checkpoint['meta']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if lr > 0 and reset_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        print_log(f'resumed epoch {self.epoch}, iter {self.iter}', logger=self.logger)

        return checkpoint['meta']['state_dataloader']

    def run_iter(self, data_batch, train_mode, **kwargs):
        if train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                        **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update_dict(outputs['log_vars'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        cfg = self.cfg
        if hasattr(self.model, 'train'):
            self.model.train()
        else:
            self.model.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.max_iters = self.max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        tic = time.time()
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.run_optimizer()
            self.call_hook('after_train_iter')
            self.iter += 1

        self.metrics = {k: meter.avg for k, meter in self.log_buffer.meters.items()}
        self.metrics.update(epoch_time=time.time() - tic)
        self.call_hook('after_train_epoch')
        self.epoch += 1

    def val(self, data_loader, **kwargs):

        cfg = self.cfg
        kwargs['test_mode'] = False if not kwargs.get('test_mode', None) else True
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif isinstance(self.model.model, dict):
            for name in self.model.model.keys():
                self.model.model[name].eval()
        else:
            self.model.model.eval()
        self.mode = 'val' if not kwargs.get('test_mode', None) else 'test'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        tic = time.time()
        for i, data_batch in enumerate(self.data_loader):
            self.inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False, idx=i,
                          img_range=cfg['img_range'], eval=cfg['eval'], test=cfg['test'],
                          save_fmt=cfg['save_fmt'], filename=data_batch.get('filename', [None])[0],
                          save_dir=self.save_dir,
                          **kwargs)
            self.call_hook('after_val_iter')
            # break
        print_log(f"test time: {time.time() - tic}", logger=self.logger)
        self.call_hook('after_val_epoch')
        if cfg['eval'] or not self.eval_flag:
            self.epoch += 1
            self.eval_flag = True

        @torch.no_grad()
        def test(self, data_loader, **kwargs):
            return self.val(data_loader, test_mode=True, **kwargs)

    def run(self, data_loaders, workflow, **kwargs):
        self.eval_flag = train_flag = any('train' in mode for mode, _ in workflow)
        self.data_loaders = data_loaders
        self.data_length = {}
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            self.data_length[mode] = len(data_loaders[mode])
            if mode == "train":
                self.train_interval = epochs
            if mode == "test":
                self.test_interval = epochs
        if 'train' in data_loaders.keys():
            self.max_iters = self.max_epochs * len(data_loaders['train'])

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        print_log(f'Start running, host: {get_host_info()}, work_dir: {work_dir}',
                  logger=self.logger)
        print_log(f'Hooks will be executed in the following order:\n{self.get_hook_info()}',
                  logger=self.logger)
        print_log(f'workflow: {workflow}, max: {self.max_epochs} epochs',
                  logger=self.logger)
        self.call_hook('before_run')
        tic = time.time()
        while (self.epoch < self.max_epochs and train_flag) or not self.eval_flag:
            for i, flow in enumerate(workflow):
                mode, epochs = flow

                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for epoch in range(epochs):
                    if mode == 'train' and self.epoch >= self.max_epochs:
                        break
                    epoch_runner(data_loaders[mode], **kwargs)
            if self.earlyStop:
                print_log("model train has diverged, python will stop training", logger=self.logger)
                break
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        total_time = time.time() - tic
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_log('Training time {}'.format(total_time_str), logger=self.logger)

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
