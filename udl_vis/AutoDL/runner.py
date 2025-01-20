# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
import torch
from udl_vis.mmcv.runner import Hook_v2, hooks, clip_grads, detect_anomalous_parameters
from udl_vis.mmcv.runner.epoch_based_runner import EpochBasedRunner
# simplified hooks + base_runner + epoch_based_runner + trainer

class Runner(EpochBasedRunner, Hook_v2):

    def run(self, data_loaders, workflow, **kwargs):

        cfg = kwargs['cfg']
        hook_dict = dict(
            ModelCheckpoint=dict(type=hooks.ModelCheckpoint, priority='NORMAL',
                                 indicator='loss', save_top_k=cfg.save_top_k,
                                 use_save=cfg.use_save, save_interval=cfg.save_interval, earlyStopping=cfg.earlyStopping,
                                 start_save_epoch=cfg.start_save_epoch, flag_fast_train=cfg.flag_fast_train),
            TextLogger=dict(type=hooks.TextLoggerHook, epoch_interval=cfg.log_epoch_interval,
                            iter_interval=cfg.log_iter_interval, priority='VERY_LOW'),
            IterTimer=dict(type=hooks.IterTimerHook, priority='LOW')
        )
        if cfg.use_tfb:
            hook_dict['TensorboardLogger'] = dict(type='TensorboardLoggerHook')
        if cfg.mode == 'nni':
            hook_dict['NNIHook'] = dict(type='NNIHook', priority='very_low')
        # hook_dict.update(hook)

        super(Runner, self).init_hook(hook_dict)
        super(Runner, self).run(data_loaders, workflow)

    def run_optimizer(self):

        self.optimizer.zero_grad()
        if self.detect_anomalous_params:
            detect_anomalous_parameters(self.model, self.outputs['loss'], self.logger)
        self.outputs['loss'].backward()
        if not hasattr(self.model, 'train'):
            grad_norm = clip_grads(self.grad_clip, self.model.model.parameters())
        else:
            grad_norm = clip_grads(self.grad_clip, self.model.parameters())

        self.log_buffer.update_dict({'grad_norm': float(grad_norm)})
        self.optimizer.step()


    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass