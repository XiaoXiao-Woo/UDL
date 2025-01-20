# Copyright (c) OpenMMLab. All rights reserved.
import time

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_epoch(self, runner):
        self.t = time.time()
        self.epoch_time = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update(data_time=time.time() - self.t)#{'data_time': time.time() - self.t}

    def after_iter(self, runner):
        runner.log_buffer.update(time=time.time() - self.t)#{'time': time.time() - self.t}
        self.t = time.time()
        
    def after_epoch(self, runner):
        runner.epoch_time = time.time() - self.epoch_time
