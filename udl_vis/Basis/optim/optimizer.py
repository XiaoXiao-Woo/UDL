# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
import logging
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
from torch import distributed as dist
from udl_vis.mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


def get_grad_norm(parameters, norm_type=2, grad_clip_value=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.requires_grad and p.grad is not None, parameters))
    norm_type = float(norm_type)
    if grad_clip_value is not None:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    else:
        total_norm = 0
    return parameters, total_norm

def clip_norm_(params, optimizer, grad_clip_norm, fp_scaler=None, accelerator=None):
    if fp_scaler:
        fp_scaler.unscale_(optimizer)
        
    if accelerator is None:
        return clip_grad.clip_grad_norm_(params, grad_clip_norm)
    else:
        if accelerator.sync_gradients:
            return accelerator.clip_grad_norm_(params, grad_clip_norm)


def clip_value_(params, optimizer, grad_clip_value, fp_scaler=None, accelerator=None):
    
    if fp_scaler:
        fp_scaler.unscale_(optimizer)
        
    if accelerator is None:
        clip_grad.clip_grad_value_(params, grad_clip_value)
    else:
        if accelerator.sync_gradients:
            accelerator.clip_grad_value_(params, grad_clip_value)

def clip_grad_(params, optimizer, grad_clip_norm, grad_clip_value, fp_scaler=None, accelerator=None):
    params, grad_norm = get_grad_norm(params, grad_clip_value)
    if grad_clip_norm > 0:
        return clip_norm_(params, optimizer, grad_clip_norm, fp_scaler, accelerator)
    elif grad_clip_value > 0:
        clip_value_(params, optimizer, grad_clip_value, fp_scaler, accelerator)
        return grad_norm
    else:
        return grad_norm


def cal_clip_grad(model, optimizer, grad_clip_norm, grad_clip_value, fp_scaler=None, accelerator=None):
    if not hasattr(model, 'train'):
        params = model.model.parameters()
        grad_norm = clip_grad_(params, optimizer, grad_clip_norm, grad_clip_value, fp_scaler, accelerator)
    else:
        params = model.parameters()
        grad_norm = clip_grad_(params, optimizer, grad_clip_norm, grad_clip_value, fp_scaler, accelerator)
    return {'grad_norm': float(grad_norm)}

def detect_anomalous_parameters(model, loss, logger):
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce gradients.

    Args:
        params (list[torch.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))

class Optimizer():
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A float to control the clip_grad.
            Default: None. (not a config dict)
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self, model, optimizer, distributed, fp16=None, fp_scaler=None, accelerator=None, grad_clip_norm=None, grad_clip_value=None, detect_anomalous_params=False):

        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.grad_clip_norm = 0 if grad_clip_norm is None else float(grad_clip_norm)
        self.grad_clip_value = 0 if grad_clip_value is None else float(grad_clip_value)
        self.detect_anomalous_params = detect_anomalous_params
        self.distributed = distributed

        if grad_clip_norm is not None and grad_clip_value is not None:
            raise ValueError(f"grad_clip_norm: {grad_clip_norm} and grad_clip_value: {grad_clip_value} can not be set together")

        if (fp16 and fp_scaler is None) or (not fp16 and fp_scaler):
            if accelerator is None:
                raise ValueError("fp16 and grad_scaler should be set together")
            else:
                fp16 = False

        self.loss_scaler = fp_scaler if fp16 else False
        if self.loss_scaler:
            if fp_scaler == 'dynamic':
                self.loss_scaler = GradScaler()
            elif isinstance(fp_scaler, float):
                self._scale_update_param = fp_scaler
                self.loss_scaler = GradScaler(init_scale=fp_scaler)
            elif isinstance(fp_scaler, dict):
                self.loss_scaler = GradScaler(**fp_scaler)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                    f'"dynamic", got {fp_scaler}')

    def prepare_step(self, outputs):

        has_overflow = False

        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce,
                            self.bucket_size_mb)

        if self.loss_scaler:
            has_overflow = self.loss_scaler.has_overflow(fp32_weights)

        return has_overflow

    def step(self, outputs):

        # if self.detect_anomalous_params:
        # self.detect_anomalous_parameters(runner.outputs['loss'], runner)

        if self.accelerator is None:
            outputs['loss'].backward()
        else:
            self.accelerator.backward(outputs['loss'])

        grad_norm = cal_clip_grad(self.model, self.optimizer, self.grad_clip_norm, 
                                  self.grad_clip_value, self.loss_scaler, self.accelerator)

        # short operator to pass gradient_accumulation_steps
        if (
            self.accelerator is not None and 
            self.accelerator.gradient_accumulation_steps == 1
        ):
            self.optimizer.step()

        self.optimizer.zero_grad()

        return grad_norm


class GradientCumulativeOptimizerHook:
    """Optimizer Hook implements multi-iters gradient cumulating.

    Args:
        cumulative_iters (int, optional): Num of gradient cumulative iters.
            The optimizer will step every `cumulative_iters` iters.
            Defaults to 1.

    Examples:
        >>> # Use cumulative_iters to simulate a large batch size
        >>> # It is helpful when the hardware cannot handle a large batch size.
        >>> loader = DataLoader(data, batch_size=64)
        >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
        >>> # almost equals to
        >>> loader = DataLoader(data, batch_size=256)
        >>> optim_hook = OptimizerHook()
    """

    def __init__(self, cumulative_iters=1, **kwargs):
        super(GradientCumulativeOptimizerHook, self).__init__(**kwargs)

        assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
            f'cumulative_iters only accepts positive int, but got ' \
            f'{type(cumulative_iters)} instead.'

        self.cumulative_iters = cumulative_iters
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.cumulative_iters != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by cumulative_iters in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters - runner.iter

        self.divisible_iters = (
            residual_iters // self.cumulative_iters * self.cumulative_iters)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def step(self, runner):
        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.cumulative_iters
        else:
            loss_factor = self.remainder_iters
        loss = runner.outputs['loss']
        loss = loss / loss_factor
        loss.backward()

        if (self.every_n_iters(runner, self.cumulative_iters)
                or self.is_last_iter(runner)):

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            optimizer.step()
            optimizer.zero_grad()
