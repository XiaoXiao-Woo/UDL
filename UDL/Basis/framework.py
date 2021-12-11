# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
import datetime
import time
import torch
import torch.distributed as dist
import shutil
import os
from .auxiliary import *
from torch.backends import cudnn
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from .dist_utils import dist_train_v1

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    print("Currently using torch.cuda.amp")
    try:
        from torch.cuda import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex or use pytorch1.6+.")


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss


class model_amp(nn.Module):
    def __init__(self, args, model, criterion, regularization=False):
        super(model_amp, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.reg = regularization
        if regularization:
            log_string("using l2_regularization for nn.Conv2D")

    def dist_train(self):
        self.model = dist_train_v1(self.args, self.model)

    def __call__(self, x, *args, **kwargs):
        if not self.args.amp or self.args.amp is None:
            output, loss = self.model.train_step(x)
            if self.reg:
                loss = self.l2_regularization(loss, self.model)
            else:
                loss['reg_loss'] = loss['Loss']
        else:
            # torch.amp optimization
            with amp.autocast():
                output, loss = self.model.train_step(x)
                # output = self.model(x)
                # loss = self.criterion(output, gt)
                if self.reg:
                    loss = self.l2_regularization(loss, self.model)
                else:
                    loss['reg_loss'] = loss['Loss']

        if hasattr(self.model, 'ddp_step'):
            self.model.ddp_step(loss)
        return output, loss

    def backward(self, optimizer, loss, scaler=None):
        if self.args.amp is not None:
            if not self.args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # optimizer.step()
                if self.args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.clip_max_norm)
            else:
                # torch.amp optimization
                scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
        else:
            loss.backward()
            # optimizer.step()

    def l2_regularization(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)
        if isinstance(criterion, dict):
            criterion['reg_loss'] = criterion['Loss'] + sum(regularizations)
        else:
            criterion = criterion + sum(regularizations)

        return criterion

    def apex_initialize(self, optimizer):

        scaler = None
        if self.args.amp is not None:
            cudnn.deterministic = False
            cudnn.benchmark = True
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

            if not self.args.amp:
                log_string("apex optimization")
                self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.amp_opt_level)
                # opt_level=args.opt_level,
                # keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                # loss_scale=args.loss_scale
                # )
            else:
                log_string("torch.amp optimization")
                scaler = amp.GradScaler()

        return optimizer, scaler


# def get_checkpoint_dir(_DIR_NAME = "checkpoints"):
#     """Retrieves the location for storing checkpoints."""
#     return os.path.join(cfg.OUT_DIR, _DIR_NAME)
#
# from iopath.common.file_io import g_pathmgr
#
# def get_last_checkpoint(_NAME_PREFIX = "ckpt_ep_"):
#     """Retrieves the most recent checkpoint (highest epoch number)."""
#     checkpoint_dir = get_checkpoint_dir()
#     checkpoints = [f for f in g_pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
#     last_checkpoint_name = sorted(checkpoints)[-1]
#     return os.path.join(checkpoint_dir, last_checkpoint_name)

# def reduce_mean(tensor, nprocs):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= nprocs
#     return rt

def save_checkpoint(state, model_save_dir, is_best, filename='amp_checkpoint.pth.tar'):
    filename = os.path.join(model_save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_dir, 'amp_model_best.pth.tar'))


def partial_load_checkpoint(base_wrap, state_dict, amp, eval, dismatch_list=[]):
    pretrained_dict = {}
    dismatch_list = ['agg_conv']

    if amp is not None:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            k = '.'.join(['amp', k])
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})
    else:

        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            # if args.eval is not None:
            #     k = k.split('.')
            #     k = '.'.join(k[1:])
            if base_wrap:
                if 'model' not in k:
                    k = '.'.join(['model', k])
                else:
                    k = k.split('.')
                    k = '.'.join(
                        [k_item for k_item in k if k_item != 'module' and k_item != 'ddp'])
            else:
                k = k.split('.')
                k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'model' and k_item != 'ddp'])




            # k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'model' and k_item != 'ddp'])  #
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})

    return pretrained_dict


def load_checkpoint(args, model, optimizer, ignore_params=[]):
    global_rank = args.global_rank
    checkpoint = {}
    if args.resume:
        if os.path.isfile(args.resume):
            if args.distributed:
                dist.barrier()
            init_checkpoint = torch.load(args.resume, map_location=f"cuda:{args.local_rank}")
            if init_checkpoint.get('state_dict') is None:
                checkpoint['state_dict'] = init_checkpoint
                del init_checkpoint
                torch.cuda.empty_cache()
            else:
                checkpoint = init_checkpoint
                print(checkpoint.keys())
            args.start_epoch = args.best_epoch = checkpoint.setdefault('epoch', 0) + 1
            args.best_epoch = checkpoint.setdefault('best_epoch', 0)
            args.best_prec1 = checkpoint.setdefault('best_metric', 0)
            if args.amp is not None:
                print(checkpoint.keys())
                try:
                    amp.load_state_dict(checkpoint['amp'])
                except:
                    Warning("no loading amp_state_dict")
            # if ignore_params is not None:
            # if checkpoint.get('state_dict') is not None:
            #     ckpt = partial_load_checkpoint(checkpoint['state_dict'], args.amp, ignore_params)
            # else:
            #     print(checkpoint.keys())
            #     ckpt = partial_load_checkpoint(checkpoint, args.amp, ignore_params)
            print(checkpoint['state_dict'].keys())
            base_wrap = "model" in list(model.state_dict().keys())[0]
            ckpt = partial_load_checkpoint(base_wrap, checkpoint['state_dict'], args.amp, args.eval, ignore_params)
            if args.distributed:
                model.module.load_state_dict(ckpt, strict=args.load_model_strict)  # , strict=False
            else:
                model.load_state_dict(ckpt, strict=args.load_model_strict)  # , strict=False
            if global_rank == 0:
                log_string(f"=> loading checkpoint '{args.resume}'\n ignored_params: \n{ignore_params}")
            # else:
            #     if global_rank == 0:
            #         log_string(f"=> loading checkpoint '{args.resume}'")
            #     if checkpoint.get('state_dict') is None:
            #         model.load_state_dict(checkpoint)
            #     else:
            #         model.load_state_dict(checkpoint['state_dict'])
            # print(checkpoint['state_dict'].keys())
            # print(model.state_dict().keys())
            if optimizer is not None:
                if checkpoint.get('optimizer') is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])

                lr = args.lr
                if args.lr > 0 and args.reset_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

            if global_rank == 0:
                log_string("=> loaded checkpoint '{}' (epoch {})"
                           .format(args.resume, checkpoint['epoch']))



            del checkpoint
            torch.cuda.empty_cache()

        else:
            if global_rank == 0:
                log_string("=> no checkpoint found at '{}'".format(args.resume))

        return model, optimizer


def show_maps(axes, O, B, outputs):
    pred = outputs[0, ...].cpu().detach().numpy().transpose(1, 2, 0)
    gt = B[0, ...].cpu().numpy().transpose(1, 2, 0)
    axes[0, 0].imshow(O[0, ...].cpu().numpy().transpose(1, 2, 0))
    axes[0, 1].imshow(pred)
    axes[1, 0].imshow(B[0, ...].cpu().numpy().transpose(1, 2, 0))
    axes[1, 1].imshow(np.abs(pred - gt))

    # pred = outputs[1, ...].cpu().detach().numpy().transpose(1, 2, 0)
    # gt = B[1, ...].cpu().numpy().transpose(1, 2, 0)
    # axes[1, 0].imshow(O[1, ...].cpu().numpy().transpose(1, 2, 0))
    # axes[1, 1].imshow(pred)
    # axes[1, 2].imshow(B[1, ...].cpu().numpy().transpose(1, 2, 0))
    # axes[1, 3].imshow(np.abs(pred - gt))


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
