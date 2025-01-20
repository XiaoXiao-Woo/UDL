import torch
import torch.nn as nn
import math
import numpy as np
import os
# TorchDEQ
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm
from pancollection.models.FusionNet.model_fusionnet import FusionNet
from udl_vis.Basis.module import PatchMergeModule
from udl_vis.Basis.criterion_metrics import SetCriterion
from udl_vis.Basis.metrics.cal_ssim import ssim
from torch import optim
import logging
from udl_vis.Basis.logger import print_log

import sys
sys.path.append(os.path.dirname(__file__))

from Unet import Unet

logger = logging.getLogger(__name__)

class Unfolding(nn.Module):
    def __init__(self, task, scheme, stages, ms_channel, hs_spectral, num_channel, factor, lambda1, lambda2, adaptation=None):
        super().__init__()

        getattr(self, f'init_{task}_recon')(ms_channel, hs_spectral, num_channel, factor, lambda1, lambda2)
        self.ms_channel = ms_channel
        self.hs_spectral = hs_spectral
        self.adaptation = adaptation
        self.task = task
        self.stages = stages
        self.scheme = scheme

        self.reset_parameters()

    def init_pansharpening_recon(self, ms_channel, hs_spectral, num_channel, factor, lambda1, lambda2):
        return self.init_mhif_recon(ms_channel, hs_spectral, num_channel, factor, lambda1, lambda2)

    def init_mhif_recon(self, ms_channel, hs_spectral, num_channel, factor, lambda1, lambda2):
        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))
        # self.lambda1 = torch.nn.Parameter(torch.tensor(1e-2))
        # self.lambda2 = torch.nn.Parameter(torch.tensor(1e-3))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        KERNEL_SIZE = 13  # set as MoGDCN
        # self.conv1 =  torch.nn.Conv2d(in_channels=num_spectral, out_channels=num_channel, kernel_size=3, stride=1, padding=3 // 2)
        # self.conv2 =  torch.nn.Conv2d(in_channels=num_channel, out_channels=num_spectral, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_downsample = torch.nn.Conv2d(
            in_channels=hs_spectral,
            out_channels=hs_spectral,
            kernel_size=KERNEL_SIZE,
            stride=factor,
            padding=KERNEL_SIZE // 2,
        )
        self.conv_upsample = torch.nn.ConvTranspose2d(
            in_channels=hs_spectral,
            out_channels=hs_spectral,
            kernel_size=KERNEL_SIZE,
            stride=factor,
            padding=KERNEL_SIZE // 2,
        )

        self.conv_topan = torch.nn.Conv2d(
            in_channels=hs_spectral,
            out_channels=ms_channel,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )
        self.conv_tolms = torch.nn.Conv2d(
            in_channels=ms_channel,
            out_channels=hs_spectral,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )

    def init_mff_recon(self, ms_channel, hs_spectral, num_channel, factor, lambda1, lambda2):
        self.delta_0 = torch.nn.Parameter(torch.tensor(0.1))
        self.eta_0 = torch.nn.Parameter(torch.tensor(0.9))
        # self.lambda1 = torch.nn.Parameter(torch.tensor(1e-2))
        # self.lambda2 = torch.nn.Parameter(torch.tensor(1e-3))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.conv1 = torch.nn.Conv2d(
            in_channels=ms_channel,
            out_channels=ms_channel,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=hs_spectral,
            out_channels=hs_spectral,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
        )

    def mff_recon(self, features, vi, vis, ir):  # z = pan, y2 = ms
        DELTA = self.delta_0
        ETA = self.eta_0
        LAMBDA1 = self.lambda1
        LAMBDA2 = self.lambda2

        err1 = self.conv1(vi - ir)
        err3 = self.conv2(vi - ir)

        out = (
            vi
            - DELTA * (LAMBDA1 * err1 + LAMBDA2 * err3)
            + DELTA * ETA * (features - vi)
        )
        return out

    def pansharpening_recon(self, features, lms, ms, pan):  # z = pan, y2 = ms
        return self.mhif_recon(features, lms, ms, pan)

    def mhif_recon(self, features, up, lrhsi, rgb):  # z = pan, y2 = ms
        DELTA = self.delta_0
        ETA = self.eta_0
        LAMBDA1 = self.lambda1
        LAMBDA2 = self.lambda2

        # y1: ms upsampled
        # y2: ms
        # z: pan
        sz = up.shape

        # upsample ms -> ms_d
        down = self.conv_downsample(up)
        # (ms_d - ms) -> upsample -> err of ms
        err1 = self.conv_upsample(down - lrhsi, output_size=sz)

        # ms upsampled -> to_pan -> pan_conv
        to_pan = self.conv_topan(up)
        # error of pan
        err_pan = rgb - to_pan
        # 1 band -> conv -> lms space
        err3 = self.conv_tolms(err_pan)
        # err3 = err3.reshape(sz)
        ################################################################
        # features - recon: 误差？ F范数？ x

        # DELTA: 0.1, LAMBDA1/2: ??, ETA: 0.9
        # DELTA: GD step size
        # LAMBDA 1, 2: error of lms/pan balance
        # ETA: unet(y1) - y1; change of info in unet
        out = (
            up
            - DELTA * (LAMBDA1 * err1 + LAMBDA2 * err3)
            + DELTA * ETA * (features - up)
        )
        ################################################################
        # out = (err3+err1)
        #
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def pansharpening_unfolding_step(self, x_now, lms, ms, pan, x_net):
        # x_now, lms, ms, pan, x_net
        # unet(lms) + residual
        # output = self.adaptation(lms)[0] + lms

        # for _ in range(self.stages):
        #     sr = getattr(self, f'{self.task}_recon')(x_now, lms, ms, pan)
        #     x_now = self.adaptation(sr, pan) + x_net
        sr = getattr(self, f'{self.task}_recon')(x_now, lms, ms, pan)
        x_now = self.adaptation(sr, pan) + x_net
        return x_now

    def mff_unfolding_step(self, x_now, lms, ms, pan, x_net):
        # x_now, lms, ms, pan, x_net=lms

        sr = getattr(self, f"{self.task}_recon")(x_now, lms, ms, pan)
        conv_out = self.adaptation(sr, pan) + x_net

        return conv_out

    def mhif_unfolding_step(self, x_now, lms, ms, pan, x_net):
        # x_now, lms, ms, pan, x_net

        sr = getattr(self, f"{self.task}_recon")(x_now, lms, ms, pan)
        conv_out = self.adaptation(sr, pan) + x_net

        return conv_out

    def forward_once(self, x_now, lms, ms, pan, x_net):
        sr = getattr(self, f"{self.task}_recon")(x_now, lms, ms, pan)
        conv_out = self.adaptation(sr, pan) + x_net
        return conv_out


# ==================================================================================
# X_net
#

# UnfoldingDEQ:
#   1. unet(lms)  ->   init x_now
#   2. deq for-loop  ->    update x_now
#       2.1 recon(feature, y1, y2, z)
#       2.2 unet(feature - x_net)


class UnfoldingAdaption(PatchMergeModule):

    def __init__(
        self,
        strategy,
        scheme, 
        stages,
        ms_channel,
        hs_channel,
        num_channel,
        factor,
        lambda1,
        lambda2,
        adaptation=None,
        args=None,
    ):
        super(UnfoldingAdaption, self).__init__(
            args.crop_batch_size,
            args.patch_size_list,
            args.scale)

        self.task = args.task
        self.strategy = strategy
        self.args = args

        self.stage = Unfolding(
            self.task,
            scheme,
            stages,
            ms_channel,
            hs_channel,
            num_channel,
            factor,
            lambda1,
            lambda2,
            adaptation=Unet(hs_channel, ms_channel)  # .requires_grad_(False),
        )

        # if scheme == "FIXED":
        #     adaptation.eval()
        #     adaptation.requires_grad_(False)
        #     self.adaptation = adaptation
        # elif scheme == "APT":
        #     self.stage.requires_grad_(False)
        #     self.stage.adaptation.requires_grad_(True)
        # elif scheme == "FPT":
        #     self.requires_grad_(True)
        # else:
        #     raise ValueError(f"Invalid scheme: {scheme}")

        self.adaptation = FusionNet(hs_channel, num_channel)
        state_dict = torch.load(
            "/home/dsq/nips/huggingface/PanCollection/wv3/FusionNet/FusionNet.pth.tar"
        )
        self.adaptation.load_state_dict(state_dict["state_dict"])
        # set to none grad and not to update the parameters
        self.adaptation.eval()
        self.adaptation.zero_grad(set_to_none=True)
        self.adaptation.requires_grad_(False)
        self.sr = nn.Parameter(torch.randn(1, 8, 64, 64), requires_grad=True)

        # print_log(
        #     f"[UnfoldingAdaption]: stages: {stages}, scheme: {scheme}, strategy: {self.strategy}",
        #     logger=logger,
        # )

        if self.strategy == "deq":
            # Only support dict/Namespace but not raise error
            self.deq = get_deq(dict(vars(args)["_cfg_dict"]) if not isinstance(args, argparse.Namespace) else args)
            apply_norm(self.stage)

    def _forward_implem(self, lms, ms, pan):
        if self.strategy == "deq":
            reset_norm(self.stage)
            x_net = self.adaptation(lms, pan)
            x_now = self.stage.adaptation(lms, pan) + lms
            def Unfolding_func(x_now):
                return self.stage.forward_once(x_now, lms, ms, pan, x_net)            
            x_out, info = self.deq(
                Unfolding_func, x_now, solver_kwargs={"tau": self.args.tau}
            )
            if self.training:
                return x_out[-1]
            else:
                x_out = x_out[-1]
                return x_out
        elif self.strategy == "unfolding":
            # x_now = torch.ones_like(lms)
            if self.adaptation is not None:
                x_now = self.stage.adaptation(lms, pan) + lms
                x_net = self.adaptation(lms, pan)
            else:
                x_now = torch.zeros_like(lms)
            x_out = getattr(self.stage, f"{self.task}_unfolding_step")(
                x_now, lms, ms, pan, x_net
            )

        return x_out

    def forward_pansharpening(self, data):
        return self._forward_implem(**data).clip(0, 1)

    def forward_mff(self, data):
        # data['x_net'] = (data['vi'] + data['ir']) / 2
        return self.forward_chop(data)

    def forward_mhif(self, data):
        return self._forward_implem(**data)

    def train_step(self, data, **kwargs):
        gt = data.pop('gt')
        sr =  getattr(self, f"forward_{self.task}")(data)
        loss_dicts = self.criterion(sr, gt)

        return loss_dicts

    def val_step(self, data, **kwargs):
        return getattr(self, f"forward_{self.task}")(data)

import ipdb
from pancollection.models.base_model import PanSharpeningModel
class build_pan(PanSharpeningModel, name='Unfolding'):
    def __call__(self, cfg):
        scheduler = None

        loss = nn.L1Loss(size_average=True)  ## Define the Loss function
        weight_dict = {'loss': 1, 'ssim_loss': 0.1}
        losses = {'loss': loss, 'ssim_loss': lambda x, y: 1 - ssim(x, y)}
        criterion = SetCriterion(losses, weight_dict)
        adaptation = FusionNet(cfg.hs_channel)  # .requires_grad_(False)
        if cfg.adaptation_path is not None:
            print("loading pretrained adaptation model: ", cfg.adaptation_path)
            adaptation.load_state_dict(torch.load(cfg.adaptation_path, map_location='cpu')['state_dict'])
        model = UnfoldingAdaption(cfg.strategy,
                                  cfg.scheme,
                                  cfg.stages,
                                  cfg.ms_channel, 
                                  cfg.hs_channel,
                                  cfg.num_channel,
                                  cfg.factor,
                                  cfg.lambda1,
                                  cfg.lambda2,
                                  adaptation=adaptation,
                                  args=cfg)
        model.criterion = criterion
        # optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-6)  ## optimizer 1: Adam

        from udl_vis.Basis.optim.SOAP import SOAP
        optimizer = SOAP(
            model.parameters(), lr=cfg.lr, betas=(0.95, 0.99), weight_decay=1e-5
        )

        from udl_vis.Basis.optim.scheduler import CosineAnnealingWarmRestartsReduce

        scheduler = CosineAnnealingWarmRestartsReduce(
            optimizer, T_0=20, T_mult=2, lr_mult=0.7, eta_min=8e-5, warmup_epochs=0
        )

        return model, criterion, optimizer, scheduler


from msif.models.base_model import MSFusionModel
class build_MSFusion(MSFusionModel, name='Unfolding'):
    def __call__(self, cfg):
        scheduler = None

        loss = nn.L1Loss(size_average=True)  ## Define the Loss function
        weight_dict = {'loss': 1, 'ssim_loss': 0.1}
        losses = {'loss': loss, 'ssim_loss': lambda x, y: 1 - ssim(x, y)}
        criterion = SetCriterion(losses, weight_dict)
        adaptation = Unet(cfg.hs_channel, cfg.ms_channel)  # .requires_grad_(False)
        model = UnfoldingAdaption(
            cfg.strategy,
            cfg.scheme,
            cfg.stages,
            cfg.ms_channel,
            cfg.hs_channel,
            cfg.num_channel,
            cfg.factor,
            cfg.lambda1,
            cfg.lambda2,
            adaptation=adaptation,
            args=cfg,
        )
        model.criterion = criterion
        # optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-6)  ## optimizer 1: Adam

        from udl_vis.Basis.optim.SOAP import SOAP

        optimizer = SOAP(
            model.parameters(), lr=cfg.lr, betas=(0.95, 0.99), weight_decay=1e-5
        )

        from udl_vis.Basis.optim.scheduler import CosineAnnealingWarmRestartsReduce

        scheduler = CosineAnnealingWarmRestartsReduce(
            optimizer, T_0=20, T_mult=2, lr_mult=0.7, eta_min=8e-5, warmup_epochs=0
        )

        return model, criterion, optimizer, scheduler


from hisr.models.base_model import HISRModel
class build_MHIF(HISRModel, name="Unfolding"):
    def __call__(self, cfg):
        scheduler = None

        loss = nn.L1Loss(size_average=True)  ## Define the Loss function
        weight_dict = {"loss": 1, "ssim_loss": 0.1}
        losses = {"loss": loss, "ssim_loss": lambda x, y: 1 - ssim(x, y)}
        criterion = SetCriterion(losses, weight_dict)
        adaptation = Unet(cfg.hs_channel, cfg.ms_channel)  # .requires_grad_(False)
        model = UnfoldingAdaption(
            cfg.strategy,
            cfg.scheme,
            cfg.stages,
            cfg.ms_channel,
            cfg.hs_channel,
            cfg.num_channel,
            cfg.factor,
            cfg.lambda1,
            cfg.lambda2,
            adaptation=adaptation,
            args=cfg,
        )
        model.criterion = criterion
        
        
        # optimizer = optim.AdamW(
        #     model.parameters(), lr=cfg.lr, weight_decay=1e-6
        # )  ## optimizer 1: Adam

        from udl_vis.Basis.optim.SOAP import SOAP
        optimizer = SOAP(
        model.parameters(), lr=cfg.lr, betas=(0.95, 0.99), weight_decay=1e-5
        )

        from udl_vis.Basis.optim.scheduler import CosineAnnealingWarmRestartsReduce
        scheduler = CosineAnnealingWarmRestartsReduce(
            optimizer, T_0=20, T_mult=2, lr_mult=0.7, eta_min=8e-5, warmup_epochs=0
        )

        return model, criterion, optimizer, scheduler


def find_leanable_parameters(model, logger=None):
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                print_log(f"* {name} has not grad", logger=logger)
            else:
                print_log(f"- {name} has grad shape as {p.grad.shape}", logger=logger)


if __name__ == "__main__":
    from accelerate import Accelerator, DataLoaderConfiguration
    from pancollection.common.psdata import PansharpeningSession

    accelerator = Accelerator(
            # device_placement=True,  # according to CUDA_VISIBLE_DEVICES, it is invalid when using DDP
            cpu=False,
            mixed_precision="no",
            project_dir=os.path.dirname(__file__),
            dataloader_config=DataLoaderConfiguration(
                use_stateful_dataloader=True,
                use_seedable_sampler=True,
                non_blocking=True,
                split_batches=False,  # True means that samplers_per_gpu
                even_batches=True,
            ),
        )

    scale = 4
    data = {
        'lms': torch.randn(1, 8, 64 * scale, 64 * scale).to(accelerator.device),
        'pan': torch.randn(1, 1, 64 * scale, 64 * scale).to(accelerator.device),
        'ms': torch.randn(1, 8, 16 * scale, 16 * scale).to(accelerator.device),
    }

    import argparse
    from omegaconf import OmegaConf

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--tau", default=0.5, type=float)
    args = args_parser.parse_args()
    args.crop_batch_size = 32
    args.patch_size_list =  {"lms": 64, "pan": 64, "ms": 16, "x_net": 64}
    args.scale = 1
    args.task = "pansharpening"
    args.workers_per_gpu = 8
    args.samples_per_gpu = 32
    args.seed = 10
    args.dataset_type = "PanCollection"
    cfg = OmegaConf.load("/home/dsq/nips/work/configs/unfolding/datasets_node02.yaml")
    adaptation_path = "/home/dsq/nips/huggingface/PanCollection/wv3/FusionNet/FusionNet.pth.tar"
    args.dataset = cfg.dataset
    args.img_range = 2047.0
    # adaptation = FusionNet(8)  # .requires_grad_(False)
    # print(adaptation.state_dict().keys())
    # print(torch.load(adaptation_path)['state_dict'].keys())
    # adaptation.load_state_dict(torch.load(adaptation_path)['state_dict'])
    # adaptation.load_state_dict()
    # adaptation = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)

    model = UnfoldingAdaption(
        strategy="deq",
        scheme="FPT",
        stages=1,  # scheme: fully parameter fine-tuning (FPT), FIXED, APT
        hs_channel=8,
        ms_channel=1,
        num_channel=32,
        factor=4,
        lambda1=1,
        lambda2=1,
        # adaptation=adaptation,
        args=args,
    )

    model = accelerator.prepare(model)
    model = model.to(accelerator.device)
    x_net = torch.randn(1, 8, 64, 64).to(accelerator.device)
    import time

    t1 = time.time()

    # for _ in range(100):
    #     y = model.forward_pansharpening(data)
    #     # print(y[0].shape)
    #     y[0].mean().backward()

    # print(f"100 iters use: {time.time() - t1}")


    # 包装您的前向过程

    # unfolding: 0.13
    # Engine: 0.3
    train_loader, _, _ = PansharpeningSession(args).get_dataloader("wv3", distributed=False, state_dataloader=None)
    for data in train_loader:
        data = {k: v.to(accelerator.device) for k, v in data.items()}
        model_time = time.time()
        gt = data.pop('gt')
        y = model.forward_pansharpening(data)
        print(f"model_time: {time.time() - model_time}")
        y[0].mean().backward()
