import os
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from pancollection import build_model


def run_accelerate_engine(
    cfg,
    logger,
    task_model,
    build_model,
    getDataSession,
    callbacks=[],
    **kwargs,
):
    import sys

    sys.path.append("/home/dsq/nips/UDL/udl_vis/plugin")
    from Unfolding_Unet import UnfoldingAdaption

    # train_loader, _, _ = getDataSession(cfg).get_dataloader(
    #     "wv3", distributed=False, state_dataloader=None
    # )
    _iter = 0
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
    print(cfg.pretty_text)
    # model, optimizer, _, _ = build_model(cfg)
    model = UnfoldingAdaption(
        strategy="unfolding",
        scheme="FPT",
        stages=1,  # scheme: fully parameter fine-tuning (FPT), FIXED, APT
        hs_channel=8,
        ms_channel=1,
        num_channel=32,
        factor=4,
        lambda1=1,
        lambda2=1,
        args=cfg,
    ).to(accelerator.device)

    model = accelerator.prepare(model)
    import time

    tic = time.time()
    log_dict = {}
    # runner.before_train_epoch()
    scale = 1
    data = {
        "lms": torch.randn(32, 8, 64 * scale, 64 * scale).to(accelerator.device),
        "pan": torch.randn(32, 1, 64 * scale, 64 * scale).to(accelerator.device),
        "ms": torch.randn(32, 8, 16 * scale, 16 * scale).to(accelerator.device),
    }
    # print(model)

    for _ in range(1000):
        # for _inner_iter, batch in enumerate(train_loader):
        # runner.before_train_iter()
        # data = {k: v.to(accelerator.device) for k, v in data.items()}
        log_dict["data_time"] = time.time() - tic
        # gt = batch.pop('gt')
        model_time = time.time()
        model.forward_pansharpening(data)
        log_dict["model_time"] = time.time() - model_time
        tic = time.time()
        print(log_dict)
        # runner.after_train_iter(log_dict, _iter, _inner_iter)


def test_pansharpening():
    from rich.traceback import install

    install()

    from pancollection.python_scripts.accelerate_pansharpening import trainer
    from omegaconf import OmegaConf, DictConfig
    from udl_vis.Basis.option import Config
    from pancollection import getDataSession
    from pancollection.models.FusionNet.model_fusionnet import build_fusionnet
    from pancollection.models.base_model import PanSharpeningModel

    # from udl_vis import trainer
    import hydra
    from hydra.core.hydra_config import HydraConfig
    import os
    from udl_vis.mmcv.utils import create_logger
    from torch import distributed as dist

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "23456"
    # os.environ["WORLD_SIZE"] = "2"
    # os.environ["RANK"] = "0"
    @hydra.main(
        config_path="/home/dsq/nips/work/configs/unfolding",
        config_name="unfolding_pansharpening",
    )
    def hydra_run(cfg: DictConfig):
        if isinstance(cfg, DictConfig):
            cfg = Config(OmegaConf.to_container(cfg, resolve=True))
            cfg.merge_from_dict(cfg.args)
            cfg.__delattr__("args")
            cfg.merge_from_dict(cfg.base)
            cfg.__delattr__("base")
            hydra_cfg = HydraConfig.get()
            cfg.work_dir = hydra_cfg.runtime.output_dir
        print(cfg.pretty_text)
        cfg.backend = "accelerate"
        # cfg.dataset_type = "Dummy"
        cfg.distributed = False
        cfg.plugins = []
        cfg.save_interval = 1
        cfg.max_epochs = 30
        cfg.resume_from = ""
        cfg.use_ema = False
        cfg.eval = False
        cfg.expeimental_desc = "test"
        # trainer.main(cfg, PanSharpeningModel, build_fusionnet, getDataSession)

        # dist.init_process_group(backend="nccl")

        # logger = create_logger(cfg, work_dir=cfg.work_dir)
        # run_accelerate_engine(
        #     cfg, None, PanSharpeningModel, build_fusionnet(cfg), getDataSession
        # )
        trainer.main(cfg, PanSharpeningModel, build_fusionnet(cfg), getDataSession)

    return hydra_run()


if __name__ == "__main__":
    test_pansharpening()
