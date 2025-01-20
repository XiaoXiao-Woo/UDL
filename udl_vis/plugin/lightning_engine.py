# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:

import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from udl_vis import TaskDispatcher


class MyDataModule(pl.LightningDataModule):

    def __init__(self, cfg, getDataSession):
        super().__init__()
        self.cfg = cfg
        self.sess = getDataSession(cfg)

    def train_dataloader(self):
        train_loader, train_sampler, generator = self.sess.get_dataloader(
            self.cfg.dataset.train_name, self.cfg.distributed, state_dataloader=None
        )

        return train_loader

    def val_dataloader(self):
        valid_loader, valid_sampler = self.sess.get_valid_dataloader(
            self.cfg.dataset.val_name, self.cfg.distributed
        )
        return valid_loader

    def test_dataloader(self):
        test_loader, _ = self.sess.get_eval_dataloader(
            self.cfg.dataset.test_name, self.cfg.distributed
        )
        return test_loader


class LitModel(pl.LightningModule):
    def __init__(self, cfg, task_model, build_model):
        super().__init__()

        self.save_hyperparameters()
        model, criterion, self.optimizer, _ = build_model(device=cfg.device)(cfg)
        self.task_model = task_model(cfg.device, model, criterion)
        self.model = model

    def training_step(self, batch, batch_idx):
        metrics = self.task_model.train_step(batch)
        self.log("train_loss", metrics["loss"])
        return metrics["loss"]

    def evaluate(self, batch, stage=None):
        metrics = self.task_model.val_step(batch)

        # if stage:
        #     self.log(f"{stage}_loss", metrics[""], prog_bar=True)
        #     self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):

        return {"optimizer": self.optimizer}


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(
            pl_module.hparams.output_dir, "test_results.txt"
        )
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def run_lightning_engine(
    cfg, logger, task_model, build_model, getDataSession, callbacks=[], **kwargs
):
    pl.seed_everything(cfg.seed)
    os.makedirs(cfg.work_dir, exist_ok=True)

    model = LitModel(cfg, task_model, build_model)

    # add custom checkpoints
    # if cfg.use_save:
    #     checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #         filepath=cfg.work_dir,
    #         prefix="checkpoint",
    #         monitor="val_loss",
    #         mode="min",
    #         save_top_k=1,
    #     )
    # # if cfg.early_stopping:
    # #     callbacks.append(early_stopping_callback)
    # if cfg.use_log is None:
    #     logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if cfg.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = cfg.fp16_opt_level

    if len(cfg.gpu_ids) > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = cfg.accumulated_step
    train_params["accelerator"] = cfg.get("accelerator", None)
    train_params["profiler"] = cfg.get("profiler", None)

    # trainer = pl.Trainer.from_argparse_args(
    #     cfg,
    #     weights_summary=None,
    #     callbacks=[logging_callback] + callbacks,
    #     logger=logger,
    #     checkpoint_callback=checkpoint_callback,
    #     **train_params,
    # )

    data_module = MyDataModule(cfg, getDataSession)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        limit_val_batches=0,
        log_every_n_steps=cfg.log_epoch_interval,
    )
    trainer.fit(model, datamodule=data_module)


def test_pansharpening():
    from udl_vis.mmcv.utils.logging import print_log, create_logger
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from udl_vis.Basis.option import Config
    from pancollection import getDataSession
    from pancollection.models.FusionNet.model_fusionnet import build_fusionnet
    from pancollection.models.base_model import PanSharpeningModel
    
    cfg = Config().fromfile("/Data2/woo/PanCollection/pancollection/configs/fusionnet.yaml")
    

    cfg = Config(dict(task="pansharpening", 
                 samples_per_gpu=32, 
                 workers_per_gpu=8,
                 test_samples_per_gpu=1,
                 seed=42,
                 lr=1e-4,
                 gpu_ids=[0],
                 device="cuda",
                 use_save=False,
                 use_log=True,
                 fp16=False,
                 distributed=False,
                 log_epoch_interval=10,
                 accumulate_grad=1,
                 experimental_desc="test_lightning",
                 dataloader_name="PanCollection_dataloader",
                 img_range=2047.0,
                 max_epochs=400,
                 work_dir="/Data2/woo/DDP_example/results",
                 dataset=dict(
                     train_name="wv3",
                     wv3_train_path="/Data/Datasets/pansharpening_2/PanCollection/training_data/train_wv3_9714.h5",
                     wv3_test_path="/Data/Datasets/pansharpening_2/PanCollection/test_data/test_wv3_multiExm1.h5"
                 )))
    
    
    print(cfg.pretty_text)
    
    logger = create_logger(cfg, work_dir=cfg.work_dir)
    run_lightning_engine(cfg, logger, getDataSession, PanSharpeningModel, build_fusionnet)


if __name__ == "__main__":
    test_pansharpening()
