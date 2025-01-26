if __name__ == "__main__":
    from rich.traceback import install

    install()

import torch
from udl_vis.Basis.dist_utils import master_only, allreduce_params
from udl_vis.plugin.mmcv1_engine import (
    set_random_seed,
    get_data_loader,
    print_log,
)
from udl_vis.Basis.checkpoint import ModelCheckpoint
import time
import os
from udl_vis.Basis.optim.optimizer import Optimizer
from udl_vis.plugin.base import run_engine
import gc
from .base import val
from udl_vis.Basis.dev_utils.deprecated import deprecated, deprecated_context


def parser_mixed_precision(mixed_precision):
    if mixed_precision == "fp32" or mixed_precision == "no" or mixed_precision is None:
        return torch.float32
    elif mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid mixed precision value: {mixed_precision}")


class NaiveEngine:
    def __init__(self, cfg, task_model, 
                 build_model, logger,
                 sync_buffer=False):
        super().__init__()

        self.cfg = cfg
        self.load_model_status = False
        self.sync_buffer = sync_buffer
        self.metrics = cfg.metrics
        self.formatter = cfg.formatter
        self.model_dir = self.cfg.model_dir
        self.results_dir = os.path.dirname(self.cfg.results_dir)
        self.summaries_dir = self.cfg.summaries_dir
        self.logger = logger
        self.by_epoch = cfg.by_epoch
        self.checkpoint = ModelCheckpoint(
            indicator=self.metrics["train"],
            save_latest_limit=cfg.save_latest_limit,
            save_top_k=cfg.save_latest_limit,
            logger=logger,
        )

        self.dtype = parser_mixed_precision(cfg.mixed_precision)

        # CUDA_VISIBLE_DEVICES
        # torch.cuda.set_device() (DDP)
        self.device = torch.cuda.current_device() 
        self.model, criterion, self._optimizer, scheduler = build_model(cfg, logger)
        self.model.to(self.device)
        self.optimizer_wrapper = Optimizer(
            self.model,
            self._optimizer,
            cfg.distributed,
            fp16=cfg.fp16,
            fp_scaler=cfg.fp_scaler,
            grad_clip_norm=cfg.grad_clip_norm,
            grad_clip_value=cfg.grad_clip_value,
            detect_anomalous_params=cfg.detect_anomalous_params,
            accelerator=None,
        )
        self.scheduler = scheduler
        self.task_model = task_model(self.device, self.model, criterion)

    def train_step(self, batch, iteration, **kwargs):
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            log_vars = self.task_model.train_step(batch, **kwargs)
            if torch.isnan(log_vars["loss"]):
                self.model.zero_grad(set_to_none=True)
                self._optimizer.zero_grad(set_to_none=True)
                if self.ema_net is not None:
                    self.ema_net._zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        # print(f"loss is nan, skip {_skip_n} batch(es) in this epoch")
        # self.ema_net.after_train_iter(iteration)
        # optimizer.step() is out of autocast
        self.iter_time = time.time() - self.tic
        grad_norm = self.optimizer_wrapper.step(log_vars)
        return {
            **log_vars,
            **grad_norm,
            "data_time": self.data_time,
            "iter_time": self.iter_time,
        }

    def val_step(self, batch, **kwargs):
        return self.task_model.val_step(batch, **kwargs)

    def test_step(self, batch, **kwargs):

        return self.task_model.val_step(batch, **kwargs)

    def before_run(self):

        if self.cfg.use_ema:
            if not deep_speed_zero_n_init(
                self.accelerator, n=[2, 3]
            ) and "FSDP" not in self.cfg.get("plugins", []):
                print_log(f"Use EMA model and register for checkpointing")
                # self.ema_net = EMA(self.model, beta=self.cfg.ema_decay, update_every=2 * self.accelerator.gradient_accumulation_steps)
                self.ema_net = EMAHook(
                    model=self.model,
                    momentum=1 - self.cfg.ema_decay,
                    interval=2 * self.accelerator.gradient_accumulation_steps,
                    strict=False,
                )
                # self.ema_net = DeepspeedEMA(model=self.model, momentum=1-self.cfg.ema_decay, interval=2 * self.accelerator.gradient_accumulation_steps)
                self.ema_net.before_run()
                self.accelerator.register_for_checkpointing(self.ema_net)
            else:
                self.ema_net = None
                print_log("Can't use EMA model in FSDP mode", logger=self.logger)
        else:
            self.ema_net = None

        self.resume_checkpoints(self.cfg.resume_from)
        # self.accelerator.wait_for_everyone()

    def before_train_epoch(self):
        self.tic = time.time()

    def after_train_epoch(self, loss_dicts, _iter, epoch, _inner_iter):

        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step(loss_dicts["loss"])
                else:
                    print_log(
                        f"The optimizer step was skipped due to mixed precision overflow"
                    )

            else:
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step(
                        epoch - 1 if self.by_epoch else _iter - 1
                    )  # epoch defaluts to 0
                else:
                    print_log(
                        f"The optimizer step was skipped due to mixed precision overflow"
                    )

    def before_train_iter(self):
        self.data_time = time.time() - self.tic

    def after_train_iter(self, loss_dicts, _iter, _inner_iter):
        self.tic = time.time()

    def resume_checkpoints(self, path):
        retry_count = 0  # 0: load latest/best, 1: load sub-latest/sub-best
        while retry_count < 2:
            # maybe the model is not ready, so we need to retry to load sub-latest or sub-best model.
            if path is None or path == "":
                if self.cfg.resume_mode == "latest":
                    path = self.checkpoint.get_latest_checkpoints(
                        self.model_dir, retry_count
                    )
                    self.cfg.resume_mode = "best"  # "best" contains "latest", but we should first load model from "latest" is better
                elif self.cfg.resume_mode == "best":
                    path = self.checkpoint.get_best_checkpoints(
                        self.model_dir, retry_count
                    )
            if path is not None and os.path.isdir(path):
                self.resume(path, retry_count)
                fname = os.path.basename(path)

                tup = fname.split("_")
                save_version = len(tup)
                with deprecated_context(
                    "save_version",
                    "model_{epoch}_{metrics} will be deprecated in a future version. Please use model_{epoch}_{iter}_{metrics} instead.",
                ):
                    if save_version == 3:
                        if self.cfg.workflow[0][0] == "test":
                            self.start_iter = 1
                            self.start_epoch = int(tup[1])
                        else:
                            self.start_iter = 1
                            self.start_epoch = int(tup[1]) + 1
                        return
                    elif save_version == 4:
                        if self.cfg.workflow[0][0] == "test":
                            self.start_iter = int(tup[2])
                            self.start_epoch = int(tup[1])
                        else:
                            self.start_iter = int(tup[2]) + 1
                            self.start_epoch = int(tup[1]) + 1
                        return
                    else:
                        raise ValueError(f"Invalid save version: {save_version}")

            else:
                print_log(
                    f"Loading path:{path} failed. Maybe the path is not a directory. The model will be trained/inferred from scratch"
                )
                return
            # try:
            #     if path is not None and os.path.isdir(path):
            #         self.resume(path, retry_count)
            #         fname = os.path.basename(path)
            #         if self.cfg.workflow[0][0] == "test":
            #             self.start_epoch = int(fname.split("_")[1])
            #         else:
            #             self.start_epoch = int(fname.split("_")[1]) + 1
            #         return
            #     else:
            #         print_log(f"Loading path:{path} failed. Maybe the path is not a directory. The model will be trained/inferred from scratch")
            #         return
            # except Exception as e:
            #     retry_count += 1

            #     raise ValueError(f"Loading path:{path}. {e}")

    def resume(self, path, retry_count=0):

        # self.checkpoint.load_checkpoint(path)
        self.accelerator.load_state(input_dir=path, strict=False)
        # self.accelerator.state.epoch = self.cfg.start_epoch
        if retry_count == 0:
            print_log(
                f"{self.accelerator.process_index} loaded {self.cfg.resume_mode.lower()} state from {path} done.",
                logger=self.logger,
            )
        else:
            print_log(
                f"{self.accelerator.process_index} loaded sub-{self.cfg.resume_mode.lower()} state from {path} done.",
                logger=self.logger,
            )
        self.load_model_status = True

    def save_ckpt(self, epoch, log_vars, _iter):

        if self.sync_buffer:
            allreduce_params(self.model.buffers())

        if self.ema_net is not None:
            saved_model = self.ema_net.ema_model
        else:
            saved_model = self.model

        saved_path = os.path.join(
            self.model_dir,
            self.formatter.format(
                iter=_iter - 1,
                epoch=epoch,
                metrics=log_vars[f'{self.metrics["train"]}'],
            ),
        )

        # TODO: Use accelerate to save model
        # self.accelerator.save_model(
        #     saved_model,
        #     saved_path,
        #     safe_serialization=True,
        # )
        # # Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.
        # self.accelerator.save_state(output_dir=saved_path, safe_serialization=True)

        # if self.accelerator.is_main_process:
        #     # save parial results/checkpoints according to the training results
        #     self.checkpoint.after_train_epoch(self.model_dir, self.results_dir)

        # self.accelerator.wait_for_everyone()
        # self.resume(self.model_dir)

    def end_of_run(self, mode, data_loader):
        # mode: "test" or "val"
        # data_loader: data_loaders[mode]
        from udl_vis.Basis.auxiliary import MetricLogger

        log_buffer = MetricLogger(logger=self.logger, delimiter="  ")
        path = self.checkpoint.get_best_checkpoints(self.model_dir)
        print_log(f"Loading best checkpoint from {path}", logger=self.logger)
        self.resume(path)
        fname = os.path.basename(path)
        version = len(fname.split("_"))
        with deprecated_context(
            "end_of_run",
            "model_{epoch}_{metrics} will be deprecated in a future version. Please use model_{epoch}_{iter}_{metrics} instead.",
        ):
            if version == 3:
                best_epoch = int(fname.split("_")[1])
                best_iter = 0
            elif version == 4:
                best_epoch = int(fname.split("_")[1])
                best_iter = int(fname.split("_")[2])
        _, _, log_buffer = val(
            runner=self,
            data_loader=data_loader,
            epoch=best_epoch,
            max_epochs=self.cfg.max_epochs,
            logger=self.logger,
            log_buffer=log_buffer,
            img_range=self.cfg.img_range,
            eval_flag=True,
            save_fmt=self.formatter,
            test=self.cfg.test,
            _iter=best_iter,
            test_mode=True,
            mode=mode,
            data_length={mode: len(data_loader)},
            # not save, only obtain best results to store into db (udl_cil)
            results_dir=os.path.join(self.cfg.work_dir, "results/best_{epoch}"),
            log_epoch_interval=self.cfg.log_epoch_interval,
            train_log_iter_interval=self.cfg.train_log_iter_interval,
            val_log_iter_interval=self.cfg.val_log_iter_interval,
            test_log_iter_interval=self.cfg.test_log_iter_interval,
            save_interval=self.cfg.save_interval,
            dataset_cfg=self.cfg.dataset.dataset_cfg.get(mode, {}),
        )

        log_buffer.synchronize_between_processes()
        metrics = {
            k: meter.avg if not hasattr(meter, "image") else meter.image
            for k, meter in log_buffer.meters.items()
        }

        best_metric_name = f"{mode}_{self.metrics[mode]}"
        print_log(
            f"Metrics: {metrics}, {best_metric_name} is chosen as the best metric",
            logger=self.logger,
        )
        # TODO: multi-objective optimization
        import ipdb

        ipdb.set_trace()
        return {"best_value": metrics[best_metric_name]}


def run_naive_engine(cfg, logger, task_model, build_model, getDataSession, **kwargs):

    cfg.model_dir = os.path.join(cfg.work_dir, "checkpoints")
    cfg.results_dir = os.path.join(cfg.work_dir, "results")
    cfg.summaries_dir = os.path.join(cfg.work_dir, "summaries")

    set_random_seed(cfg.seed)

    runner = NaiveEngine(cfg, task_model, build_model, logger)

    state_dataloader = runner.before_run()

    run_engine(cfg, runner, getDataSession, logger, state_dataloader)


def test_pansharpening():
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
        config_path="/Data2/woo/NIPS/PanCollection/pancollection/configs",
        config_name="model",
    )
    def hydra_run(cfg: DictConfig):
        if isinstance(cfg, DictConfig):
            cfg = Config(OmegaConf.to_container(cfg, resolve=True))
            cfg.merge_from_dict(cfg.args)
            cfg.__delattr__("args")
            hydra_cfg = HydraConfig.get()
            cfg.work_dir = hydra_cfg.runtime.output_dir

        cfg.backend = "naive"
        cfg.workflow = [("train", 1), ("test", 1)]
        cfg.dataset_type = "Dummy"
        cfg.distributed = False
        cfg.plugins = []
        print(cfg.pretty_text)

        logger = create_logger(cfg, work_dir=cfg.work_dir)
        run_naive_engine(
            cfg, logger, PanSharpeningModel, build_fusionnet, getDataSession
        )

    return hydra_run()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    test_pansharpening()
