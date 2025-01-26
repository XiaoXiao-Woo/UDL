import os

import torch
from accelerate import (
    Accelerator,
    PartialState,
    DistributedType,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils import extract_model_from_parallel, GradientAccumulationPlugin

# from udl_vis.Basis.criterion_metrics import SetCriterion
from udl_vis.Basis.auxiliary import set_random_seed
from typing import Dict

# from udl_vis.plugin.mmcv1_engine import run_mmcv1_engine
# from udl_vis.plugin.naive_engine import run_naive_engine
from udl_vis.mmcv.utils import print_log
from udl_vis.Basis.dist_utils import allreduce_params, master_only
from udl_vis.plugin.base import run_engine
from udl_vis.Basis.optim.optimizer import Optimizer
from udl_vis.Basis.ema import EMA, EMAHook  # DeepspeedEMA
from udl_vis.Basis.checkpoint import ModelCheckpoint
import gc
import shutil
import time
from .base import val
from udl_vis.Basis.dev_utils.deprecated import deprecated, deprecated_context

os.environ["NCCL_TIMEOUT"] = "600"


def print_gpu_device(accelerator, logger=None):
    try:
        gpu_ids = [
            "cuda:" + x for x in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
        ]
    except:
        gpu_ids = ["cuda:0"]
    if accelerator.state.num_processes > 1:
        index = (
            accelerator.state.local_process_index
        )  # int(str(accelerator.state.device).split(":")[1])
    else:
        index = 0

    if accelerator.state.num_processes > len(gpu_ids):
        raise ValueError(
            f"Number of processes {accelerator.state.num_processes} does not match given GPUs {gpu_ids}"
        )

    print_log(
        f"[Accelerate]: Using CUDA_VISIBLE_DEVICES: {gpu_ids[index]} to set device {accelerator.state.device}",
        logger=logger,
    )
    print_log(
        f"[Accelerate]: Number of devices: {accelerator.state.num_processes}",
        logger=logger,
    )


def deepspeed_to_device(tensors: Dict[str, torch.Tensor], dtype: "torch.dtype"):
    state = PartialState()
    if state.distributed_type == DistributedType.DEEPSPEED:
        for k, t in tensors.items():
            tensors[k] = t.to(state.device, dtype=dtype)
    return tensors


def deep_speed_zero_n_init(accelerator: "Accelerator", n: "int | list[int]" = 3):
    if isinstance(n, int):
        n = [n]
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        if accelerator.state.deepspeed_plugin.zero_stage in n:
            return True
    return False


def parser_mixed_precision(mixed_precision):
    if mixed_precision == "fp32" or mixed_precision == "no" or mixed_precision is None:
        return torch.float32
    elif mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid mixed precision value: {mixed_precision}")


class AcceleratorEngine:

    def __init__(
        self,
        cfg,
        task_model,
        build_model,
        logger,
        sync_buffer=False,
    ):
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

        # try:
        #     self.model, criterion, optimizer, scheduler = build_model(cfg)
        # except Exception as e:
        #     import inspect
        #     parameters = inspect.signature(build_model).parameters
        #     raise TypeError(f"{e}, build_model need parameters: {parameters}")
        # self.model.to(f"cuda:{int(os.environ.get("LOCAL_RANK", 0))}")``
        self.model, criterion, optimizer, scheduler = build_model(cfg, logger)

        plugin_dicts = {}
        for name in cfg.get("plugins", []):
            print_log(f"Setting accelerate plugin: {name}", logger=self.logger)

            plugin_dicts[name.lower() + "_plugin"] = getattr(
                self, f"setup_{name.lower()}_plugin"
            )()

        self.model, self._optimizer = self.init_accelerator(
            plugin_dicts, self.model, optimizer
        )

        self.device = self.accelerator.device

        self.model.to(self.device)

        self.scheduler = scheduler

        self.optimizer_wrapper = Optimizer(
            self.model,
            self._optimizer,
            cfg.distributed,
            fp16=cfg.fp16,
            fp_scaler=cfg.fp_scaler,
            grad_clip_norm=cfg.grad_clip_norm,
            grad_clip_value=cfg.grad_clip_value,
            detect_anomalous_params=cfg.detect_anomalous_params,
            accelerator=self.accelerator,
        )
        self.task_model = task_model(self.device, self.model, criterion.to(self.device))

    def init_accelerator(self, plugin_dicts, *args, **kwargs):

        accelerator = Accelerator(
            # device_placement=True,  # BUG: according to CUDA_VISIBLE_DEVICES, it is invalid when using multi-gpu
            cpu=False,
            mixed_precision=self.cfg.mixed_precision,
            project_dir=self.model_dir,
            dataloader_config=DataLoaderConfiguration(
                use_stateful_dataloader=True,
                use_seedable_sampler=True,
                non_blocking=True,
                split_batches=False,  # True means that samplers_per_gpu
                even_batches=True,
            ),
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=self.cfg.accumulated_step,
                adjust_scheduler=False,
                sync_with_dataloader=False,
                sync_each_batch=False,
            ),
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            **plugin_dicts,
        )

        if "deepspeed" in plugin_dicts.keys():
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = self.cfg.samples_per_gpu
        self.accelerator = accelerator
        # accelerator.state.device = torch.device(self.cfg.device)
        print_gpu_device(accelerator, logger=self.logger)
        return accelerator.prepare(*args, **kwargs)

    def prepare_dataloader(self, dataloader):
        return self.accelerator.prepare_data_loader(dataloader)

    def setup_deepspeed_plugin(self):
        from accelerate import DeepSpeedPlugin

        # BUG: parial settings is invalid
        # config = {
        #     "train_batch_size": 32,
        #     "zero_optimization": {
        #         "stage": 2,
        #         "allgather_partitions": True,
        #         "reduce_scatter": True,
        #         "overlap_comm": True,
        #         "offload_param": {"device": "cpu", "pin_memory": True},
        #         "offload_optimizer": {"device": "cpu", "pin_memory": True},
        #     },
        #     # "communication_data_type": "NCCL",
        #     # "fp16": {"enabled": True, "loss_scale": "dynamic"},
        # }
        return DeepSpeedPlugin()  # zero_stage=2, gradient_accumulation_steps=1)

    def setup_fsdp_plugin(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            # FullOptimStateDictConfig,
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
            # FullStateDictConfig,
        )
        from accelerate import (
            DistributedType,
            FullyShardedDataParallelPlugin,
        )

        # self.model = FSDP(self.model)

        return FullyShardedDataParallelPlugin(
            state_dict_config=ShardedStateDictConfig(
                offload_to_cpu=False,  # rank0_only=False
            ),
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=False,  # rank0_only=False
            ),
        )

    def train_step(self, batch, iteration, **kwargs):

        batch = deepspeed_to_device(batch, dtype=self.dtype)
        with self.accelerator.autocast() and self.accelerator.accumulate(self.model):
            log_vars = self.task_model.train_step(batch, **kwargs)
            if self.accelerator.gather(log_vars["loss"]).isnan().any():
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
        self.accelerator.wait_for_everyone()

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

        self.accelerator.save_model(
            saved_model,
            saved_path,
            safe_serialization=True,
        )
        # Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.
        self.accelerator.save_state(output_dir=saved_path, safe_serialization=True)

        if self.accelerator.is_main_process:
            # save parial results/checkpoints according to the training results
            self.checkpoint.after_train_epoch(self.model_dir, self.results_dir)

        self.accelerator.wait_for_everyone()
        # self.resume(self.model_dir)

    def end_of_run(self, mode, data_loader):
        # mode: "test" or "val"
        # data_loader: data_loaders[mode]
        from udl_vis.Basis.auxiliary import MetricLogger

        log_buffer = MetricLogger(logger=self.logger, delimiter="  ")
        resume_from = self.cfg.resume_from
        model_dir = self.model_dir
        if resume_from != "":
            base_resume_from = os.path.basename(resume_from)
            epoch = int(base_resume_from.split("_")[1])
            if epoch == self.cfg.max_epochs:
                model_dir = os.path.dirname(resume_from)
            

        path = self.checkpoint.get_best_checkpoints(model_dir)
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
        
        results_dir = os.path.join(self.cfg.work_dir, f"results/best_{best_epoch}_{best_iter}")
        
        _, _, log_buffer = val(
            runner=self,
            data_loader=data_loader,
            epoch=best_epoch,
            max_epochs=self.cfg.max_epochs,
            logger=self.logger,
            log_buffer=log_buffer,
            img_range=self.cfg.img_range,
            eval_flag=True,
            save_fmt=self.cfg.save_fmt,
            test=self.cfg.test,
            _iter=best_iter,
            test_mode=True,
            mode=mode,
            data_length={mode: len(data_loader)},
            # not save, only obtain best results to store into db (udl_cil)
            results_dir=results_dir,
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

        # best_metric_name = f"{mode}_{self.metrics[mode]}"
        
        shutil.move(results_dir, (results_dir+"_{metrics_name}_{metrics}") \
                    .format(epoch=best_epoch, iter=best_iter, 
                            metrics_name=self.metrics[mode], metrics=metrics[self.metrics[mode]]))
        print_log(
            f"Metrics: {metrics}, {self.metrics[mode]} is chosen as the best metric",
            logger=self.logger,
        )
        # TODO: multi-objective optimization
        return {"best_value": metrics[self.metrics[mode]]}


def run_accelerate_engine(
    cfg,
    logger,
    task_model,
    build_model,
    getDataSession,
    callbacks=[],
    **kwargs,
):
    if cfg.local_rank == 0:
        cfg.code_dir.append(os.path.abspath(__file__))
    if cfg.eval:
        cfg.results_dir = os.path.join(cfg.work_dir, "results/eval_{epoch}")
    else:
        cfg.results_dir = os.path.join(cfg.work_dir, "results/{epoch}")

    cfg.model_dir = os.path.join(cfg.work_dir, "checkpoints")
    cfg.summaries_dir = os.path.join(cfg.work_dir, "summaries")
    os.makedirs(os.path.dirname(cfg.results_dir), exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.summaries_dir, exist_ok=True)

    set_random_seed(cfg.seed)
    # task_model = TaskDispatcher(cfg.task) # PanSharpeningModel(model, criterion)

    runner = AcceleratorEngine(cfg, task_model, build_model, logger)

    runner.before_run()

    print_log("Running process ...", logger=logger)

    run_engine(cfg, runner, getDataSession, logger, state_dataloader=None)
