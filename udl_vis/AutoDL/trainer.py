# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
import os
import warnings
warnings.filterwarnings("ignore")
from udl_vis.Basis.auxiliary import init_random_seed, set_random_seed
from udl_vis.mmcv.utils.logging import print_log, create_logger
from udl_vis.mmcv.runner import init_dist, get_dist_info
from udl_vis.plugin.engine import get_engine
from udl_vis.Basis.config import merge_keys
from udl_vis.Basis.option import Config
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import types

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.get("MASTER_ADDR", "localhost")
os.environ.get("MASTER_PORT", "24567")

def run_hydra(
    full_config_path="configs/config",
    import_path=None,
    taskModel=None,
    build_model=None,
    getDataSession=None,
):
    config_path = os.path.dirname(full_config_path)
    config_name = os.path.basename(full_config_path)

    @hydra.main(config_path=config_path, config_name=config_name)
    def inner_func(cfg: DictConfig):
        if isinstance(cfg, DictConfig):
            cfg = Config(OmegaConf.to_container(cfg, resolve=True))
            hydra_cfg = HydraConfig.get()
            cfg.work_dir = cfg.get("work_dir", hydra_cfg.runtime.output_dir)
        cfg.backend = "accelerate"
        cfg.launcher = "accelerate"
        if import_path is not None:
            cfg.import_path = import_path
        main(cfg, taskModel, build_model, getDataSession)

    return inner_func()


def trainer(
    cfg,
    logger,
    task_model,
    build_model,
    getDataSession,
    runner=None,
    meta=None,
    **kwargs,
):
    run_engine = get_engine(cfg.backend)

    if not isinstance(build_model, types.FunctionType):
        build_model = build_model()

    run_engine(
        cfg,
        logger,
        task_model,
        build_model,
        getDataSession,
        runner=runner,
        meta=meta,
        **kwargs,
    )


def train_loop(local_rank, cfg, build_task, build_model, getDataSession, runner):
    # rank, _ = get_dist_info()

    if cfg.launcher == "accelerator":
        from accelerate.utils import (
            InitProcessGroupKwargs,
            prepare_multi_gpu_env,
            patch_environment,
        )

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "23456"
        # os.environ["LOCAL_RANK"] = "0"
        # os.environ["WORLD_SIZE"] = str(len(cfg.gpu_ids))
        # os.environ["RANK"] = str(local_rank)
        cfg.accelerator_kwargs_handlers = InitProcessGroupKwargs()
        cfg.accelerator_kwargs_handlers.world_size = len(cfg.gpu_ids)
        cfg.accelerator_kwargs_handlers.rank = str(local_rank)
        # current_env = prepare_multi_gpu_env(cfg)
        # with patch_environment(**current_env):
        #    main(cfg, build_model, getDataSession)

    else:  # pytorch DDP
        cfg.dist_params = dict(
            rank=local_rank, init_method="env://", world_size=len(cfg.gpu_ids)
        )
        os.environ["RANK"] = str(local_rank)
        main(cfg, build_task, build_model, getDataSession)


def main_spawn(cfg, build_task, build_model, getDataSession=None, runner=None):
    from torch import multiprocessing as mp

    # if cfg.launcher == "accelerate":
    #     from accelerate import notebook_launcher
    #     port = os.environ.get("MASTER_PORT", "36790")
    #     notebook_launcher(
    #         main, (cfg, build_task, build_model, getDataSession, runner), 
    #         num_processes=len(cfg.gpu_ids), use_port=port
    #     )
    # else:
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = os.environ.get("MASTER_PORT", "36790")
    mp.spawn(
            train_loop,
            args=(cfg, build_task, build_model, getDataSession, runner),
            nprocs=len(cfg.gpu_ids),
        )


def main(cfg, build_task, build_model, getDataSession=None, runner=None, **kwargs):
    # init distributed env first, since logger depends on the dist info.
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Example: cfg.args and cfg.base
    # only easy for user to view, not easy for user to use
    merge_keys(cfg)
    
    if cfg.debug:
        print(cfg.pretty_text)


    logger = create_logger(cfg, cfg.experimental_desc, work_dir=cfg.work_dir)
    cfg.seed = init_random_seed(cfg.seed)

    if cfg.local_rank == 0:
        print_log(f"Set random seed to {cfg.seed}", logger=logger)
        print_log(f"Work Directory: {cfg.work_dir}", logger=logger)
        cfg.code_dir.append(os.path.abspath(__file__))

    # ipdb.set_trace()
    if cfg.backend == "accelerate":
        cfg.launcher = "none"
        
        cfg.distributed = False
        print_log(
            f"Prepare training environment by Accelerator when the backend is set to {cfg.backend}."
        )

    elif cfg.launcher in ["none", "dp"]:
        cfg.distributed = False
        print_log(
            f"Prepare single-node training environment because of the launcher is set to {cfg.launcher}."
        )
        cfg.gpu_ids = [cfg.gpu_ids]

    else:
        cfg.distributed = True
        print_log(f"Manually set distributed training environment (dist_params: {cfg.dist_params})", logger=logger)
        init_dist(cfg.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    set_random_seed(cfg.seed)

    trainer(
        cfg,
        logger,
        build_task,
        build_model,
        getDataSession,
        runner,
        meta={},
        **kwargs,
    )
