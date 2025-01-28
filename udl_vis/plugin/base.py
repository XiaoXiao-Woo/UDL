import torch
from functools import partial
import time
import numpy as np
from collections import OrderedDict
from udl_vis.mmcv.utils import print_log
from udl_vis.Basis.auxiliary import MetricLogger, get_max_memory
from torch.utils.data import DataLoader
import os
import datetime
import shutil


try:
    print("udl_cil will receive experiment results")
    from udl_cil.v2.cilent import websocket_communicator
except:
    import functools

    def websocket_communicator(arg=None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


def get_data_loader(cfg, getDataSession, state_dataloader):

    generator = None

    if getDataSession is not None:
        sess = getDataSession(cfg)
    # cfg.valid_or_test = False
    if cfg.eval:
        cfg.workflow = [("test", 1)]

    data_loaders = {}
    for idx, flow in enumerate(cfg.workflow):
        if getDataSession is None:
            mode, epoch = flow[:-1], flow[-1]
            try:
                mode, mode_func = mode
            except:
                raise ValueError(
                    f"workflow={cfg.workflow} should be [(('train', train_loader), 1), ...] when getDataSession is None"
                )
        else:
            mode, epoch = flow

        if "test" in mode:
            # cfg.dataset = cfg.dataset + '_OrigScale_multiExm1.h5'
            # cfg.dataset = cfg.dataset + '_multiExm1.h5'
            if getDataSession is not None:
                eval_loader, eval_sampler = sess.get_test_dataloader(
                    cfg.dataset.test_name.lower(), cfg.distributed)
                
            else:
                eval_loader, eval_sampler = mode_func
            eval_cfg = cfg.get("evaluation", {})
            eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
            data_loaders["test"] = eval_loader
            cfg.workflow[idx] = ("test", epoch)

        if "valid" in mode:
            if getDataSession is not None:
                valid_loader, valid_sampler = sess.get_valid_dataloader(
                    cfg.dataset.val_name, cfg.distributed
                )
            else:
                valid_loader, valid_sampler = mode_func
            if cfg.once_epoch:
                valid_loader = lambda: iter(list(valid_loader))
            data_loaders["val"] = valid_loader
            # reset workflow to standard workflow
            cfg.workflow[idx] = ("val", epoch)

        if "train" in mode:
            if getDataSession is not None:
                train_loader, train_sampler, generator = sess.get_train_dataloader(
                    cfg.dataset.train_name, cfg.distributed, state_dataloader
                )
            else:

                train_loader, train_sampler, generator = mode_func
            # 保存generator状态用于恢复数据批次/轮次
            if cfg.once_epoch:
                train_loader = iter(list(train_loader))
            data_loaders[mode] = train_loader

            if len(cfg.workflow) == 0:
                cfg.workflow.append(("simple_train", 1))
            # reset workflow to standard workflow
            cfg.workflow[idx] = (mode, epoch)

    return cfg, data_loaders, generator


@websocket_communicator()
def run_engine(cfg, runner, getDataSession, logger, state_dataloader):
    websocket_communicator.set_value(cfg, logger)

    cfg.max_epochs = int(cfg.max_epochs)  # udl_cil may set max_epochs as float

    if cfg.local_rank == 0:
        cfg.code_dir.append(os.path.abspath(__file__))
        os.makedirs(os.path.join(cfg.work_dir, "codes"), exist_ok=True)
        for file in cfg.code_dir:
            shutil.copy(file, os.path.join(cfg.work_dir, "codes"))

    cfg.runner = {"type": "EpochBasedRunner", "max_epochs": cfg.max_epochs}  # argparser

    log_buffer = MetricLogger(logger=logger, delimiter="  ")

    cfg, data_loaders, generator = get_data_loader(
        cfg, getDataSession, state_dataloader
    )

    workflow = cfg.workflow
    runner.state_dataloader = generator
    cfg.start_epoch = getattr(runner, "start_epoch", 1)
    cfg.start_iter = getattr(runner, "start_iter", 1)

    assert len(data_loaders) == len(workflow), print_log(
        f"{len(data_loaders)} == {len(workflow)}"
    )

    assert (
        cfg.get("max_epochs") is not None
    ), "max_epochs must be specified during instantiation"

    train_flag = any("train" in mode for mode, _ in workflow)
    # eval_flag = not train_flag
    cfg.eval = not train_flag if not cfg.eval else True

    data_length = {"train": 1, "val": 1, "test": 1}

    for i, flow in enumerate(workflow):
        mode, interval = flow
        if not isinstance(data_loaders[mode], DataLoader) and callable(
            data_loaders[mode]
        ):
            data_loaders[mode] = data_loaders[mode]()

        if cfg.backend == "accelerate" and mode == "train":
            data_loaders[mode] = runner.prepare_dataloader(data_loaders[mode])

        data_length[mode] = len(data_loaders[mode])
    # if mode == "train":
    #     train_interval = interval
    # if mode == "test":
    #     test_interval = interval

    tic = time.time()
    if cfg.runner_mode == "iter":

        if "train" in data_loaders.keys():
            max_iters = cfg.max_epochs * data_length["train"]
        train = train_iter
        run_by_iter(cfg, model, logger, train_flag, workflow, data_loaders)

    else:
        run_by_epoch(
            cfg,
            runner,
            logger,
            data_length,
            train_flag,
            workflow,
            data_loaders,
            log_buffer,
        )
    total_time = time.time() - tic
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_log("Training time {}".format(total_time_str), logger=logger)

    if "test" in data_loaders.keys():
        return runner.end_of_run("test", data_loaders["test"])
    if "val" in data_loaders.keys():
        return runner.end_of_run("val", data_loaders["val"])
    else:
        return {"best_value": np.nan}

def run_by_epoch(
    cfg, runner, logger, data_length, train_flag, workflow, data_loaders, log_buffer
):

    epoch = cfg.start_epoch
    _iter = cfg.start_iter

    train_val_func = {}
    train_val_func["train"] = partial(train)
    train_val_func["val"] = partial(val, test_mode=False)
    train_val_func["test"] = partial(
        val,
        test_mode=True,
        save_interval=cfg.save_interval,
    )
    mode = None

    while epoch < cfg.max_epochs and train_flag:
        for i, flow in enumerate(workflow):
            mode, epochs = flow

            if isinstance(mode, str):  # self.train()
                if not hasattr(cfg, "runner_mode"):
                    raise ValueError(
                        f'runner has no method named "{mode}" to run an ' "epoch"
                    )
                epoch_runner = train_val_func[mode]
            else:
                raise TypeError(
                    "mode in workflow must be a str, but got {}".format(type(mode))
                )

            for _ in range(epochs):
                if mode == "train" and epoch > cfg.max_epochs:
                    break
                if mode != "train":
                    log_buffer.clear()

                epoch, new_iter, log_buffer = epoch_runner(
                    runner=runner,
                    data_loader=data_loaders[mode],
                    epoch=epoch,
                    logger=logger,
                    log_buffer=log_buffer,
                    img_range=cfg.img_range,
                    eval_flag=cfg.eval,
                    save_fmt=cfg.save_fmt,
                    test=cfg.test,
                    results_dir=cfg.results_dir.format(epoch=epoch),
                    _iter=_iter,
                    mode=mode,
                    max_epochs=cfg.max_epochs,
                    data_length=data_length,
                    train_log_iter_interval=cfg.train_log_iter_interval,
                    val_log_iter_interval=cfg.val_log_iter_interval,
                    test_log_iter_interval=cfg.test_log_iter_interval,
                    log_epoch_interval=cfg.log_epoch_interval,
                    dataset_cfg=cfg.dataset.dataset_cfg.get(mode, {}),
                )
                if mode == "train":
                    _iter = new_iter
                    if epoch % cfg.save_interval == 0 or epoch == cfg.max_epochs:
                        # log_buffer.synchronize_between_processes()
                        metrics = {
                            k: meter.avg if not hasattr(meter, "image") else meter.image
                            for k, meter in log_buffer.meters.items()
                        }
                        runner.save_ckpt(epoch, metrics, _iter)
                    epoch += 1
                elif mode == "test" and runner.load_model_status:
                    # only run once when first load the last epoch
                    cfg.start_epoch = 1
                    epoch += 1

        # if earlyStop:
        #     print_log(
        #             "model train has diverged, python will stop training",
        #             logger=logger,
        #         )
        #     break

    if not runner.load_model_status:
        epoch -= 1

    if cfg.eval or mode == "train":
        if "val" in data_loaders.keys():
            val(
                runner=runner,
                data_loader=data_loaders["val"],
                logger=logger,
                max_epochs=cfg.max_epochs,
                img_range=cfg.img_range,
                eval_flag=cfg.eval,
                save_fmt=cfg.save_fmt,
                results_dir=cfg.results_dir.format(epoch=epoch),
                test_mode=False,
                _iter=_iter,
                epoch=epoch,
                mode="val",
                data_length=data_length,
                log_epoch_interval=cfg.log_epoch_interval,
                train_log_iter_interval=cfg.train_log_iter_interval,
                val_log_iter_interval=cfg.val_log_iter_interval,
                test_log_iter_interval=cfg.test_log_iter_interval,
                save_interval=cfg.save_interval,
                log_buffer=log_buffer,
                test=cfg.test,
                dataset_cfg=cfg.dataset.dataset_cfg.get("val", {}),
            )
        if "test" in data_loaders.keys():
            val(
                runner=runner,
                data_loader=data_loaders["test"],
                logger=logger,
                max_epochs=cfg.max_epochs,
                img_range=cfg.img_range,
                eval_flag=cfg.eval,
                save_fmt=cfg.save_fmt,
                results_dir=cfg.results_dir.format(epoch=epoch),
                test_mode=True,
                _iter=_iter,
                epoch=epoch,
                mode="test",
                data_length=data_length,
                log_epoch_interval=cfg.log_epoch_interval,
                train_log_iter_interval=cfg.train_log_iter_interval,
                val_log_iter_interval=cfg.val_log_iter_interval,
                test_log_iter_interval=cfg.test_log_iter_interval,
                save_interval=cfg.save_interval,
                log_buffer=log_buffer,
                test=cfg.test,
                dataset_cfg=cfg.dataset.dataset_cfg.get("test", {}),
            )


def train(
    runner,
    data_loader,
    epoch,
    logger,
    log_buffer,
    _iter,
    mode,
    data_length,
    max_epochs,
    log_epoch_interval,
    train_log_iter_interval,
    val_log_iter_interval,
    test_log_iter_interval,
    results_dir,
    dataset_cfg,
    **kwargs,
):
    # epochs = kwargs['epochs']
    if hasattr(runner.model, "train"):
        runner.model.train()
    else:
        runner.model.model.train()

    mode = "train"
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        torch.manual_seed(1)
        data_loader.sampler.set_epoch(epoch)
    else:
        # accelerate
        ...
    # # self.call_hook("before_train_epoch")
    # # self.data_loader.sampler.generator.manual_seed(self.epoch)
    # # tic = time.time()
    # # time.sleep(2)  # Prevent possible deadlock during epoch transition
    # if hasattr(data_loader.dataset, "gen_transform"):
    #     data_loader.dataset.gen_transform(data_loader.generator)
    runner.before_train_epoch()
    for i, train_batch in enumerate(data_loader, start=1):  # train_loader
        runner.before_train_iter()
        _inner_iter = i
        # self.call_hook("before_train_iter")
        log_buffer = run_iter(
            runner=runner,
            data_batch=train_batch,
            epoch=epoch,
            _inner_iter=_inner_iter,
            _iter=_iter,
            log_buffer=log_buffer,
            logger=logger,
            mode=mode,
            data_length=data_length,
            max_epochs=max_epochs,
            log_epoch_interval=log_epoch_interval,
            train_log_iter_interval=train_log_iter_interval,
            val_log_iter_interval=val_log_iter_interval,
            test_log_iter_interval=test_log_iter_interval,
            dataset_cfg=dataset_cfg,
        )
        # self.call_hook("after_train_iter")
        _iter += 1

    # self.call_hook("after_train_epoch")
    # epoch += 1

    return epoch, _iter, log_buffer


@torch.no_grad()
def val(
    runner,
    data_loader,
    epoch,
    max_epochs,
    logger,
    log_buffer,
    img_range,
    eval_flag,
    save_fmt,
    test,
    _iter,
    # hooks,
    test_mode,
    mode: "train | val | test",  # type: ignore
    data_length,
    results_dir,
    log_epoch_interval,
    train_log_iter_interval,
    val_log_iter_interval,
    test_log_iter_interval,
    save_interval,  # used in train, val is not used
    dataset_cfg,
):
    # kwargs["test_mode"] = False if not kwargs.get("test_mode", None) else True
    if hasattr(runner.model, "eval"):
        runner.model.eval()
    else:
        runner.model.model.eval()
    # mode = "val" if not kwargs.get("test_mode", None) else "test"
    # hooks.call_hook("before_val_epoch")
    # time.sleep(2)  # Prevent possible deadlock during epoch transition
    tic = time.time()
    for i, val_batch in enumerate(data_loader, start=1):  # val_loader
        _inner_iter = i
        # self.call_hook("before_val_iter")

        log_buffer = run_iter(
            runner=runner,
            data_batch=val_batch,
            logger=logger,
            idx=i,
            img_range=img_range,
            eval_flag=eval_flag,
            test=test,
            save_fmt=save_fmt,
            filename=(
                val_batch.get("filename", [None])[0]
                if isinstance(val_batch, dict)
                else None
            ),
            log_buffer=log_buffer,
            epoch=epoch,
            _inner_iter=_inner_iter,
            _iter=_iter,
            test_mode=test_mode,
            mode=mode,
            data_length=data_length,
            max_epochs=max_epochs,
            log_epoch_interval=log_epoch_interval,
            train_log_iter_interval=train_log_iter_interval,
            val_log_iter_interval=val_log_iter_interval,
            test_log_iter_interval=test_log_iter_interval,
            results_dir=results_dir,
            dataset_cfg=dataset_cfg,
        )
        # val_mode=self.opt_cfg['val_mode'])
        # self.call_hook("after_val_iter")
        # break
    print_log(f"test time: {time.time() - tic}", logger=logger)
    # self.call_hook("after_val_epoch")

    return epoch, _iter, log_buffer


def get_current_lr(optimizer):
    if isinstance(optimizer, torch.optim.Optimizer):
        lr = [group["lr"] for group in optimizer.param_groups]
    elif isinstance(optimizer, dict):
        lr = dict()
        for name, optim in optimizer.items():
            lr[name] = [group["lr"] for group in optim.param_groups]
    else:
        raise RuntimeError("lr is not applicable because optimizer does not exist.")
    return lr


def log_info(
    optimizer,
    logger,
    log_buffer,
    log_dict,
    device,
    epoch,
    inner_iter,
    _iter,
    max_epochs,
    mode,
    data_length,
    precision=4,
    by_epoch=True,
    end_of_inner_iter=False,
):

    log_dict = OrderedDict(
        mode=mode,
        epoch=epoch,
        inner_iter=inner_iter,
        iter=_iter,
        **log_dict,
        LOCAL_RANK=os.environ.get("LOCAL_RANK", 0),
        # memory=get_max_memory(device),
    )
    if end_of_inner_iter:
        # import ipdb; ipdb.set_trace()
        # tmp = {k: meter.avg if not hasattr(meter, 'image') else meter.image for k, meter in log_buffer.meters.items()}
        # print_log(f"{os.environ.get('LOCAL_RANK')}, {tmp}", logger=logger)
        log_buffer.synchronize_between_processes()
        metrics = {
            k: meter.avg if not hasattr(meter, "image") else meter.image
            for k, meter in log_buffer.meters.items()
        }
    else:
        metrics = {
            k: meter.val if not hasattr(meter, "image") else meter.image
            for k, meter in log_buffer.meters.items()
        }

    log_dict = dict(log_dict, **metrics)

    # # only record lr of the first param group
    cur_lr = get_current_lr(optimizer)
    if isinstance(cur_lr, list):
        log_dict["lr"] = cur_lr[0]
    else:
        assert isinstance(cur_lr, dict)
        log_dict["lr"] = {}
        for k, lr_ in cur_lr.items():
            assert isinstance(lr_, list)
            log_dict["lr"].update({k: lr_[0]})

    if log_dict["mode"] == "train":
        if isinstance(log_dict["lr"], dict):
            lr_str = []
            for k, val in log_dict["lr"].items():
                lr_str.append(f"lr_{k}: {val:.3e}")
            lr_str = " ".join(lr_str)
        else:
            lr_str = f'lr: {log_dict["lr"]:.3e}'

        log_str = (
            f'[{log_dict["LOCAL_RANK"]}] Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) '
            f'[{log_dict["epoch"]}/{max_epochs}]'
            f'[{log_dict["inner_iter"]}/{data_length[log_dict["mode"]]}]\t'
        )

        log_str += f"{lr_str}, "

        # if any("time" in k for k in log_dict.keys()):
        #     log_str += f'iter_time: {log_dict["iter_time"]:.5f}, data_time: {log_dict["data_time"]:.5f}, '
        #     self.time_sec_tot += (log_dict['time'] * iter_interval)
        #     time_sec_avg = time_sec_tot / (runner.iter - self.start_iter + 1) #
        #     eta_sec = time_sec_avg * (max_iters - runner.iter - 1)
        #     eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        #     log_str += f'eta: {eta_str}, '
        #     log_str += f'time: {log_dict["time"]:.3f}, ' \
        #                     f'data_time: {log_dict["data_time"]:.3f}, '
        # # statistic memory
        # if torch.cuda.is_available():
        #     log_str += f'memory: {log_dict["memory"]}MB, '

    else:
        if by_epoch:
            log_str = (
                f'[{log_dict["LOCAL_RANK"]}] Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) '
                f'[{log_dict["epoch"]}/{max_epochs}]'
                f'[{log_dict["inner_iter"]}/{data_length[log_dict["mode"]]}]\t'
            )
        else:
            # log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'
            log_str = (
                f'[{log_dict["LOCAL_RANK"]}] Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) '
                f"[{log_dict['epoch']}/{max_epochs}]\t"
                f'[{log_dict["inner_iter"]}/{data_length[log_dict["mode"]]}]\t'
            )

    log_items = []
    for name, val in log_dict.items():
        # TODO: resolve this hack
        # these items have been in log_str
        if name in [
            "mode",
            "Epoch",
            "iter",
            "lr",
            "time",
            "data_time",
            "memory",
            "epoch",
            "inner_iter",
            "LOCAL_RANK",
        ]:
            continue
        if isinstance(val, np.ndarray):
            continue
        if isinstance(val, float):
            val = f"{val:.{precision}f}"
        log_items.append(f"{name}: {val}")
    log_str += ", ".join(log_items)
    print_log(log_str, logger=logger)


def run_iter(
    runner,
    data_batch,
    logger,
    epoch,
    _inner_iter,
    _iter,
    log_buffer,
    data_length,
    max_epochs,
    log_epoch_interval,
    train_log_iter_interval,
    val_log_iter_interval,
    test_log_iter_interval,
    dataset_cfg,
    results_dir="",
    idx=0,
    img_range=1.0,
    filename=None,
    save_fmt=None,
    eval_flag=False,
    test=None,
    test_mode=False,
    mode="train",  # "train | val | test",
):

    # log_dict = {}

    if mode == "train":
        log_dict = runner.train_step(
            data_batch, mode=mode, iteration=_iter, epoch=epoch, **dataset_cfg
        )
        # log_dict.update(runner.optimizer_wrapper.step(outputs))

    else:
        os.makedirs(results_dir, exist_ok=True)
        log_dict = runner.val_step(
            data_batch,
            mode=mode,
            iteration=_iter,
            epoch=epoch,
            idx=idx,
            test=test,
            img_range=img_range,
            save_dir=results_dir,
            save_fmt=save_fmt,
            test_mode=test_mode,
            filename=filename,
            eval=eval_flag,
            **dataset_cfg,
        )

    if not isinstance(log_dict, dict):  # outputs is not None and
        raise TypeError(
            f'"batch_processor()" or "model.train_step()"'
            'and "model.val_step()" must return a dict'
        )
    # ipdb.set_trace()
    # log_buffer will sync outputs per card/machine
    if log_dict is not None and "log_vars" in log_dict:
        log_buffer.update_dict(log_dict.pop("log_vars"))

    if mode == "train":
        runner.after_train_iter(log_buffer, _iter, _inner_iter)

    # self.metrics = {k: meter.avg for k, meter in self.log_buffer.meters.items()}
    # self.metrics = {k: meter.avg if not hasattr(meter, 'image') else meter.image for k, meter in self.log_buffer.meters.items()}
    # {'loss': loss, 'log_vars': {'loss': loss, 'metric_1': ..., 'metric_2': ....} }
    if epoch == 1 or epoch % log_epoch_interval == 0:
        if mode == "train" and (
            _inner_iter % train_log_iter_interval == 0
            or _inner_iter == data_length["train"]
        ):
            log_info(
                runner._optimizer,
                logger,  # if _inner_iter == data_length["train"] else None,
                log_buffer,
                log_dict,
                runner.device,
                epoch,
                _inner_iter,
                _iter,
                max_epochs,
                mode,
                data_length,
                end_of_inner_iter=_inner_iter == data_length["train"],
            )
        elif mode == "val" and (
            _inner_iter == 1
            or _inner_iter % val_log_iter_interval == 0
            or _inner_iter == data_length["val"]
        ):
            log_info(
                runner._optimizer,
                logger,
                log_buffer,
                log_dict,
                runner.device,
                epoch,
                _inner_iter,
                _iter,
                max_epochs,
                mode,
                data_length,
                end_of_inner_iter=_inner_iter == data_length["val"],
            )
        elif mode == "test" and (
            _inner_iter == 1
            or _inner_iter % test_log_iter_interval == 0
            or _inner_iter == data_length["test"]
        ):
            log_info(
                runner._optimizer,
                logger,
                log_buffer,
                log_dict,
                runner.device,
                epoch,
                _inner_iter,
                _iter,
                max_epochs,
                mode,
                data_length,
                end_of_inner_iter=_inner_iter == data_length["test"],
            )

    if hasattr(data_length, "train") and _inner_iter == data_length["train"]:
        runner.after_train_epoch(log_buffer, _iter, epoch, _inner_iter)

    return log_buffer
