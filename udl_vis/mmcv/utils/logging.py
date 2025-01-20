# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:

import json
from collections import defaultdict
import logging
import os
import functools
import torch.distributed as dist
import colorlog
import time
from pathlib import Path
from datetime import datetime

# from importlib import reload

# from udl_vis.mmcv.runner.dist_utils import master_only


logger_initialized = {}

log_colors_config = {
    "DEBUG": "cyan",
    "INFO": "white",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}


# TODO: Depre
# the same as "get_root_logger"
def create_logger(
    cfg=None, cfg_name=None, dist_print=0, work_dir=None, log_level=logging.INFO
):
    return get_logger(cfg.experimental_desc, cfg, cfg_name, work_dir, log_level)


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    name, final_log_file, color=True, fixed_time=None, fixed_time_str=None
):
    # LOG_DIR = cfg.log_dir
    # LOG_FOUT = open(final_log_file, 'w')
    # head = '%(asctime)-15s %(message)s'

    # logging.basicConfig(filename=str(final_log_file).replace('\\', '/'), format='%(message)s', level=logging.INFO

    logger = logging.getLogger(name)
    # if name in logger_initialized:
    #     return logger

    # for handler in logger.root.handlers:
    #     if type(handler) is logging.StreamHandler:
    #         handler.setLevel(logging.ERROR)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    fixed_time = datetime.fromtimestamp(fixed_time)
    if color:
        # 设置LogRecord对象的固定时间戳
        class FixedTimeFormatter(colorlog.ColoredFormatter):
            def formatTime(self, record, datefmt=None):
                # cnt_t = record.created - fixed_time
                cnt_t = datetime.fromtimestamp(record.created) - fixed_time
                # ct = time.gmtime(cnt_t)
                # s = time.strftime('%H:%M:%S', ct)
                return fixed_time_str + "," + f"{cnt_t.days}:{str(cnt_t)}"

        color_formatter = FixedTimeFormatter(
            "%(log_color)s %(name)s - %(asctime)s - %(message)s",
            log_colors=log_colors_config,
        )  # 日志输出格式

        formatter = logging.Formatter("%(name)s - %(message)s")
    else:
        color_formatter = formatter = logging.Formatter(
            "%(name)s - %(asctime)s - %(message)s"
        )

    if rank == 0:
        # console = logging.StreamHandler()
        console = colorlog.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(color_formatter)

        handler = logging.FileHandler(str(final_log_file).replace("\\", "/"))
        handler.setLevel(logging.INFO)  # log_level
        handler.setFormatter(formatter)

        logger.addHandler(console)
        logger.addHandler(handler)

    # if rank == 0:
    #     logger.setLevel(logging.INFO)  # log_level
    # else:
    #     logger.setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logger_initialized[name] = True

    return logger


def get_logger(
    name=None,
    cfg=None,
    cfg_name=None,
    work_dir=None,
    log_level=logging.INFO,  # phase='train',
    file_mode="w",
):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    # reload(logging)

    # if name in logger_initialized or cfg is None:
    #     # if cfg is None:  # cfg.use_log
    #     print(f"logger_initialized: {logger_initialized}")
    #     logger = logging.getLogger(name)
    #     logger.handlers.clear()
    #     return logger

    # else:
    #     return None
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    # for logger_name in logger_initialized:
    #     if name.startswith(logger_name):
    #         if cfg.use_log:
    #             return logging.getLogger(name)
    #         else:
    #             return None
    if work_dir is not None:
        tic = time.time()
        time_str = datetime.fromtimestamp(tic).strftime("%Y-%m-%d-%H-%M-%S")
        # work_dir = work_dir + "/" + time_str
        final_log_file = work_dir + "/1.log"
        os.makedirs(work_dir, exist_ok=True)
        logger = setup_logger(name, final_log_file, True, tic, time_str)
        return logger
        # return (
        #     logger,
        #     work_dir,
        #     os.path.join(work_dir, "checkpoints"),
        #     os.path.join(work_dir, "summaries"),
        # )

    logger = tensorboard_log_dir = final_output_dir = model_save_dir = None

    root_output_dir = Path(cfg.work_dir)
    # set up logger in root_path
    if not root_output_dir.exists():
        # if not dist_print: #rank 0-N, 0 is False
        print("=> creating {}".format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    assert isinstance(dataset, dict), print(
        f"{dataset}'s type is {type(dataset)}, not a dict. "
    )
    tic = time.time()
    time_str = datetime.fromtimestamp(tic).strftime("%Y-%m-%d-%H-%M-%S")
    # if not dist_print:
    if os.path.exists(cfg.resume_from) and (
        dataset.get("train", None) is None or cfg.eval
    ):
        # model_save_dir = '/'.join([cfg.out_dir.replace('\\', '/'), cfg.arch])
        model_save_dir = os.path.dirname(cfg.resume_from.replace("\\", "/"))
        log_file = "{}_{}.log".format(
            cfg_name, model_save_dir.split("/")[-1].split("_")[-1]
        )
        final_output_dir = model_save_dir
        final_log_file = Path(model_save_dir) / log_file

    else:
        if (
            "train" not in dataset.keys()
            or "valid" not in dataset.keys()
            or "test" not in dataset.keys()
        ):
            dataset = list(dataset.values())[0]
            dataset, file_extension = os.path.splitext(dataset)
        else:
            if cfg.eval:
                dataset = (
                    dataset.get("test")
                    if dataset.get("test", None) is not None
                    else dataset.get("valid")
                )
            else:
                dataset = (
                    dataset.get("train")
                    if dataset.get("train", None) is not None
                    else dataset.get("valid")
                )
        model = cfg.arch
        cfg_name_base = os.path.basename(cfg_name).split(".")[0]

        # store all output except tb_log file
        final_output_dir = root_output_dir / dataset / model / cfg_name
        if cfg.eval:
            model_save_tmp = os.path.dirname(cfg.resume_from).split("/")[-1]
        else:
            model_save_tmp = "model_{}".format(time_str)

        model_save_dir = final_output_dir / model_save_tmp

        print_log("=> creating {}".format(final_output_dir))
        if cfg.use_save:
            model_save_dir.mkdir(parents=True, exist_ok=True)
            # os.makedirs(str(model_save_dir), exist_ok=True)
        if cfg.use_log:
            final_output_dir.mkdir(parents=True, exist_ok=True)

        cfg_name = "{}_{}".format(cfg_name_base, time_str)
        # a logger to save results
        log_file = "{}.log".format(cfg_name)
        # if cfg.eval:
        #     final_log_file = model_save_dir / log_file
        # else:
        #     final_log_file = final_output_dir / log_file
        final_log_file = final_output_dir / log_file
        # tensorboard_log
        tensorboard_log_dir = (
            root_output_dir / Path(cfg.log_dir) / dataset / model / cfg_name
        )
        # if not dist_print:
        # print_log('=> creating tfb logs {}'.format(tensorboard_log_dir))
        # tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_log:
        logger = setup_logger(name, final_log_file, cfg.use_colorlog, tic, time_str)
    if not cfg.use_save:
        tensorboard_log_dir = model_save_dir = None
        return logger, final_output_dir, model_save_dir, tensorboard_log_dir
    else:
        return (
            logger,
            str(final_output_dir),
            str(model_save_dir),
            str(tensorboard_log_dir),
        )  # logger,


def print_log(msg, logger=None, level=logging.INFO, clear_logger=False):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )
    if clear_logger:
        logger_initialized = {}


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict[int, dict[str, list]]:
            Key is the epoch, value is a sub dict. The keys in each sub dict
            are different metrics, e.g. memory, bbox_mAP, and the value is a
            list of corresponding values in all iterations in this epoch.

            .. code-block:: python

                # An example output
                {
                    1: {'iter': [100, 200, 300], 'loss': [6.94, 6.73, 6.53]},
                    2: {'iter': [100, 200, 300], 'loss': [6.33, 6.20, 6.07]},
                    ...
                }
    """
    log_dict = dict()
    with open(json_log, "r") as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if "epoch" not in log:
                continue
            epoch = log.pop("epoch")
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict


# class RichLogger:
#     def __init__(self): ...

#     @functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
#     def setup_logger(
#         name, final_log_file, color=True, fixed_time=None, fixed_time_str=None
#     ):
#         # LOG_DIR = cfg.log_dir
#         # LOG_FOUT = open(final_log_file, 'w')
#         # head = '%(asctime)-15s %(message)s'

#         # logging.basicConfig(filename=str(final_log_file).replace('\\', '/'), format='%(message)s', level=logging.INFO

#         logger = logging.getLogger(name)
#         # if name in logger_initialized:
#         #     return logger

#         # for handler in logger.root.handlers:
#         #     if type(handler) is logging.StreamHandler:
#         #         handler.setLevel(logging.ERROR)

#         if dist.is_available() and dist.is_initialized():
#             rank = dist.get_rank()
#         else:
#             rank = 0

#         fixed_time = datetime.fromtimestamp(fixed_time)
#         if color:
#             # 设置LogRecord对象的固定时间戳
#             class FixedTimeFormatter(colorlog.ColoredFormatter):
#                 def formatTime(self, record, datefmt=None):
#                     # cnt_t = record.created - fixed_time
#                     cnt_t = datetime.fromtimestamp(record.created) - fixed_time
#                     # ct = time.gmtime(cnt_t)
#                     # s = time.strftime('%H:%M:%S', ct)
#                     return fixed_time_str + "," + f"{cnt_t.days}:{str(cnt_t)}"

#             color_formatter = FixedTimeFormatter(
#                 "%(log_color)s %(name)s - %(asctime)s - %(message)s",
#                 log_colors=log_colors_config,
#             )  # 日志输出格式

#             formatter = logging.Formatter("%(name)s - %(message)s")
#         else:
#             color_formatter = formatter = logging.Formatter(
#                 "%(name)s - %(asctime)s - %(message)s"
#             )

#         if rank == 0:
#             # console = logging.StreamHandler()
#             console = colorlog.StreamHandler()
#             console.setLevel(logging.INFO)
#             console.setFormatter(color_formatter)

#             handler = logging.FileHandler(str(final_log_file).replace("\\", "/"))
#             handler.setLevel(logging.INFO)  # log_level
#             handler.setFormatter(formatter)

#             logger.addHandler(console)
#             logger.addHandler(handler)

#         # if rank == 0:
#         #     logger.setLevel(logging.INFO)  # log_level
#         # else:
#         #     logger.setLevel(logging.ERROR)
#         logger.setLevel(logging.INFO)
#         logger_initialized[name] = True

#         return logger

#     def get_logger(
#         self,
#         name=None,
#         cfg=None,
#         cfg_name=None,
#         work_dir=None,
#         log_level=logging.INFO,  # phase='train',
#         file_mode="w",
#     ):
#         """Initialize and get a logger by name.

#         If the logger has not been initialized, this method will initialize the
#         logger by adding one or two handlers, otherwise the initialized logger will
#         be directly returned. During initialization, a StreamHandler will always be
#         added. If `log_file` is specified and the process rank is 0, a FileHandler
#         will also be added.

#         Args:
#             name (str): Logger name.
#             log_file (str | None): The log filename. If specified, a FileHandler
#                 will be added to the logger.
#             log_level (int): The logger level. Note that only the process of
#                 rank 0 is affected, and other processes will set the level to
#                 "Error" thus be silent most of the time.
#             file_mode (str): The file mode used in opening log file.
#                 Defaults to 'w'.

#         Returns:
#             logging.Logger: The expected logger.
#         """
#         reload(logging)

#         if name in logger_initialized or cfg is None:
#             # if cfg is None:  # cfg.use_log
#             print(f"logger_initialized: {logger_initialized}")
#             logger = logging.getLogger(name)
#             logger.handlers.clear()
#             return logger

#             # else:
#             #     return None
#         # handle hierarchical names
#         # e.g., logger "a" is initialized, then logger "a.b" will skip the
#         # initialization since it is a child of "a".
#         # for logger_name in logger_initialized:
#         #     if name.startswith(logger_name):
#         #         if cfg.use_log:
#         #             return logging.getLogger(name)
#         #         else:
#         #             return None
#         if work_dir is not None:
#             final_log_file = work_dir + "/1.log"
#             tic = time.time()
#             time_str = datetime.fromtimestamp(tic).strftime("%Y-%m-%d-%H-%M-%S")
#             logger = setup_logger(name, final_log_file, cfg.use_colorlog, tic, time_str)
#             return (
#                 logger,
#                 work_dir,
#                 os.path.join(work_dir, "checkpoints"),
#                 os.path.join(work_dir, "summaries"),
#             )

#         logger = tensorboard_log_dir = final_output_dir = model_save_dir = None

#         root_output_dir = Path(cfg.out_dir)
#         # set up logger in root_path
#         if not root_output_dir.exists():
#             # if not dist_print: #rank 0-N, 0 is False
#             print("=> creating {}".format(root_output_dir))
#             root_output_dir.mkdir(parents=True, exist_ok=True)

#         dataset = cfg.dataset
#         assert isinstance(dataset, dict), print(
#             f"{dataset}'s type is {type(dataset)}, not a dict. "
#         )
#         tic = time.time()
#         time_str = datetime.fromtimestamp(tic).strftime("%Y-%m-%d-%H-%M-%S")
#         # if not dist_print:
#         if os.path.exists(cfg.resume_from) and (
#             dataset.get("train", None) is None or cfg.eval
#         ):
#             # model_save_dir = '/'.join([cfg.out_dir.replace('\\', '/'), cfg.arch])
#             model_save_dir = os.path.dirname(cfg.resume_from.replace("\\", "/"))
#             log_file = "{}_{}.log".format(
#                 cfg_name, model_save_dir.split("/")[-1].split("_")[-1]
#             )
#             final_output_dir = model_save_dir
#             final_log_file = Path(model_save_dir) / log_file

#         else:
#             if (
#                 "train" not in dataset.keys()
#                 or "valid" not in dataset.keys()
#                 or "test" not in dataset.keys()
#             ):
#                 dataset = list(dataset.values())[0]
#                 dataset, file_extension = os.path.splitext(dataset)
#             else:
#                 if cfg.eval:
#                     dataset = (
#                         dataset.get("test")
#                         if dataset.get("test", None) is not None
#                         else dataset.get("valid")
#                     )
#                 else:
#                     dataset = (
#                         dataset.get("train")
#                         if dataset.get("train", None) is not None
#                         else dataset.get("valid")
#                     )
#             model = cfg.arch
#             cfg_name_base = os.path.basename(cfg_name).split(".")[0]

#             # store all output except tb_log file
#             final_output_dir = root_output_dir / dataset / model / cfg_name
#             if cfg.eval:
#                 model_save_tmp = os.path.dirname(cfg.resume_from).split("/")[-1]
#             else:
#                 model_save_tmp = "model_{}".format(time_str)

#             model_save_dir = final_output_dir / model_save_tmp

#             print_log("=> creating {}".format(final_output_dir))
#             if cfg.use_save:
#                 model_save_dir.mkdir(parents=True, exist_ok=True)
#                 # os.makedirs(str(model_save_dir), exist_ok=True)
#             if cfg.use_log:
#                 final_output_dir.mkdir(parents=True, exist_ok=True)

#             cfg_name = "{}_{}".format(cfg_name_base, time_str)
#             # a logger to save results
#             log_file = "{}.log".format(cfg_name)
#             # if cfg.eval:
#             #     final_log_file = model_save_dir / log_file
#             # else:
#             #     final_log_file = final_output_dir / log_file
#             final_log_file = final_output_dir / log_file
#             # tensorboard_log
#             tensorboard_log_dir = (
#                 root_output_dir / Path(cfg.log_dir) / dataset / model / cfg_name
#             )
#             # if not dist_print:
#             # print_log('=> creating tfb logs {}'.format(tensorboard_log_dir))
#             # tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

#         if cfg.use_log:
#             logger = setup_logger(name, final_log_file, cfg.use_colorlog, tic, time_str)
#         if not cfg.use_save:
#             tensorboard_log_dir = model_save_dir = None
#             return logger, final_output_dir, model_save_dir, tensorboard_log_dir
#         else:
#             return (
#                 logger,
#                 str(final_output_dir),
#                 str(model_save_dir),
#                 str(tensorboard_log_dir),
#             )  # logger,

#     def print(): ...

#     def print_log(): ...

#     def console(): ...

#     @master_only
#     def log(self, runner):
#         tags = self.get_loggable_tags(runner, allow_text=True)
#         for tag, val in tags.items():
#             if isinstance(val, str):
#                 self.writer.add_text(tag, val, self.get_iter(runner))
#             elif not isinstance(val, np.ndarray):
#                 self.writer.add_scalar(tag, val, self.get_iter(runner))

#             if isinstance(val, np.ndarray):
#                 if runner.epoch % self.interval == 0:
#                     self.writer.add_image(
#                         tag, val, dataformats="HWC", global_step=self.get_iter(runner)
#                     )


# if __name__ == "__main__":
#     logger = RichLogger()
