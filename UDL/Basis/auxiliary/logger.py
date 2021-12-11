import logging
import os
from pathlib import Path
import time
import colorlog
# from termcolor import colored
import functools
'''
- root_path:
    - dataset
        - model_1
           - exp_name
               - model
               - log
    - dataset
        - model_1
           - exp_name
               - model
               - log
    - tb_log
        - dataset+model_1
        - exp_name
'''

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

# import logging
def log_string(out_str):
    # LOG_FOUT.write(out_str + '\n')
    # LOG_FOUT.flush()
    # print(out_str)
    logging.info(out_str)

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

@functools.lru_cache() # so that calling setup_logger multiple times won't add many handlers
def setup_logger(final_log_file, distributed_rank=0, color=True):
    # LOG_DIR = cfg.log_dir
    # LOG_FOUT = open(final_log_file, 'w')
    # head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=str(final_log_file), format='%(message)s')
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # console = logging.StreamHandler()
    # logging.getLogger('').addHandler(console)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # formatter = colorlog.ColoredFormatter(
    #     '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
    #     log_colors=log_colors_config)  # 日志输出格式

    if distributed_rank == 0:
        console = colorlog.StreamHandler()
        console.setLevel(logging.DEBUG)
        # if color:
        #     formatter = _ColorfulFormatter(
        #         colored("%(message)s", "green")
        #     )
        # else:
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s- %(message)s',
            log_colors=log_colors_config)  # 日志输出格式

        console.setFormatter(formatter)
        logger.addHandler(console)


def create_logger(cfg, cfg_name, phase='train', dist_print=0):

    root_output_dir = Path(cfg.out_dir)
    # set up logger in root_path
    if not root_output_dir.exists():
        if not dist_print: #rank 0-N, 0 is False
            print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    model = cfg.arch
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    # store all output except tb_log file
    final_output_dir = root_output_dir / dataset / model / cfg_name
    if cfg.eval:
        model_save_tmp = os.path.dirname(cfg.resume).split('/')[-1]
    else:
        model_save_tmp = "model_{}".format(time_str)

    model_save_dir = final_output_dir / model_save_tmp
    if not dist_print:
        print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = '{}_{}'.format(cfg_name, time_str)
    # a logger to save results
    log_file = '{}_{}.log'.format(cfg_name, phase)
    if cfg.eval:
        final_log_file = model_save_dir / log_file
        tensorboard_log_dir = None
    else:
        final_log_file = final_output_dir / log_file
        # tensorboard_log
        tensorboard_log_dir = root_output_dir / Path(cfg.log_dir) / dataset / model / cfg_name
        if not dist_print:
            print('=> creating tfb logs {}'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)


    setup_logger(final_log_file)

    return str(final_output_dir), str(model_save_dir), str(tensorboard_log_dir) #logger,

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DIR', default='dataset',
                        help='path to dataset')
    parser.add_argument('--arch', metavar='DIR', default='PanHrnet',
                        help='path to save model')
    parser.add_argument('--log_dir', metavar='DIR', default='tfb_log',
                        help='path to save log')
    parser.add_argument('--out_dir', metavar='DIR', default='output1',
                        help='useless in this script.')
    args = parser.parse_args()
    create_logger(args, "test")
    logging.info(111)
    logging.debug(111)
    logging.warning(111)