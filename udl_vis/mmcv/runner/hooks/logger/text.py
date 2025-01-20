# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist

from udl_vis import mmcv
from udl_vis.mmcv.fileio.file_client import FileClient
from udl_vis.mmcv.utils import is_tuple_of, scandir
from ..hook import HOOKS
from .base import LoggerHook
from udl_vis.mmcv.utils.logging import print_log


@HOOKS.register_module()
class TextLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        interval_exp_name (int, optional): Logging interval for experiment
            name. This feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
        out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
            If ``out_dir`` is specified, logs will be copied to a new directory
            which is the concatenation of ``out_dir`` and the last level
            directory of ``runner.work_dir``. Default: None.
            `New in version 1.3.16.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.3.16.`
        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
            `New in version 1.3.16.`
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """

    def __init__(self,
                 runner_mode="epoch",
                 epoch_interval=10,
                 iter_interval=10,
                 precision=5,
                 use_json=False,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000,
                 out_dir=None,
                 out_suffix=('.log.json', '.log', '.py'),
                 keep_local=True,
                 file_client_args=None):
        super(TextLoggerHook, self).__init__(epoch_interval, iter_interval, ignore_last, reset_flag,
                                             runner_mode)
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        self.use_json = use_json
        self.status_end_of_n_inner_iters = False
        self.precision = precision

        if out_dir is None and file_client_args is not None:
            raise ValueError(
                'file_client_args should be "None" when `out_dir` is not'
                'specified.')
        self.out_dir = out_dir

        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be  "None" or string or tuple of '
                            'string, but got {out_dir}')
        self.out_suffix = out_suffix

        self.keep_local = keep_local
        self.file_client_args = file_client_args
        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(file_client_args,
                                                       self.out_dir)

    def before_run(self, runner):
        super(TextLoggerHook, self).before_run(runner)

        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(self.file_client_args,
                                                       self.out_dir)
            # The final `self.out_dir` is the concatenation of `self.out_dir`
            # and the last level directory of `runner.work_dir`
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)
            print_log(
                (f'Text logs will be saved to {self.out_dir} by '
                 f'{self.file_client.name} after the training process.'), logger=runner.logger)

        self.start_iter = runner.iter
        # self.data_length = runner.data_length
        self.max_epochs = runner.max_epochs
        self.json_log_path = None
        if runner.work_dir is not None:
            self.json_log_path = osp.join(runner.work_dir,
                                          f'{runner.timestamp}.log.json')
        if runner.meta is not None and runner.logger is not None and self.use_json:
            self._dump_log(runner.meta, runner)

    def _get_max_memory(self, runner):
        device = getattr(runner.model.model, 'output_device', None) # model
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                print_log(exp_info, logger=runner.logger)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            # if self.by_epoch:
            #     log_str = f'Iter [{log_dict["iter"]}] Epoch [{log_dict["epoch"]}/{self.max_epochs}]' \
            #               f'[{log_dict["inner_iter"]}/{self.data_length["train"]}]\t'
            # else:
            #     log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            # if self.by_epoch:
            #     log_str = f'Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) ' \
            #               f'[{log_dict["epoch"]}]/[{self.max_epochs}] ' \
            #               f'[{log_dict["inner_iter"]}/{runner.data_length[log_dict["mode"]]}]\t'
            # else:
            #     # self.epoch = int(np.ceil(log_dict["iter"] / self.data_length[log_dict["mode"]]))
            #     # runner.epoch = self.epoch
            #     # log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'
            #     log_str = f'Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) ' \
            #               f'[{runner.epoch + 1}]/[{self.max_epochs}] ' \
            #               f'[{log_dict["inner_iter"]}]/[{runner.data_length[log_dict["mode"]]}]\t'
            log_str = f'Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) ' \
                      f'[{log_dict["epoch"]}]/[{self.max_epochs}] ' \
                      f'[{log_dict["inner_iter"]}/{runner.data_length[log_dict["mode"]]}]\t'

            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.iter_interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1) #
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, ' \
                           f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}MB, '
                # if self.status_end_of_n_inner_iters:
                #     new_opt_param_groups = np.sum([param.cpu().detach().numpy().sum() for param in
                #                         runner.optimizer.param_groups[0]['params']])
                #     new_opt_state = np.sum([param.cpu().detach().numpy().sum() for _, v in
                #                         runner.optimizer.state.items()  for param in list(v.values())])
                #     log_str += f"opt_param_groups: {new_opt_param_groups}, opt_state: {new_opt_state} "
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) ' \
                          f'[{log_dict["epoch"]}]/[{self.max_epochs}] ' \
                          f'[{log_dict["inner_iter"]}/{runner.data_length[log_dict["mode"]]}]\t'
            else:
                # log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'
                log_str = f'Iter [{log_dict["iter"]}] Epoch({log_dict["mode"]}) ' \
                          f'[{runner.epoch + 1}/[{self.max_epochs}]\t' \
                          f'[{log_dict["inner_iter"]}/{runner.data_length[log_dict["mode"]]}]\t'

        log_items = []
        precision = self.precision
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch', 'inner_iter'
            ]:
                continue
            if isinstance(val, np.ndarray):
                continue
            if isinstance(val, float):
                val = f'{val:.{precision}f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        print_log(log_str, logger=runner.logger)

    def _dump_log(self, log_dict, runner):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        # only append log at last line
        if runner.rank == 0:
            with open(self.json_log_path, 'a+') as f:
                mmcv.dump(json_log, f, file_format='json')
                f.write('\n')

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, runner):

        if 'eval_iter_num' in runner.log_buffer.meters: #output
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.meters.pop('eval_iter_num') #output
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)
        
        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            inner_iter=cur_iter,
            iter=self.get_iter(runner))

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.meters:#output
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        # if self.by_epoch or log_dict['mode'] != 'train':
        # metrics = {k: meter.avg if not hasattr(meter, 'image') else meter.image for k, meter in runner.log_buffer.meters.items()}
        log_dict = dict(log_dict, **self.metrics) #output
        # if self.status_end_of_n_inner_iters:
        #     values = [np.mean(v['exp_avg'].cpu().numpy()) for k, v in runner.optimizer.state_dict()['state'].items()]
        #     log_dict.update(opt=np.mean(values))


        self._log_info(log_dict, runner)
        if self.use_json and runner.logger is not None:
            self._dump_log(log_dict, runner)
        return log_dict

    def after_run(self, runner):
        # copy or upload logs to self.out_dir
        if self.out_dir is not None:
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                out_filepath = self.file_client.join_path(
                    self.out_dir, filename)
                with open(local_filepath, 'r') as f:
                    self.file_client.put_text(f.read(), out_filepath)

                print_log(
                    (f'The file {local_filepath} has been uploaded to '
                     f'{out_filepath}.'), logger=runner.logger)

                if not self.keep_local:
                    os.remove(local_filepath)
                    print_log(
                        (f'{local_filepath} was removed due to the '
                         '`self.keep_local=False`'), logger=runner.logger)
