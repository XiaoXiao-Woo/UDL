# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import shutil
import time
import warnings
import time
import datetime
import torch
from udl_vis import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
from udl_vis.mmcv.utils.logging import print_log
# from udl_vis.Basis.dist_utils import get_dist_info
from torch.utils.data import DataLoader
# from torch.utils.data import SequentialSampler, RandomSampler

# from udl_vis.Basis.auxiliary import set_random_seed
# import random


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def run_iter(self, data_batch, train_mode, iteration, **kwargs):
        with self.amp.autocast(
            enabled=self.opt_cfg["mixed_precision"] != None,
            device_type="cuda",
        ):
            if self.batch_processor is not None:
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=train_mode, **kwargs
                )
            elif train_mode:

                outputs = self.model.train_step(
                    data_batch,
                    iteration=iteration,
                    epoch=self.epoch,
                    **kwargs,
                )

            else:
                outputs = self.model.val_step(
                    data_batch,
                    # self.optimizer,
                    iteration=iteration,
                    epoch=self.epoch,
                    **kwargs,
                )

        if not isinstance(outputs, dict):  # outputs is not None and
            raise TypeError(
                f'"batch_processor()" or "model.train_step()"'
                'and "model.val_step()" must return a dict'
            )

        if outputs is not None and "log_vars" in outputs:
            self.log_buffer.update_dict(outputs["log_vars"])
        # self.metrics = {k: meter.avg for k, meter in self.log_buffer.meters.items()}
        # self.metrics = {k: meter.avg if not hasattr(meter, 'image') else meter.image for k, meter in self.log_buffer.meters.items()}
        # {'loss': loss, 'log_vars': {'loss': loss, 'metric_1': ..., 'metric_2': ....} }
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        # epochs = kwargs['epochs']
        if hasattr(self.model, "train"):
            self.model.train()
        elif isinstance(self.model.model, dict):
            for name in self.model.model.keys():
                self.model.model[name].train()
        else:
            self.model.model.train()
        # if not isinstance(self.model, dict):
        #     self.model.train()
        # else:
        #     for name in self.model.keys():
        #         self.model[name].train()

        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        # self.data_loader.dataset.set_epoch(self.epoch)
        if hasattr(self.data_loader, "sampler") and hasattr(
            self.data_loader.sampler, "set_epoch"
        ):
            # pytorch
            # set_random_seed(1)
            # print(self.rank)
            # random.seed(1)
            torch.manual_seed(1)
            # torch.cuda.manual_seed(1)
            # torch.cuda.manual_seed_all(1)
            self.data_loader.sampler.set_epoch(self.epoch)
            # print(f"set_epoch: {self.epoch}")
        else:
            # accelerate
            ...
        self.call_hook("before_train_epoch")
        # self.data_loader.sampler.generator.manual_seed(self.epoch)
        # tic = time.time()
        # time.sleep(2)  # Prevent possible deadlock during epoch transition
        if hasattr(self.data_loader.dataset, "gen_transform"):
            self.data_loader.dataset.gen_transform(self.generator)
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, iteration=i, **kwargs)
            self.call_hook("after_train_iter")
            self._iter += 1
            # break
        # self.metrics.update(epoch_time=time.time() - tic)
        # self.metrics = {k: meter.avg for k, meter in self.log_buffer.meters.items()}
        self.call_hook("after_train_epoch")
        self._epoch += 1

    def train_iter(self, data_loader, **kwargs):
        if hasattr(self.model, "train"):
            self.model.train()
        elif isinstance(self.model.model, dict):
            for name in self.model.model.keys():
                self.model.model[name].train()
        else:
            self.model.model.train()
        self.mode = "train"
        # self.data_loader = data_loader
        # iter_loader = iter(loader)
        try:
            data_batch = next(self.old_data_loader)
        except StopIteration:
            # del data_loaders
            # torch.cuda.empty_cache()
            data_loader = iter(self.data_loader)
            self.old_data_loader = data_loader
            data_batch = next(data_loader)
        except TypeError:
            data_loader = iter(self.data_loader)
            self.old_data_loader = data_loader
            data_batch = next(data_loader)
        # try:
        #     data_batch = next(self.old_data_loader)
        # except StopIteration:
        #     self.old_data_loader = iter(self.data_loader)
        #     data_batch = next(self.old_data_loader)

        self.call_hook("before_train_iter")

        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError("model.train_step() must return a dict")
        if "log_vars" in outputs:
            self.log_buffer.update_dict(outputs["log_vars"])
        self.outputs = outputs

        # if (self.every_n_inner_iters(runner, self.iter_interval) or self.end_of_n_inner_iters(
        #         runner) and self.every_n_epochs(runner, self.epoch_interval)):
        #     self.metrics = {k: meter.avg if not hasattr(meter, 'image') else meter.image for k, meter in
        #                     runner.log_buffer.meters.items()}
        #     runner.log_buffer.ready = True
        # self.metrics = {k: meter.avg if not hasattr(meter, 'image') else meter.image for k, meter in
        #                 runner.log_buffer.meters.items()}

        self.call_hook("after_train_iter")
        # self.data_length['train'] = 1
        if (self._inner_iter + 1) % self.data_length["train"] == 0:
            self.call_hook("after_train_epoch")
            self._inner_iter = 0
            self.epoch += 1

        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        kwargs["test_mode"] = False if not kwargs.get("test_mode", None) else True
        if hasattr(self.model, "eval"):
            self.model.eval()
        elif isinstance(self.model.model, dict):
            for name in self.model.model.keys():
                self.model.model[name].eval()
        else:
            self.model.model.eval()
        self.mode = "val" if not kwargs.get("test_mode", None) else "test"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        # time.sleep(2)  # Prevent possible deadlock during epoch transition
        tic = time.time()
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")
            self.run_iter(
                data_batch,
                train_mode=False,
                idx=i,
                img_range=self.opt_cfg["img_range"],
                eval=self.opt_cfg["eval"],
                test=self.opt_cfg["test"],
                save_fmt=self.opt_cfg["save_fmt"],
                filename=(
                    data_batch.get("filename", [None])[0]
                    if isinstance(data_batch, dict)
                    else None
                ),
                save_dir=self.save_dir,
                iteration=(
                    self._epoch if self.runner_mode == "epoch" else self._inner_iter
                ),
                device=self.opt_cfg["device"],
                **kwargs,
            )
            # val_mode=self.opt_cfg['val_mode'])
            self.call_hook("after_val_iter")
            # break
        print_log(f"test time: {time.time() - tic}", logger=self.logger)
        self.call_hook("after_val_epoch")
        # if self.opt_cfg['eval'] or self.eval_flag:
        # self.eval_flag = False

    # @torch.no_grad()
    def test(self, data_loader, **kwargs):
        return self.val(data_loader, test_mode=True, **kwargs)

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, dict)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow), print_log(
            f"{len(data_loaders)} == {len(workflow)}"
        )
        if max_epochs is not None:
            warnings.warn(
                "setting max_epochs in run is deprecated, "
                "please set max_epochs in runner_config",
                DeprecationWarning,
            )
            self._max_epochs = max_epochs

        assert (
            self._max_epochs is not None
        ), "max_epochs must be specified during instantiation"
        train_flag = any("train" in mode for mode, _ in workflow)
        self.eval_flag = not train_flag

        self.data_length = {"train": 1, "test": 1}
        for i, flow in enumerate(workflow):
            mode, interval = flow
            if not isinstance(data_loaders[mode], DataLoader) and callable(
                data_loaders[mode]
            ):
                data_loaders[mode] = data_loaders[mode]()
            self.data_length[mode] = len(data_loaders[mode])
            if mode == "train":
                self.train_interval = interval
            if mode == "test":
                self.test_interval = interval
        if "train" in data_loaders.keys():
            self._max_iters = self._max_epochs * self.data_length["train"]

        self.data_loaders = data_loaders
        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        print_log(
            f"Start running, host: {get_host_info()}, work_dir: {work_dir}",
            logger=self.logger,
        )
        print_log(
            f"Hooks will be executed in the following order:\n{self.get_hook_info()}",
            logger=self.logger,
        )
        print_log(
            f"workflow: {workflow}, max: {self._max_epochs} epochs", logger=self.logger
        )
        self.call_hook("before_run")
        tic = time.time()
        # from 1 to self._max_epochs, not from 0

        if self.runner_mode == "iter":
            self.train = self.train_iter
            self.run_by_iter(train_flag, workflow, data_loaders, **kwargs)
        else:
            self.run_by_epoch(train_flag, workflow, data_loaders, **kwargs)
        # time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_run")
        total_time = time.time() - tic
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_log("Training time {}".format(total_time_str), logger=self.logger)

    def run_by_epoch(self, train_flag, workflow, data_loaders, **kwargs):
        mode = "train"
        while self.epoch < self._max_epochs and train_flag:  # or self.eval_flag:
            for i, flow in enumerate(workflow):
                mode, epochs = flow

                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an ' "epoch"
                        )
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        "mode in workflow must be a str, but got {}".format(type(mode))
                    )

                for epoch in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[mode], **kwargs)

            if self.earlyStop:
                print_log(
                    "model train has diverged, python will stop training",
                    logger=self.logger,
                )
                break
        if self.eval_flag and mode == "train":
            if "val" in self.data_loaders.keys():
                self.val(self.data_loaders["val"], **kwargs)
            if "test" in self.data_loaders.keys():
                self.val(self.data_loaders["test"], test_mode=True, **kwargs)

    def run_by_iter(self, train_flag, workflow, data_loaders, **kwargs):
        self.data_loader = data_loaders.get("train", None)
        self.old_data_loader = None  # iter(self.data_loader)
        # old_data_loader = None
        while (self.iter < self._max_iters and train_flag) or not self.eval_flag:
            for i, flow in enumerate(workflow):
                mode, iters = flow

                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an ' "epoch"
                        )
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        "mode in workflow must be a str, but got {}".format(type(mode))
                    )
                # if mode == 'train' and not self.iter >= self._max_iters:
                # if self._iter %  self.data_length['train'] == 0:
                #     data_loader = iter(self.data_loader)#iter(data_loaders[mode])
                #     self.old_data_loader = data_loader
                # else:
                #     data_loader = self.old_data_loader
                # elif mode != 'train':
                # data_loader = data_loaders[mode]
                if mode == "train":
                    self.call_hook("before_train_epoch")
                for it in range(iters):
                    if mode == "train" and self.iter >= self._max_iters:
                        break
                    epoch_runner(data_loaders[mode], **kwargs)
                # if mode == 'train':
                #     old_data_loader = self.old_data_loader
            if self.earlyStop:
                print_log(
                    "model train has diverged, python will stop training",
                    logger=self.logger,
                )
                break

    def save_checkpoint(
        self,
        out_dir,
        filename_tmpl="epoch_{}.pth",
        save_optimizer=True,
        meta=None,
        create_symlink=True,
    ):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"meta should be a dict or None, but got {type(meta)}")
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = os.path.join(out_dir, "latest.pth")
            if platform.system() != "Windows":
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Runner was deprecated, please use EpochBasedRunner instead",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
