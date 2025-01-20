# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from pathlib import Path
import warnings
from udl_vis.mmcv.utils.logging import print_log
from udl_vis.Basis.dist_utils import master_only
from math import inf
import os
import platform
from udl_vis import mmcv
import shutil
import numpy as np
import re
from typing import List
from udl_vis.Basis.dev_utils.deprecated import deprecated_context


class ModelCheckpoint(object):

    _default_greater_keys = [
        "acc",
        "top",
        "AR@",
        "auc",
        "precision",
        "mAP",
        "mDice",
        "mIoU",
        "mAcc",
        "aAcc",
        "psnr",
        "ssim",
        "q",
    ]
    _default_best_prec1 = {"greater": -inf, "less": inf}
    _default_less_keys = ["loss", "sam", "ergas"]

    def __init__(
        self,
        indicator: str,
        save_latest_limit=1,
        save_top_k: int = 1,
        logger=None,
        greater_keys=None,
        less_keys=None,
    ):
        """
        Args:
            save_interval:
            save_top_k: ``save_top_k == k``,
                        if ``save_top_k == 0``, no models are saved.
                        if ``save_top_k == -1``, all models are saved.
                        Please note that the monitors are checked every ``every_n_epochs`` epochs.
            Returns:
        """
        self.logger = logger
        self.save_latest_limit = save_latest_limit
        self.save_top_k = save_top_k
        indicator = indicator.lower()
        self.indicator = "top-1" if indicator == "top" else indicator

        if greater_keys is None:
            greater_keys = ModelCheckpoint._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys,)
            greater_keys = [key.lower() for key in greater_keys]

        if less_keys is None:
            less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys,)
            # assert is_seq_of(less_keys, str)
            less_keys = [key.lower() for key in less_keys]

        if indicator in greater_keys:
            rule = "greater"
        elif indicator in less_keys:
            rule = "less"
        elif any(key in indicator for key in greater_keys):
            rule = "greater"
        elif any(key in indicator for key in less_keys):
            rule = "less"
        else:
            raise ValueError(
                f"Cannot infer the rule for key "
                f"{indicator}, thus a specific rule "
                f"must be specified."
            )

        self.rule = rule
        self.best_model_checkpoint = []

    def after_train_epoch(self, model_dir, results_dir):

        # output_dir = "/home/dsq/NIPS/results/test/init"
        self._rotate_checkpoints(model_dir)
        self._rotate_latest_results(results_dir)

    @master_only
    def _rotate_latest_results(self, results_dir) -> None:

        glob_checkpoints = [
            str(x) for x in Path(results_dir).glob(f"*") if os.path.isdir(x)
        ]

        best_results_according_to_train_loss = []
        for path in self.best_model_checkpoint:
            path = os.path.basename(path)
            tup = path.split("_")
            save_version = len(tup)
            with deprecated_context(
                "save_version",
                "model_{epoch}_{metrics} will be deprecated in a future version. Please use model_{epoch}_{iter}_{metrics} instead.",
            ):
                if save_version == 3:
                    _, epoch, _ = path.split("_")
                elif save_version == 4:
                    _, _, _, epoch = path.split("_")
                best_results_according_to_train_loss.append(epoch)

        limits = self.save_latest_limit

        if limits is None or limits <= 0:
            return

        # Check if we should delete older checkpoint(s)
        results_sorted = self._sorted_latest_checkpoints(glob_checkpoints)

        if len(results_sorted) <= limits:
            return

        number_of_to_delete = max(0, len(results_sorted) - limits)
        results_to_be_deleted = results_sorted[:number_of_to_delete]

        for result in results_to_be_deleted:
            base_result = os.path.basename(result[1])
            if (
                best_results_according_to_train_loss
                and base_result not in best_results_according_to_train_loss
            ):
                print_log(
                    f"[Save latest mode], deleting older result [{result[1]}] due to limits={limits}",
                    logger=self.logger,
                )
                shutil.rmtree(result[1], ignore_errors=True)

    def _sorted_topk_checkpoints(self, glob_checkpoints) -> List[str]:
        ordering_and_checkpoint_path = []

        for path in glob_checkpoints:
            # obtain all ckpts with metric
            fname = os.path.basename(path)
            # fname = fname.replace(".npy", "")
            with deprecated_context(
                "_sorted_topk_checkpoints",
                "model_{epoch}_{metrics} will be deprecated in a future version. Please use model_{epoch}_{iter}_{metrics} instead.",
            ):
                version = len(fname.split("_"))
                if version == 3:
                    _, _, value = fname.split("_")
                elif version == 4:
                    _, _, _, value = fname.split("_")
            ordering_and_checkpoint_path.append((float(value), path))

        # up order
        checkpoints_sorted_by_best_metric_index = sorted(
            range(len(ordering_and_checkpoint_path)),
            key=lambda k: float(ordering_and_checkpoint_path[k][0]),
            reverse=self.rule == "less",
        )

        return checkpoints_sorted_by_best_metric_index

    def _sorted_latest_checkpoints(self, glob_checkpoints) -> List[str]:
        # obtain latest ckpt
        ordering_and_checkpoint_path = [
            ((os.path.getmtime(path), path)) for path in glob_checkpoints
        ]
        return sorted(ordering_and_checkpoint_path)

    def _rotate_checkpoints(self, dir) -> None:
        # glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*") if os.path.isdir(x)]
        glob_checkpoints = [str(x) for x in Path(dir).glob(f"*") if os.path.isdir(x)]
        self._rotate_best_checkpoints(glob_checkpoints)
        self._rotate_latest_checkpoints(glob_checkpoints)

    def get_latest_checkpoints(self, model_dir, retry_count=0):
        glob_checkpoints = [
            str(x) for x in Path(model_dir).glob(f"*") if os.path.isdir(x)
        ]
        if len(glob_checkpoints) == 0:
            print_log(
                f"[Load latest mode] No checkpoints found in {model_dir}, the model is not trained from scratch",
                logger=self.logger,
            )
            return None
        return self._sorted_latest_checkpoints(glob_checkpoints)[-1][-1 - retry_count]

    def get_best_checkpoints(self, model_dir, retry_count=0):
        glob_checkpoints = [
            str(x) for x in Path(model_dir).glob(f"*") if os.path.isdir(x)
        ]
        if len(glob_checkpoints) == 0:
            print_log(
                f"[Load best mode] No checkpoints found in {model_dir}, the model is not trained from scratch",
                logger=self.logger,
            )
            return None
        return glob_checkpoints[
            self._sorted_topk_checkpoints(glob_checkpoints)[-1 - retry_count]
        ]

    @master_only
    def _rotate_latest_checkpoints(self, glob_checkpoints) -> None:

        limits = self.save_latest_limit

        if limits is None or limits <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_latest_checkpoints(glob_checkpoints)

        if (
            limits == 1
            and self.best_model_checkpoint
            and checkpoints_sorted[-1] not in self.best_model_checkpoint
        ):
            limits = self.save_latest_limit = 2

        if len(checkpoints_sorted) <= limits:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - limits)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

        for checkpoint in checkpoints_to_be_deleted:
            if (
                self.best_model_checkpoint
                and checkpoint[1] not in self.best_model_checkpoint
            ):
                print_log(
                    f"[Save latest mode], deleting older checkpoint [{checkpoint[1]}] due to limits={limits}",
                    logger=self.logger,
                )
                if os.path.isdir(checkpoint[1]):
                    shutil.rmtree(checkpoint[1], ignore_errors=True)
                else:
                    os.remove(checkpoint[1])

        # debug
        # output_dir = "/home/dsq/NIPS/results/test/init"
        # # glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*") if os.path.isdir(x)]
        # glob_checkpoints = [
        #     str(x) for x in Path(output_dir).glob(f"{self.checkpoint_prefix}*")
        # ]
        # checkpoints_sorted = self._sorted_checkpoints(glob_checkpoints)
        # number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - limits)
        # print(
        #     "rest:",
        #         checkpoints_sorted[number_of_checkpoints_to_delete:
        #     ],
        # )

    @master_only
    def _rotate_best_checkpoints(self, glob_checkpoints) -> None:

        limits = self.save_top_k

        if self.save_top_k is None or self.save_top_k <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_topk_checkpoints(glob_checkpoints)

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        if limits == 1:
            limits = self.save_latest_limit = 2

        if len(checkpoints_sorted) <= limits:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - limits)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

        not_init_flag = True
        # if we have many checkpoints, but no best_model_checkpoint, then copy all checkpoints to best_model_checkpoint
        if glob_checkpoints and not self.best_model_checkpoint:
            not_init_flag = False
            self.best_model_checkpoint = copy.deepcopy(glob_checkpoints)

        # if best_model_checkpoint is less than limits, then keep the newest one
        # add the newest checkpoints to best_model_checkpoint
        for i in checkpoints_sorted[number_of_checkpoints_to_delete:]:
            checkpoint_name = glob_checkpoints[i]
            if checkpoint_name not in self.best_model_checkpoint:
                self.best_model_checkpoint.append(checkpoint_name)
                print_log(
                    f"[Save best mode], add the best checkpoint [{checkpoint_name}] (limits={limits})",
                    logger=self.logger,
                )

        # if best_model_checkpoint is greater than limits, then delete the oldest checkpoints
        if self.best_model_checkpoint and len(self.best_model_checkpoint) > limits:
            checkpoints_to_be_deleted = checkpoints_sorted[
                :number_of_checkpoints_to_delete
            ]
            for idx in checkpoints_to_be_deleted:
                checkpoint_name = glob_checkpoints[idx]
                if checkpoint_name in self.best_model_checkpoint:
                    self.best_model_checkpoint.remove(checkpoint_name)

        if not not_init_flag:
            for checkpoint_name in self.best_model_checkpoint:
                print_log(
                    f"[Save best mode], keep the best checkpoint [{checkpoint_name}] (limits={limits})",
                    logger=self.logger,
                )

        # print("rest best:", self.best_model_checkpoint)


if __name__ == "__main__":
    import time

    checker = ModelCheckpoint("acc")
    os.makedirs("/home/dsq/NIPS/results/test/init/checkpoints", exist_ok=True)
    for epoch in range(40, 50):
        # acc = np.load(f"/home/dsq/NIPS/results/test/init/model_{epoch}.npy")
        acc = np.random.random(1)
        print(epoch, acc)
        np.save(
            f"/home/dsq/NIPS/results/test/init/checkpoints/model_{epoch}_{acc[0]}.npy",
            acc,
        )
        os.makedirs(f"/home/dsq/NIPS/results/test/init/results/{epoch}", exist_ok=True)
        time.sleep(1)
        checker.after_train_epoch(
            "/home/dsq/NIPS/results/test/init/checkpoints",
            f"/home/dsq/NIPS/results/test/init/results",
        )
