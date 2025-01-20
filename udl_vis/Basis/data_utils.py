import os
import torch
from torch.utils.data import Dataset
from collections import Counter
from pathlib import Path
import json
import functools
import imageio
import numpy as np
import glob
from PIL import Image


def to_tensor(nda3d):
    return torch.from_numpy(nda3d).float()


def recursive_glob(rootdir, pattern):
    sample_list = sorted(list(Path(rootdir).rglob(pattern)))
    sample_list = [str(item) for item in sample_list]
    return sample_list


def read_image(path, modal=""):
    return to_tensor(np.array(Image.open(path))).permute(2, 0, 1)


def read_all_image(source_data, pattern, prefetch_data):
    for modal in source_data.keys():
        for dir in source_data[modal]["num"].keys():
            path_list = list(sorted(glob.glob(os.path.join(dir, pattern))))
            if prefetch_data:
                data = [read_image(path) for path in path_list]
                source_data[modal]["data"] = data
            source_data[modal]["path"] = path_list
    return source_data


class PairedImageDataset(Dataset):

    def __init__(self, dataset_path_dicts, read_data_func):
        super(PairedImageDataset, self).__init__()
        dataset_path_dicts.setdefault("pattern", "*")
        dataset_path_dicts.setdefault("prefetch_data", True)
        self.pattern = dataset_path_dicts.pop("pattern")
        prefetch_data_flag = dataset_path_dicts.pop("prefetch_data")
        self.prefetch_data_flag = prefetch_data_flag
        self.prefetch_func = read_data_func["prefetch"]

        if prefetch_data_flag:
            print("loading all data into memory")
        else:
            self.read_data_func = read_data_func["func"]

        if isinstance(dataset_path_dicts, str):
            raise TypeError(
                f"dataset_path_dicts: {dataset_path_dicts} should contain different modalities, e.g. source1, source2, etc."
            )

        print(
            f"dataset_path_dicts contains multiple modalities: {dataset_path_dicts.keys()}"
        )
        print(f"dataset_kwargs: {dataset_path_dicts.get('dataset_kwargs', {})}")
        print(f"extra_kwargs: {dataset_path_dicts.get('extra_kwargs', {})}")

        source_data = {}
        for modal, source_path in dataset_path_dicts["source_path"].items():
            modal = modal.lower()
            print(f"loading mode={modal}, path={source_path}")
            # source_data[modal] = {"path": recursive_glob(source_path, pattern[modal])}
            # print(f"path len={len(source_data[modal]['path'])}")
            if isinstance(self.pattern, str):
                path = recursive_glob(source_path, self.pattern)
            else:
                path = recursive_glob(source_path, self.pattern[modal])
            sample_list = [os.path.dirname(file) for file in path]
            source_data[modal] = {"num": dict(Counter(sample_list))}
            source_data[modal]["path"] = list(source_data[modal]["num"].keys())
        self.data_length = len(sample_list)

        print("source_data:", json.dumps(source_data, indent=4))
        source_data = self.prefetch_func(source_data, self.pattern, prefetch_data_flag)

        self.dataset_kwargs = dataset_path_dicts.get("dataset_kwargs", {})
        self.extra_kwargs = dataset_path_dicts.get("extra_kwargs", {})
        self.source_data = source_data

    def __getitem__(self, index):

        batch = {}
        for key, value in self.dataset_kwargs.items():
            batch[key] = value

        for modal in self.source_data.keys():
            # batch[f"{modal}_filename"] = os.path.basename(
            #     self.source_data[modal]["path"][index]
            # )
            batch[f"{modal}_filename"] = (
                self.source_data[modal]["path"][index]
                # .replace("/home/dsq/nips/datasets/heartHX", "")
                # .replace("/", "_")
            )
            if not self.prefetch_data_flag:
                batch[modal] = self.read_data_func(
                    self.source_data[modal]["path"][index], modal
                )
            else:
                batch[modal] = self.source_data[modal]["data"][index]
        return batch

    def __len__(self):
        return self.data_length
