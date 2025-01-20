import torch
from torch.utils import data

# from typing_extensions import Optional
import math
from torch.utils.data import Dataset


class RandomSampler(data.RandomSampler):

    def __init__(
        self,
        data_source,
        seed=0,
        replacement: bool = False,
        num_samples: int = None,
        generator=None,
    ) -> None:
        super(RandomSampler, self).__init__(
            data_source, replacement, num_samples, generator
        )
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
            self.generator.manual_seed(self.seed + self.epoch)
        # if self.replacement:
        #     for _ in range(self.num_samples // 32):
        #         yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
        #     yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        # else:
        #     for _ in range(self.num_samples // n):
        #         yield from torch.randperm(n, generator=generator).tolist()
        #     yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
        # yield from torch.randperm(len(self.data_source), generator=generator).tolist()  # type: ignore[arg-type]
        # return iter(indices)

        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        # yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(data.DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        replacement: int = False,
        generator: torch.Generator = None,
    ) -> None:
        super(DistributedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, drop_last=True
        )
        self.generator = generator
        self.replacement = replacement

    def random_sample(self, g):
        n = len(self.dataset)
        
        num_samples = self.num_replicas * self.num_samples
        if self.replacement:
            for _ in range(num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=g
                ).tolist()
            yield from torch.randint(
                high=n, size=(num_samples % 32,), dtype=torch.int64, generator=g
            ).tolist()
        else:
            for _ in range(num_samples // n):
                yield from torch.randperm(n, generator=g).tolist()
            yield from torch.randperm(n, generator=g).tolist()[: num_samples % n]

    def __iter__(self):

        g = self.generator
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            if g is None:
                g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            indices = list(self.random_sample(g))

        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
