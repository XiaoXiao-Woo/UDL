# Copyright (c) Xiao Wu, LJ Deng (UESTC-MMHCISP). All rights reserved.
import glob
import torch
from torch.utils.data import DataLoader


class PansharpeningSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        # self.patch_size = args.patch_size
        self.writers = {}
        self.args = args

    def get_dataloader(self, dataset_name, distributed):

        if any(list(map(lambda x: x in dataset_name, ['wv2', 'wv3', 'wv4', 'qv', 'gf']))):
            if "hp" in dataset_name:
                # high-pass filter
                from .dataset_hp import Dataset_Pro
                dataset_name = dataset_name.split('_')[0] #'wv2_hp'
                dataset = Dataset_Pro('/'.join([self.args.data_dir, 'training_data', f'train_{dataset_name}.h5']))
            else:
                #
                from .dataset import Dataset_Pro
                dataset = Dataset_Pro('/'.join([self.args.data_dir, 'training_data', f'train_{dataset_name}.h5']))
        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError


        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.samples_per_gpu,
                           persistent_workers=(True if self.workers_per_gpu > 0 else False), pin_memory=True,
                           shuffle=(sampler is None), num_workers=self.workers_per_gpu, drop_last=True, sampler=sampler)

        return self.dataloaders[dataset_name], sampler

    def get_test_dataloader(self, dataset_name, distributed):
        # creat data for validation
        if dataset_name == 'wv3':
            dataset = Dataset_Pro(
                '/Data/DataSet/pansharpening/training_data/valid.h5')
        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.samples_per_gpu, pin_memory=True,
                           shuffle=False, num_workers=self.workers_per_gpu, drop_last=True, sampler=sampler)

        return self.dataloaders[dataset_name], sampler

    def get_eval_dataloader(self, dataset_name, distributed):

        if 'multi_exm1258' in dataset_name:
            from ..evaluation.ps_evaluate import MultiExmTest_h5
            dataset = MultiExmTest_h5('/'.join([self.args.data_dir, "test_data/WV3_Simu_mulExm/test1_mulExm1258.mat"]), dataset_name, suffix='.mat')

        elif 'singleMat' in dataset_name:
            from ..evaluation.ps_evaluate import SingleDataset
            dataset = SingleDataset(glob.glob('/'.join([self.args.data_dir, "test_data", "*.mat"])), dataset_name)

        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=1, drop_last=False, sampler=sampler)
        return self.dataloaders[dataset_name], sampler



if __name__ == '__main__':
    from option import args
    sess = PansharpeningSession(args)
    train_loader = sess.get_dataloader(args.dataset, False)
    print(len(train_loader))






