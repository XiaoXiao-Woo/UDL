from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

class DatasetH5_Parallel(Dataset):
    def __init__(self, file_path):
        super(DatasetH5_Parallel, self).__init__()
        # self.data = h5py.File(file_path, 'r')#, swmr=True, libver='latest')  # NxCxHxW = 0x1x2x3
        self.data = None
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as data:
            self.len = data["gt"].shape[0]
        print(self.len)

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
        return torch.from_numpy(self.data["img"][index, ...]), \
               torch.from_numpy(np.array(self.data["gt"][index, ...])).long()
    def __len__(self):
        return self.len

import scipy.io as sio
class DatasetH5(Dataset):
    def __init__(self, file_path):
        super(DatasetH5, self).__init__()
        # self.data = h5py.File(file_path, 'r')#, swmr=True, libver='latest')  # NxCxHxW = 0x1x2x3
        self.data = None
        self.file_path = file_path
        data = h5py.File(self.file_path, 'r')
        # data = sio.loadmat(self.file_path)
        self.inputs = torch.from_numpy(data["img"][...])
        self.targets = torch.from_numpy(data["gt"][...])
        print(self.inputs.shape, self.targets.shape)
        self.length = len(self.inputs)

    def __getitem__(self, index):
        # if self.data is None:
        #     self.data = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
        #     self.inputs = torch.from_numpy(self.data["img"][...])
        #     self.targets = torch.from_numpy(self.data["gt"][...])
        #     self.__len__ = len(self.inputs)
        #     print("len: ", self.__len__)

        index = index % self.length

        return {'img': self.inputs[index, ...], 'gt':self.targets[index, ...]}

    def __len__(self):

        return self.length



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    dataset = DatasetH5(file_path="./Rain200L_train_chunks.h5")

    loader = DataLoader(dataset, num_workers=4, batch_size=4)
    print(len(loader))
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=2)

    x = np.ones([4, 3, 64, 64]) * -1

    for idx, batch in enumerate(loader):

        (O, B) = batch['img'], batch['gt']
        print(idx, O.shape, B.shape, O.min(), O.max(), B.min(), B.max())
        if np.allclose(x, O):
            axes[0, 0].imshow(O[0].numpy().transpose(1, 2, 0))
            axes[0, 1].imshow(O[1].numpy().transpose(1, 2, 0))
            axes[1, 0].imshow(O[2].numpy().transpose(1, 2, 0))
            axes[1, 1].imshow(O[3].numpy().transpose(1, 2, 0))
            plt.pause(0.4)



    plt.ioff()