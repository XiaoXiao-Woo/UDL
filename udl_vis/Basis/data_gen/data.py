import scipy.io as sio
import torch
from torch.utils.data import Dataset
import h5py


class TestDatasetMat(Dataset):
    def __init__(self, file_path):
        super(TestDatasetMat, self).__init__()
        self.data = None
        self.file_path = file_path
        data = sio.loadmat(self.file_path)
        self.inputs = torch.from_numpy(data["img"][...])
        self.targets = torch.from_numpy(data["gt"][...])
        print(self.inputs.shape, self.targets.shape)
        self.length = len(self.inputs)

    def __getitem__(self, index):
        index = index % self.length

        return {'img': self.inputs[index, ...], 'gt': self.targets[index, ...]}

    def __len__(self):
        return self.length


class TrainValDatasetH5(Dataset):
    def __init__(self, file_path):
        super(TrainValDatasetH5, self).__init__()
        self.data = None
        self.file_path = file_path
        data = h5py.File(self.file_path, 'r')
        self.inputs = torch.from_numpy(data["img"][...])
        self.targets = torch.from_numpy(data["gt"][...])
        print(self.inputs.shape, self.targets.shape)
        self.length = len(self.inputs)

    def __getitem__(self, index):
        index = index % self.length

        return {'img': self.inputs[index, ...], 'gt': self.targets[index, ...]}

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt

    dataset = TrainValDatasetH5(file_path="./Rain200L_train_chunks.h5")

    loader = DataLoader(dataset, num_workers=4, batch_size=4)
    print(len(loader))
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=2)

    for idx, batch in enumerate(loader):
        (O, B) = batch['img'], batch['gt']
        print(idx, O.shape, B.shape, O.min(), O.max(), B.min(), B.max())
        axes[0, 0].imshow(O[0].numpy().transpose(1, 2, 0))
        axes[0, 1].imshow(O[1].numpy().transpose(1, 2, 0))
        axes[1, 0].imshow(O[2].numpy().transpose(1, 2, 0))
        axes[1, 1].imshow(O[3].numpy().transpose(1, 2, 0))
        plt.pause(0.4)

    plt.ioff()
