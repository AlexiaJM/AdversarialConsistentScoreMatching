## Code from https://github.com/LiamZ0302/IVGAN/blob/master/IVGAN_stackMNIST/dataset.py

import os

import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class Stacked_MNIST(Dataset):
    def __init__(self, root="./dataset", load=True, source_root=None, imageSize=28,
                 train=True):  # load=True means loading the dataset from existed files.
        super(Stacked_MNIST, self).__init__()
        if load:
            self.data = torch.load(os.path.join(root, "data.pt"))
            self.targets = torch.load(os.path.join(root, "targets.pt"))
        else:
            if source_root is None:
                source_root = "~/datasets"
            source_data = torchvision.datasets.MNIST(source_root, train=train, transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]), download=True)
            self.data = torch.zeros((0, 3, imageSize, imageSize))
            self.targets = torch.zeros((0), dtype=torch.int64)
            # has 60000 images in total
            dataloader_R = DataLoader(source_data, batch_size=100, shuffle=True)
            dataloader_G = DataLoader(source_data, batch_size=100, shuffle=True)
            dataloader_B = DataLoader(source_data, batch_size=100, shuffle=True)

            for (xR, yR), (xG, yG), (xB, yB) in tqdm(zip(dataloader_R, dataloader_G, dataloader_B)):
                x = torch.cat((xR, xG, xB), dim=1)
                y = 100 * yR + 10 * yG + yB
                self.data = torch.cat((self.data, x), dim=0)
                self.targets = torch.cat((self.targets, y), dim=0)
            if not os.path.isdir(root):
                os.makedirs(root)
            torch.save(self.data, os.path.join(root, "data.pt"))
            torch.save(self.targets, os.path.join(root, "targets.pt"))
            vutils.save_image(x, "ali.png", nrow=10)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        return img, targets

    def __len__(self):
        return len(self.targets)
