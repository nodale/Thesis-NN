import torch
import h5py
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

class QuickDataset(Dataset):
    def __init__(self, path, batch_size=1024):
        self.path = path
        self.batch_size = batch_size

        self.file = h5py.File(self.path, "r")
        self.dataset_len = self.file["xy"].shape[0]

        self.num_batches = (self.dataset_len + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.dataset_len)

        xy = self.file["xy"][start:end]
        params = self.file["params"][start:end]

        return (
            torch.from_numpy(xy).float(),
            torch.from_numpy(params).float()
        )

#main for testing purposes only
def main():
    t0 = time.perf_counter()

    train_dataset = QuickDataset(path='dataset/train_data.h5')

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    t1 = time.perf_counter()
    print("duration : ", t1-t0, " s")

if __name__ == "__main__":
    main()

