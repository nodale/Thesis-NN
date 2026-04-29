import torch
import h5py
import zarr
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

class QuickDataset(Dataset):
    def __init__(self, path):
        self.path = path

        self.root = zarr.open_group(self.path, mode="r")
        self.xy = self.root["data"]["xy"]
        self.params = self.root["data"]["params"]

    def __len__(self):
        return self.xy.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.xy[idx]).float(),
            torch.from_numpy(self.params[idx]).float()
        )

#main for testing purposes only
def main():
    t0 = time.perf_counter()

    train_dataset = QuickDataset(path='dataset/train_data.zarr/')

    train_loader = DataLoader(
        train_dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    t1 = time.perf_counter()
    print("duration : ", t1-t0, " s")

if __name__ == "__main__":
    main()

