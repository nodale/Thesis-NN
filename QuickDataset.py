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
        self.prio = self.root["data"]["prio"]

        self.chunk_size = self.prio.chunks[0] 
        self.num_chunks = self.prio.shape[0] // self.chunk_size

#shape : [num_sequence][len_sequence][sequence_dim]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size

        chunk = self.prio[start:end] 
        return torch.from_numpy(chunk).float()

#main for testing purposes only
def main():

    train_dataset = QuickDataset(path='dataset/train_data.zarr/')

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    loader_iter = iter(train_loader)
    for _ in range(2):
        next(loader_iter)

    t0 = time.perf_counter()

    loader_iter = iter(train_loader)
    num_batches = 500

    for i in range(num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

    t1 = time.perf_counter()

    print(f"Avg time per batch: {(t1 - t0)/num_batches:.6f} s")
    print(f"Total time: {t1 - t0:.3f} s")

if __name__ == "__main__":
    main()

