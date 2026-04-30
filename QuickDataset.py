import torch
import random
import zarr
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

class QuickDataset(Dataset):
    def __init__(self, path, output_len):
        self.path = path

        self.root = zarr.open_group(self.path, mode="r")
        self.data = self.root["data"]

        self.x = self.data["x"]
        self.y = self.data["y"]

        self.chunk_size = self.x.chunks[0]
        self.length = self.x.shape[0]
        self.output_len = output_len

        self.max_start = self.length - self.output_len

    def __len__(self):
        #return self.length // self.chunk_size
        return self.length - self.output_len

    def __getitem__(self, idx):
        x_seq = self.x[idx] 
        y_seq = self.y[idx]

        max_start = x_seq.shape[0] - self.output_len
        start = np.random.randint(0, max_start + 1)
        end = start + self.output_len

        x = x_seq[start:end]
        y = y_seq[start:end]

        return (
                torch.from_numpy(x).to(torch.float32),
                torch.from_numpy(y).to(torch.float32),
        )

#main for testing purposes only
def main():

    train_dataset = QuickDataset(path='dataset/train_data.zarr/', output_len=40)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    loader_iter = iter(train_loader)
    for _ in range(2):
        next(loader_iter)

    t0 = time.perf_counter()

    loader_iter = iter(train_loader)
    num_batches = 50

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

