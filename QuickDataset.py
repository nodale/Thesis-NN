import torch
import random
import zarr
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import numpy as np

class QuickDataset(Dataset):
    def __init__(self, path, output_len):
        self.path = path

        self.root = zarr.open_group(self.path, mode="r", cache_attrs=True)
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
        start = np.random.randint(0, self.max_start + 1)
        end = start + self.output_len

        x = self.x[idx, start:end]
        y = self.y[idx, start:end]

        return (
                torch.from_numpy(x),
                torch.from_numpy(y),
        )

class QuickDataset2(IterableDataset):
    def __init__(self, path, output_len):
        super().__init__()

        self.root = zarr.open_group(path, mode="r")
        self.x = self.root["data"]["x"]
        self.y = self.root["data"]["y"]

        self.output_len = output_len
        self.batch_num, self.seq_len = self.x.shape

        self.chunk_size = self.x.chunks[0]

    def __iter__(self):
        for chunk_start in range(0, self.batch_num, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.batch_num)

            x_chunk = self.x[chunk_start:chunk_end]
            y_chunk = self.y[chunk_start:chunk_end]

            for i in range(chunk_end - chunk_start):
                x_seq = x_chunk[i]
                y_seq = y_chunk[i]

                start = np.random.randint(0, self.seq_len - self.output_len + 1)
                end = start + self.output_len

                yield (
                    torch.from_numpy(x_seq[start:end]).float(),
                    torch.from_numpy(y_seq[start:end]).float(),
                )

#main for testing purposes only
def main():

    train_dataset = QuickDataset2(path='dataset/train_data.zarr/', output_len=40)

    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
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

