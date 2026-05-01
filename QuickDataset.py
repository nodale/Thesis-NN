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
    def __init__(self, path, output_len, batch_size=5000):
        super().__init__()

        self.root = zarr.open_group(path, mode="r")
        self.x = self.root["data"]["x"]
        self.y = self.root["data"]["y"]

        self.output_len = output_len
        self.batch_size = batch_size

        self.N, self.seq_len = self.x.shape
        self.chunk_size = self.x.chunks[0]

    def __iter__(self):
        for chunk_start in range(0, self.N, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.N)

            x_chunk = self.x[chunk_start:chunk_end]
            y_chunk = self.y[chunk_start:chunk_end]

            chunk_len = chunk_end - chunk_start

            for b_start in range(0, chunk_len, self.batch_size):
                b_end = min(b_start + self.batch_size, chunk_len)

                x_batch = x_chunk[b_start:b_end]
                y_batch = y_chunk[b_start:b_end]

                B = x_batch.shape[0]

                starts = np.random.randint(
                    0,
                    self.seq_len - self.output_len + 1,
                    size=B
                )

                idx = starts[:, None] + np.arange(self.output_len)[None, :]

                x_out = x_batch[np.arange(B)[:, None], idx]
                y_out = y_batch[np.arange(B)[:, None], idx]

                yield (
                    torch.from_numpy(x_out),
                    torch.from_numpy(y_out),
                )





#main for testing purposes only
def main():
    train_dataset = QuickDataset2(path='dataset/train_data.zarr/', output_len=40)

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )

    loader_iter = iter(train_loader)
    for _ in range(2):
        next(loader_iter)

    t0 = time.perf_counter()

    for m, n in train_loader:
        m = m.cuda(non_blocking=True)
        n = n.cuda(non_blocking=True)

    t1 = time.perf_counter()

    print(f"Total time: {t1 - t0:.3f} s")

if __name__ == "__main__":
    main()

