import torch
import h5py
import time
import glob

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

import torch
import webdataset as wds

class QuickDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, batch_size=256, shuffle_buffer=10000):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def _build_pipeline(self):
        dataset = (
            wds.WebDataset(self.path)
            .shuffle(self.shuffle_buffer)
            .to_tuple("pred.npy", "prio.npy")
            .map_tuple(torch.from_numpy, torch.from_numpy)
            .batched(self.batch_size, partial=False),
        )
        return dataset

    def __iter__(self):
        return iter(self._build_pipeline())


def main():
    t0 = time.perf_counter()

    train_dataset = QuickDataset(
        path=sorted(glob.glob("dataset/train-*.tar")),
        batch_size=25600
    )

    train_loader = torch.utils.data.DataLoader(
            wds.WebDataset(glob.glob("dataset/train-*.tar"))
            .shuffle(10000)
            .to_tuple("pred.npy", "prio.npy")
            .batched(256),
            batch_size=None,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
    for prio, pred in train_loader:
        print("yes")
        break

    t1 = time.perf_counter()
    print("duration:", t1 - t0, "s")

if __name__ == "__main__":
    main()
