import torch
import random
import zarr
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import numpy as np

import torch
import zarr
import numpy as np
from torch.utils.data import IterableDataset

class QuickDataset2(IterableDataset):
    def __init__(self, path, window_size=32, seed=0):
        self.path = path
        self.window_size = window_size

        self.store = zarr.storage.LocalStore(self.path)
        self.root = zarr.open(store=self.store, mode='r')
        self.data = self.root['episodes']  # (num_batch, seq_len, n_dim)

        self.num_batch, self.seq_len, self.n_dim = self.data.shape
        print(self.num_batch)

        self.rng = np.random.default_rng(seed)
        self.indices = self._generate_random_idx()

    def _generate_random_idx(self):
        ep_idx = self.rng.integers(0, self.num_batch, size=self.num_batch)
        t_idx = self.rng.integers(0, self.seq_len - self.window_size + 1, size=self.num_batch)

        return np.stack([ep_idx, t_idx], axis=1)

    def __iter__(self):
        for ep_idx, t_idx in self.indices:

            window = self.data[
                ep_idx,
                t_idx:t_idx + self.window_size,
                :
            ]  # (M, n_dim)

            yield torch.tensor(window, dtype=torch.float32)



#main for testing purposes only
def main():
    train_dataset = QuickDataset2(path='dataset/patient_one_data.zarr/')

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

    for m in train_loader:
        m = m.cuda(non_blocking=True)

    t1 = time.perf_counter()

    print(f"Total time: {t1 - t0:.3f} s")

if __name__ == "__main__":
    main()

