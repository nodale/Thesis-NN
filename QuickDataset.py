import torch
import random
import zarr
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import numpy as np

class QuickDataset2(IterableDataset):
    def __init__(self, path, training_size, window_size=12, seed=0, normalise=False):
        self.path = path
        self.window_size = window_size

        self.store = zarr.storage.LocalStore(self.path)
        self.root = zarr.open(store=self.store, mode='r')
        self.data = self.root['episodes']  # (num_batch, seq_len, n_dim)

        self.num_batch, self.seq_len, self.n_dim = self.data.shape
        if training_size is None:
            self.training_size = self.num_batch
        else:
            self.training_size = training_size

        self.normalise = normalise
        if self.normalise is True:
            all_data = self.data[:].astype(np.float32)
            self.mean = float(all_data.mean())
            self.std = float(all_data.std())
            self.std = max(self.std, 1e-8)

        self.rng = np.random.default_rng(seed)
        self.indices = self._generate_random_idx()

    def __len__(self):
        return self.training_size

    def _generate_random_idx(self):
        ep_idx = self.rng.integers(0, self.num_batch, size=self.training_size)
        t_idx = self.rng.integers(0, self.seq_len - self.window_size + 1, size=self.training_size)

        return np.stack([ep_idx, t_idx], axis=1)

    def __iter__(self):

        store = zarr.storage.LocalStore(self.path)
        root = zarr.open(store=store, mode="r")
        data = root["episodes"]

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = np.random.default_rng(worker_id)

        for i in range(worker_id, self.training_size, num_workers):

            ep_idx = rng.integers(0, self.num_batch)
            #ep_idx = rng.integers(0, 20)
            t_idx = rng.integers(
                0,
                self.seq_len - self.window_size + 1,
            )

            window = data[
                ep_idx,
                t_idx:t_idx + self.window_size,
                :
            ].astype(np.float32)

            if self.normalise:
                window = (window - self.mean) / self.std

            yield torch.from_numpy(window)


class QuickDatasetStraight(IterableDataset):
    def __init__(self, path, episode_idx=0, window_size=12):
        self.path = path
        self.window_size = window_size

        root = zarr.open(
            zarr.storage.LocalStore(path),
            mode="r"
        )

        self.data = root["episodes"]

        self.episode_idx = episode_idx
        self.seq_len = self.data.shape[1]


    def __len__(self):
        return self.seq_len - self.window_size + 1


    def __iter__(self):

        root = zarr.open(
            zarr.storage.LocalStore(self.path),
            mode="r"
        )

        data = root["episodes"]

        for i in range(len(self)):

            window = data[
                self.episode_idx,
                i:i+self.window_size,
                :
            ].astype(np.float32)

            yield torch.from_numpy(window)


#main for testing purposes only
def main():
    train_dataset = QuickDataset2(path='dataset/patient_one_data.zarr/', training_size=1000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
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

