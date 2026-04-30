import os
import torch
import random
import zarr

from torch import nn

import numpy as np

class PolynomialGenerator:
    def __init__(self, path, num_points, begin, end, len_batch, chunk=32):
        self.path = path
        self.num_points = num_points
        self.begin = begin
        self.end = end
        self.len_batch = len_batch
        self.chunk = chunk

        if self.path is not None:
            self.store = zarr.storage.LocalStore(self.path)
            self.root = zarr.group(store=self.store, overwrite=True)
            self.data = self.root.create_group('data')

            self.prio = self.data.create_array(
                name="prio",
                shape=(self.len_batch, self.num_points, 2),
                chunks=(self.chunk, self.num_points, 2),
                dtype="f4",
                overwrite=True
            )

            self.generate()


    def quadratic(self, x, a, b, c):
        return a * x**2 + b * x + c

    def generate(self):
        B, N, chunk = self.len_batch, self.num_points, 1024

        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            bs = j - i

            params = torch.empty(bs, 3).uniform_(-5, 5)
            a = params[:, 0]
            b = params[:, 1]
            c = params[:, 2]

            x = torch.sort(torch.empty(bs, N).uniform_(self.begin, self.end), dim=1).values

            y = a[:, None] * x**2 + b[:, None] * x + c[:, None]

            self.prio[i:j] = torch.stack([x[:, :self.num_points], y[:, :self.num_points]], dim=-1).numpy().astype("float32")

            print(i, " out of ", B)

        self.store.close()

def main():
    train_data = PolynomialGenerator(path='dataset/train_data.zarr/', num_points=400, begin=-10, end=10, len_batch=1000000)
    test_data = PolynomialGenerator(path='dataset/test_data.zarr/', num_points=400, begin=-10, end=10, len_batch=10000)


if __name__ == "__main__":
    main()


