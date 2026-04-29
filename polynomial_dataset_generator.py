import os
import torch
import random
import h5py
import zarr

from torch import nn

import numpy as np

class PolynomialGenerator:
    def __init__(self, path, num_points, begin, end, len_batch, chunk=4096):
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

            self.xy = self.data.create_array(
                name="xy",
                shape=(self.len_batch, self.num_points * 2),
                chunks=(self.chunk, self.num_points * 2),
                dtype="f4"
            )

            self.params = self.data.create_array(
                name="params",
                shape=(self.len_batch, 3),
                chunks=(self.chunk, 3),
                dtype="f4"
            )

            self.generate()


    def quadratic(self, x, a, b, c):
        return a * x**2 + b * x + c

    def generate(self):
            B = self.len_batch
            N = self.num_points
            chunk = 1024

            for i in range(0, B, chunk):
                end = min(i + chunk, B)
                bs = end - i

                # ----------------------------
                # 1. vectorized x sampling
                # ----------------------------
                x = torch.empty(bs, N).uniform_(self.begin, self.end)
                x, _ = torch.sort(x, dim=1)

                # ----------------------------
                # 2. vectorized parameters
                # ----------------------------
                params = torch.empty(bs, 3).uniform_(-5, 5)
                a = params[:, 0]
                b = params[:, 1]
                c = params[:, 2]

                # ----------------------------
                # 3. vectorized polynomial
                # ----------------------------
                y = a[:, None] * x**2 + b[:, None] * x + c[:, None]

                # ----------------------------
                # 4. flatten (x,y) → interleaved
                # ----------------------------
                xy = torch.stack([x, y], dim=2).reshape(bs, 2 * N)

                # ----------------------------
                # 5. write chunk to Zarr
                # ----------------------------
                self.xy[i:end] = xy.numpy()
                self.params[i:end] = params.numpy().astype("float32")

                if i % (chunk * 5) == 0:
                    print(f"progress: {i}/{B}")

            self.store.close()


def main():
    train_data = PolynomialGenerator(path='dataset/train_data.zarr/', num_points=400, begin=-10, end=10, len_batch=1000000)
    test_data = PolynomialGenerator(path='dataset/test_data.zarr/', num_points=400, begin=-10, end=10, len_batch=10000)


if __name__ == "__main__":
    main()


