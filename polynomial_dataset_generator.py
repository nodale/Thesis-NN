import os
import torch
import random
import h5py

from torch import nn

import numpy as np

class PolynomialGenerator:
    def __init__(self, path, num_points, begin, end, len_batch):
        self.path = path
        self.num_points = num_points
        self.begin = begin
        self.end = end
        self.len_batch = len_batch

        if self.path is not None:
            self.file = h5py.File(self.path, "w")
            self.xy_dset = self.file.create_dataset(
                "xy",
                shape=(self.len_batch, self.num_points * 2),
                dtype="float32"
            )
            self.param_dset = self.file.create_dataset(
                "params",
                shape=(self.len_batch, 3),
               dtype="float32"
            )

        self.generate()

    def quadratic(self, x, a, b, c):
        return a * x**2 + b * x + c

    def generate(self):
        #x_batches = []
        #y_batches = []
        #xy_batches = []
        #params = []

        for i in range(self.len_batch):
            #x = torch.linspace(begin, end, num_points)
            x = torch.empty(self.num_points).uniform_(self.begin, self.end)
            x, _ = torch.sort(x)

            a = torch.empty(1).uniform_(-5, 5).item()
            b = torch.empty(1).uniform_(-5, 5).item()
            c = torch.empty(1).uniform_(-5, 5).item()

            y = self.quadratic(x, a, b, c)
            xy = torch.stack([x, y], dim=1) 
            xy = xy.flatten()

            #x_batches.append(x)
            #y_batches.append(y)
            #xy_batches.append(xy)
            #params.append((a, b, c))

            if self.xy_dset is not None:
                self.xy_dset[i] = xy.numpy()
                self.param_dset[i] = np.array([a, b, c], dtype=np.float32)

            if i % 10000 == 0:
                print(f"progress : {i}/{self.len_batch}")

        if self.file is not None:
            self.file.close()

        #self.x_batch = torch.stack(x_batches)
        #self.y_batch = torch.stack(y_batches)
        #self.xy_batch = torch.stack(xy_batches) 
        #self.param = torch.tensor(params)     


def main():
    train_data = PolynomialGenerator(path='dataset/train_data.h5', num_points=400, begin=-10, end=10, len_batch=1000000)
    test_data = PolynomialGenerator(path='dataset/test_data.h5', num_points=400, begin=-10, end=10, len_batch=10000)


if __name__ == "__main__":
    main()


