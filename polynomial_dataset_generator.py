import os
import torch
import random
import h5py
import webdataset as wds

from torch import nn

import numpy as np

class PolynomialGenerator:
    def __init__(self, path, num_points, begin, end, len_batch, shard_size=25600, pred_len_percentage=0.2):
        self.path = path
        self.num_points = num_points
        self.begin = begin
        self.end = end
        self.len_batch = len_batch
        
        self.output = int(num_points * 0.2)
        self.input = num_points - self.output

        if self.path is not None:
            self.file = h5py.File(self.path, "w")
            self.xy_dset = self.file.create_dataset(
                "xy",
                shape=(self.len_batch, self.num_points, 2),
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

            a = torch.empty(1).uniform_(-2, 2).item()
            b = torch.empty(1).uniform_(-2, 2).item()
            c = torch.empty(1).uniform_(-2, 2).item()

            y = self.quadratic(x, a, b, c)

            xy = torch.stack([x[:self.input], y[:self.input]], dim=0) 
            pred = torch.stack([x[:self.output], y[:self.output]], dim=0) 

            if i % 10000 == 0:
                print(f"progress : {i}/{self.len_batch}")

        if self.file is not None:
            self.file.close()



def main():
    train_data = PolynomialGenerator(path='dataset/train_data.h5', num_points=400, begin=-10, end=10, len_batch=100000)
    test_data = PolynomialGenerator(path='dataset/test_data.h5', num_points=400, begin=-10, end=10, len_batch=10000)


if __name__ == "__main__":
    main()


