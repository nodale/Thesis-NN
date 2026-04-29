import os
import torch
import random
import h5py
import webdataset as wds

from torch import nn

import numpy as np

class PolynomialGenerator:
    def __init__(self, out_path, num_points, begin, end, len_batch,
                 shard_size=10000, pred_len_percentage=0.2):

        self.out_path = out_path
        self.num_points = num_points
        self.begin = begin
        self.end = end
        self.len_batch = len_batch
        self.shard_size = shard_size

        self.output = int(num_points * pred_len_percentage)
        self.input = num_points - self.output

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        self.generate()

    def quadratic(self, x, a, b, c):
        return a * x**2 + b * x + c

    def generate(self):
        with wds.ShardWriter(self.out_path, maxcount=self.shard_size) as sink:
            for i in range(self.len_batch):
                x = torch.empty(self.num_points).uniform_(self.begin, self.end)
                x, _ = torch.sort(x)

                a = torch.empty(1).uniform_(-2, 2).item()
                b = torch.empty(1).uniform_(-2, 2).item()
                c = torch.empty(1).uniform_(-2, 2).item()

                y = self.quadratic(x, a, b, c)

                prio = torch.stack([x[:self.input], y[:self.input]], dim=0)
                pred = torch.stack([x[self.output:], y[self.output:]], dim=0)

                sample = {
                    "__key__": f"{i:08d}",
                    "prio.npy": prio.numpy(),
                    "pred.npy": pred.numpy(),
                    "params.npy": torch.tensor([a, b, c], dtype=torch.float32).numpy(),
                }

                sink.write(sample)

                if i % 10000 == 0:
                    print(f"progress: {i}/{self.len_batch}")


def main():
    PolynomialGenerator(
        out_path="dataset/train-%06d.tar",
        num_points=400,
        begin=-10,
        end=10,
        len_batch=100000,
        shard_size=5000
    )

    PolynomialGenerator(
        out_path="dataset/test-%06d.tar",
        num_points=400,
        begin=-10,
        end=10,
        len_batch=10000,
        shard_size=5000
    )

if __name__ == "__main__":
    main()


