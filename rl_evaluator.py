import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2
from trainer import NeuralNetwork
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import IterableDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")


class FlightLog(IterableDataset):
    def __init__(self, data, window_size=12):
        self.window_size = window_size
        self.data = data.cpu()
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __iter__(self):
        for idx in range(self.len - self.window_size + 1):
            data = self.data[idx:idx+self.window_size, :]
            yield data

def main():
    input_len = 28
    output_len = 1
    total_len = input_len + output_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_len=input_len, output_len=output_len, n_dim=26, out_dim=3).to(device)
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = torch.compile(model)
    model.eval()

    log = torch.load("rl_dataset/converted.pt")

    dataset = FlightLog(data=log, window_size=total_len)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    for d in loader:
        _in = d[:input_len, :].cuda()
        _in = _in.unsqueeze(0)
        out = model(_in)

        print(_in[:3])



if __name__ == "__main__":
    main()
