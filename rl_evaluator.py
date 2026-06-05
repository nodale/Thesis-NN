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

def plot(ts1, ts2):
    a = ts1.detach().cpu().numpy()
    b = ts2.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(a[:, 0], a[:, 1], a[:, 2], label='ts1')
    ax.plot(b[:, 0], b[:, 1], b[:, 2], label='ts2')

    ax.legend()
    #plt.savefig("test.png")
    plt.show()

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
    input_len = 24
    output_len = 1
    total_len = input_len + output_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_len=input_len, output_len=output_len, n_dim=26, out_dim=6).to(device)
    state_dict = torch.load("model_12steps.pth", map_location=device)
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

    predicted   = []
    truth       = []

    #init_pos = torch.zeros((input_len, 3), dtype=torch.float32, device="cuda")
    init_pos = log[:input_len, :3].clone().to(device)

    with torch.inference_mode():
        for d in loader:
            _in = d[:input_len, :].cuda()
            _in[:, :3] = init_pos
            _in = _in.unsqueeze(0)
            out = model(_in)
            out = out.squeeze(0)

            new_pos = (init_pos[-1] + out[-1, :3]).detach()
            init_pos = torch.roll(init_pos, shifts=-1, dims=0)
            init_pos[-1] = new_pos
            #new_pos = _in[-1, -1, :3] + out[0, :3]

            predicted.append(new_pos.unsqueeze(0).cpu())
            truth.append(d[input_len:total_len, :3].cpu())

            print("pos diff : ", _in[-1, -1, :3] - new_pos, "   log_var : ", out[-1, 3:])

    predicted = torch.cat(predicted, dim=0)
    truth = torch.cat(truth, dim=0)

    print(predicted.shape)
    print(truth.shape)

    plot(predicted, truth)




if __name__ == "__main__":
    main()
