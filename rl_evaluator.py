import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import IterableDataset
from include.mama import JeuralJetwork

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
    input_len = 12
    output_len = 6
    total_len = input_len + output_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = NeuralNetwork(input_len=input_len, output_len=output_len, n_dim=26, out_dim=12).to(device)

    cfg = {
            "d_model": 64,
            "n_encoder_layers": 0,
            "n_decoder_layers": 0,
            "time_adapter": "conv",
            "block_type": "simple",
            "use_norm": False,
            "layer_scale": 1e-4,
            "drop_path": 0.05,
            "mamba_type": "mamba2",
        }

    model = JeuralJetwork(
            n_dim=26,
            out_dim=6,
            input_len=input_len,
            output_len=output_len,
            **cfg
            ).to(device)

    state_dict = torch.load("model_cfg_test.pth", map_location=device)
    #new_state_dict = {
    #    k.replace("_orig_mod.", ""): v
    #    for k, v in state_dict.items()
    #}
    #model.load_state_dict(new_state_dict)
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
    init_state = log[:input_len, :3].clone().to(device)

    predicted = []
    truth = []

    history = log[:input_len].clone().to(device)

    with torch.inference_mode():

        for d in loader:

            inp = history.unsqueeze(0)

            out = model(inp)
            print("first pred delta")
            print(out[0,0,:3])

            print("history std")
            print(history.std())

            delta = out[0, 0, :3]

            curr_state = history[-1, :3]

            new_state = curr_state.clone()

            new_state[:3] += delta[:3]

            predicted.append(
                new_state[:3].cpu().unsqueeze(0)
            )

            gt_frame = d[input_len].clone().to(device)

            truth.append(
                gt_frame[:3].cpu().unsqueeze(0)
            )

            next_frame = gt_frame.clone()

            next_frame[:3] = new_state

            history = torch.cat(
                    [
                        history[1:],
                        next_frame.unsqueeze(0)
                    ],
                    dim=0
                )



    predicted = torch.cat(predicted, dim=0)
    truth = torch.cat(truth, dim=0)

    #print("truth mean", truth.mean())
    #print("truth std ", truth.std())

    print(predicted.shape)
    print(truth.shape)

    plot(predicted, truth)



if __name__ == "__main__":
    main()
