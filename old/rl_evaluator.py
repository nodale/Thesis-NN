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
from include.mama import JeuralJetwork

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import zarr

matplotlib.use("QtAgg")

def plot(ts1, ts2):
    a = ts1.detach().cpu().numpy()
    b = ts2.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(a[:, 0], a[:, 1], a[:, 2], label='pred')
    ax.plot(b[:, 0], b[:, 1], b[:, 2], label='truth')

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



class QuickDatasetStraight(IterableDataset):
    def __init__(
        self,
        path,
        episode_idx=0,
        window_size=12,
        normalise=False,
    ):
        self.path = path
        self.window_size = window_size
        self.episode_idx = episode_idx
        self.normalise = normalise

        self.store = zarr.storage.LocalStore(self.path)
        self.root = zarr.open(store=self.store, mode="r")
        self.data = self.root["episodes"]

        self.num_batch, self.seq_len, self.n_dim = self.data.shape

        if episode_idx >= self.num_batch:
            raise ValueError(
                f"episode_idx={episode_idx} exceeds dataset size "
                f"({self.num_batch} episodes)"
            )

        if self.normalise:
            all_data = self.data[:].astype(np.float32)
            self.mean = float(all_data.mean())
            self.std = float(all_data.std())
            self.std = max(self.std, 1e-8)

    def __len__(self):
        return self.seq_len - self.window_size + 1

    def __iter__(self):

        store = zarr.storage.LocalStore(self.path)
        root = zarr.open(store=store, mode="r")
        data = root["episodes"]

        for idx in range(self.seq_len - self.window_size + 1):

            window = data[
                self.episode_idx,
                idx : idx + self.window_size,
                :
            ].astype(np.float32)

            if self.normalise:
                window = (window - self.mean) / self.std

            yield torch.from_numpy(window)

def main():
    input_len = 12
    output_len = 3
    total_len = input_len + output_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = NeuralNetwork(input_len=input_len, output_len=output_len, n_dim=26, out_dim=12).to(device)

    #model = NeuralNetwork(
    #        n_dim=26,
    #        out_dim=13,
    #        input_len=input_len,
    #        output_len=output_len
    #        )

    cfg = {
            "d_model": 64,
            "n_encoder_layers": 2,
            "n_decoder_layers": 0,
            "block_type": "simple",
            "time_adapter": "last_token",
            "in_norm": "rms",
            "out_norm": "none",
            "layer_scale": 0,
            "drop_path": 0.0,
            "mamba_type": "mamba",
            "mamba_kwargs": 
            {
                "d_state": 28,
                "d_conv": 4,
                "expand": 2,
                "norm" : "rms",
            },
        }

    model = JeuralJetwork(
            n_dim=26,
            out_dim=13,
            input_len=input_len,
            output_len=output_len,
            **cfg
            )

    state_dict = torch.load("models/model_original+replica+rollout.pth", map_location=device)
    #new_state_dict = {
    #    k.replace("_orig_mod.", ""): v
    #    for k, v in state_dict.items()
    #}
    #model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = torch.compile(model)
    model.eval()

    #log = torch.load("rl_dataset/converted.pt")
    #dataset = FlightLog(data=log, window_size=total_len)

    dataset = QuickDatasetStraight(
        path="/home/joey/Thesis/data/patient_one_data.zarr/",
        episode_idx=0,
        window_size=total_len
        )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    predicted   = []
    truth       = []

    init_pos = torch.zeros((input_len, 3), dtype=torch.float32, device="cuda")
    init_pos = next(iter(loader))[:input_len, :3].clone().to(device)
    #init_state = log[:input_len, :3].clone().to(device)

    predicted = []
    truth = []

    with torch.inference_mode():
        for d in loader:
            _in = d[:input_len, :].cuda()
            _in[:input_len, :3] = init_pos
            _in = _in.unsqueeze(0)
            out = model(_in)
            out = out.squeeze(0)

            new_pos = (init_pos[-1] + out[0, :3]).detach()
            init_pos = torch.roll(init_pos, shifts=-1, dims=0)
            init_pos[-1] = new_pos

            predicted.append(new_pos.unsqueeze(0).cpu())
            truth.append(d[input_len:total_len, :3].cpu())

    #history = log[:input_len].clone().to(device)
    #with torch.inference_mode():
    #    for d in loader:
    #        inp = history.unsqueeze(0)
    #        out = model(inp)
    #        #print("first pred delta")
    #        #print(out[0,0,:3])
    #        #print("history std")
    #        #print(history.std())
    #        delta = out[0, 0, :3]
    #        curr_state = history[-1, :3]
    #        new_state = curr_state.clone()
    #        new_state[:3] += delta[:3]
    #        predicted.append(
    #            new_state[:3].cpu().unsqueeze(0)
    #        )
    #        gt_frame = d[input_len].clone().to(device)
    #        truth.append(
    #            gt_frame[:3].cpu().unsqueeze(0)
    #        )
    #        next_frame = gt_frame.clone()
    #        next_frame[:3] = new_state
    #        history = torch.cat(
    #                [
    #                    history[1:],
    #                    next_frame.unsqueeze(0)
    #                ],
    #                dim=0
    #            )



    predicted = torch.cat(predicted, dim=0)
    truth = torch.cat(truth, dim=0)

    plot(predicted, truth)



if __name__ == "__main__":
    main()
