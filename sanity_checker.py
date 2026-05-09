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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_len = 16
    output_len = 2
    total_len = input_len + output_len

    model = NeuralNetwork(input_len=input_len, output_len=output_len, n_dim=26, out_dim=3).to(device)
    state_dict = torch.load("model_tstmp.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = torch.compile(model)
    model.eval()


    train_dataset = QuickDataset2(path='dataset/patient_one_data.zarr/', training_size = 10, window_size=total_len, seed=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    vec = next(iter(train_loader))

    vec = vec.to(device, non_blocking=True)
    in_vec = vec[:, :model.input_len, :].cuda()
    truth_vec = vec[:, model.input_len:, :3].cuda()
    pred_vec = model(in_vec)




    prior_np = vec[:, :model.input_len, :3].cpu().numpy()
    truth_np = truth_vec.detach().cpu().numpy()
    pred_np = pred_vec.detach().cpu().numpy()
    pred_np = pred_np + vec[:, model.input_len - 1:model.input_len, :3].cpu().numpy()

    print(truth_np)
    print(pred_np)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        prior_np[:, :, 0],
        prior_np[:, :, 1],
        prior_np[:, :, 2],
        label='Prior',
        linewidth=2
    )
    
    ax.plot(
        truth_np[:, :, 0],
        truth_np[:, :, 1],
        truth_np[:, :, 2],
        label='Truth',
        linewidth=2
    )

    ax.plot(
        pred_np[:, :, 0],
        pred_np[:, :, 1],
        pred_np[:, :, 2],
        label='Prediction',
        linewidth=2
    )

    ax.legend()
    plt.tight_layout()
    plt.savefig('vec_trajectory.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
