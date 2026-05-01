import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2
from trainer import NeuralNetwork

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_len = 12
    output_len = 4
    total_len = input_len + output_len

    model = NeuralNetwork(input_len=input_len, output_len=output_len)
    state_dict = torch.load("model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = torch.compile(model)
    model.eval()


    train_dataset = QuickDataset2(path='dataset/test_data.zarr/', output_len=total_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    x, y = next(iter(train_loader))

    x = x.to("cuda", non_blocking=True)
    y = y.to("cuda", non_blocking=True)

    x0, x1 = x[:1, :model.input_len], x[:1, model.input_len:]
    y0, y1 = y[:1, :model.input_len], y[:1, model.input_len:]

    pred_x, pred_y = model(x0, y0)







    # move to CPU + numpy
    x0_np = x0.squeeze().detach().cpu().numpy()
    y0_np = y0.squeeze().detach().cpu().numpy()

    x1_np = x1.squeeze().detach().cpu().numpy()
    y1_np = y1.squeeze().detach().cpu().numpy()

    pred_x_np = pred_x.squeeze().detach().cpu().numpy()
    pred_y_np = pred_y.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(8, 8))

    # INPUT trajectory
    plt.plot(x0_np, y0_np, label="input (x0, y0)", linestyle="--", marker="o")

    # GROUND TRUTH trajectory
    plt.plot(x1_np, y1_np, label="ground truth (x1, y1)", linewidth=2)

    # PREDICTION trajectory
    plt.plot(pred_x_np, pred_y_np, label="prediction", marker="x")

    # mark start points
    plt.scatter(x0_np[0], y0_np[0], color="blue", label="start input")
    plt.scatter(x1_np[0], y1_np[0], color="green", label="start truth")
    plt.scatter(pred_x_np[0], pred_y_np[0], color="red", label="start pred")

    plt.legend()
    plt.title("Trajectory Comparison")
    plt.axis("equal")   # important for geometry correctness
    plt.grid(True)

    plt.savefig("debug_plot.png", dpi=500)
    plt.show()



if __name__ == "__main__":
    main()
