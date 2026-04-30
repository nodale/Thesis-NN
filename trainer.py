import os
import torch
import random
import h5py

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    learning_rate : float = 1e-4

    def __init__(self, input_len, output_len, neural_count=80):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

        self.linear_x = nn.Sequential(
            nn.Linear(input_len, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, neural_count),
        )

        self.linear_y = nn.Sequential(
            nn.Linear(input_len, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, neural_count),
        )

        self.linear_xy = nn.Sequential(
            nn.Linear(neural_count, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, output_len*2),
        )

    def forward(self, x, y):
        x_forward = self.linear_x(x)
        y_forward = self.linear_y(y)

        xy_forward = self.linear_xy(x_forward + y_forward)
        
        _x = xy_forward[:self.output_len]
        _y = xy_forward[self.output_len:]

        return _x, _y

def loss_fn(pred_x, pred_y, truth_x, truth_y):
    loss_x = torch.sum((pred_x - truth_x) ** 2)
    loss_y = torch.sum((pred_y - truth_y) ** 2)
    return loss_x + loss_y

def train_loop(loader, model, optimizer):
    model.train()

    running_loss = 0.0  
    count = 0          

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x0 = x[:model.input_len]
        y0 = y[:model.input_len]

        x1 = x[model.input_len:]
        y1 = y[model.input_len:]

        pred_x, pred_y = model(x0, y0)
        loss = loss_fn(pred_x, pred_y, x1, y1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item(); count += 1
        
        if count % 10000 == 0:  # +1
            print(f"avg_loss: {running_loss / 10000:.6f}")
            running_loss = 0.0

def test_loop(loader, model):
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x0 = x[:model.input_len]
            y0 = y[:model.input_len]

            x1 = x[model.input_len:]
            y1 = y[model.input_len:]

            pred_x, pred_y = model(x0, y0)
            loss = loss_fn(pred_x, pred_y, x1, y1)

            test_loss += loss.item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:.6f}")

def main():
    model = NeuralNetwork(input_len=20, output_len=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = QuickDataset(path='dataset/train_data.zarr/', output_len=22)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataset = QuickDataset(path='dataset/test_data.zarr/', output_len=22)
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, optimizer)
        test_loop(test_loader, model)
    print("Done!")


if __name__ == "__main__":
    main()
