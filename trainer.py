import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    learning_rate : float = 1e-4

    def __init__(self, input_len, output_len, neural_count=2048):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

        self.linear_x = nn.Sequential(
            nn.Linear(input_len, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, neural_count),
        )

        self.linear_y = nn.Sequential(
            nn.Linear(input_len, neural_count),
            nn.ReLU(),
            nn.Linear(neural_count, neural_count),
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
        
        _x = xy_forward[:, :self.output_len]
        _y = xy_forward[:, self.output_len:]

        return _x, _y

def loss_fn(pred_x, pred_y, truth_x, truth_y):
    loss = (
        torch.nn.functional.mse_loss(pred_x, truth_x, reduction='sum') +
        torch.nn.functional.mse_loss(pred_y, truth_y, reduction='sum')
    )
    return loss

def train_loop(loader, model, optimizer):
    model.train()

    running_loss = 0.0  
    count = 0          

    for x, y in loader:
        t0 = time.perf_counter()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x0, x1 = x[:, :model.input_len], x[:, model.input_len:]
        y0, y1 = y[:, :model.input_len], y[:, model.input_len:]

        pred_x, pred_y = model(x0, y0)
        loss = loss_fn(pred_x, pred_y, x1, y1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1
        if count % 1000 == 0:
            print(f"avg_loss: {running_loss / 10000:.6f}")
            running_loss = 0.0

            t1 = time.perf_counter()
            print(t1 - t0)

def test_loop(loader, model):
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x0, x1 = x[:, :model.input_len], x[:, model.input_len:]
            y0, y1 = y[:, :model.input_len], y[:, model.input_len:]

            pred_x, pred_y = model(x0, y0)
            loss = loss_fn(pred_x, pred_y, x1, y1)

            test_loss += loss.item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:.6f}")

def main():
    input_len = 80
    output_len = 2
    total_len = input_len + output_len

    model = NeuralNetwork(input_len=input_len, output_len=output_len).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = QuickDataset2(path='dataset/train_data.zarr/', output_len=total_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4 
    )

    test_dataset = QuickDataset2(path='dataset/test_data.zarr/', output_len=total_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4 
    )

    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, optimizer)
        test_loop(test_loader, model)
    print("Done!")


if __name__ == "__main__":
    main()
