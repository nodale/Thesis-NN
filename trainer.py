import os
import torch
import random
import h5py
import time 

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

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(800, 700),
            nn.ReLU(),
            nn.Linear(700, 600),
            nn.ReLU(),
            nn.Linear(600, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(loader, model, loss_fn, optimizer):
    model.train()

    for x, p in loader:
        x = x.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, p)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(loader, model, loss_fn):
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, p in loader:
            x = x.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, p)

            test_loss += loss.item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:.6f}")

def main():
    t0 = time.perf_counter()

    model = NeuralNetwork().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = QuickDataset(path='dataset/train_data.zarr/')
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataset = QuickDataset(path='dataset/test_data.zarr/')
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
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    
    t1 = time.perf_counter()
    print('time : ', t1 - t0)


if __name__ == "__main__":
    main()
