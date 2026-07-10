import os
import torch
import random

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        N = 4
        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(2, 64, 8, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(N),
            nn.Flatten(),
            nn.Linear(64 * N, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits














class PolyDataset(Dataset):
    def __init__(self, xy, params):
        self.xy = xy
        self.params = params

    def __len__(self):
        return self.xy.shape[0]

    def __getitem__(self, idx):
        return self.xy[idx], self.params[idx]

class PolynomialGenerator:
    def __init__(self, num_points, begin, end, len_batch):
        self.num_points = num_points
        self.begin = begin
        self.end = end
        self.len_batch = len_batch


        self.generate()

    def quadratic(self, x, a, b, c):
        return a * x**2 + b * x + c

    def generate(self):
        x_batches = []
        y_batches = []
        xy_batches = []
        params = []

        for _ in range(self.len_batch):
            #x = torch.linspace(begin, end, num_points)
            x = torch.empty(self.num_points).uniform_(self.begin, self.end)
            x, _ = torch.sort(x)

            a = torch.empty(1).uniform_(-5, 5).item()
            b = torch.empty(1).uniform_(-5, 5).item()
            c = torch.empty(1).uniform_(-5, 5).item()

            y = self.quadratic(x, a, b, c)
            xy = torch.stack([x, y], dim=0) 

            x_batches.append(x)
            y_batches.append(y)
            xy_batches.append(xy)
            params.append((a, b, c))

        self.x_batch = torch.stack(x_batches)
        self.y_batch = torch.stack(y_batches)
        self.xy_batch = torch.stack(xy_batches) 
        self.param = torch.tensor(params)     














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
    model = NeuralNetwork().to(device)

    train_data = PolynomialGenerator(num_points=200, begin=-10, end=10, len_batch=800000)
    test_data = PolynomialGenerator(num_points=200, begin=-10, end=10, len_batch=5000)

    train_dataset = PolyDataset(
        train_data.xy_batch,
        train_data.param
    )
    test_dataset = PolyDataset(
        test_data.xy_batch,
        test_data.param
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
