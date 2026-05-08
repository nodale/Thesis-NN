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
    def __init__(self, n_dim, input_len, output_len, neural_count=800):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_dim = n_dim

        self.channel = nn.ModuleDict()

        for c in range(self.n_dim):
            self.channel[str(c)] = nn.Sequential(
                nn.Linear(input_len, neural_count),
                nn.ReLU(),
                nn.Linear(neural_count, neural_count),
            )

        self.almagation = nn.Sequential(
                nn.Linear(neural_count, neural_count),
                nn.ReLU(),
                nn.Linear(neural_count, output_len * int(3)),
        )

    def forward(self, vec):
        outs = []

        for dim in self.channel:
            outs.append(self.channel[dim](vec[:, int(dim)]))

        out = torch.stack(outs).mean(dim=0)
        pred_flat = self.almagation(out)

        return pred_flat.reshape(self.output_len, 3)

def loss_fn(pred, truth):
    return torch.nn.functional.mse_loss(pred, truth)


def train_loop(loader, model, optimizer):
    model.train()

    running_loss = 0.0  
    count = 0          

    for vec in loader:
        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:model.input_len, :].cuda()
        truth_vec = vec[model.input_len:, :3].cuda()
    
        pred_vec = model(in_vec)
        loss = loss_fn(pred_vec, truth_vec)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1
        if count % 10000 == 0:
            print(f"avg_loss: {running_loss / 10000:.6f}")
            running_loss = 0.0

def test_loop(loader, model):
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for vec in loader:
            vec = vec.to(device, non_blocking=True)

            in_vec = vec[:model.input_len, :].cuda()
            truth_vec = vec[model.input_len:, :3].cuda()
        
            pred_vec = model(in_vec)
            loss = loss_fn(pred_vec, truth_vec)

            test_loss += loss.item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:.6f}")

def main():
    input_len = 10
    output_len = 2
    total_len = input_len + output_len

    model = NeuralNetwork(input_len=input_len, output_len=output_len, n_dim=25).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    train_dataset = QuickDataset2(path='dataset/patient_one_data.zarr/', training_size = 100000, window_size=total_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4 
    )

    test_dataset = QuickDataset2(path='dataset/patient_one_data.zarr/', training_size = 10000, window_size=total_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4 
    )

    epochs = 20
    for t in range(epochs):
        t0 = time.perf_counter()

        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, optimizer)
        test_loop(test_loader, model)

        t1 = time.perf_counter()
        print("time per epoch : ", t1 - t0)

    torch.save(model._orig_mod.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
