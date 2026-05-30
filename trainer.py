import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2
from mamba_ssm import Mamba

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, n_dim, out_dim, input_len, output_len, neural_count=64):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_dim = n_dim
        self.out_dim = out_dim

        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.n_dim = n_dim
        self.out_dim = out_dim

        self.input_proj = nn.Sequential(
            nn.Linear(n_dim, neural_count),
            nn.LayerNorm(neural_count)
        )

        self.mamba = Mamba(
            d_model=neural_count,
            d_state=32,
            d_conv=8,
            expand=2,
        )

        self.head = nn.Sequential(
            nn.Linear(neural_count, neural_count),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(neural_count, neural_count),
            nn.GELU(),

            nn.Linear(neural_count, out_dim * output_len),
        )

    def forward(self, vec):
        x = self.input_proj(vec) # [B, T, D] -> [B, T, H]
        x = self.mamba(x) # [B, T, H]
        x = x[:, -1] # [B, H]
        #x = x.mean(dim=1)
        x = self.head(x) # [B, out_dim * output_len]
        x = x.view(vec.size(0), self.output_len, self.out_dim) # [B, output_len, out_dim]

        return x


def loss_fn(pred, truth):
    return (1.0 * torch.nn.functional.mse_loss(pred, truth))

#def loss_fn(pred, truth):
#    error = pred - truth
#    return 1e+5 * torch.mean(torch.abs(error) ** 4)

def train_loop(loader, model, optimizer, batch_size=100, std_min=1e-8,std_max=2e-4):
    model.train()

    running_loss = 0.0  
    count = 0          

    t0 = time.perf_counter()

    tot_len = len(loader)

    for vec in loader:
        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:, :model.input_len, :]
        #noise_std = std_min + (std_max - std_min) * torch.rand(1, device=in_vec.device).item()
        #in_vec[:, :, :12] += noise_std * torch.randn_like(in_vec)[:, :, :12]

        #truth_vec = vec[:, model.input_len:, :3] - vec[:, model.input_len - 1, :3] # makeing it relative to the last prio given
        truth_vec = vec[:, model.input_len:, :3] - vec[:, model.input_len - 1:model.input_len, :3]
    
        pred_vec = model(in_vec)
        loss = loss_fn(pred_vec, truth_vec)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0


        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1
        if count % batch_size == 0:
            print(f"avg_loss: {running_loss / batch_size:.12f}         time_per_window : {dt/batch_size:.6f}        progress : {count/tot_len:.3f}")
            running_loss = 0.0


def train_loop_stepwise(loader, model, optimizer, batch_size=100):
    model.train()

    running_loss = 0.0  
    count = 0          

    t0 = time.perf_counter()
    for vec in loader:
        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:, :model.input_len, :]

        # future absolute positions
        future = vec[:, model.input_len:, :3]

        # prepend last input frame to compute first delta correctly
        prev = vec[:, model.input_len - 1:model.input_len, :3]

        # concatenate so we can do timestep differences cleanly
        full = torch.cat([prev, future], dim=1)

        # timestep-wise relative targets
        truth_vec = full[:, 1:, :] - full[:, :-1, :]

        pred_vec = model(in_vec)

        loss = loss_fn(pred_vec, truth_vec)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach()
        count += 1

        if count % batch_size == 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1
            print(f"avg_loss: {running_loss / batch_size:.12f}         time_per_batch: {dt:.6f}")
            running_loss = 0.0

def test_loop(loader, model):
    model.eval()

    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for vec in loader:
            vec = vec.to(device, non_blocking=True)

            in_vec = vec[:, :model.input_len, :].cuda()
            truth_vec = vec[:, model.input_len:, :3].cuda()
        
            pred_vec = model(in_vec)
            loss = loss_fn(pred_vec, truth_vec)

            test_loss += loss.item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:.6f}")

def main():
    input_len = 20
    output_len = 1
    total_len = input_len + output_len
    batch_size = 128

    model = NeuralNetwork(
            input_len=input_len, 
            output_len=output_len, 
            n_dim=27, 
            out_dim=3,
            ).to(device)

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4,
            weight_decay=1e-8,
            eps=1e-38
            )
            #betas=(0.98, 0.999),

    training_size = 15000000

    train_dataset = QuickDataset2(path='/home/joey/Thesis/data/patient_one_data.zarr/', training_size=training_size, window_size=total_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    test_dataset = QuickDataset2(path='/home/joey/Thesis/data/patient_one_data.zarr/', training_size=1000, window_size=total_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8 
    )

    epochs = 5
    for t in range(epochs):
        t0 = time.perf_counter()

        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, optimizer, batch_size=batch_size)
        test_loop(test_loader, model)

        t1 = time.perf_counter()
        print("time per epoch : ", t1 - t0)

    torch.save(model._orig_mod.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
