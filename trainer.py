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

        self.mamba = nn.Sequential(
            Mamba(
                d_model=neural_count,
                d_state=28,
                d_conv=4,
                expand=2,
            ),
            nn.LayerNorm(neural_count),
            Mamba(
                d_model=neural_count,
                d_state=28,
                d_conv=4,
                expand=2,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(neural_count, neural_count),
            nn.GELU(),
            nn.Dropout(0.10),

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
        
        #print(x.shape) #maybe train the data on sequences instead of windows?
        #x = x.mean(dim=1)
        
        x = x.view(vec.size(0), self.output_len, self.out_dim) # [B, output_len, out_dim]

        return x


def loss_fn(pred, truth):
    return (torch.nn.functional.mse_loss(pred, truth))

def loss_fn_gml(pred_vec, truth_vec):
    pred_mean = pred_vec[..., :3]

    raw_var = pred_vec[..., 3:]
    var = torch.nn.functional.softplus(raw_var) + 1e-6

    e = truth_vec - pred_mean

    logdet = torch.log(var).sum(dim=-1)
    mahal = (e.square() / var).sum(dim=-1)

    loss = 0.5 * (logdet + mahal)

    print("diff : ", e[-1, :3].square().detach().cpu(), "   var : ", raw_var[-1, :3].detach().cpu())

    # average over horizon and batch
    return loss.mean()

def train_loop(loader, model, optimizer, batch_size=100, std_min=1e-8,std_max=2e-4):
    model.train()

    running_loss = 0.0  
    count = 0          

    t0 = time.perf_counter()

    tot_len = len(loader)

    for vec in loader:
        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:, :model.input_len, :]
        truth_vec = vec[:, model.input_len:, :3] - vec[:, model.input_len - 1:model.input_len, :3]
    
        pred_vec = model(in_vec)
        loss = loss_fn(pred_vec[:, :, :3], truth_vec)

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


def train_loop_gml(loader, model, optimizer, batch_size=100):
    model.train()

    running_loss = 0.0  
    count = 0          

    t0 = time.perf_counter()

    tot_len = len(loader)

    for vec in loader:
        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:, :model.input_len, :]
        truth_vec = vec[:, model.input_len:, :3] - vec[:, model.input_len - 1:model.input_len, :3]
        pred_vec = model(in_vec)

        loss = loss_fn_gml(pred_vec, truth_vec)

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
    input_len = 12
    output_len = 1
    total_len = input_len + output_len
    batch_size = 128

    model = NeuralNetwork(
            input_len=input_len, 
            output_len=output_len, 
            n_dim=26, 
            out_dim=6,
            ).to(device)
    state_dict = torch.load("model_12steps.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=5e-6,
            eps=1e-38,
            weight_decay=2e-3,
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
        prefetch_factor=16
    )

    train_gml_dataset = QuickDataset2(path='/home/joey/Thesis/data/patient_one_data.zarr/', training_size=training_size, window_size=total_len)
    train_gml_loader = DataLoader(
        train_gml_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=16
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

    epochs = 1
    for t in range(epochs):
        t0 = time.perf_counter()

        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, optimizer, batch_size=batch_size)
        #print("STARTING GML TRAINING")
        #train_loop_gml(train_gml_loader, model, optimizer, batch_size=batch_size)
        #test_loop(test_loader, model)

        t1 = time.perf_counter()
        print("time per epoch : ", t1 - t0)

    torch.save(model._orig_mod.state_dict(), "model_12steps.pth")

if __name__ == "__main__":
    main()
