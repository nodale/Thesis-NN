import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2
from mamba_ssm import Mamba
from include.mama import JeuralJetwork
from mamba_ssm.ops.triton.layer_norm import RMSNorm

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def loss_fn(pred, truth):
    return (torch.nn.functional.mse_loss(pred, truth))

def loss_fn_gml(pred_vec, truth_vec):
    pred_mean = pred_vec[..., :3]
    raw_var = pred_vec[..., 3:]
    var = torch.nn.functional.softplus(raw_var) + 1e-12
    e = truth_vec - pred_mean
    logdet = torch.log(var).sum(dim=-1)
    mahal = (e.square() / var).sum(dim=-1)
    loss = 0.5 * (logdet + mahal)

    return loss.mean(), var[-1, :].detach().cpu()

def train_loop(loader, model, optimizer, batch_size=100):
    model.train()

    running_loss = 0.0  
    count = 0          

    t0 = time.perf_counter()

    tot_len = len(loader)

    for vec in loader:
        delta = (
            vec[:, model.input_len:, :model.out_dim]
            - vec[:, model.input_len-1:model.input_len, :model.out_dim]
        )

        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:, :model.input_len, :]
        truth_vec = vec[:, model.input_len:, :model.out_dim] - vec[:, model.input_len - 1:model.input_len, :model.out_dim]
    
        pred_vec = model(in_vec)
        loss = loss_fn(pred_vec[:, :, :model.out_dim], truth_vec)

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


def train_rollout_loop(loader, model, optimizer, generator, batch_size=100, schedule_prob=0.01):
    model.train()
    running_loss = 0.0  
    count = 0          
    t0 = time.perf_counter()
    tot_len = len(loader)
    for vec in loader:
        delta = (
            vec[:, model.input_len:, :3]
            - vec[:, model.input_len-1:model.input_len, :3]
        )

        vec = vec.to(device, non_blocking=True)
        history = vec[:, :model.input_len, :].clone()
        loss = 0
        for step in range(model.output_len):
            pred = model(history)
            pred_delta = pred[:,0,:3]
            curr_state = history[:,-1,:3]
            pred_state = curr_state + pred_delta
            gt_state = vec[:, model.input_len + step, :3]
            loss += loss_fn(pred_state, gt_state)
            next_frame = vec[:, model.input_len + step, :].clone()
            use_pred = (
                torch.rand(
                    history.shape[0],
                    device=history.device,
                    generator=generator
                ) < schedule_prob
            )

            next_frame[:, :3] = torch.where(
                use_pred.unsqueeze(1),   # (B,1)
                pred_state,              # (B,3)
                gt_state                 # (B,3)
            )

            history = torch.cat(
                [history[:,1:,:],
                 next_frame.unsqueeze(1)],
                dim=1
            )
        loss /= model.output_len
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

        loss, var = loss_fn_gml(pred_vec, truth_vec)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0


        t1 = time.perf_counter()
        dt = t1 - t0
        t0 = t1
        if count % batch_size == 0:
            print(f"avg_loss: {running_loss / batch_size:.12f}          random_var: {var}         time_per_window : {dt/batch_size:.6f}        progress : {count/tot_len:.3f}")
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
    output_len = 3
    total_len = input_len + output_len
    batch_size = 1024

    gen = torch.Generator(device="cuda").manual_seed(0)

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
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4,
            eps=1e-14,
            weight_decay=1e-5,
            )
            #betas=(0.98, 0.999),

    training_size = 1500000

    train_dataset = QuickDataset2(path='/home/joey/Thesis/data/patient_one_data.zarr/', training_size=training_size, window_size=total_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=32
    )

    epochs = 5
    epochs_iter = 0
    for t in range(epochs):
        t0 = time.perf_counter()

        print(f"Epoch {t+1}\n-------------------------------")
        _schedule_prob = epochs_iter * 0.25
        #train_rollout_loop(train_loader, model, optimizer, generator=gen, batch_size=batch_size, schedule_prob=_schedule_prob)
        #train_loop_gml(train_loader, model, optimizer, batch_size=batch_size)
        train_loop(train_loader, model, optimizer, batch_size=batch_size)

        t1 = time.perf_counter()
        print("time per epoch : ", t1 - t0)
        epochs_iter += 1

        torch.save(model._orig_mod.state_dict(), "models/model_original+replica+rollout.pth")

if __name__ == "__main__":
    main()
