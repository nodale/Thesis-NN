import hydra
import torch
import time
import os
import math

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from include.mama import JeuralJetwork
from QuickDataset import QuickDataset2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

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

def train_loop(loader, model, optimizer, batch_size=100, process_name=" ", pred_dim=6, plot=False):
    #monitoring
    if plot is True:
        plt.ion()
        fig, ax = plt.subplots()
    
    losses = []

    #training
    model.train()
    running_loss = 0.0  
    count = 0          
    t0 = time.perf_counter()
    tot_len = len(loader)

    for vec in loader:
        delta = (vec[:, model.input_len:model.input_len+1, :pred_dim] - vec[:, model.input_len-1:model.input_len, :pred_dim])
        vec = vec.to(device, non_blocking=True)
        in_vec = vec[:, :model.input_len, :]
        truth_vec = vec[:, model.input_len:model.input_len+1, :pred_dim] - vec[:, model.input_len - 1:model.input_len, :pred_dim]
        pred_vec = model(in_vec)
        loss = loss_fn(pred_vec[:, :, :pred_dim], truth_vec)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0

        torch.cuda.synchronize()
        t1=time.perf_counter()
        dt = t1 - t0
        t0 = t1

        losses.append(running_loss.cpu())
        if count % batch_size == 0:
            print(f"name: {process_name}    avg_loss: {running_loss / batch_size:.12f}  time_per_window : {dt/batch_size:.6f}   progress : {count/tot_len:.3f}")


            if plot is True:

                ax.clear()
                ax.plot(losses)
                ax.text(
                    0.02,
                    0.95,
                    process_name,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top"
                )
                ax.set_yscale("log")
                fig.canvas.flush_events()
                plt.pause(0.05)

            running_loss = 0.0
    return losses

def train_rollout_loop(loader, model, optimizer, generator, batch_size=100, schedule_prob=0.01, rollout_max_steps=24, process_name=" ", pred_dim=6, plot=False):
    #monitoring
    if plot is True:
        plt.ion()
        fig, ax = plt.subplots()
    
    losses = []

    #training
    model.train()
    running_loss = 0.0  
    count = 0          
    t0 = time.perf_counter()
    tot_len = len(loader)
    for vec in loader:
        delta = (
            vec[:, model.input_len:, :pred_dim]
            - vec[:, model.input_len-1:model.input_len, :pred_dim]
        )

        vec = vec.to(device, non_blocking=True)
        history = vec[:, :model.input_len, :].clone()
        loss = 0
        #rollout_steps = torch.randint(low=3,high=rollout_max_steps, size=(), generator=generator, device="cuda")
        rollout_steps = rollout_max_steps
        for step in range(rollout_steps):
            pred = model(history)
            pred_delta = pred[:,0,:pred_dim]
            curr_state = history[:,-1,:pred_dim]
            pred_state = curr_state + pred_delta
            gt_state = vec[:, model.input_len + step, :pred_dim]
            loss += loss_fn(pred_state, gt_state)
            next_frame = vec[:, model.input_len + step, :].clone()

            use_pred = (
                torch.rand(
                    history.shape[0],
                    device=history.device,
                    generator=generator
                ) < schedule_prob
            )

            next_frame[:, :pred_dim] = torch.where(
                use_pred.unsqueeze(1),
                pred_state,
                gt_state
            )

            history = torch.cat(
                [
                    history[:,1:,:],
                    next_frame.unsqueeze(1)
                ],
                dim=1
            )

        loss /= rollout_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0
        torch.cuda.synchronize()
        t1=time.perf_counter()
        dt = t1 - t0
        t0 = t1


        losses.append(running_loss.cpu())
        if count % batch_size == 0:
            print(f"name: {process_name}    avg_loss: {running_loss / batch_size:.12f}  time_per_window : {dt/batch_size:.6f}   progress : {count/tot_len:.3f}")

            if plot is True:
                ax.clear()
                ax.plot(losses)
                ax.text(
                    0.02,
                    0.95,
                    process_name,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top"
                )
                ax.set_yscale("log")
                fig.canvas.flush_events()
                plt.pause(0.05)

            running_loss = 0.0

    return losses


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


        torch.cuda.synchronize()
        t1=time.perf_counter()
        dt = t1 - t0
        t0 = t1
        if count % batch_size == 0:
            print(f"avg_loss: {running_loss / batch_size:.12f}          random_var: {var}         time_per_window : {dt/batch_size:.6f}        progress : {count/tot_len:.3f}")
            running_loss = 0.0


@hydra.main(
    version_base=None,
    config_path="hydra-cfgs",
    config_name="config",
)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    run_dir = HydraConfig.get().runtime.output_dir
    process_name = "\n".join(HydraConfig.get().overrides.task)

    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

    total_len = cfg.input_len + cfg.output_len
    all_losses = []

    model = JeuralJetwork(
        n_dim=cfg.models.n_dim,
        out_dim=cfg.models.out_dim,
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        **cfg.models.architecture,).to(device)

    if cfg.checkpoint.load:
        state_dict = torch.load(cfg.checkpoint.path, map_location=device)
        model.load_state_dict(state_dict)

    if cfg.compile:
        model = torch.compile(model)


    train_dataset = QuickDataset2(
        path=cfg.dataset.path,
        training_size=cfg.dataset.training_size,
        window_size=total_len+cfg.training.rollout_steps,)
            
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=32,
        pin_memory=True,
        multiprocessing_context='fork',
        persistent_workers=True,
        prefetch_factor=8,
        )

    val_dataset = QuickDataset2(
        path=cfg.evaluation.path,
        training_size=cfg.evaluation.evaluation_size,
        window_size=total_len+cfg.training.rollout_steps,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=32,
        pin_memory=True,
        multiprocessing_context='fork',
        persistent_workers=True,
        prefetch_factor=8,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        eps=1e-12,
        )
    sched_prob = 1.0/(1.0 + cfg.epochs)

    for epoch in range(cfg.epochs):
        sched_prob_sigmoid = 1 / (1 + math.exp(-12*(epoch*sched_prob-0.5)))

        if cfg.training.mode == "standard":
            losses = train_loop(
                train_loader,
                model,
                optimizer,
                batch_size=cfg.batch_size,
                process_name=process_name,
                )

        elif cfg.training.mode == "rollout":
            losses = train_rollout_loop(
                train_loader,
                model,
                optimizer,
                generator=gen,
                batch_size=cfg.batch_size,
                schedule_prob=epoch * sched_prob,
                process_name=process_name,
                )

        elif cfg.training.mode == "gml":
            train_loop_gml(
                train_loader,
                model,
                optimizer,
                batch_size=cfg.batch_size,
                )

        all_losses.extend(losses)

    save_path = os.path.join(run_dir, cfg.checkpoint.save_path,)

    torch.save(
        model._orig_mod.state_dict()
        if hasattr(model, "_orig_mod")
        else model.state_dict(),
        save_path,
    )

    np.save(os.path.join(run_dir, "loss_history.npy"), torch.stack(all_losses).cpu().numpy())

    print("DONE!!!")

    plt.ioff()
    plt.show()

    return 0


if __name__ == "__main__":
    main()

