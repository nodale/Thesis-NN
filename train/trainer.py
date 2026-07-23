import hydra
import torch
import time
import os
import math

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from model.network import JeuralJetwork
from data.dataset import QuickDataset2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def loss_fn(pred, truth):
    return (torch.nn.functional.mse_loss(pred, truth))

def loss_fn_gml(pred, target, eps=1e-6, use_logvar=False):
    h = pred.shape[-1] // 2
    mu, raw = pred[..., :h], pred[..., h:]
    err = mu - target[..., :h]

    if use_logvar:                       # raw head predicts log-variance (unconstrained)
        var = torch.exp(raw)
        log_det = raw.sum(dim=-1)        # sum(log(var)) == sum(logvar)
    else:                                # raw head predicts variance directly
        var = torch.clamp(raw, min=eps)
        log_det = torch.log(var).sum(dim=-1)

    mahalanobis = (err.pow(2) / var).sum(dim=-1)
    loss = 0.5 * log_det + 0.5 * mahalanobis
    return loss.mean()

def train_loop(loader, model, optimizer, batch_size=100, process_name=" ", pred_dim=6, plot=False):
    #monitoring
    if plot is True:
        plt.ion()
        fig, ax = plt.subplots()

    losses = []

    #training
    model.train()
    scaler = torch.amp.GradScaler("cuda:0")
    running_loss = 0.0
    count = 0
    tot_len = len(loader)

    for vec in loader:
        t0 = time.perf_counter()

        vec = vec.to(device, non_blocking=True)
        in_vec = vec[:, :model.input_len, :]
        truth_vec = vec[:, model.input_len:model.input_len+1, :pred_dim] - vec[:, model.input_len - 1:model.input_len, :pred_dim]
        with torch.amp.autocast("cuda:0", dtype=torch.float16):
            pred_vec = model(in_vec)
            loss = loss_fn(pred_vec[:, :, :pred_dim], truth_vec)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0

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

def train_rollout_loop(loader, model, optimizer, generator, batch_size=100, schedule_prob=0.01, rollout_max_steps=24, process_name=" ", pred_dim=13, est_dim=6, plot=False):
    #monitoring
    if plot is True:
        plt.ion()
        fig, ax = plt.subplots()
    losses = []

    #training
    model.train()
    scaler = torch.amp.GradScaler("cuda:0")
    running_loss = 0.0
    count = 0
    tot_len = len(loader)
    for vec in loader:
        t0 = time.perf_counter()

        vec = vec.to(device, non_blocking=True)
        history = vec[:, :model.input_len, :].clone()
        loss = 0
        #rollout_steps = torch.randint(low=3,high=rollout_max_steps, size=(), generator=generator, device="cuda:0")
        rollout_steps = rollout_max_steps
        for step in range(rollout_steps):
            with torch.amp.autocast("cuda:0", dtype=torch.float16):
                pred = model(history)
                pred_delta = pred[:,0,:pred_dim]
                curr_state = history[:,-1,:pred_dim]
                pred_state = curr_state + pred_delta
                gt_state = vec[:, model.input_len + step, :pred_dim]
                loss += loss_fn(pred_state, gt_state)

                est_state = pred_state[:, :est_dim]
                real_state = vec[:, model.input_len + step, :est_dim]

            next_frame = vec[:, model.input_len + step, :].clone()

            use_pred = (
                torch.rand(
                    history.shape[0],
                    device=history.device,
                    generator=generator
                ) < schedule_prob
            )
            next_frame[:, :est_dim] = torch.where(
                use_pred.unsqueeze(1),
                est_state,
                real_state
            )
            history = torch.roll(history, -1, dims=1)
            history[:,-1,:] = next_frame

        loss /= rollout_steps
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0
        t1=time.perf_counter()
        dt = t1 - t0

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

def train_rollout_horizon_loop(loader, model, optimizer, generator, batch_size=100, schedule_prob=0.01, rollout_max_steps=24, process_name=" ", pred_dim=13, est_dim=6, plot=False):
    #monitoring
    if plot is True:
        plt.ion()
        fig, ax = plt.subplots()

    losses = []

    #training
    model.train()
    scaler = torch.amp.GradScaler("cuda:0")
    running_loss = 0.0
    count = 0
    tot_len = len(loader)
    for vec in loader:
        t0 = time.perf_counter()

        vec = vec.to(device, non_blocking=True)
        history = vec[:, :model.input_len, :].clone()
        loss = 0
        #rollout_steps = torch.randint(low=3,high=rollout_max_steps, size=(), generator=generator, device="cuda:0")
        rollout_steps = rollout_max_steps

        for step in range(rollout_steps):
            with torch.amp.autocast("cuda:0", dtype=torch.float16):
                pred = model(history)
                pred_delta = pred[:,0,:pred_dim]
                curr_state = history[:,-1,:pred_dim]
                pred_state = curr_state + pred_delta
                gt_state = vec[:, model.input_len + step, :pred_dim]
                weight = (step + 1) / (rollout_steps * (rollout_steps + 1) / 2)
                loss += weight * loss_fn(pred_state, gt_state)

                est_state = pred_state[:, :est_dim]
                real_state = vec[:, model.input_len + step, :est_dim]

            next_frame = vec[:, model.input_len + step, :].clone()

            use_pred = (
                torch.rand(
                    history.shape[0],
                    device=history.device,
                    generator=generator
                ) < schedule_prob
            )
            next_frame[:, :est_dim] = torch.where(
                use_pred.unsqueeze(1),
                est_state,
                real_state
            )
            history = torch.roll(history, -1, dims=1)
            history[:,-1,:] = next_frame

        #loss /= rollout_steps
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1.0
        t1=time.perf_counter()
        dt = t1 - t0

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

def train_gml_loop(loader, model, optimizer, batch_size=100, device="cuda:0"):
    model.train()
    scaler = torch.amp.GradScaler("cuda:0")
    running_loss = 0.0
    losses = []
    count = 0
    tot_len = len(loader)

    for vec in loader:
        t0 = time.perf_counter()

        vec = vec.to(device, non_blocking=True)

        in_vec = vec[:, :model.input_len, :]
        truth_vec = vec[:, model.input_len:, :]

        with torch.amp.autocast("cuda:0", dtype=torch.float16):
            pred_vec = model(in_vec)
            print(pred_vec)
            loss = loss_fn_gml(pred_vec, truth_vec)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach()
        count += 1

        losses.append(running_loss.cpu())
        if count % batch_size == 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1

            print(
                f"avg_loss: {running_loss / batch_size:.8f} "
                f"var: {var:.6f} "
                f"time/window: {dt/batch_size:.6f} "
                f"progress: {count/tot_len:.3f}"
            )
            running_loss = 0.0

    return losses

def train_gml_rollout_loop(loader, model, optimizer, generator, batch_size=100, schedule_prob=0.01, rollout_max_steps=24, process_name=" ", pred_dim=13, est_dim=6, plot=False):
    #monitoring
    if plot is True:
        plt.ion()
        fig, ax = plt.subplots()

    losses = []

    #training
    model.train()
    scaler = torch.amp.GradScaler("cuda:0")
    running_loss = 0.0
    count = 0
    tot_len = len(loader)
    for vec in loader:
        t0 = time.perf_counter()

        vec = vec.to(device, non_blocking=True)
        history = vec[:, :model.input_len, :].clone()
        loss = 0
        #rollout_steps = torch.randint(low=3,high=rollout_max_steps, size=(), generator=generator, device="cuda:0")
        rollout_steps = rollout_max_steps

        for step in range(rollout_max_steps):
            with torch.amp.autocast("cuda:0", dtype=torch.float16):
                pred=model(history)
                mu_delta=pred[:,0,:pred_dim]
                logvar=pred[:,0,pred_dim:]
                curr=history[:,-1,:pred_dim]
                mu_state=curr+mu_delta
                gt_state=vec[:,model.input_len+step,:pred_dim]
                pred_gaussian=torch.cat([mu_state,logvar],dim=-1)
                loss+=loss_fn_gml(pred_gaussian,gt_state)
                #loss+=loss_fn_gml(torch.cat([mu_state, logvar], -1), gt_state)

                est_state = mu_state[:, :est_dim]
                real_state = vec[:, model.input_len + step, :est_dim]

            next_frame=vec[:,model.input_len+step,:].clone()
            use_pred=(torch.rand(history.shape[0],device=history.device,generator=generator)<schedule_prob)
            next_frame[:, :est_dim]=torch.where(use_pred.unsqueeze(1),est_state,real_state)
            #next_frame[:, :est_dim]=real_state
            history=torch.cat([history[:,1:,:],next_frame.unsqueeze(1)],dim=1)

        loss /= rollout_steps
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.detach()
        count += 1.0
        t1 = time.perf_counter()
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
                ax.set_yscale("linear")
                fig.canvas.flush_events()
                plt.pause(0.05)

            running_loss = 0.0

    return losses

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    run_dir = HydraConfig.get().runtime.output_dir
    process_name = "\n".join(HydraConfig.get().overrides.task)

    gen = torch.Generator(device="cuda:0").manual_seed(cfg.seed)

    total_len = cfg.input_len + cfg.output_len
    all_losses = []

    if cfg.training.mode == "rollout" or cfg.training.mode == "standard":
        model = JeuralJetwork(
            n_dim=cfg.models.n_dim,
            out_dim=cfg.models.out_dim,
            input_len=cfg.input_len,
            output_len=cfg.output_len,
            **cfg.models.architecture,).to(device)
    else:
        model = JeuralJetwork(
            n_dim=cfg.models.n_dim,
            out_dim=cfg.models.out_dim * 2,
            input_len=cfg.input_len,
            output_len=cfg.output_len,
            **cfg.models.architecture,).to(device)
    if cfg.checkpoint.load:
        state_dict = torch.load(cfg.checkpoint.path, map_location=device)
        model.load_state_dict(state_dict)

    if cfg.compile:
        model=torch.compile(
                model,
                mode="reduce-overhead"
            )


    train_dataset = QuickDataset2(
        path=cfg.dataset.path,
        training_size=cfg.dataset.training_size,
        window_size=total_len+cfg.input_len,)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=6,
        pin_memory=True,
        multiprocessing_context='fork',
        persistent_workers=True,
        prefetch_factor=4,
        )

    val_dataset = QuickDataset2(
        path=cfg.evaluation.path,
        training_size=cfg.evaluation.evaluation_size,
        window_size=total_len+cfg.input_len,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context='fork',
        persistent_workers=True,
        prefetch_factor=2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        eps=1e-12,
        )
    sched_prob = 1.0/(-1.0 + cfg.epochs)

    plot = False
    for epoch in range(cfg.epochs):
        sched_prob_imp = 1 / (1 + math.exp(-12*(epoch*sched_prob-0.5)))
        #p = epoch/cfg.epochs
        #sched_prob_imp = p**2
        #sched_prob_imp = sched_prob * epoch
        #sched_prob_imp = 0.0

        if cfg.training.mode == "standard":
            losses = train_loop(
                train_loader,
                model,
                optimizer,
                batch_size=cfg.batch_size,
                process_name=process_name,
                plot=plot
                )

        elif cfg.training.mode == "rollout":
            losses = train_rollout_loop(
                train_loader,
                model,
                optimizer,
                generator=gen,
                batch_size=cfg.batch_size,
                pred_dim=cfg.models.out_dim,
                rollout_max_steps=cfg.training.rollout_steps,
                schedule_prob=sched_prob_imp,
                process_name=process_name,
                plot=plot
                )

        elif cfg.training.mode == "rollout_horz":
            losses = train_rollout_horizon_loop(
                train_loader,
                model,
                optimizer,
                generator=gen,
                batch_size=cfg.batch_size,
                pred_dim=cfg.models.out_dim,
                rollout_max_steps=cfg.training.rollout_steps,
                schedule_prob=sched_prob_imp,
                process_name=process_name,
                plot=plot
                )

        elif cfg.training.mode == "gml":
            train_loop_gml(
                train_loader,
                model,
                optimizer,
                batch_size=cfg.batch_size,
                )

        elif cfg.training.mode == "gml_rollout":
            gml_sched_prob = 1.0/(-1.0 + (cfg.epochs//2))
            if epoch < cfg.epochs - 1:
                #sched_prob_imp = 1 / (1 + math.exp(-12*(epoch*gml_sched_prob-0.5)))
                losses = train_rollout_loop(
                    train_loader,
                    model,
                    optimizer,
                    generator=gen,
                    batch_size=cfg.batch_size,
                    pred_dim=cfg.models.out_dim,
                    rollout_max_steps=cfg.training.rollout_steps,
                    schedule_prob=sched_prob_imp,
                    process_name=process_name,
                    plot=plot
                    )
            else:
                gml_sched_prob_imp = 1 / (1 + math.exp(-12*((epoch-4)*gml_sched_prob-0.5)))
                gml_sched_prob_imp = 0.0
                losses = train_gml_rollout_loop(
                    train_loader,
                    model,
                    optimizer,
                    generator=gen,
                    batch_size=cfg.batch_size,
                    pred_dim=cfg.models.out_dim,
                    rollout_max_steps=cfg.training.rollout_steps,
                    schedule_prob=gml_sched_prob_imp,
                    process_name=process_name,
                    plot=plot
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
