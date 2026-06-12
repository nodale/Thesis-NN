import os
import glob
import yaml
import torch
import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from torch.utils.data import DataLoader, IterableDataset
from QuickDataset import QuickDatasetStraight
from include.mama import JeuralJetwork
from pathlib import Path

matplotlib.use("QtAgg")

def get_latest_multirun():
    runs = list(Path("multirun").glob("*/*"))

    if not runs:
        raise RuntimeError("No Hydra runs found")

    latest = max(
        runs,
        key=lambda p: p.stat().st_mtime
    )

    return latest


def latest_sweep():
    sweeps = list(Path("multirun").glob("*/*"))
    if not sweeps:
        raise RuntimeError("No Hydra sweeps found")

    return max(
        sweeps,
        key=lambda p: p.stat().st_mtime
    )

def load_run(run_dir, device):

    # load hydra saved config
    with open(run_dir / ".hydra/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # find checkpoint
    ckpt = list(run_dir.glob("**/*.pth"))[0]

    print("Loading:", ckpt)

    model = JeuralJetwork(
        n_dim=cfg["models"]["n_dim"],
        out_dim=cfg["models"]["out_dim"],
        input_len=cfg["input_len"],
        output_len=cfg["output_len"],
        **cfg["models"]["architecture"]
    )

    state = torch.load(
        ckpt,
        map_location=device
    )

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model, cfg


def plot(pred, truth, name):
    a = pred.detach().cpu().numpy()
    b = truth.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(a[:,0], a[:,1], a[:,2], label="pred")
    ax.plot(b[:,0], b[:,1], b[:,2], label="truth")

    ax.set_title(name)
    ax.legend()
    plt.show()






def load_model(checkpoint, cfg, device):

    model = JeuralJetwork(
        n_dim=cfg["models"]["n_dim"],
        out_dim=cfg["models"]["out_dim"],
        input_len=cfg["input_len"],
        output_len=cfg["output_len"],
        **cfg["models"]["architecture"],
    )


    state = torch.load(
        checkpoint,
        map_location=device
    )

    # handle torch.compile checkpoints
    state = {
        k.replace("_orig_mod.", ""):v
        for k,v in state.items()
    }

    model.load_state_dict(state)

    model = model.to(device)

    if cfg.get("compile", False):
        model = torch.compile(model)

    model.eval()

    return model



def evaluate(model, loader, input_len, output_len, device):

    total_len = input_len + output_len

    init_pos = next(iter(loader))[:input_len,:3]
    init_pos = init_pos.to(device)


    predicted=[]
    truth=[]

    with torch.inference_mode():
        for d in loader:

            _in = d[:input_len, :].to(device)
            _in[:input_len, :3] = init_pos

            out = model(_in.unsqueeze(0))
            out = out.squeeze(0)

            new_pos = init_pos[-1] + out[0, :3]

            init_pos = torch.roll(
                init_pos,
                shifts=-1,
                dims=0
            )
            init_pos[-1] = new_pos

            predicted.append(
                new_pos.cpu().unsqueeze(0)
            )

            truth.append(
                d[-1, :3].cpu().unsqueeze(0)
            )

    predicted = torch.cat(predicted, dim=0)
    truth = torch.cat(truth)

    print(predicted.shape)
    print(truth.shape)

    return predicted, truth


def main():

    device = torch.device("cuda")

    sweep = latest_sweep()

    print(
        "Evaluating sweep:",
        sweep
    )


    for run in sorted(sweep.iterdir()):

        if not run.is_dir():
            continue

        if not (run / ".hydra").exists():
            continue


        print("\n===================")
        print("RUN:", run.name)


        model, cfg = load_run(
            run,
            device
        )


        input_len = cfg["input_len"]
        output_len = cfg["output_len"]

        dataset = QuickDatasetStraight(
            path=cfg["dataset"]["path"],
            episode_idx=0,
            window_size=input_len + output_len
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0
        )


        predicted, truth = evaluate(
            model,
            loader,
            input_len,
            output_len,
            device
        )


        print(
            "MS Error:",
            torch.mean(
                (predicted - truth)**2
            )
        )

        plot(
            name=f"{run.name}",
            pred=predicted,
            truth=truth
        )


if __name__=="__main__":
    main()
