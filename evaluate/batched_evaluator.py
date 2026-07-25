import os
import json
import yaml
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import numpy as np
import zarr

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader

from data.dataset import QuickDatasetStraight
from model.network import JeuralJetwork
from evaluate.metrics import print_all_metrics, MetricsAccumulator
from data.denormaliser import denormalise

import matplotlib.pyplot as plt

def plot(pred_list, truth_list, name="trajectories"):
    """
    pred_list: list of tensors [T, 3]
    truth_list: list of tensors [T, 3]
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot all trajectories
    for pred, truth in zip(pred_list, truth_list):

        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()

        #ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], alpha=0.8)
        ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], linestyle="--", alpha=0.5)

    ax.set_title(name)
    ax.legend(["pred", "truth"])

    plt.show()


# -------------------------
# utilities (unchanged)
# -------------------------
def get_overrides(run_dir):
    path = run_dir / ".hydra/overrides.yaml"
    if not path.exists():
        return ""
    with open(path) as f:
        return ", ".join(yaml.safe_load(f))

def load_model(run_dir, device):

    with open(run_dir / ".hydra/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt = list(run_dir.glob("**/*.pth"))[0]
    print("Loading:", ckpt)

    if cfg["training"]["mode"] == "gml_rollout" or cfg["training"]["mode"] == "gml":
        model = JeuralJetwork(
            n_dim=cfg["models"]["n_dim"],
            out_dim=cfg["models"]["out_dim"]*2,
            input_len=cfg["input_len"],
            output_len=cfg["output_len"],
            **cfg["models"]["architecture"]
        )
    else:
        model = JeuralJetwork(
            n_dim=cfg["models"]["n_dim"],
            out_dim=cfg["models"]["out_dim"],
            input_len=cfg["input_len"],
            output_len=cfg["output_len"],
            **cfg["models"]["architecture"]
        )

    state = torch.load(ckpt, map_location=device)

    # handle torch.compile prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model = model.to(device)

    #if cfg.get("compile", False):
    #    model = torch.compile(model)

    model.eval()

    return model, cfg

def load_episode_tensor(root, eps_indices, window_size):
    data = []

    for e in eps_indices:
        ep = root[e].astype(np.float32)  # [T_total, C]
        data.append(ep)

    return torch.from_numpy(np.stack(data))  # [B, T_total, C]

def rollout_batched(model, episodes, input_len, output_len, device, pred_dim=6):

    B, T_total, C = episodes.shape
    T = input_len + output_len

    init_pos = episodes[:, :input_len, :pred_dim].to(device)

    preds = [[] for _ in range(B)]
    truths = [[] for _ in range(B)]

    with torch.inference_mode():

        for t in range(T_total - T):

            # === ONLY PAST CONTEXT (NO FUTURE LEAKAGE) ===
            context = episodes[:, t:t+input_len, :].to(device).clone()
            context[:, :, :pred_dim] = init_pos

            # === MODEL STEP (BATCHED OVER EPISODES) ===
            out = model(context)  # or your required shape
            delta = out[:, 0, :pred_dim]

            # === ORIGINAL STATE UPDATE ===
            new_pos = init_pos[:, -1] + delta

            init_pos = torch.roll(init_pos, -1, dims=1)
            init_pos[:, -1] = new_pos

            # === LOGGING ===
            for b in range(B):
                preds[b].append(new_pos[b].cpu())
                truths[b].append(episodes[b, t + input_len, :pred_dim].cpu())

    preds = [torch.stack(p) for p in preds]
    truths = [torch.stack(t) for t in truths]

    return preds, truths

def evaluate_batched(model, data, eps_indices, input_len, output_len, device, pred_dim=6):

    T = input_len + output_len
    B = len(eps_indices)

    # init states
    init = [
        data[e, :input_len, :pred_dim].astype(np.float32)
        for e in eps_indices
    ]
    init_pos = torch.from_numpy(np.stack(init)).to(device)  # [B, T_in, C]

    preds = [[] for _ in range(B)]
    truths = [[] for _ in range(B)]

    with torch.inference_mode():

        for i in range(len(data[0]) - T):

            batch = []
            gt_batch = []

            for b, e in enumerate(eps_indices):

                window = data[
                    e,
                    i:i+T,
                    :
                ].astype(np.float32)

                window = torch.from_numpy(window).to(device)

                window[:input_len, :pred_dim] = init_pos[b]

                batch.append(window)
                gt_batch.append(window[-1, :pred_dim].detach().cpu())

            batch = torch.stack(batch)  # [B, T, C]

            out = model(batch)         # [B, ?, C]
            out = out[:, 0, :pred_dim]

            new_pos = init_pos[:, -1] + out

            init_pos = torch.roll(init_pos, -1, dims=1)
            init_pos[:, -1] = new_pos

            for b in range(B):
                preds[b].append(new_pos[b].cpu())
                truths[b].append(gt_batch[b])

    preds = [torch.stack(p) for p in preds]
    truths = [torch.stack(t) for t in truths]

    return preds, truths

def evaluate_model(model, root, eps_indices, input_len, output_len, device):

    episodes = load_episode_tensor(
        root,
        eps_indices,
        input_len + output_len + 4500  # buffer for rollout
    )

    preds, truths = rollout_batched(
        model,
        episodes,
        input_len,
        output_len,
        device
    )

    #plot([denormalise(p) for p in preds], [denormalise(t) for t in truths], name=" ")

    acc = MetricsAccumulator()

    for p, t in zip(preds, truths):
        p = denormalise(p)
        t = denormalise(t)
        acc.update(p, t)

    return acc

# -------------------------
# worker (runs in parallel)
# -------------------------
def run_eval(run_path, eps_indices):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # original loader restored
    model, cfg = load_model(run_path, device)

    root = zarr.open(
        zarr.storage.LocalStore(
            cfg["evaluation"]["path"]
        ),
        mode="r"
    )["episodes"]

    acc = evaluate_model(
        model,
        root,
        eps_indices,
        cfg["input_len"],
        cfg["output_len"],
        device
    )

    print(f"\nRUN: {run_path.name}")
    acc.print()

    return run_path.name, acc.average()

# -------------------------
# main parallel loop
# -------------------------
from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    sweep_override = os.environ.get("SWEEP_DIR")
    if sweep_override:
        sweep = Path(sweep_override)
    else:
        sweep = max(Path("multirun").glob("*/*"), key=lambda p: p.stat().st_mtime)

    runs = [
        r for r in sweep.iterdir()
        if r.is_dir() and (r / ".hydra").exists()
    ]

    eps_indices = list(range(int(os.environ.get("EVAL_EPISODES", 1000))))

    with ProcessPoolExecutor(max_workers=min(len(runs), os.cpu_count())) as ex:
        futures = [
            ex.submit(run_eval, run, eps_indices)
            for run in runs
        ]

        results = []
        for f in as_completed(futures):
            results.append(f.result())

    print("\nDONE")
    for name, avg_metrics in results:
        print(name, avg_metrics)

    out_path = os.environ.get("EVAL_OUTPUT")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sweep": str(sweep),
            "runs": {
                name: {
                    "overrides": get_overrides(next(r for r in runs if r.name == name)),
                    "metrics": avg_metrics,
                }
                for name, avg_metrics in results
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
