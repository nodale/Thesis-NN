"""Compare offline (ablation_seeded) training loss vs online (in-sim)
fine-tuning loss, for both architectures and both head types.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
ROOT = Path(__file__).parent

OFFLINE_LOGS = {
    "simple_mamba_out6":        REPO_ROOT / "ablation_seeded/2026-07-27_00-17-03/simple_mamba_out6.train.log",
    "attention_1e0d_out10":     REPO_ROOT / "ablation_seeded/2026-07-27_00-16-14/attention_1e0d_out10\t\t.train.log",
    "simple_gml_mamba_out10":   REPO_ROOT / "ablation_seeded/2026-07-27_00-17-03/simple_mamba_gml_rollout.train.log",
    "attention_1e0d_gml_out10": REPO_ROOT / "ablation_seeded/2026-07-27_00-16-14/attention_1e0d_gml_out10\t\t.train.log",
}
ONLINE_DIRS = {
    "simple_mamba_out6":        ROOT / "sim_losslog_simple_mamba_out6",
    "attention_1e0d_out10":     ROOT / "sim_losslog_attention_1e0d_out10",
    "simple_gml_mamba_out10":   ROOT / "sim_losslog_simple_gml_mamba_out10",
    "attention_1e0d_gml_out10": ROOT / "sim_losslog_attention_1e0d_gml_out10",
}
COLOR = {
    "simple_mamba_out6": "#2a78d6",
    "attention_1e0d_out10": "#1baf7a",
    "simple_gml_mamba_out10": "#2a78d6",
    "attention_1e0d_gml_out10": "#1baf7a",
}


def extract_offline(path):
    vals = []
    with open(path) as f:
        for line in f:
            m = re.search(r"avg_loss:\s*([0-9.eE+\-]+)", line)
            if m:
                vals.append(float(m.group(1)))
    return np.array(vals)


def load_online(run_dir):
    files = sorted((run_dir / "generations" / "train_losses").glob("cycle_*.npy"))
    return [np.load(f) for f in files]


offline = {name: extract_offline(p) for name, p in OFFLINE_LOGS.items()}
online = {name: load_online(d) for name, d in ONLINE_DIRS.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- A: offline, MSE head ---
ax = axes[0, 0]
for name in ["simple_mamba_out6", "attention_1e0d_out10"]:
    v = offline[name]
    ax.plot(np.arange(len(v)) / len(v), v, color=COLOR[name], linewidth=0.8, label=name)
ax.set_yscale("log")
ax.set_xlabel("training progress (fraction of offline run)")
ax.set_ylabel("avg_loss (log scale)")
ax.set_title("A. Offline training — MSE head")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# --- B: online, MSE head ---
ax = axes[0, 1]
for name in ["simple_mamba_out6", "attention_1e0d_out10"]:
    means = [c.mean() for c in online[name]]
    ax.plot(range(len(means)), means, "-o", ms=4, color=COLOR[name], label=name)
ax.set_yscale("log")
ax.set_xlabel("online-learning cycle #")
ax.set_ylabel("mean cycle loss (log scale)")
ax.set_title("B. Online (in-sim) fine-tuning — MSE head")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# --- C: offline, GML head ---
ax = axes[1, 0]
for name in ["simple_gml_mamba_out10", "attention_1e0d_gml_out10"]:
    v = offline[name]
    ax.plot(np.arange(len(v)) / len(v), v, color=COLOR[name], linewidth=0.8, label=name)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlabel("training progress (fraction of offline run)")
ax.set_ylabel("avg_loss (symlog scale)")
ax.set_title("C. Offline training — GML head")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# --- D: online, GML head (cycle 0 warmup excluded) ---
ax = axes[1, 1]
for name in ["simple_gml_mamba_out10", "attention_1e0d_gml_out10"]:
    means = [c.mean() for c in online[name]][1:]
    ax.plot(range(1, len(means) + 1), means, "-o", ms=4, color=COLOR[name], label=name)
ax.set_xlabel("online-learning cycle #")
ax.set_ylabel("mean cycle loss (Gaussian NLL)")
ax.set_title("D. Online (in-sim) fine-tuning — GML head (cycle 0 warmup excluded)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("Offline vs online training loss: simple_mamba vs attention_1e0d", fontsize=13)
fig.tight_layout()
out = ROOT / "offline_vs_online.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")

print("\nsummary:")
for name in OFFLINE_LOGS:
    off = offline[name]
    on = online[name]
    on_means = [c.mean() for c in on]
    print(f"  {name}:")
    print(f"    offline: n={len(off)} first={off[0]:.4g} last={off[-1]:.4g} min={off.min():.4g}")
    print(f"    online:  n_cycles={len(on)} cycle0_mean={on_means[0]:.4g} last_cycle_mean={on_means[-1]:.4g}")
