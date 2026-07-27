import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
RUNS = {
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


def load_cycles(run_dir):
    files = sorted((run_dir / "generations" / "train_losses").glob("cycle_*.npy"))
    return [np.load(f) for f in files]


data = {name: load_cycles(d) for name, d in RUNS.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- A: MSE-head runs, per-cycle mean loss, log-y ---
ax = axes[0, 0]
for name in ["simple_mamba_out6", "attention_1e0d_out10"]:
    means = [c.mean() for c in data[name]]
    ax.plot(range(len(means)), means, "-o", ms=4, color=COLOR[name], label=name)
ax.set_yscale("log")
ax.set_xlabel("cycle #")
ax.set_ylabel("mean MSE loss (log scale)")
ax.set_title("A. Plain-MSE head: per-cycle training loss")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# --- B: GML-head runs, per-cycle mean loss (cycle 0 excluded — MSE warmup) ---
ax = axes[0, 1]
for name in ["simple_gml_mamba_out10", "attention_1e0d_gml_out10"]:
    means = [c.mean() for c in data[name]][1:]
    ax.plot(range(1, len(means) + 1), means, "-o", ms=4, color=COLOR[name], label=name)
ax.set_xlabel("cycle #")
ax.set_ylabel("mean Gaussian NLL (more negative = better fit)")
ax.set_title("B. GML head: per-cycle training loss (cycle 0 warmup excluded)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# --- C: cycle-0 apples-to-apples (all 4 runs, all still MSE-scale here) ---
ax = axes[1, 0]
names = list(RUNS.keys())
c0_means = [data[n][0].mean() for n in names]
colors = [COLOR[n] for n in names]
x = np.arange(len(names))
ax.bar(x, c0_means, color=colors)
for i, v in enumerate(c0_means):
    ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("cycle-0 mean loss (log scale)")
ax.set_title("C. Cycle 0 — all four runs, still MSE-scale, before any online update")
ax.grid(True, axis="y", alpha=0.3)

# --- D: within-cycle spread (boxplot) at the last cycle each MSE run reached ---
ax = axes[1, 1]
box_data, box_labels, box_colors = [], [], []
for name in ["simple_mamba_out6", "attention_1e0d_out10"]:
    box_data.append(data[name][-1])
    box_labels.append(f"{name}\n(cycle {len(data[name]) - 1})")
    box_colors.append(COLOR[name])
bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_yscale("log")
ax.set_ylabel("MSE loss within final cycle (log scale)")
ax.set_title("D. Within-cycle spread at each run's last cycle")
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Training loss comparison: simple_mamba vs attention_1e0d (MSE and GML heads)", fontsize=13)
fig.tight_layout()
out = ROOT / "loss_comparison.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")

print("\nsummary (mean loss per cycle):")
for name in names:
    means = [round(c.mean(), 6) for c in data[name]]
    print(f"  {name}: {means}")
