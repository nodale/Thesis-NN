"""Compare online (in-sim) learning performance across all four ablation
variants using the champion/challenger ATE gate logged by the online
learner, not the raw training loss (which lives on incompatible scales
between MSE and GML heads).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent

RUNS = {
    "simple_mamba_out6":        (ROOT / "sim_losslog_simple_mamba_out6/logs/online_learner_20260727_181725.log", "mamba", False),
    "simple_gml_mamba_out10":   (ROOT / "sim_losslog_simple_gml_mamba_out10/logs/online_learner_20260727_180945.log", "mamba", True),
    "attention_1e0d_out10":     (ROOT / "sim_losslog_attention_1e0d_out10/logs/online_learner_20260727_190131.log", "attention", False),
    "attention_1e0d_gml_out10": (ROOT / "sim_losslog_attention_1e0d_gml_out10/logs/online_learner_20260727_185400.log", "attention", True),
}

# categorical slots (dataviz palette): blue = mamba, aqua = attention
COLOR = {"mamba": "#2a78d6", "attention": "#1baf7a"}
STYLE = {False: "-", True: "--"}  # solid = plain MSE head, dashed = GML head


def parse(path):
    cur, new, swapped = [], [], []
    for line in open(path):
        m = re.search(r"cur_ATE=([\d.]+)\s+new_ATE=([\d.]+)", line)
        if m:
            cur.append(float(m.group(1)))
            new.append(float(m.group(2)))
        if "new gen wins" in line:
            swapped.append(True)
        elif "current gen retained" in line:
            swapped.append(False)
    n_episodes = sum(1 for l in open(path) if "episode saved" in l)
    return np.array(cur), np.array(new), swapped, n_episodes


data = {name: parse(path) for name, (path, *_rest) in RUNS.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- A: deployed champion ATE per cycle (the actual online performance) ---
ax = axes[0, 0]
for name, (_, arch, gml) in RUNS.items():
    cur, _, _, _ = data[name]
    ax.plot(range(len(cur)), cur, STYLE[gml], color=COLOR[arch], marker="o", ms=4, label=name)
ax.set_yscale("log")
ax.set_xlabel("cycle #")
ax.set_ylabel("deployed champion ATE (log scale)")
ax.set_title("A. Deployed model ATE per online-learning cycle")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# --- B: challenger ATE per cycle (shows stability of the online update) ---
ax = axes[0, 1]
for name, (_, arch, gml) in RUNS.items():
    _, new, _, _ = data[name]
    ax.plot(range(len(new)), new, STYLE[gml], color=COLOR[arch], marker="o", ms=4, label=name)
ax.set_yscale("log")
ax.set_xlabel("cycle #")
ax.set_ylabel("challenger ATE (log scale)")
ax.set_title("B. Challenger ATE per cycle (divergence = unstable online update)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

# --- C: throughput — cycles completed for ~same episode budget ---
ax = axes[1, 0]
names = list(RUNS.keys())
n_cycles = [len(data[n][0]) for n in names]
n_eps = [data[n][3] for n in names]
colors = [COLOR[RUNS[n][1]] for n in names]
x = np.arange(len(names))
ax.bar(x, n_cycles, color=colors)
for i, (c, e) in enumerate(zip(n_cycles, n_eps)):
    ax.text(i, c + 0.3, f"{c} cyc\n({e} eps)", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("online-learning cycles completed")
ax.set_title("C. Throughput: cycles fit into the same ~380-episode session")
ax.grid(True, axis="y", alpha=0.3)

# --- D: final deployed champion ATE ---
ax = axes[1, 1]
final_ate = [data[n][0][-1] if not data[n][2][-1] else data[n][1][-1] for n in names]
ax.bar(x, final_ate, color=colors)
for i, v in enumerate(final_ate):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("final deployed champion ATE")
ax.set_title("D. Final online ATE at end of session (lower = better)")
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Online learning: mamba vs attention_1e0d, MSE vs GML head", fontsize=13)
fig.tight_layout()
out = ROOT / "online_learning_comparison.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")

print("\nsummary:")
for n in names:
    cur, new, swapped, eps = data[n]
    print(f"  {n}: cycles={len(cur)} episodes={eps} swaps={sum(swapped)}/{len(swapped)} "
          f"final_champion_ATE={final_ate[names.index(n)]:.4f}")
