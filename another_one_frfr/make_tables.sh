#!/usr/bin/env bash
# Build one flat LaTeX table out of every ablation-session folder in this
# directory, sorted best-first by ATE RMSE (so the top 5 configs across all
# sweeps land at the top), styled to match the thesis's siunitx table look.
#
# Usage:
#   ./make_tables.sh                 # combine every subfolder that has *.json
#   ./make_tables.sh 2026-07-23_*    # combine only the given folder(s)
set -euo pipefail

cd "$(dirname "$0")"

OUT_TEX="table.tex"

if [ $# -ge 1 ]; then
  DIRS=("$@")
else
  mapfile -t DIRS < <(find . -mindepth 1 -maxdepth 1 -type d | sort)
fi

FILTERED=()
for d in "${DIRS[@]}"; do
  if compgen -G "${d%/}"/*.json > /dev/null 2>&1; then
    FILTERED+=("$d")
  fi
done
DIRS=("${FILTERED[@]}")

if [ ${#DIRS[@]} -eq 0 ]; then
  echo "No folders with *.json result files found." >&2
  exit 1
fi

echo "Combining ${#DIRS[@]} folder(s) into ${OUT_TEX}" >&2

python3 - "$OUT_TEX" "${DIRS[@]}" <<'PYEOF'
import json, sys, pathlib, re

out_tex = pathlib.Path(sys.argv[1])
dirs = [pathlib.Path(p) for p in sys.argv[2:]]

METRIC_KEYS = ["ate_rmse", "mean_error", "max_error", "endpoint_error", "drift_percent", "drift_m_per_km"]

BLOCK_LABEL = {
    "simple": "Simple",
    "simple_cls": "SimpleCLS",
    "cls": "CLS",
    "attention": "Attention",
}

MODE_LABEL = {
    "rollout": "rollout",
    "gml_rollout": "GML",
    "rollout_horz": "rollout-horz",
    "standard": "standard",
}

# Hydra defaults a config falls back to when a hyperparameter isn't part of
# its override string (from config/config.yaml + config/models/rollout.yaml
# in the main repo). Used to backfill so every row shows the same knobs.
DEFAULTS = {
    "out_dim": "10",
    "rollout_steps": "16",
    "input_len": "32",
    "n_encoder_layers": "2",
    "n_decoder_layers": "0",
    "mamba_type": "mamba",
}

# Order + display label for the "extra" numeric knobs (anything besides
# block_type/mode, which always get their own slot up front).
EXTRA_KEY_ORDER = [
    ("out_dim", "out"),
    ("rollout_steps", "steps"),
    ("input_len", "in_len"),
    ("n_encoder_layers", "enc"),
    ("n_decoder_layers", "dec"),
    ("mamba_type", "mamba"),
]

def parse_kv(overrides: str) -> dict:
    kv = {}
    for p in overrides.split(","):
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.split(".")[-1]] = v
    return kv

def extra_cells(kv: dict, varying_keys: set) -> list:
    """One cell per varying knob (besides block_type/mode, which get their
    own dedicated columns). A run that didn't override a knob is backfilled
    with its Hydra default, rendered in gray to mark it as implicit."""
    cells = []
    for key, _label in EXTRA_KEY_ORDER:
        if key not in varying_keys:
            continue
        if key in kv:
            cells.append(kv[key])
        else:
            cells.append(("gray", DEFAULTS.get(key, "")))
    return cells

def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&")

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""

raw_rows = []
for d in sorted(dirs):
    for f in sorted(d.glob("*.json")):
        data = json.loads(f.read_text())
        run = next(iter(data["runs"].values()))
        m = run["metrics"]
        try:
            ate = float(m.get("ate_rmse", "nan"))
        except (TypeError, ValueError):
            ate = float("nan")
        raw_rows.append({
            "sweep": d.name,
            "kv": parse_kv(run.get("overrides", "")),
            "metrics": m,
            "ate": ate,
        })

# A knob only earns its own column/slot if it's actually ablated somewhere
# (i.e. takes more than one distinct value across the whole dataset).
value_sets = {}
for r in raw_rows:
    for k, v in r["kv"].items():
        value_sets.setdefault(k, set()).add(v)
varying_keys = {k for k, vs in value_sets.items() if len(vs) > 1}

extra_keys_present = [key for key, _label in EXTRA_KEY_ORDER if key in varying_keys]

rows = []
for r in raw_rows:
    kv = r["kv"]
    block = kv.get("block_type", "")
    mode = kv.get("mode", "")
    rows.append({
        "sweep": r["sweep"],
        "block": BLOCK_LABEL.get(block, block.capitalize() if block else "?"),
        "mode": MODE_LABEL.get(mode, mode),
        "extra": extra_cells(kv, varying_keys),
        "metrics": r["metrics"],
        "ate": r["ate"],
    })

# Best (lowest ATE RMSE) first; broken/NaN runs sink to the bottom.
rows.sort(key=lambda r: (r["ate"] != r["ate"], r["ate"]))

finite_ates = [r["ate"] for r in rows if r["ate"] == r["ate"]]
best_ate = min(finite_ates) if finite_ates else None

EXTRA_LABEL = dict(EXTRA_KEY_ORDER)

lines = []
lines.append("\\begin{table}[H]")
lines.append("\\centering")
lines.append("\\caption{Ablation results across all sweeps, sorted by ATE RMSE.}")
lines.append("\\label{tab:ablation_all}")
lines.append("\\scriptsize")
lines.append("\\setlength{\\tabcolsep}{4pt}")
lines.append("")
lines.append("\\begin{tabular}{")
lines.append("l")  # Block
lines.append("l")  # Mode
for _ in extra_keys_present:
    lines.append("r")
lines.append("S[table-format=1.4]")
lines.append("S[table-format=1.4]")
lines.append("S[table-format=2.4]")
lines.append("S[table-format=2.4]")
lines.append("S[table-format=3.4]")
lines.append("S[table-format=4.4]")
lines.append("}")
lines.append("\\toprule")
lines.append("\\textbf{Block} &")
lines.append("\\textbf{Mode} &")
for key in extra_keys_present:
    lines.append(f"\\textbf{{{tex_escape(EXTRA_LABEL[key])}}} &")
lines.append("\\textbf{ATE} &")
lines.append("\\textbf{Mean} &")
lines.append("\\textbf{Max} &")
lines.append("\\textbf{EP} &")
lines.append("\\textbf{Drift \\%} &")
lines.append("\\textbf{m/km}")
lines.append("\\\\")
lines.append("\\midrule")
lines.append("")

def render_extra(cell, bold: bool) -> str:
    if isinstance(cell, tuple):
        _, value = cell
        inner = f"\\textbf{{{value}}}" if bold else value
        return f"\\textcolor{{gray!55}}{{{inner}}}"
    return f"\\textbf{{{cell}}}" if bold else cell

for r in rows:
    block_tex = tex_escape(r["block"])
    mode_tex = tex_escape(r["mode"])
    metric_cells = [fmt(r["metrics"].get(k, None)) for k in METRIC_KEYS]

    is_best = best_ate is not None and r["ate"] == best_ate
    if is_best:
        lines.append("\\rowcolor{gray!10}")
        block_tex = f"\\textbf{{{block_tex}}}"
        mode_tex = f"\\textbf{{{mode_tex}}}"
        metric_cells = [f"\\bfseries {c}" if c else c for c in metric_cells]

    extra_tex = [render_extra(c, is_best) for c in r["extra"]]

    row_cells = [f"\\texttt{{{block_tex}}}", f"\\texttt{{{mode_tex}}}"] + extra_tex + metric_cells
    lines.append(" & ".join(row_cells) + " \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

out_tex.write_text("\n".join(lines) + "\n")
print(f"Written: {out_tex}  ({len(rows)} rows)", file=sys.stderr)
PYEOF
