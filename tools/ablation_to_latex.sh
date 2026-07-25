#!/usr/bin/env bash
# Usage: ./tools/ablation_to_latex.sh [ablation_results/SESSION_DIR]
# If no argument given, uses the most recent session directory.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -ge 1 ]; then
  SESSION_DIR="$1"
else
  SESSION_DIR="ablation_results)all/$(ls ablation_results/ | sort | tail -1)"
fi

echo "Using session: $SESSION_DIR" >&2

OUT_TEX="${SESSION_DIR}/table.tex"

# Collect all JSON result files
mapfile -t JSON_FILES < <(ls "${SESSION_DIR}"/*.json 2>/dev/null | sort)

if [ ${#JSON_FILES[@]} -eq 0 ]; then
  echo "No JSON files found in ${SESSION_DIR}" >&2
  exit 1
fi

# Parse one JSON with python and emit a TSV row: name \t overrides \t metrics...
parse_json() {
  python3 - "$1" <<'PYEOF'
import json, sys, pathlib

f = pathlib.Path(sys.argv[1])
name = f.stem
data = json.loads(f.read_text())

# Take first run (ablation configs have one run each)
run = next(iter(data["runs"].values()))
overrides = run["overrides"]
m = run["metrics"]

# Shorten overrides: strip long Hydra prefixes, keep only changed keys
parts = [p.strip() for p in overrides.split(",")]
short_parts = []
for p in parts:
    k, v = p.split("=", 1)
    k = k.split(".")[-1]   # last segment only
    short_parts.append(f"{k}={v}")
short_overrides = ", ".join(short_parts)

fields = [
    name,
    short_overrides,
    m.get("ate_rmse", "—"),
    m.get("mean_error", "—"),
    m.get("max_error", "—"),
    m.get("endpoint_error", "—"),
    m.get("drift_percent", "—"),
    m.get("drift_m_per_km", "—"),
]
print("\t".join(fields))
PYEOF
}

# ---- Build LaTeX ---------------------------------------------------------
cat > "$OUT_TEX" <<'HEADER'
\begin{table}[ht]
\centering
\caption{Ablation study results}
\label{tab:ablation}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll rr rr rr}
\toprule
\textbf{Config} & \textbf{Overrides} &
\textbf{ATE RMSE} & \textbf{Mean Err} &
\textbf{Max Err} & \textbf{EP Err} &
\textbf{Drift \%} & \textbf{Drift m/km} \\
\midrule
HEADER

for json_file in "${JSON_FILES[@]}"; do
  row=$(parse_json "$json_file")
  IFS=$'\t' read -r name overrides ate_rmse mean_err max_err ep_err drift_pct drift_mkm <<< "$row"

  # Escape underscores for LaTeX
  name_tex="${name//_/\\_}"
  overrides_tex="${overrides//_/\\_}"

  # Round floats to 4 decimal places via python
  fmt() {
    python3 -c "v='$1'; print(f'{float(v):.4f}') if v not in ('—','') else print('—')"
  }

  printf '%s & \\texttt{%s} & %s & %s & %s & %s & %s & %s \\\\\n' \
    "$name_tex" \
    "$overrides_tex" \
    "$(fmt "$ate_rmse")" \
    "$(fmt "$mean_err")" \
    "$(fmt "$max_err")" \
    "$(fmt "$ep_err")" \
    "$(fmt "$drift_pct")" \
    "$(fmt "$drift_mkm")" \
    >> "$OUT_TEX"
done

cat >> "$OUT_TEX" <<'FOOTER'
\bottomrule
\end{tabular}
\end{table}
FOOTER

echo "Written: $OUT_TEX"
cat "$OUT_TEX"
