#!/usr/bin/env bash
# Compile avg_loss values out of training logs (or the newest log matching
# each name pattern) into a short summary: sample count, first/last/min/max
# avg_loss, mean over the whole run, and mean over the final 10% of samples
# (a cheap proxy for "did it converge").
#
# Usage:
#   tools/compile_avg_losses.sh [pattern-or-log-file ...]
#
# Each argument is either an existing file (used directly) or a substring
# pattern (default: simple_mamba_out6 simple_gml_mamba_out10) whose most
# recently modified "*<pattern>*.train.log" under any top-level ablation*/
# another_one* directory is used (trainer.log is Hydra's own job log and
# never contains avg_loss, since trainer.py logs via print(), so it's
# excluded). One summary block is printed per pattern/file.
set -uo pipefail
cd "$(dirname "$0")/.."

SEARCH_DIRS=($(find . -maxdepth 1 -type d \( -iname "ablation*" -o -iname "another_one*" \)))

PATTERNS=("$@")
if [ "${#PATTERNS[@]}" -eq 0 ]; then
  PATTERNS=(simple_mamba_out6 simple_gml_mamba_out10)
fi

summarize() {
  local log_file="$1"
  grep -oE 'avg_loss: *[0-9.eE+-]+' "$log_file" | awk '
    {
      v = $2 + 0
      vals[++n] = v
      if (n == 1 || v < min) min = v
      if (n == 1 || v > max) max = v
      sum += v
    }
    END {
      if (n == 0) {
        print "No avg_loss lines found."
        exit 1
      }
      tail_n = int(n * 0.1); if (tail_n < 1) tail_n = 1
      tail_sum = 0
      for (i = n - tail_n + 1; i <= n; i++) tail_sum += vals[i]

      printf "samples:        %d\n", n
      printf "first avg_loss: %.10f\n", vals[1]
      printf "last avg_loss:  %.10f\n", vals[n]
      printf "min avg_loss:   %.10f\n", min
      printf "max avg_loss:   %.10f\n", max
      printf "mean avg_loss:  %.10f\n", sum / n
      printf "tail mean (last %d%%, n=%d): %.10f\n", 10, tail_n, tail_sum / tail_n
    }
  '
}

first=1
for pattern in "${PATTERNS[@]}"; do
  [ "$first" -eq 1 ] || echo
  first=0
  echo "=== ${pattern} ==="

  if [ -f "$pattern" ]; then
    log_file="$pattern"
  else
    log_file="$(
      find "${SEARCH_DIRS[@]}" -type f -name "*${pattern}*.train.log" \
        -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2-
    )"
    if [ -z "$log_file" ]; then
      echo "No log matching '${pattern}' found under: ${SEARCH_DIRS[*]}"
      continue
    fi
  fi

  echo "Log: ${log_file}"
  summarize "$log_file"
done
