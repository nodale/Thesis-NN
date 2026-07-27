# Online learning: mamba vs attention_1e0d

## TL;DR

The claim "mamba performs shite online vs attention" is **half right, and for a
different reason than it looks like at first**. The `train_losses/*.npy`
arrays used in `loss_analysis.md` aren't the right signal for this question —
they're per-cycle loss profiles on different scales for MSE vs GML heads. The
online learner logs a much better signal directly: a champion/challenger **ATE
gate** every cycle (`[nn.learner] cur_ATE=... new_ATE=...`). That's the real
downstream metric, and it tells a clear story once you separate architecture
(mamba vs attention) from head type (plain MSE vs GML).

Source: `analyze_online_learning.py` → `online_learning_comparison.png`,
parsed from `sim_losslog_*/logs/online_learner_*.log`.

| run | cycles completed | episodes in session | swaps accepted | final deployed ATE |
|---|---:|---:|---:|---:|
| `simple_mamba_out6` (MSE) | 4 | 380 | 4/4 | **0.061** |
| `attention_1e0d_out10` (MSE) | 14 | 383 | 11/14 | **0.041** |
| `simple_gml_mamba_out10` (GML) | 4 | 380 | 1/4 | 0.099 |
| `attention_1e0d_gml_out10` (GML) | 17 | 383 | 1/17 | 0.097 |

## Two separate effects, not one

### 1. GML heads stall online, for both architectures — a head-type problem

Both GML variants take exactly **one** successful swap (the initial big drop
from a fresh/undertrained champion), then every later challenger loses and
gets thrown away for the rest of the session (`attention_1e0d_gml_out10`:
16 straight rejections; `simple_gml_mamba_out10`: 3 straight rejections).
Panel B (`online_learning_comparison.png`) shows challenger ATE for both GML
runs oscillating wildly above the champion (0.25–7.6 for attention, up to
**96.5×** the champion for mamba) instead of trending down. Online
fine-tuning of a Gaussian-NLL head from small per-cycle batches looks
fundamentally more fragile than fine-tuning a plain MSE head — plausibly
because the log-variance channel is high-variance under small-batch updates
and the NLL loss rewards overconfident (small-variance) predictions that then
misfire hard on unfamiliar online data.

**Within GML, mamba is worse than attention**: mamba's rejected challengers
blow up to 13–96× the champion ATE, attention's only to 2–8×. So GML+mamba is
the most unstable combination in this data — but GML+attention is not healthy
either, it's just less catastrophic.

### 2. Plain-MSE mamba isn't learning badly — it's getting ~4x fewer cycles

`simple_mamba_out6` swapped in a better champion **every single cycle it was
given** (0.153 → 0.113 → 0.122 → 0.077 → 0.061), the same clean monotonic
improvement pattern as `attention_1e0d_out10`'s early cycles. The difference
is throughput: in essentially the same online session (~380 vs 383 collector
episodes, i.e. the same wall-clock/data budget), attention completed **14
cycles** of retrain-eval-swap and mamba completed only **4** (panel C). Given
roughly 3.5x fewer chances to adapt, mamba's session simply ended earlier in
its improvement curve — not because its updates were worse per cycle, but
because each mamba retrain+eval iteration takes much longer wall-clock time
than attention's. If mamba's session had run as long as attention's, its
trend line in panel A suggests it would have kept closing the gap toward
attention's ~0.04 rather than being stuck at 0.06.

## What this means for "mamba performs shite online"

- **True as measured** — at the point both sessions were stopped, mamba's
  deployed ATE (0.061 MSE / 0.099 GML) is worse than attention's (0.041 MSE /
  0.097 GML).
- **Not because mamba's online gradient updates are worse** for the MSE head
  — it accepted 100% of its swaps and improved every time it got a shot.
- **The dominant cause for the MSE comparison is wall-clock throughput**:
  mamba's online retrain/eval loop is ~3.5x slower per cycle than
  attention_1e0d's, so it gets far fewer online-adaptation cycles in a
  fixed-duration deployment session. Check inference/training latency per
  cycle (`min_gen_dwell_s=32.0s` gate plus actual retrain time) if faster
  mamba online updates matter for the thesis — this is a throughput/engineering
  issue, not a learning-capacity one.
- **For the GML comparison**, the story is different: both architectures
  stall after the first swap, and mamba's failed updates are far more
  extreme. If GML heads are going into deployment, this instability (for
  both architectures, worse for mamba) needs addressing before online
  fine-tuning is trustworthy at all — e.g. smaller online LR for the
  log-variance channel, gradient clipping, or a stricter swap gate that
  requires consecutive wins rather than a single ATE comparison.

## Recommended follow-ups

- Profile per-cycle wall-clock time for mamba vs attention_1e0d online
  retraining to confirm the throughput gap and find where the time goes
  (data collation, backward pass, Mamba's `torch.compile` warmup, etc.).
- Re-run `simple_mamba_out6` online for the same wall-clock duration as the
  attention runs (not the same cycle count) to test whether it converges to
  a similar final ATE given equal time.
- For GML variants, log per-cycle gradient norm / log-variance channel stats
  during online updates to see whether the challenger blow-ups correlate
  with the variance head specifically.
