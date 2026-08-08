# MiniMax M2.7 — Galactus results

**Model:** MiniMax M2.7, Unsloth UD-Q5_K_M — `minimax-m2` arch, 228.69 B params (≈10 B active), 157.23 GiB.
**System:** Galactus (EPYC 7713, DDR4-2933 8-channel, 4 × Radeon Pro V620). See [platform-and-method](../platform-and-method.md).
**Build:** `0893f50f2 (8746)`. **Date:** 2026-04-17.
**Raw log:** `../../raw-logs/model-benchmark-logs/minimax-m2.7-raw.md`.

## Result in one line

The fastest decode of the pre-speculation models, because it is the smallest (≈10 B active, Q5). All-CPU-experts decode holds ~14.4–14.8 t/s; resident-expert offload lifts it to **17.37 t/s decode / 154.93 t/s prefill**.

## Baseline thread sweep — all experts on CPU

Config: `-ngl 99 -nopo 1 -mmp 0 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 -ot "exps=CPU" -p 512 -n 128 -r 3`.

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 39.48 | 14.39 |
| 32 | 71.82 | **14.76** |
| 48 | 88.67 | 14.60 |
| 64 | 100.82 | 14.42 |
| 96 | **101.71** | 14.05 |
| 128 | 99.96 | 8.96 (SMT collapse) |

Prefill peaks near t=96; decode is flat 14.4–14.8 and best at t=32. Decode falls apart into the SMT range (t=128), the same shape seen on every model here.

## Resident-expert offload

Placing expert layers on the four GPUs (`-ngl 42`, `blk.0–41` distributed across ROCm0–3, remaining experts on CPU):

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| all experts on CPU (best) | 101.71 | 14.76 |
| resident experts on GPU (ngl 42) | **154.93** | **17.37** |

Resident-expert offload is worth +53 t/s prefill and +2.6 t/s decode here — the model is small enough that a large share of experts fits in the ~120 GiB of VRAM.

## Takeaways

- Best all-CPU config: **t=32**, 14.76 t/s decode / 71.82 t/s prefill.
- Best overall config: **resident-expert offload, ngl 42** — 17.37 t/s decode / 154.93 t/s prefill.
- Never schedule into SMT siblings (t=128 halves decode).
