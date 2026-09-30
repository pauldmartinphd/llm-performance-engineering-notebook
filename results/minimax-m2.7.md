# MiniMax M2.7 — Galactus results

## Final measurement — 2 TB common baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

This is the final measurement, on the identical file to April (UD-Q5_K_M, 157.23 GiB): pp8192 418.83 ± 24.11, the machine record and 4.1× the April baseline-class 101.71, and tg128 15.18 ± 0.18 (t=64), which is +2.8% over the April baseline-class 14.76. MiniMax is the only model to beat its April decode baseline, using the same model file. The build, memory population, and benchmark conditions changed, so this is not an isolated test of the build alone.

MiniMax was retired and removed from the machine on August 16. The April resident-offload result of 17.37 t/s was not remeasured before retirement. See [Entry 12](lab-notebook/12-common-baseline-2tb.md) for the complete conditions.

## Earlier measurements

The sections below record earlier runs under their original conditions.

**Model:** MiniMax M2.7, Unsloth UD-Q5_K_M — `minimax-m2` arch, 228.69 B params (≈10 B active), 157.23 GiB.
**System:** Galactus (EPYC 7713, DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Build:** `0893f50f2 (8746)`. **Date:** 2026-04-17.
**Raw log:** [benchmark capture](raw-logs/model-benchmark-logs/minimax-m2.7-raw.md).

### April summary

This is the fastest decode of the pre-speculation models, because it is the smallest (about 10 B active, Q5). All-CPU-experts decode holds around 14.4 to 14.8 t/s, and resident-expert offload lifts it to **17.37 t/s decode / 154.93 t/s prefill**.

### Baseline thread sweep — all experts on CPU

Config: `-ngl 99 -nopo 1 -mmp 0 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 -ot "exps=CPU" -p 512 -n 128 -r 3`.

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 39.48 | 14.39 |
| 32 | 71.82 | **14.76** |
| 48 | 88.67 | 14.60 |
| 64 | 100.82 | 14.42 |
| 96 | **101.71** | 14.05 |
| 128 | 99.96 | 8.96 (SMT collapse) |

Prefill peaks near t=96, and decode is flat at 14.4 to 14.8 t/s and best at t=32. Decode collapses in the SMT range (t=128), the same shape seen on every model here.

### Resident-expert offload

I placed expert layers on the four GPUs (`-ngl 42`, with `blk.0–41` distributed across ROCm0–3 and the remaining experts on CPU):

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| all experts on CPU (best) | 101.71 | 14.76 |
| resident experts on GPU (ngl 42) | **154.93** | **17.37** |

Resident-expert offload is worth +53 t/s prefill and +2.6 t/s decode here, because the model is small enough that a large share of experts fits in the about 120 GiB of VRAM.

### Conclusions from the April tests

- The best all-CPU configuration is t=32: 14.76 t/s decode and 71.82 t/s prefill.
- The best overall configuration is resident-expert offload at ngl 42: 17.37 t/s decode and 154.93 t/s prefill.
- Using t=128 reduced decode sharply in this sweep; the best settings avoided SMT siblings.
