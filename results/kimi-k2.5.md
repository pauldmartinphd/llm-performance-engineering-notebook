# Kimi K2.5/K2.6 — Galactus results

**Model:** Kimi K2.5, Unsloth UD-Q4_K_XL — `deepseek2` arch (Kimi K2 is built on the DeepSeek-V3 MLA backbone), 1026.41 B params (≈1.03 T), 579.28 GiB.
**System:** Galactus (EPYC 7713, DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Build:** `58190cc84 (8671)`. **Date:** 2026-04-07.
**Raw log:** `raw-logs/model-benchmark-logs/kimi-k2.5-raw.md`.

## Result in one line

The largest model tested on Galactus (1.03 T params, 579 GiB, Q4). Decode holds ~6.7 t/s, prefill peaks ~44 t/s; the ROCm backend beats Vulkan by ~18% on decode.

## Baseline thread sweep — all experts on CPU

Config: `-ngl 99 -nopo 1 -mmp 0 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 -ot "exps=CPU" -p 512 -n 128 -r 3`.

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 17.74 | 6.58 |
| 32 | 31.36 | **6.76** |
| 48 | 38.52 | 6.72 |
| 64 | 43.37 | 6.68 |
| 96 | **44.41** | 6.56 |
| 128 | 42.82 | 3.25 (SMT collapse) |

Prefill peaks near t=96; decode is flat ~6.6–6.8 and best at t=32. At 579 GiB the model is bounded by how fast the experts stream from DRAM — the same wall as GLM-5.2, and the reason decode is low.

## Backend comparison (t=32)

| Backend | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| ROCm | 31.31 | **6.74** |
| Vulkan (Vulkan0–3) | 29.81 | 5.44 |

ROCm is ~18% faster on decode and slightly faster on prefill. Use ROCm for this workload.

## Takeaways

- Best config: **t=32, ROCm** — 6.76 t/s decode / 31.36 t/s prefill.
- Prefill scales with threads to t=96; decode does not.
- Vulkan is a working fallback but slower; SMT (t=128) collapses decode.
- Not yet tested: resident-expert offload (little VRAM headroom at 579 GiB) and speculative decode.

## K2.6 — 2 TB common baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

First fully-conditioned Kimi row, measured on **K2.6** (Unsloth UD-Q8_K_XL, 553.71 GiB, native-INT4 MoE weights + BF16 rest; llama-bench's `deepseek2 671B BF16` label is cosmetic): **pp8192 94.23 ± 4.45, tg128 5.79 ± 0.01** (t=64). The April tables above are K2.5 under April conditions and are not class-comparable; treating K2.6 ≈ K2.5 is a recorded assumption. Decode decomposes as S ≈ 101 ms (≈15 GB/token of INT4 routed experts at 148 GB/s) + C ≈ 72 ms — K2.6 out-decodes GLM-5.2 despite the larger total size, on the smaller GPU-side constant. Entry: `lab-notebook/12-common-baseline-2tb.md`. Open: the K2.6 vs K2.7-Code disposition.
