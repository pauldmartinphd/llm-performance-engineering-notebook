# Kimi K2.5/K2.6 — Galactus results

## K2.6 — 2 TB common baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

This is the first fully documented Kimi row, measured on K2.6 (Unsloth UD-Q8_K_XL, 553.71 GiB, native-INT4 MoE weights with BF16 for the rest; the llama-bench `deepseek2 671B BF16` label is cosmetic): pp8192 94.23 ± 4.45 and tg128 5.79 ± 0.01 (t=64). The April tables below are K2.5 under April conditions and cannot serve as a controlled comparison with K2.6. For planning, I treated the models as roughly equivalent, but that was an assumption rather than a controlled comparison.

The fitted decode model gives S ≈ 101 ms (about 15 GB per token of INT4 routed experts at 148 GB/s) plus C ≈ 72 ms. K2.6 decodes faster than GLM-5.2 despite its larger total size; the fitted model attributes this to the smaller GPU-side term. See [Entry 12](lab-notebook/12-common-baseline-2tb.md) for the complete conditions. The choice between K2.6 and K2.7-Code remains open.

## Earlier measurements

The sections below record earlier runs under their original conditions.

**Model:** Kimi K2.5, Unsloth UD-Q4_K_XL — `deepseek2` arch (Kimi K2 is built on the DeepSeek-V3 MLA backbone), 1026.41 B params (≈1.03 T), 579.28 GiB.
**System:** Galactus (EPYC 7713, DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Build:** `58190cc84 (8671)`. **Date:** 2026-04-07.
**Raw log:** [benchmark capture](raw-logs/model-benchmark-logs/kimi-k2.5-raw.md).

### K2.5 summary (April)

This is the largest model tested on Galactus (1.03 T params, 579 GiB, Q4). Decode holds around 6.7 t/s and prefill peaks around 44 t/s. In the paired backend comparison, Vulkan is about 19% slower than ROCm (5.44 versus 6.74 t/s).

### Baseline thread sweep — all experts on CPU

Config: `-ngl 99 -nopo 1 -mmp 0 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 -ot "exps=CPU" -p 512 -n 128 -r 3`.

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 17.74 | 6.58 |
| 32 | 31.36 | **6.76** |
| 48 | 38.52 | 6.72 |
| 64 | 43.37 | 6.68 |
| 96 | **44.41** | 6.56 |
| 128 | 42.82 | 3.25 (SMT collapse) |

Prefill peaks near t=96, and decode is flat at about 6.6 to 6.8 t/s and best at t=32. The full file occupies 579 GiB, while decode reads only the active experts for each token. DRAM bandwidth constrains that streaming cost, as in GLM-5.2.

### Backend comparison (t=32)

| Backend | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| ROCm | 31.31 | **6.74** |
| Vulkan (Vulkan0–3) | 29.81 | 5.44 |

ROCm is about 24% faster than Vulkan on decode in this comparison (6.74 versus 5.44 t/s), and slightly faster on prefill. I used ROCm for this configuration.

### Conclusions from the April K2.5 tests

- The best configuration is t=32 with ROCm: 6.76 t/s decode and 31.36 t/s prefill.
- Prefill scales with threads up to t=96; decode does not.
- Vulkan is a working fallback but slower, and SMT (t=128) collapses decode.
- Two things remain untested: resident-expert offload (there is little VRAM headroom at 579 GiB) and speculative decode.
