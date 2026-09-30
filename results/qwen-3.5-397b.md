# Qwen 3.5 397B.A17B — Galactus results

## 2 TB common baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

The file changed from April: Unsloth UD-Q6_K_XL (337.43 GiB) replaces bartowski Q6_K_L (319.21 GiB). The stock results are pp8192 249.61 ± 19.78, the second-fastest prefill on the machine, behind MiniMax's final measurement, and tg128 9.37 ± 0.16 (t=64), which is −2.0% relative to the April baseline of 9.56 t/s. The model export, bandwidth, and build changed, so the similar result does not isolate any one effect.

Attempts to repeat the April resident-offload configuration at ub 8192 failed: the 5-layer placement failed at weight load and the 4-layer placement failed at context creation, because large-batch prefill and resident experts compete for VRAM. I set further offload testing aside for this workload.

Qwen MTP does not arm on this export (`failed to create MTP context`; the cause is not isolated). See [Entry 12](lab-notebook/12-common-baseline-2tb.md) for the complete conditions. I kept Qwen for further testing.

## Earlier measurements

The sections below record earlier runs under their original conditions.

**Model:** Qwen 3.5 397B.A17B, Q6_K_L — `qwen35moe` arch, 396.35 B params (≈17 B active), 319.21 GiB.
**System:** Galactus (EPYC 7713, DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Builds:** `58190cc84 (8671)` and `0893f50f2 (8746)`. **Dates:** 2026-04-07 and 2026-04-10.
**Raw log:** [benchmark capture](raw-logs/model-benchmark-logs/qwen-3.5-397b-raw.md).

### April summary

All-CPU-experts decode holds around 9.5 to 9.7 t/s and prefill peaks around 87 t/s. Resident-expert offload (5 layers per card) with DDR4-2933 lifts it to about 11.5 to 11.7 t/s decode and about 110 t/s prefill.

### Baseline thread sweep — all experts on CPU (build 8671, 2026-04-07)

Config: `-ngl 99 -nopo 1 -mmp 0 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 -ot "exps=CPU" -p 512 -n 128 -r 3`.

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 36.28 | 9.46 |
| 32 | 62.81 | **9.69** |
| 48 | 76.05 | 9.63 |
| 64 | 84.64 | 9.56 |
| 96 | **87.54** | 9.42 |
| 128 | 85.61 | 4.57 (SMT collapse) |

I selected t=64 as the best prefill/decode balance at the time.

### Resident-expert offload (build 8671)

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| all experts on CPU (t=96 best pp) | 87.54 | 9.42 |
| 4 expert layers per card resident | 102.68 | 10.81 |
| 5 expert layers per card resident | **107.95** | **11.10** |
| 5 layers + transparent hugepages | 107.61 | 11.14 (no change vs no THP) |

Increasing placement from four to five expert layers per card raised decode from 10.81 to 11.10 t/s. Relative to the 9.42 t/s row shown here, the gains were 1.39 and 1.68 t/s respectively. Transparent huge pages made no difference, which is consistent with the GLM-5.2 finding that mmap-backed weights are file pages, not anonymous pages.

### DDR4-2933 re-run, 5-layer resident offload (build 8746, 2026-04-10)

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 51.14 | 11.52 |
| 32 | 82.87 | **11.67** |
| 48 | 97.35 | 11.63 |
| 64 | 107.21 | 11.51 |
| 96 | **110.23** | 11.34 |
| 128 | 104.75 | 5.78 (SMT collapse) |

### Conclusions from the April tests

- The best all-CPU configuration is t=32–64: about 9.7 t/s decode and 63–85 t/s prefill.
- The best overall configuration is 5-layer resident offload at DDR4-2933: about 11.5 to 11.7 t/s decode (t=32) and about 110 t/s prefill (t=96).
- Transparent huge pages have no effect. SMT (t=128) collapses decode.
