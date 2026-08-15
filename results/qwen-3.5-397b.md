# Qwen 3.5 397B.A17B — Galactus results

**Model:** Qwen 3.5 397B.A17B, Q6_K_L — `qwen35moe` arch, 396.35 B params (≈17 B active), 319.21 GiB.
**System:** Galactus (EPYC 7713, DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Builds:** `58190cc84 (8671)` and `0893f50f2 (8746)`. **Dates:** 2026-04-07 and 2026-04-10.
**Raw log:** `raw-logs/model-benchmark-logs/qwen-3.5-397b-raw.md`.

## Result in one line

All-CPU-experts decode holds ~9.5–9.7 t/s and prefill peaks ~87 t/s; resident-expert offload (5 layers per card) plus DDR4-2933 lifts it to **~11.5–11.7 t/s decode / ~110 t/s prefill**.

## Baseline thread sweep — all experts on CPU (build 8671, 2026-04-07)

Config: `-ngl 99 -nopo 1 -mmp 0 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 -ot "exps=CPU" -p 512 -n 128 -r 3`.

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 36.28 | 9.46 |
| 32 | 62.81 | **9.69** |
| 48 | 76.05 | 9.63 |
| 64 | 84.64 | 9.56 |
| 96 | **87.54** | 9.42 |
| 128 | 85.61 | 4.57 (SMT collapse) |

Recorded conclusion at the time: t=64 is the best prefill/decode balance.

## Resident-expert offload (build 8671)

| Config | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| all experts on CPU (t=96 best pp) | 87.54 | 9.42 |
| 4 expert layers per card resident | 102.68 | 10.81 |
| 5 expert layers per card resident | **107.95** | **11.10** |
| 5 layers + transparent hugepages | 107.61 | 11.14 (no change vs no THP) |

Each extra resident layer buys ~1 t/s. Transparent hugepages made no difference — consistent with the GLM-5.2 finding that mmap-backed weights are file pages, not anonymous pages.

## DDR4-2933 re-run, 5-layer resident offload (build 8746, 2026-04-10)

| Threads | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| 16 | 51.14 | 11.52 |
| 32 | 82.87 | **11.67** |
| 48 | 97.35 | 11.63 |
| 64 | 107.21 | 11.51 |
| 96 | **110.23** | 11.34 |
| 128 | 104.75 | 5.78 (SMT collapse) |

## Takeaways

- Best all-CPU config: **t=32–64**, ~9.7 t/s decode / 63–85 t/s prefill.
- Best overall config: **5-layer resident offload at DDR4-2933** — ~11.5–11.7 t/s decode (t=32) / ~110 t/s prefill (t=96).
- THP: no effect. SMT (t=128): decode collapses.
