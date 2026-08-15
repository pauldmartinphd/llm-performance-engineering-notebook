# DeepSeek-V4-Flash-0731 — Galactus results

**Model:** DeepSeek-V4-Flash-0731, Unsloth UD-Q8_K_XL — 162 GB, MXFP4 routed experts (~13 B active).
**System:** Galactus (EPYC 7713, 2 TB DDR4-2933 8-channel, 4 × Radeon Pro V620). See [platform-and-method](../platform-and-method.md).
**Drafter:** am17an `DeepseekV4-Flash-20260731-DSpark.gguf` — `dflash` arch, block size 5, ~10.9 GB, in VRAM.
**Date:** 2026-08-08 (Session 10). Full narrative: `../lab-notebook/10-dspark-deepseek-v4-flash.md`.

## Result in one line

The fastest decode Galactus has produced on any model: **14.7 ± 0.2 t/s** with DSpark speculative decode (n=3, p-min off), ≈ +45% over the 10.1 t/s baseline. Roughly 2× since July, entirely from software.

## Placement

`-ngl 99 --cpu-moe` keeps the routed experts (~135 GiB) in system RAM and everything else — attention, shared experts, the drafter — in VRAM. `--cpu-moe`'s regex catches routed experts only; shared experts (which fire every token) stay on GPU, which is both correct and the only arithmetic that fits.

## DSpark draft-depth sweep

Config: `-ngl 99 --cpu-moe -fa on -t 64 -c 8192 -b 8192 -ub 8192 --spec-type draft-dspark -md <drafter> -ngld 99 --spec-draft-n-max <N>`. Single technical-prose prompt, greedy. Acceptance rates were **not captured** (instrumentation failure — see the notebook). The `--fit` state was not held constant across runs; the sweep is provisional pending the identical-conditions rerun (`scripts/session-10-rerun.sh`).

| Configuration | tg (t/s) |
|---|---|
| baseline (no speculation) | 9.8 / 10.3 |
| n=1 | 12.8 |
| n=2 | 14.3 (--fit off) / 14.5 |
| n=2, p-min 0.3 | 14.4 |
| n=2, p-min 0.5 | 13.9 |
| **n=3** | **14.6 / 14.8** |
| n=3, p-min 0.3 | 14.7 |
| n=3, p-min 0.8 | 12.6 |
| n=5 | 11.4 |
| n=5, p-min 0.5 | 13.3 |

Baseline progression: **7.16** (July, build 9942) → **~10.1** (upstream DSv4 fused kernels + the 0731 checkpoint) → **14.7** (DSpark).

## Analysis

- **Optimum: fixed draft depth 2–3, plateau ~14.5–14.8.** Production setting `--spec-draft-n-max 3`, p-min off: **14.7 ± 0.2 t/s**.
- **Depth curve** (12.8 / 14.5 / 14.7 / 11.4 at n = 1/2/3/5) is the verify-tax shape: on a top-k-of-many MoE, drafted tokens activate nearly disjoint expert sets, so positions 4–5 cost more expert-read bytes than their acceptance yields.
- **`p-min` is a monotone tax on technical prose**, not a tuning knob — leave it off. Truncated configs converge to ~13.3–13.4 regardless of n-max.
- **Cost decomposition (estimate):** ~70 ms/token amortizable GPU path + ~29 ms/token CPU expert streaming (≈ 4.4 GB/token at 152 GB/s, matching 13 B active MXFP4). Implied speculation ceiling ≈ 35 t/s; perfect-acceptance block-5 at n=3 ≈ 22 t/s. The 14.7 → 22 gap is drafter quality, not configuration.
- First working Radeon Pro V620 DSpark data point, and the first hybrid CPU-MoE rig to land DSpark's advertised range — MXFP4 routed experts make the per-token verify tax small, and 8-channel DDR4 absorbs verify batches better than consumer boards.

## Contrast with GLM-5.2

V4-Flash speculates better (+45% at n=3) than GLM-5.2 MTP (+31% at n=2) because its streaming share of the token budget is smaller (~29 of ~100 ms vs GLM's ~91 of ~180 ms), leaving more amortizable cost for speculation to attack.
