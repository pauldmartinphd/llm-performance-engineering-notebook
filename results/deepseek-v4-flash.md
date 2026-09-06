# DeepSeek-V4-Flash-0731 — Galactus results

**Model:** DeepSeek-V4-Flash-0731, Unsloth UD-Q8_K_XL — 162 GB, MXFP4 routed experts (~13 B active).
**System:** Galactus (EPYC 7713, 2 TB DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Drafter:** am17an `DeepseekV4-Flash-20260731-DSpark.gguf` — `dflash` arch, block size 5, ~10.9 GB, in VRAM.
**Date:** 2026-08-08 (Session 10). Full narrative: `../lab-notebook/10-dspark-deepseek-v4-flash.md`.

## Summary

This is the fastest decode Galactus has produced on any model: **14.7 ± 0.2 t/s** with DSpark speculative decode (n=3, p-min off), about +45% over the 10.1 t/s baseline. That is roughly 2× the July figure, entirely from software.

## Placement

`-ngl 99 --cpu-moe` keeps the routed experts (about 135 GiB) in system RAM and everything else — attention, shared experts, and the drafter — in VRAM. The `--cpu-moe` regex catches the routed experts only. The shared experts, which fire on every token, stay on the GPU, which is both correct and the only arithmetic that fits.

## DSpark draft-depth sweep

The configuration was `-ngl 99 --cpu-moe -fa on -t 64 -c 8192 -b 8192 -ub 8192 --spec-type draft-dspark -md <drafter> -ngld 99 --spec-draft-n-max <N>`, on a single technical-prose prompt with greedy decoding. We did not capture acceptance rates; an instrumentation failure caused this (see the notebook). The `--fit` state was not held constant across runs, so this sweep was provisional. The identical-conditions re-stamp on one build is in the section below (2026-08-16), and it supersedes `experiments/session-10-rerun.sh`.

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

The baseline progressed as follows: 7.16 t/s (July, build 9942), then about 10.1 t/s (upstream DSv4 fused kernels plus the 0731 checkpoint), then 14.7 t/s (DSpark).

## Analysis

- The optimum is a fixed draft depth of 2 to 3, with a plateau around 14.5 to 14.8 t/s. The production setting is `--spec-draft-n-max 3` with p-min off, which gives **14.7 ± 0.2 t/s**.
- The depth curve (12.8, 14.5, 14.7, and 11.4 t/s at n = 1, 2, 3, and 5) has the verify-tax shape. On a top-k-of-many MoE, the drafted tokens activate nearly disjoint expert sets, so positions 4 and 5 cost more expert-read bytes than their acceptance yields.
- `p-min` is a monotone tax on technical prose, not a tuning knob, so leave it off. The truncated configurations converge to about 13.3 to 13.4 t/s regardless of n-max.
- The cost decomposition is an estimate: about 70 ms per token for the amortizable GPU path, plus about 29 ms per token for CPU expert streaming (about 4.4 GB per token at 152 GB/s, matching the 13 B active MXFP4 experts). This implies a speculation ceiling of about 35 t/s, and a perfect-acceptance block-5 drafter at n=3 would reach about 22 t/s. The gap from 14.7 to 22 t/s is drafter quality, not configuration.
- This is the first working DSpark data point on a Radeon Pro V620, and the first hybrid CPU-MoE system to reach DSpark's advertised range. The MXFP4 routed experts make the per-token verify tax small, and 8-channel DDR4 absorbs verify batches better than consumer boards do.

## Contrast with GLM-5.2

V4-Flash speculates better than GLM-5.2 MTP (+45% at n=3 against +31% at n=2), because its streaming share of the token budget is smaller (about 29 of about 100 ms, against GLM-5.2's about 91 of about 180 ms). That leaves more amortizable cost for speculation to reduce.

## 2 TB common baseline and DSpark re-stamp (2026-08-15/16, build 3653e6d6d, stock scheduler)

Entry 12 (`lab-notebook/12-common-baseline-2tb.md`) normalized all five models on one build. V4-Flash stock measured pp8192 143.54 ± 1.64, the first recorded V4-Flash prefill at the standing configuration, and tg128 10.34 ± 0.10, which is +44% over July's 7.16 from upstream churn alone. The DSpark re-stamp on the same build, with p-min off, measured n=1 12.9, n=2 14.3, n=3 14.7/13.4 (14.1 ± 0.7, production), n=4 13.6, and n=8 (clamped to 5) 12.5. The July depth curve reproduces end to end (July: 12.8, 14.3–14.5, 14.6–14.8, and 11.4 at n=5). The speculative reps carry about 9% timing noise. This supersedes `experiments/session-10-rerun.sh`, and that open item is closed. The current-build production decode is **14.1 ± 0.7 t/s** with DSpark n=3.
