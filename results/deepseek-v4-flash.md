# DeepSeek-V4-Flash-0731 — Galactus results

## August baseline and DSpark (2 TB, build 3653e6d6d, stock scheduler)

[Entry 12](lab-notebook/12-common-baseline-2tb.md) normalized all five models on one build. V4-Flash stock measured pp8192 143.54 ± 1.64, the first recorded V4-Flash prefill at the standing configuration, and tg128 10.34 ± 0.10, 44% higher than the July result of 7.16 t/s. This historical comparison includes changes beyond the llama.cpp build.

The DSpark repeat measurement kept p-min off:

| Draft depth | Decode (t/s) |
|---|---|
| n=1 | 12.9 |
| n=2 | 14.3 |
| n=3 | 14.7 / 13.4 (14.1 ± 0.7) |
| n=4 | 13.6 |
| n=8, clamped to 5 | 12.5 |

The curve has the same broad shape as Session 10: 12.8, 14.3–14.5, 14.6–14.8, and 11.4 t/s at n=1, 2, 3, and 5 respectively. The n=2 and n=3 results are equivalent within the observed timing variation.

The repeated speculative runs varied by about 9%. Entry 12 supersedes the [Session 10 rerun protocol](../experiments/session-10-rerun.sh). The selected August configuration produced **14.1 ± 0.7 t/s** with DSpark n=3.

## Earlier measurements

The specifications and configurations below describe the earlier runs. They do not replace the August conditions above.

**Model:** DeepSeek-V4-Flash-0731, Unsloth UD-Q8_K_XL — 162 GB, MXFP4 routed experts (~13 B active).
**System:** Galactus (EPYC 7713, 2 TB DDR4-2933 8-channel, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md).
**Drafter:** am17an `DeepseekV4-Flash-20260731-DSpark.gguf` — `dflash` arch, block size 5, ~10.9 GB, in VRAM.
**Date:** 2026-08-08 (Session 10). Full narrative: [Session 10](lab-notebook/10-dspark-deepseek-v4-flash.md).

### Initial DSpark result

The initial DSpark sweep reached **14.7 ± 0.2 t/s** (n=3, p-min off), about +45% over the session’s 10.1 t/s baseline. That is roughly 2× the July figure, but the checkpoint, build, configuration, and memory population changed between those sessions. The common-build repeat measurement above is the later reference.

### Placement

`-ngl 99 --cpu-moe` keeps the routed experts (about 135 GiB) in system RAM and everything else — attention, shared experts, and the drafter — in VRAM. The `--cpu-moe` regex catches the routed experts only. The shared experts, which fire on every token, stay on the GPU, which is both correct and the only arithmetic that fits.

### DSpark draft-depth sweep

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

### Analysis

- The optimum is a fixed draft depth of 2 to 3, with a plateau around 14.5 to 14.8 t/s. The setting selected in that session was `--spec-draft-n-max 3` with p-min off, which gives **14.7 ± 0.2 t/s**.
- The depth curve (12.8, 14.5, 14.7, and 11.4 t/s at n = 1, 2, 3, and 5) has the verify-tax shape. On a top-k-of-many MoE, the drafted tokens activate nearly disjoint expert sets, so positions 4 and 5 cost more expert-read bytes than their acceptance yields.
- The session selected p-min off. Several confidence-truncated configurations converged around 13.3 to 13.4 t/s, but the table also contains values outside that range. Session 10 records conflicting reports of 13.4 and 14.7 t/s for n=3, p-min 0.3; the later value is shown here. These data do not establish a monotonic effect across every setting.
- The cost decomposition is an estimate: about 70 ms per token for the amortizable GPU path, plus about 29 ms per token for CPU expert streaming (about 4.4 GB per token at 152 GB/s, matching the 13 B active MXFP4 experts). This implies a speculation ceiling of about 35 t/s, and a perfect-acceptance block-5 drafter at n=3 would reach about 22 t/s. Drafter acceptance is one possible explanation for the gap from 14.7 to 22 t/s, but acceptance rates were not captured, so the decomposition does not isolate its cause.
- This establishes a working DSpark configuration on the Radeon Pro V620. The small MXFP4 expert footprint is consistent with a lower verification cost. The repository does not establish priority over other reports or provide a controlled comparison with consumer boards.

### Contrast with GLM-5.2

V4-Flash speculates better than GLM-5.2 MTP (+45% at n=3 against +31% at n=2), because its streaming share of the token budget is smaller (about 29 of about 100 ms, against GLM-5.2's about 91 of about 180 ms). That leaves more amortizable cost for speculation to reduce.
