# Speculative decoding on hybrid CPU-MoE systems

This note covers the cross-model mechanics of speculative decoding. It is the companion to the [methodology](../results/methodology.md) for the speculation-specific loop. The per-model numbers are in [glm-5.2](../results/glm-5.2.md) and [deepseek-v4-flash](../results/deepseek-v4-flash.md).

## Why it works on a memory-bound system

The decode cost per token is C + S. C is the amortizable part: the GPU dense path, synchronization, and launch. S is the CPU expert streaming, which is bandwidth-bound. Speculation verifies n+1 tokens in one pass, so it pays C once per cycle instead of once per token. On top-k-of-many MoE routing, however, the drafted tokens activate nearly disjoint expert sets, so the system pays S almost in full for each verified token. This is the verify tax. Three consequences follow:

- The gain scales with C ÷ (C + S), so models with a light active-expert footprint speculate better.
- The optimum draft depth is shallow: 2 to 3 on Galactus, on both models tested.
- Within this approximation, the ceiling is 1 ÷ S; it assumes expert streaming remains the per-token cost that speculation cannot amortize.

| Model (Galactus) | C (ms) | S (ms) | S share | Measured gain | Ceiling (1/S) |
|---|---|---|---|---|---|
| GLM-5.2 (Q4_K experts, 40 B active) | ~90 | ~91 | 50% | +31% (MTP n=2 → 7.1 t/s) | ~11 t/s |
| DS-V4-Flash (MXFP4 experts, 13 B active) | ~70 | ~29 | 30% | +45% (DSpark n=3 → 14.7 t/s) | ~35 t/s |

C and S are fitted estimates, and the gains in this table are the earlier measurements. The August common-build repeats were GLM-5.2 **6.6 ± 0.3 t/s** (+25% ± 6) and DeepSeek-V4-Flash **14.1 ± 0.7 t/s** (+36% ± 7). See [Entry 12](../results/lab-notebook/12-common-baseline-2tb.md). We did not capture acceptance rates in Session 10, so acceptance-based explanations remain hypotheses.

## Methods in llama.cpp (as of August 2026)

- `draft-mtp` uses the model's own NextN head and needs no external file, but it requires architecture support. glm-dsa has it; deepseek4 needs a separate MTP GGUF, and the 0731 checkpoint shipped none.
- `draft-dflash` and `draft-dspark` use an external block-diffusion drafter (`-md`) of architecture `dflash`, with the trained block size in the `dflash.block_size` metadata. DSpark adds a Markov head and a confidence head. `--spec-draft-n-max` clamps to the block size: block−1 for DFlash and block for DSpark.
- The `ngram-*` methods are model-free, and we did not test them here.

## Flags and findings

The flag set is `--spec-type <type> [-md <drafter> -ngld 99] --spec-draft-n-max N`. Sweep N from 1. The peak was at 2 to 3 on Galactus; the "hybrids only gain at n=1" pattern reported in llama.cpp PR #25784 did not hold on Galactus. `--spec-draft-p-min` (confidence truncation) did not establish an improvement on the technical-prose prompt. Several truncated configurations converged around 13.3–13.4 t/s, but the results were not uniformly monotonic and one setting had conflicting reports. An overly pessimistic confidence head was the proposed explanation, not a measured diagnosis. One untested hypothesis is that p-min helps on genuinely low-acceptance domains, such as creative text. Benchmark at `--temp 0`. Acceptance is strongly domain-dependent; the PR #25784 data ranges from about 0.22 for creative text to about 0.77 for math.

## Measurement rules

In the tested builds, llama-cli required terminal-aware capture: `tee` breaks its terminal display, and `--log-file` drops the info-level acceptance and timing lines. Capture with `script -q <file> -c "<command>"`, or benchmark through `llama-server` (the `/completion` JSON `timings`; the logs print `draft acceptance`). `llama-bench` does not support speculation. Record the full flag set with every number. Session 10’s table needed a conditions ledger added after the fact. The proposed [rerun protocol](../experiments/session-10-rerun.sh) is retained as history; Entry 12 superseded it with the common-build measurements. The original p-min conflict remains part of Session 10’s record.
