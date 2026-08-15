# Speculative decoding on hybrid CPU-MoE rigs

Cross-model mechanics — the companion to [methodology](../results/methodology.md) for the speculation-specific loop. Per-model numbers live in [glm-5.2](../results/glm-5.2.md) and [deepseek-v4-flash](../results/deepseek-v4-flash.md).

## Why it works on a memory-bound rig

Decode cost per token = **C** (amortizable: GPU dense path, sync, launch) + **S** (CPU expert streaming, bandwidth-bound). Speculation verifies n+1 tokens in one pass: C is paid once per cycle instead of once per token, but on top-k-of-many MoE routing the drafted tokens activate nearly disjoint expert sets, so S is paid almost in full per verified token — the **verify tax**. Consequences:

- Gains scale with C ÷ (C + S) — models with light active-expert footprints speculate better.
- The optimum draft depth is shallow (2–3 on Galactus, on both models tested).
- The hard ceiling is 1 ÷ S regardless of drafter quality.

| Model (Galactus) | C (ms) | S (ms) | S share | Measured gain | Ceiling (1/S) |
|---|---|---|---|---|---|
| GLM-5.2 (Q4_K experts, 40 B active) | ~90 | ~91 | 50% | +31% (MTP n=2 → 7.1 t/s) | ~11 t/s |
| DS-V4-Flash (MXFP4 experts, 13 B active) | ~70 | ~29 | 30% | +45% (DSpark n=3 → 14.7 t/s) | ~35 t/s |

*C/S are fitted estimates; acceptance rates were not captured in Session 10 (instrumentation debt).*

## Methods in llama.cpp (as of Aug 2026)

- **draft-mtp** — the model's own NextN head; no external file; needs arch support (glm-dsa: yes; deepseek4: via a separate MTP GGUF — the 0731 checkpoint shipped none).
- **draft-dflash / draft-dspark** — external block-diffusion drafter (`-md`), arch `dflash`, trained block size in `dflash.block_size` metadata; DSpark adds a Markov head + confidence head. `--spec-draft-n-max` clamps to the block (DFlash: block−1; DSpark: block).
- **ngram-\*** — model-free; untested here.

## Flags and findings

`--spec-type <type> [-md <drafter> -ngld 99] --spec-draft-n-max N`. Sweep N from 1; expect the peak at 2–3 on this hardware class — the "hybrids only gain at n=1" pattern reported in llama.cpp PR #25784 did not hold on Galactus. `--spec-draft-p-min` (confidence truncation) measured as a **monotone tax** on technical prose with the V4-Flash drafter: truncated configs converge to the same throughput regardless of n-max once the head is in charge (head miscalibrated pessimistic). Untested hypothesis: p-min earns its keep on genuinely low-acceptance domains (creative text). Benchmark at `--temp 0`; acceptance is strongly domain-dependent (PR #25784 data: ~0.22 creative → ~0.77 math).

## Measurement rules (learned the hard way)

llama-cli is not a measurement tool: `tee` breaks its TTY UI and `--log-file` drops the info-level acceptance/timing lines in practice. Capture with `script -q <file> -c "<command>"`, or benchmark through llama-server (`/completion` JSON `timings`; logs print `draft acceptance`). llama-bench does not support speculation. Record the full flag set with every number — Session 10's table needed a conditions ledger retroactively; the identical-conditions template is [scripts/session-10-rerun.sh](../experiments/session-10-rerun.sh).
