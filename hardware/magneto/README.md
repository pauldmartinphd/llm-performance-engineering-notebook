# Magneto — platform

Apple Silicon laptop in the fleet, alongside [Galactus](../galactus/README.md) and [Borg](../borg/README.md). A unified-memory machine: CPU, GPU, and model weights share one LPDDR5 pool, so the discrete-GPU offload split Galactus uses (routed experts in system RAM, dense path on GPU) does not exist here. The [methodology](../../results/methodology.md) transfers as written; the memory-bandwidth ceiling (§1) and the decode two-term model are the parts to re-measure first. Figures marked unmeasured have no benchmark on record yet.

This is the machine this notebook is currently edited from (`magneto-local`).

## Compute

Apple M2 Max (2023), 14" MacBook Pro. 12-core CPU: 8 performance + 4 efficiency cores. AVX/NEON note: Arm64, not x86 — the llama.cpp CPU path and its build flags differ from the EPYC/Threadripper machines.

## Memory

**64 GB unified LPDDR5**, shared by CPU and GPU.

Theoretical bandwidth: **~400 GB/s** — Apple's published figure; a 512-bit LPDDR5-6400 bus computes to 409.6 GB/s (64 bytes × 6400 MT/s). This is a spec-sheet number, **not** a STREAM result. Real achievable bandwidth and its efficiency ratio are **unmeasured** here, and that ratio does **not** transfer from Galactus's 81% — Apple's memory subsystem is different and must be measured on its own. For scale only: ~400 GB/s on paper is roughly 2.6× Galactus's ~150 GB/s *real* DRAM bandwidth, which is the headline reason to test MoE decode on this class of part.

## GPU

Integrated Apple GPU, **38 cores** (the M2 Max's larger GPU configuration). llama.cpp uses the Metal backend, not ROCm — so the ROCm split-histogram check and the prefill scheduler patch from the Galactus work do not apply as written; the Metal path has its own scheduler and its own knobs.

## Storage

Internal Apple NVMe SSD, **1 TB.**

## Status / LLM duty (throughput unmeasured)

- No STREAM, no theoretical-peak-vs-measured ratio, no model benchmarks on record.
- Best fleet fit: small-to-mid MoE and dense models that fit inside 64 GB unified, and a cross-architecture check of the decode two-term model (`C + bytes ÷ bandwidth`) on a high-bandwidth unified-memory part with a non-ROCm backend.

## Open items

- STREAM baseline with the RFO correction, per methodology §1 — the prerequisite for every decode budget here. Build a CPU STREAM for Arm64; note whether it emits non-temporal stores (the Copy-correction check).- Pick a model that fits 64 GB, record quant + placement, and measure `C` and bytes/token so the two-term model has Apple-Silicon constants.
