# Magneto — platform

Magneto is my Apple Silicon laptop, alongside [Galactus](../galactus/README.md) and [Borg](../borg/README.md). It is a unified-memory machine: the CPU, the GPU, and the model weights share one LPDDR5 pool, so the discrete-GPU offload split that Galactus uses (routed experts in system RAM, dense path on the GPU) does not exist here. The measurement principles in the [methodology](../../results/methodology.md) transfer, but the backend-specific procedure must be adapted; the memory-bandwidth limit (§1) and decode two-term model remain unmeasured here. Figures marked unmeasured have no benchmark on record yet.

I edit this notebook on Magneto (`magneto-local`).

## Compute

The processor is an Apple M2 Max (2023) in a 14-inch MacBook Pro, with a 12-core CPU (8 performance and 4 efficiency cores). It is Arm64, not x86, so the llama.cpp CPU path and its build flags differ from the EPYC and Threadripper machines.

## Memory

The memory is 64 GB of unified LPDDR5, shared by the CPU and the GPU.

The theoretical bandwidth is about 400 GB/s. This is Apple's published figure, and a 512-bit LPDDR5-6400 bus computes to 409.6 GB/s (64 bytes × 6400 MT/s). It is a spec-sheet number, not a STREAM result. I have no local bandwidth measurement or efficiency ratio. Galactus's 81% does not transfer to Apple's different memory subsystem. For scale only: about 400 GB/s on paper is roughly 2.6× Galactus's about 150 GB/s real DRAM bandwidth, which is the main reason to test MoE decode on this class of part.

## GPU

The integrated Apple GPU has 38 cores, the M2 Max's larger configuration. llama.cpp uses the Metal backend, not ROCm, so the ROCm split-histogram check and the prefill scheduler patch from the Galactus work do not apply as written; the Metal path has its own scheduler and its own controls.

## Storage

The internal Apple NVMe SSD is 1 TB.

## Status and LLM duty

The [September 30 oMLX record](../../results/qwen-3.8-27b.md) contains Qwen3.8-27B throughput measurements. There is still no STREAM measurement or theoretical-peak-versus-measured ratio on record. The best fleet fit is small-to-mid MoE and dense models that fit inside 64 GB of unified memory, and a cross-architecture check of the decode two-term model (`C + bytes ÷ bandwidth`) on a high-bandwidth unified-memory part with a non-ROCm backend.

## Thermal observations

The September investigation captured Heavy thermal pressure with about 21.6 W GPU power, 942 MHz active frequency, and 84.4% active residency. Earlier Nominal samples around 27–29 W are reported in the conversation but lack recovered captures. These observations do not establish a fixed chassis power limit. The oMLX leaderboard hardware label omits chassis; a 38-core/64-GB M2 Max label alone cannot identify a 14-inch MacBook Pro, 16-inch MacBook Pro, or Mac Studio. [Entry 13](../../results/lab-notebook/13-magneto-omlx-qwen38.md) separates the observations from the larger-chassis hypothesis.

## Current configuration

I’m running `Qwen3.8-27B-oQ4e-fp16-mtp` under oMLX 0.6.4 build 2529. Lightning MTP and ANE Prefill are on; DFlash, VLM MTP, TurboQuant KV, and CPU prefill sharing are off. SpecPrefill is saved as on without a draft model recorded, so its execution and benefit remain unverified. The [model note](../../results/qwen-3.8-27b.md#final-settings-for-now-september-30-2026) has the full settings. I’m stopping further measurements for now.
