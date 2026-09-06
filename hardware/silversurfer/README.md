# SilverSurfer — platform

SilverSurfer is the integrated-GPU workstation laptop in the fleet, alongside [Galactus](../galactus/README.md), [Borg](../borg/README.md), and [Magneto](../magneto/README.md). Like Magneto, it is a unified-memory machine, with one LPDDR5X pool for the CPU and a large integrated GPU, but on the x86 and ROCm side, which makes it the closest laptop analogue to the Galactus software stack. The [methodology](../../results/methodology.md) transfers as written; the first thing to measure is the memory-bandwidth limit (§1). Figures marked unmeasured have no benchmark on record yet.

## Compute

The machine is an HP ZBook Ultra G1a, 14-inch, with an AMD Ryzen AI Max+ PRO 395 ("Strix Halo"): 16 Zen 5 cores and 32 threads.

## Memory

The memory is 128 GB of LPDDR5X-8000, unified and shared by the CPU and the integrated GPU.

The theoretical bandwidth is 256 GB/s, computed from the 256-bit LPDDR5X-8000 memory bus (32 bytes × 8000 MT/s), which is the defining Strix Halo figure. It is a computed spec number, not a STREAM result. The real achievable bandwidth and its efficiency ratio are unmeasured here and do not transfer from Galactus, because the memory type and controller are different.

The firmware assigns a fixed slice of the 128 GB to the GPU, selectable in the BIOS at 512 MB, 4 GB, 8 GB, and up to 96 GB (75% of RAM); the remainder stays visible to the CPU. That split is the SilverSurfer analogue of Galactus's "experts in RAM, dense path on GPU" decision, made in firmware here rather than in llama.cpp flags, and it sets both the CPU-visible pool and the GPU-resident capacity.

## GPU

The integrated GPU is an AMD Radeon 8060S (RDNA 3.5, 40 compute units), and it is ROCm-capable. It uses the same vendor stack as Galactus's V620s, so most of the Galactus tooling (llama.cpp with ROCm, and the `GGML_SCHED_DEBUG` split-histogram check) should apply. Confirm the ROCm `gfx` target and the VMM and split-buffer behavior on the unit; Galactus's gfx1030 could not do `-sm row`, and this part may differ.

## Storage

The NVMe SSD is 8 TB.

## Status and LLM duty (throughput unmeasured)

There is no STREAM measurement, no ratio, and no model benchmark on record. The best fleet fit is MoE models up to the GPU-assigned slice (about 96 GB or less), and the x86 unified-memory counterpoint that sits between Galactus (discrete GPU with DDR4) and Magneto (Apple Silicon).

## Open items

- Run the STREAM baseline with the RFO correction, per methodology §1; it is the prerequisite for everything.
- Choose the BIOS GPU-memory split and record it before any benchmark; it changes the whole placement picture.
- Confirm the ROCm `gfx` target and whether the `-sm` and VMM limits match Galactus's gfx1030.
- Record the first model, quant, and placement, and measure `C` and the bytes per token for the two-term model on this part.
