# SilverSurfer — platform

SilverSurfer is the integrated-GPU workstation laptop in the fleet, alongside [Galactus](../galactus/README.md), [Borg](../borg/README.md), and [Magneto](../magneto/README.md). Like Magneto, it is a unified-memory machine, with one LPDDR5X pool for the CPU and a large integrated GPU, but on the x86 and ROCm side, which makes it the closest laptop analogue to the Galactus software stack. The [methodology](../../results/methodology.md) applies here too, starting with the memory-bandwidth limit (§1). Figures marked unmeasured have no benchmark on record yet.

## Compute

The machine is an HP ZBook Ultra G1a, 14-inch, with an AMD Ryzen AI Max+ PRO 395 ("Strix Halo"): 16 Zen 5 cores and 32 threads.

## Memory

The memory is 128 GB of LPDDR5X-8000, unified and shared by the CPU and the integrated GPU.

The theoretical bandwidth is 256 GB/s, computed from the 256-bit LPDDR5X-8000 memory bus (32 bytes × 8000 MT/s), which gives the Strix Halo specification. This is a calculation; I have not measured STREAM bandwidth. Since the memory type and controller differ from Galactus, its efficiency ratio does not establish the achievable bandwidth here. I have not measured either the bandwidth or the ratio on SilverSurfer.

The firmware assigns a fixed slice of the 128 GB to the GPU, selectable in the BIOS at 512 MB, 4 GB, 8 GB, and up to 96 GB (75% of RAM); the remainder stays visible to the CPU. That split is the SilverSurfer analogue of Galactus's "experts in RAM, dense path on GPU" decision, made in firmware here rather than in llama.cpp flags, and it sets both the CPU-visible pool and the GPU-resident capacity.

## GPU

The integrated GPU is an AMD Radeon 8060S (RDNA 3.5, 40 compute units), and it is ROCm-capable. It uses the same vendor stack as Galactus's V620s, so I expect most of the Galactus tooling (llama.cpp with ROCm, and the `GGML_SCHED_DEBUG` split-histogram check) to apply. The ROCm `gfx` target and VMM and split-buffer behavior remain unconfirmed on this unit. Galactus's gfx1030 could not do `-sm row`; this part may differ.

## Storage

The NVMe SSD is 8 TB.

## Status and LLM duty (throughput unmeasured)

There is no STREAM measurement, efficiency ratio, or model benchmark on record yet. Its best fit in the fleet is MoE models up to the GPU-assigned slice (about 96 GB or less), as an x86 unified-memory comparison between Galactus (discrete GPU with DDR4) and Magneto (Apple Silicon).

## Open items

- Measure the STREAM baseline with the RFO correction, per methodology §1.
- Choose the BIOS GPU-memory split and record it before any benchmark; it changes the whole placement picture.
- Confirm the ROCm `gfx` target and whether the `-sm` and VMM limits match Galactus's gfx1030.
- Record the first model, quant, and placement, and measure `C` and the bytes per token for the two-term model on this part.
