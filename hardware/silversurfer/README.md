# SilverSurfer — platform

Integrated-GPU workstation laptop in the fleet, alongside [Galactus](../galactus/README.md), [Borg](../borg/README.md), and [Magneto](../magneto/README.md). Like Magneto it is a unified-memory machine — one LPDDR5X pool for the CPU and a large integrated GPU — but on the x86 / ROCm side, which makes it the closest laptop analogue to the Galactus software stack. The [methodology](../../results/methodology.md) transfers as written; the memory-bandwidth ceiling (§1) is the first thing to measure. Figures marked unmeasured have no benchmark on record yet.

## Compute

HP ZBook Ultra G1a, 14". **AMD Ryzen AI Max+ PRO 395** ("Strix Halo"): 16 Zen 5 cores / 32 threads.

## Memory

**128 GB LPDDR5X-8000**, unified (shared by the CPU and the integrated GPU).

Theoretical bandwidth: **256 GB/s** — computed from the 256-bit LPDDR5X-8000 memory bus (32 bytes × 8000 MT/s), the defining Strix Halo figure. This is a computed spec number, **not** a STREAM result. Real achievable bandwidth and its efficiency ratio are **unmeasured** here and do not transfer from Galactus (different memory type and controller).

Firmware assigns a fixed slice of the 128 GB to the GPU — selectable in BIOS at 512 MB / 4 GB / 8 GB … up to 96 GB (75% of RAM); the remainder stays CPU-visible. That split is the SilverSurfer analogue of Galactus's "experts in RAM, dense path on GPU" decision — made in firmware here, not in llama.cpp flags — and it sets both the CPU-visible pool and the GPU-resident capacity.

## GPU

**AMD Radeon 8060S** integrated (RDNA 3.5, 40 compute units), ROCm-capable — the same vendor stack as Galactus's V620s, so most of the Galactus tooling (llama.cpp + ROCm, the `GGML_SCHED_DEBUG` split-histogram check) should apply. Confirm the ROCm `gfx` target and the VMM / split-buffer behavior on the unit; Galactus's gfx1030 could not do `-sm row`, and this part may differ.

## Storage

**8 TB** NVMe SSD.

## Status / LLM duty (throughput unmeasured)

- No STREAM, no ratio, no model benchmarks on record.
- Best fleet fit: MoE models up to the GPU-assigned slice (≤ ~96 GB), and the x86-unified-memory counterpoint sitting between Galactus (discrete GPU + DDR4) and Magneto (Apple Silicon).

## Open items

- STREAM baseline with the RFO correction, per methodology §1 — prerequisite for everything.
- Choose the BIOS GPU-memory split and record it before any benchmark; it changes the whole placement picture.
- Confirm the ROCm `gfx` target and whether the `-sm` / VMM limits match Galactus's gfx1030.
- Record the first model + quant + placement, and measure `C` and bytes/token for the two-term model on this part.
