# Galactus — hardware

Galactus is a lab server. One Proxmox host. All inference runs in one LXC container.

| Component | Detail |
|---|---|
| **CPU** | AMD EPYC 7713. 64 cores / 128 threads, Zen 3 (Milan). One NUMA node (NPS1). CPUs 0–63 are physical cores (CCD *k* = CPUs 8k…8k+7). CPUs 64–127 are SMT siblings. ISA: AVX2 and FMA. No AVX-512. |
| **RAM** | **2 TB** = 8 × 256 GB DDR4-2933 3DS RDIMM. One DIMM per channel; all 8 channels populated (8-channel mode). Run at the rated 2933 MT/s. Installed August 2026. The July GLM-5.2 investigation ran on the earlier population: 8 × 128 GB 3DS RDIMM = 1 TB, rated 2666 MT/s, configured 2933 MT/s, Rank 8. [galactus_dmidecode_memory.txt](galactus_dmidecode_memory.txt) is the capture from that 1 TB population. |
| **Measured bandwidth** | **152 GB/s** DRAM (STREAM, after the RFO correction; about 81% of the 187.7 GB/s theoretical limit for 8-channel DDR4-2933). Bandwidth reaches its maximum at 16 threads. This figure is from the 1 TB population; the re-baseline on the 2 TB DIMMs is an open item. See [galactus_triad.txt](galactus_triad.txt). |
| **GPU** | 4 × AMD Radeon Pro V620 (gfx1030), 30.7 GiB each = about **120 GiB** VRAM total. All 12 peer-to-peer directions enabled. |
| **GPU fabric** | Single-pair peer copy 16.2 GB/s. Four concurrent pairs 49.4 GB/s. Host to device 22.3 GB/s on one stream (about PCIe 4.0 ×16), and **65.7 GB/s across 4 concurrent streams**. The prefill patch uses this 4-stream headroom. |
| **Storage** | ZFS on SATA SSD. 1.5 GB/s direct, 2.1 GB/s buffered. It never limited an mmap load. |
| **Software** | llama.cpp and ROCm, in an LXC container on Proxmox. In-container STREAM equals host STREAM to within 1%. The cgroups are unlimited, so the container adds no measurable cost. |
| **Kernel tuning applied** | THP `always` and `defer+madvise`. C-state disabling was considered but not ultimately kept. See [galactus_kernel_tuning.txt](galactus_kernel_tuning.txt). Note: THP showed no measured effect for this workload, because the mmap-backed weights are file pages, not anonymous pages. |

## Cost (July 2026)

The original 1 TB build cost $9,050 in total (RAM $5,600 at $5.47/GB). RAM is now 2 TB (8 × 256 GB DDR4-2933 3DS RDIMM) at $7,800 (≈ $3.81/GB), upgraded August 2026. The four V620s cost $1,600 in total. At July 2026 street prices, DDR4 was about $5.15/GB and DDR5 about $30.94/GB. A DDR5 or Genoa machine would cost $16–21k for an estimated +46% decode. This project evaluated that option and declined it. The DDR5 evaluation is in the lab notebook: [Session 6](../../results/lab-notebook/06-session-6-ubatch-ladder-and-economics.md).

## Models tested

| Model | Quant | Size | Notes |
|---|---|---|---|
| GLM-5.2 | Unsloth UD-Q4_K_XL | 753.86 B / 435 GiB | arch `glm-dsa`, 75 MoE layers, 256 experts / 8 active, MLA attention |
| DeepSeek-V4-Flash-0731 | Unsloth UD-Q8_K_XL | 162 GB | MXFP4 routed experts; DSpark drafter |
| Kimi K2.5, MiniMax M2.7, Qwen 3.5 397B | various | — | see [../../results/](../../results/); they fit in 2 TB |
