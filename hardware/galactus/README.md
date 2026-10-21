# Galactus — hardware

Galactus is my lab server. It runs one Proxmox host, with all inference in a single LXC container.

| Component | Detail |
|---|---|
| **CPU** | AMD EPYC 7713. 64 cores / 128 threads, Zen 3 (Milan). One NUMA node (NPS1). CPUs 0–63 are physical cores (CCD *k* = CPUs 8k…8k+7). CPUs 64–127 are SMT siblings. ISA: AVX2 and FMA. No AVX-512. |
| **RAM** | **2 TB** = 8 × 256 GB DDR4-2933 3DS RDIMM. One DIMM per channel, and all 8 channels are populated (8-channel mode). It runs at the rated 2933 MT/s. I installed it in August 2026; one DIMM of this population failed, and I replaced it on 2026-08-15, so the STREAM re-baseline below post-dates the replacement. The July GLM-5.2 investigation ran on the earlier population: 8 × 128 GB 3DS RDIMM = 1 TB, rated 2666 MT/s, configured at 2933 MT/s, Rank 8. [galactus_dmidecode_memory.txt](galactus_dmidecode_memory.txt) is the capture from that 1 TB population. |
| **Measured bandwidth** | **152 GB/s** DRAM (STREAM, after the RFO correction; about 81% of the 187.7 GB/s theoretical limit for 8-channel DDR4-2933). Bandwidth reaches its maximum at 16 threads. I re-baselined on the 2 TB population on 2026-08-15, after replacing the failing DIMM the same day: **148–151 GB/s** corrected, within about 2% of the 1 TB figure, so the platform number is unchanged in practice (about 150 GB/s). Captures: [galactus_triad.txt](galactus_triad.txt) (1 TB) and [galactus_triad_2tb.txt](galactus_triad_2tb.txt) (2 TB). |
| **GPU** | 4 × AMD Radeon Pro V620 (gfx1030), 30.7 GiB each = about **120 GiB** VRAM total. All 12 peer-to-peer directions are enabled. |
| **GPU fabric** | Single-pair peer copy 16.2 GB/s. Four concurrent pairs 49.4 GB/s. Host to device 22.3 GB/s on one stream (about PCIe 4.0 ×16), and **65.7 GB/s across 4 concurrent streams**. The prefill patch uses this 4-stream headroom. |
| **Storage** | ZFS on SATA SSD. 1.5 GB/s direct, 2.1 GB/s buffered. It never limited an mmap load. |
| **Software** | llama.cpp and ROCm, in an LXC container on Proxmox. In-container STREAM equals host STREAM to within 1%. The cgroups are unlimited, so the container adds no measurable cost. |
| **Kernel tuning applied** | THP `always` and `defer+madvise`. I considered disabling C-states but did not keep it. See [galactus_kernel_tuning.txt](galactus_kernel_tuning.txt). THP showed no measured effect for this workload, because the mmap-backed weights are file pages, not anonymous pages. |

## Thermal testing and fan control

The fan tests recorded in [Program Fan Curves](https://chatgpt.com/c/6abe841f-bd64-83ea-af62-6c838ac0b595) show that airflow through the GPU duct also has a substantial effect on the adjacent memory. All thermal measurements in this section were taken on 2026-10-01. These notes were added on 2026-10-21.

### Fan mapping and overheating incident

| Header | Cooling assignment |
|---|---|
| FAN1 | CPU |
| FAN2 | Two high-pressure 80 mm server fans feeding the four-GPU duct |
| FAN3 | Case intake/exhaust |
| FAN4 / FAN5 | Bottom chamber |

Reducing chassis airflow allowed the DIMMs beside the GPU stack to exceed approximately 84–85 °C, followed by ECC/RAS trouble. The available record does not identify the error types or counts, so this records the observed sequence rather than a diagnosis of permanent DIMM damage. The hotspot was localized to the bare RDIMMs adjacent to the GPUs, particularly channels B/C/D. This incident is not established as the cause of the August DIMM replacement noted above.

### Measured response to FAN2

| FAN2 command | Approximate RPM | Hottest DIMM (DDR4C) during inference |
|---|---:|---|
| 50% | 3,900 | Approximately 74 °C under heavy inference |
| 70% | 5,100 | Generally low 70s, roughly 70–74 °C; workload-dependent and not dramatically cooler than 50% |
| 100% | 6,700 | Approximately 63 °C after about 10 minutes |

At the 13:37 observation, about ten minutes after switching FAN2 to 100%, the channel temperatures were A: 53 °C, B: 61 °C, C: 63 °C, D: 62 °C, E: 58 °C, F: 59 °C, G: 59 °C, and H: 52 °C. The surrounding zone cooled as well: card-side temperature fell from about 41 to 35 °C, motherboard temperature from about 38 to 35 °C, and CPU temperature from about 49–52 to 47 °C.

The 100% observation was about 11 °C below the approximately 74 °C heavy-inference reading. Assuming the inference load remained broadly comparable, this shows that FAN2 has substantial cooling authority over the memory hotspot as well as the GPUs. The larger improvement from 70% to 100% suggests a nonlinear response, consistent with high-pressure fans driving air through a restrictive duct, but these observations do not isolate that mechanism or establish a controlled steady-state fan sweep. In particular, the 63 °C reading is a ten-minute observation, not a demonstrated equilibrium temperature.

The practical conclusion is that the existing cooling can address this localized hotspot when FAN2 supplies sufficient airflow. An additional DIMM duct or baffle did not appear necessary from these tests.

### Proposed control strategy

Use low FAN2 duty during long idle periods, then increase airflow as DDR4C warms, with a steep ramp around 68–74 °C and full duty by the mid-70s. FAN3, FAN4, and FAN5 can remain at 100%. The first-pass FAN2 curve from the discussion is:

| DDR4C temperature | FAN2 command |
|---|---:|
| ≤50 °C | 30% |
| 55 °C | 35% |
| 60 °C | 45% |
| 65 °C | 55% |
| 68 °C | 70% |
| 70 °C | 80% |
| 72 °C | 90% |
| ≥74 °C | 100% |

This curve is a proposal to test, not a verified deployed configuration or a manufacturer temperature limit. The low-duty idle settings were not validated by the three inference measurements above. The intended behavior is quiet operation with cool memory and an aggressive increase in airflow before the DIMMs approach the temperatures associated with the ECC/RAS incident.

## Cost (July 2026)

The original 1 TB build cost $9,050 in total (RAM $5,600 at $5.47/GB). The RAM is now 2 TB (8 × 256 GB DDR4-2933 3DS RDIMM) at $7,800 (about $3.81/GB), upgraded in August 2026. The four V620s cost $1,600 in total. At July 2026 street prices, DDR4 was about $5.15/GB and DDR5 about $30.94/GB. A DDR5 or Genoa machine would cost $16–21k for an estimated +46% decode. I evaluated that option and decided against it. The DDR5 evaluation is in the lab notebook: [Session 6](../../results/lab-notebook/06-session-6-ubatch-ladder-and-economics.md).

## Models tested

| Model | Quant | Size | Notes |
|---|---|---|---|
| GLM-5.2 | Unsloth UD-Q4_K_XL | 753.86 B / 435 GiB | arch `glm-dsa`, 75 MoE layers, 256 experts / 8 active, MLA attention |
| DeepSeek-V4-Flash-0731 | Unsloth UD-Q8_K_XL | 162 GB | MXFP4 routed experts; DSpark drafter |
| Kimi K2.5/K2.6, MiniMax M2.7, Qwen 3.5 397B | various | — | see [../../results/](../../results/); they fit in 2 TB |
