# Galactus Lab Notebook — GLM-5.2 Throughput Investigation

**Period covered:** Sunday, July 13, 2026 – Tuesday, July 21, 2026
**Investigator:** Paul Martin
**Subject:** Characterizing and improving the inference throughput of GLM-5.2 (753.86 B-parameter mixture-of-experts model) running under llama.cpp/ROCm on the server "Galactus"
**Compiled:** July 24, 2026, from the original conversation exports, benchmark logs, and the automated diagnostic run log. All timestamps are US Eastern Time. Nothing in this notebook is reconstructed from memory; every command and number is taken from the primary records listed in the source table below.

---

## Headline results

| Metric | Start (7/14, 07:00) | End (7/21) | Change |
|---|---|---|---|
| Prefill, pp8192-class workload | 37.63 t/s (pp512, clamped ubatch) | 119.36 ± 0.12 t/s (pp8192, patched llama.cpp, ub 8192) | ≈ 3.2× |
| Decode (short context) | 5.15 t/s (tg128, -t 64) | 6.01 t/s (tg128, fitter config; 5.51–5.54 t/s in the -ot exps=CPU config) | +17% (config-dependent) |
| Platform memory bandwidth | unknown; NUMA suspected | 152 GB/s measured (STREAM, RFO-corrected), saturating at 16 threads | settled |
| Root cause of prefill ceiling | unknown | op_offload scheduler pins all MoE offload work to ROCm0 (731 of 1,186 GPU splits) + per-split expert-ids synchronization | proven, patched |
| Decode ceiling | unknown | DDR4 bandwidth wall: ~13.8 GB/token ÷ 152 GB/s ≈ 91 ms of the ~180 ms token budget | settled |

## System under test

| Component | Detail |
|---|---|
| CPU | AMD EPYC 7713 (Zen 3 "Milan"), 64 cores / 128 threads, single socket, 8 CCDs × 8 cores, 32 MB L3 per CCD. AVX2 + FMA; no AVX-512, no VNNI |
| RAM | 1 TB DDR4 — 8 × 128 GB 3DS RDIMMs, 8 channels, rated 2666 MT/s, configured (overclocked) to 2933 MT/s. NPS1 (single NUMA node) |
| GPUs | 4 × AMD Radeon Pro V620 (gfx1030, RDNA2), 30,704 MiB VRAM each = 122,816 MiB total; PCIe 4.0 ×16 each; VMM: no |
| Storage | Local ZFS on SATA SSD array; dd direct 1.5 GB/s, buffered 2.1 GB/s (verified not a bottleneck) |
| Host | Proxmox (kernel 7.0.14-4-pve at the time of the v3 run); llama.cpp runs in LXC container "openwebui" (Debian 13 "trixie") with full capabilities and no cgroup limits |
| GTT pool | 497.8 GiB (half of RAM, amdgpu default) — relevant to pinned-allocation failures |
| llama.cpp | Baseline: build f84a51940 (9942), backend ROCm+ZenDNN. From 7/14 mid-day: build 657e01125 (10001), rebuilt with -DGGML_ZENDNN=OFF. From 7/21: build 10001 plus the two-edit scheduler patch developed in Session 9 |

## Model under test

GLM-5.2 (Zai Org), Unsloth UD-Q4_K_XL quantization: 11 GGUF shards, 435.19 GiB on disk, 753.86 B parameters, ~4.96 bits per weight. llama.cpp architecture `glm-dsa`. 79 blocks: blk.0–2 dense, blk.3–77 MoE (75 layers), blk.78 MTP/NextN (TENSOR_SKIP — never allocated, so `--spec-type draft-mtp` cannot work). 256 routed experts, 8 active per token, 1 shared expert. MLA attention (kv_lora_rank 512, q_lora_rank 2048), DSA lightning indexer, 1 M context. Expert tensors per MoE layer: ffn_gate_exps 1728 MiB Q4_K, ffn_up_exps 1728 MiB Q4_K, ffn_down_exps 2112 MiB Q5_K (exceptions: blk.8 Q5_K/Q6_K; blk.75–77 Q6_K down). Decode reads ≈ 13.77 GB per token in the hybrid config; ≈ 27 GB per token CPU-only.

## Conventions used in this notebook

Entries are ordered by wall-clock time and grouped into nine work sessions. `### HH:MM — title` marks an entry; commands appear verbatim in fenced blocks; benchmark rows are reproduced as llama-bench printed them. Inline labels mark the epistemic status of statements at the time they were made: **Hypothesis**, **Prediction**, **Confirmed**, **Refuted**, **Dead end**, **Decision**, **Correction**. "STREAM, RFO-corrected" means Scale ×1.5 and Add/Triad ×4/3 to account for read-for-ownership traffic that STREAM does not count (Copy is compiled to non-temporal stores and needs no correction). The v3 diagnostic run (Session 4) executed unattended from 11:08 to 14:25 on 7/14 while the dialogue of Sessions 3 and 5 continued; its phases are presented as a block in wall-clock position, with per-phase times reconstructed from the log's elapsed stamps.

## Primary sources

| File (as kept in `Galactus Testing/`) | Saved (ET) | Content |
|---|---|---|
| Claude-Applying concepts to Galactus with GLM5.2.md | exported 7/16 14:45 | Main conversation, 7/13 22:43 – 7/16 14:41 (Sessions 1–3, 5–8) |
| galactus_triad.txt | 7/14 08:02 | STREAM thread sweep, 16–128 threads, bare-metal host |
| galactus_dmidecode_memory.txt | 7/14 08:05 | DIMM population: 8 × 128 GB, 2666 rated / 2933 configured |
| galactus_kernel_tuning.txt | 7/14 08:11 | THP always + defer+madvise; C-states disabled |
| galactus-diag.sh | 7/14 11:05 | v3 diagnostic script (16 phases) as launched at 11:08 |
| results.txt | 7/14 14:33 | MAIN.log of diag-20260714-110813 (the full v3 run, 175,463 lines) |
| newtest.txt | 7/14 18:13 | Failed -ot placement llama-bench log (comma-parse OOM) |
| Claude State Export.zip | 7/21 06:53 | openwebui system prompt + knowledge files (context setup) |
| llamacpp_patch.md | exported 7/21 07:37 | Patch-development session transcript (Session 9) |

---

## Sessions

1. [01-session-1-background-and-plan](01-session-1-background-and-plan.md) — Background and plan
2. [02-session-2-baselines-and-memory-bandwidth](02-session-2-baselines-and-memory-bandwidth.md) — Baselines, memory bandwidth, first diagnostic script
3. [03-session-3-v2-crash-forensics-and-v3-relaunch](03-session-3-v2-crash-forensics-and-v3-relaunch.md) — v2 crash forensics; the v3 relaunch
4. [04-session-4-v3-diagnostic-run](04-session-4-v3-diagnostic-run.md) — The v3 diagnostic run (machine log)
5. [05-session-5-interpreting-v3-and-ubatch-regime](05-session-5-interpreting-v3-and-ubatch-regime.md) — Interpreting v3; the untested ubatch regime
6. [06-session-6-ubatch-ladder-and-economics](06-session-6-ubatch-ladder-and-economics.md) — The ubatch ladder and the economics
7. [07-session-7-multi-card-placement-and-split-histogram](07-session-7-multi-card-placement-and-split-histogram.md) — Multi-card placement; the split histogram
8. [08-session-8-h2d-verdict-and-patch-interrupted](08-session-8-h2d-verdict-and-patch-interrupted.md) — The H2D verdict and the patch, interrupted
9. [09-session-9-building-and-benchmarking-the-patch](09-session-9-building-and-benchmarking-the-patch.md) — Building and benchmarking the patch
10. [10-dspark-deepseek-v4-flash](10-dspark-deepseek-v4-flash.md) — DSpark on DeepSeek-V4-Flash (Session 10) + interval notes
11. [11-stream-rebaseline-2tb](11-stream-rebaseline-2tb.md) — STREAM re-baseline on the 2 TB population (Aug 15)
12. [12-common-baseline-2tb](12-common-baseline-2tb.md) — The 2 TB common baseline: five models, one build, stock (Aug 15–16)
