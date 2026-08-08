# Galactus × GLM-5.2 — Results and Takeaways

**Investigation period:** July 13–21, 2026
**System:** AMD EPYC 7713 (64C/128T Zen 3), 1 TB DDR4-2933 (8 channels), 4 × AMD Radeon Pro V620 (gfx1030, 30.7 GiB each), llama.cpp/ROCm in LXC on Proxmox
**Model:** GLM-5.2, Unsloth UD-Q4_K_XL — 753.86 B parameters, 435.19 GiB, arch `glm-dsa`, 75 MoE layers, 256 experts / 8 active, MLA attention

This document collects the measured results and settled conclusions of the investigation. The companion lab notebook records the full chronology; the CSV extracts under `03-data-extracts/` carry every benchmark row in machine-readable form. Statements here are limited to what was measured or verified in source; where a number is a model-based estimate rather than a measurement, it is labeled as such.

---

## 1. Executive summary

The investigation began with GLM-5.2 running at 37.63 t/s prefill and 5.15 t/s decode and ended, eight days later, at 119.36 t/s prefill (≈ 3.2×) and ~5.5–6.0 t/s decode, with both ceilings explained by measurement rather than conjecture.

Decode is bounded by DRAM bandwidth and is effectively finished on this hardware. The platform delivers 152 GB/s (STREAM, RFO-corrected), decode reads ≈ 13.77 GB per token in the hybrid configuration, and a two-term model — a ~90 ms GPU-side constant plus bytes ÷ 152 GB/s — predicted the measured throughput of three independent configurations to within a few percent (predicted 5.5 / 6.2 / 3.9 t/s vs measured 5.53 / 6.01 / 3.87). No CPU-side tuning (threads beyond 24–32, polling, strict affinity, CCD placement, repacked kernels, transparent hugepages, pinned host buffers) moved it, because none of those change bytes or bandwidth.

Prefill was bounded by two artifacts of llama.cpp's scheduler, both fixed. First, an inherited `-p 512` in the benchmark scripts silently clamped `n_ubatch` to 512; raising the true microbatch to 8192 took prefill from 40.11 to 104.97 t/s with no code change. Second, the MoE offload path concentrated 731 of 1,186 GPU graph splits (62%) on ROCm0 and serialized each layer's expert-weight copy behind a per-split synchronize used to read the expert-selection ids. A two-edit patch (layer-keyed distribution of offload targets across the four cards, plus a bypass of the ids read at prefill-sized batches) raised pp8192 to 119.36 ± 0.12 t/s (+13.7%). Distribution alone was worth nothing — a pre-registered null that the measurements confirmed — because the copies were already asynchronous; the ids synchronization was the true serializer.

The remaining prefill ceiling is arithmetic, not scheduling: attention is quadratic in context (≈ 26.3 s of a pp32768 pass, 72% of the total), so the patched gains decay with depth — 119.36 / 86.11 / 55.76 t/s at 8k / 16k / 32k. The remaining decode ceiling is DDR4 itself; a DDR5/Genoa platform would buy roughly +46% decode for roughly $16–21k, which was evaluated and declined.

## 2. Platform characterization (settled facts)

**Memory bandwidth.** STREAM with the RFO correction (Scale ×1.5, Add/Triad ×4/3; Copy compiles to non-temporal stores and needs none — proof: uncorrected-Copy-plus-RFO would exceed the 204.8 GB/s theoretical ceiling) converges across all four kernels on ~150–152 GB/s, 74% of theoretical for 8-channel DDR4-2933. Bandwidth saturates at 16 threads and declines slightly beyond; per-thread placement matters at low counts (packed binding at 16 threads collapses to ~90 GB/s Copy / ~68 GB/s Triad — each Milan CCD reaches the IO die over one GMI2 link, so 3–4+ CCDs must be active). The LXC container is free: in-container STREAM equals host STREAM within 1%, and cgroups are wide open (cpu.max=max, cpuset=0-127, memory.max=max, nr_throttled=0).

**Topology.** Single NUMA node (NPS1) — the early NUMA hypothesis was refuted by `numactl --hardware`. CPUs 0–63 are physical cores (CCD k = CPUs 8k–8k+7); 64–127 are SMT siblings. Zen 3 ISA: AVX2 + FMA only.

**Memory population.** 8 × 128 GB 3DS RDIMMs, one per channel, rated 2666 MT/s, configured 2933 MT/s.

**GPU fabric.** All 12 peer-to-peer directions enabled. Measured: single-pair peer copy 16.2 GB/s; four concurrent pairs 49.4 GB/s aggregate (~76% scaling); host→device 22.3 GB/s single stream (≈ PCIe 4.0 ×16 line rate, back-calculated from the ub 512 ladder point and confirmed directly), 65.7 GB/s with four concurrent streams — the measurement that proved multi-card H2D distribution had headroom worth patching for.

**Storage.** ZFS on SATA SSDs, 1.5 GB/s direct / 2.1 GB/s buffered — never a bottleneck for mmap loads.

## 3. Decode results

| Configuration | Best decode | Conditions |
|---|---|---|
| Baseline (7/14 morning, build 9942 + ZenDNN) | 5.15 t/s tg128 | -ngl 99 -ot exps=CPU, -t 64, ub clamped |
| Hybrid -ot exps=CPU (v3 run, build 10001, no ZenDNN) | 5.53–5.54 t/s tg64 | -t 24–32; identical across mmap / --no-host / pinned |
| Fitter (no manual placement; ~106 GiB VRAM filled) | **6.01 t/s tg128** | 15–16 expert layers resident on GPU; splits 155 → 137; pp512 40.11 |
| CPU-only denominator | 3.87 t/s tg64 | -ngl 0, -t 32; graph splits = 1 |
| Patched build (7/21) | 5.51 t/s tg64 | unchanged vs 5.51 pre-patch — the patch targets prefill only |

The thread sweep peaked at t=24–32 and was catastrophic into SMT territory (t=96: 2.76 t/s; t=128: 1.29 t/s). `--poll 100` changed nothing. Context depth cost is mild (5.41 → 5.29 t/s at d=4096). The GPUs are worth +43–55% on decode (5.53–6.01 vs 3.87), refuting the mid-investigation hypothesis that they might be net-zero; at the same time they run at ~11% bandwidth utilization during decode, because MLA blocks tensor-parallel attention and the experts live in DRAM by design.

The decode budget closes as follows: ideal ≈ 91 ms CPU expert streaming (13.77 GB ÷ 152 GB/s) + ~47 ms serialized GPU dense path ≈ 138 ms (7.2 t/s) vs ~180–201 ms actual — the gap is the ~90 ms GPU-side constant (measured as C in the fit), of which graph-split overhead is only ~0.4 ms × 137–155 splits. Realistic ceiling on this platform: 7–8 t/s with further placement work; hard ceiling ~11 t/s; neither was pursued because the marginal effort exceeded the value.

## 4. Prefill results

The single largest win of the investigation was free: the benchmark scripts had inherited `-p 512`, and llama-bench clamps `n_ubatch` to `n_prompt`, so every "op_offload" test before 15:21 on 7/14 had actually run at ub 512. The ubatch ladder at -b 8192, measured 7/14 afternoon (op_offload on, exps=CPU):

| n_ubatch | pp8192 (t/s) |
|---|---|
| 512 | 25.90 ± 0.36 |
| 1024 | 41.68 ± 0.12 |
| 2048 | 62.64 ± 0.13 |
| 4096 | 84.62 ± 3.58 |
| 8192 | 104.97 ± 0.53 |

The ladder fits a two-term model, t_ubatch ≈ 14.9 s + 7.72 ms × ub: a fixed ~15 s per-ubatch streaming term (435 GiB of expert weights over one card's PCIe link at 29.4 GB/s, 93% of line rate) plus a linear GEMM term (~6.2 TFLOP/s on the single active card). Both terms are single-card artifacts of the scheduler pin.

The scheduler findings, confirmed at runtime with GGML_SCHED_DEBUG: 1,494 splits per prefill graph, ROCm0 owning 731 of the 1,186 GPU splits — the `return b` heuristic in `ggml_backend_sched` offloads every large MoE matmul to backend 0. The patch (Session 9): Edit 1+2 add a per-schedule round-robin cursor keyed on the layer number parsed from the tensor name, distributing offload targets across eligible GPUs; Edit 3 skips the expert-ids device→host read (and its per-split synchronize) when the batch is large enough that effectively all experts are active (ids count ≥ 8 × n_expert), marking all experts used instead.

| Build | pp8192 | pp16384 | pp32768 |
|---|---|---|---|
| Stock, ub 8192 | 104.97 ± 0.53 | — | — |
| Edit 2 only (distribution) | 105.71 ± 0.56 | — | — |
| Edit 2+3 | **119.36 ± 0.12** | 86.11 ± 0.10 | 55.76 ± 0.03 |
| Rollback verification (2+3 re-confirmed) | 119.29 ± 0.19 | — | — |

Edit 2 alone was a null result — predicted in advance on the grounds that `tensor_set_async` already made the copies asynchronous and only the ids synchronization serialized the pipeline; the co-author position of 170–220 t/s from distribution alone was falsified. The split histogram equalized exactly as designed (ROCm0 731 → 285; distribution 285/300/294/292 + CPU 308), demonstrating that an equalized histogram and faster prefill are separable facts.

Prompt-length scaling decomposes cleanly: a quadratic attention term A ≈ 26.3 s at 32k (72% of the pass — the DSA lightning indexer saves nothing here in this implementation) plus the linear expert terms. The pp32768 prediction from the two-point decomposition landed at 0.8% error (predicted 55.3, measured 55.76). Pipeline parallelism as a further step is closed: with `-ot`, llama.cpp silently disables it, and forcing n_copies = 2 or 4 OOMs during compute-buffer allocation because multi-copy staging marks tensors as inputs/outputs and defeats ggml-alloc buffer reuse (staging ≈ layers × copies, on the order of 200 GiB per card against 30 GiB).

op_offload at small ubatch is a net loss (Phase F/G/M: 25.9 pageable / 29.7 pinned / 7.2 managed vs 34.6 t/s CPU-only pp512) — the streaming term dominates until the microbatch amortizes it. HIP managed memory is a disaster for this workload (7.2 t/s). Pinning the host buffer (-mmp 0, either flavor) is worth ~+13–15% prefill over mmap at t=64 and nothing on decode.

## 5. Refuted hypotheses and dead ends

| Hypothesis / attempt | Verdict | Evidence |
|---|---|---|
| NUMA imbalance explains slow decode | Refuted | Single node (NPS1); `numactl --hardware` |
| Container overhead | Refuted | In-container STREAM = host within 1%; cgroups unlimited |
| Threadpool wake latency (--poll) | Refuted | poll 0 vs 100 identical across the entire thread sweep |
| Strict CPU affinity / CCD placement helps llama.cpp | Refuted | ~9% at t=32; −40% at t=16 strict; STREAM's 2× does not transfer |
| CPU_REPACK (Q4_K 8×8) speeds decode | Refuted | Engaged on 62% of expert bytes; no measurable change |
| Transparent hugepages help | Untested in effect | THP set to `always` but AnonHugePages stayed 0 in the run log (mmap-backed weights are file pages; repack path never allocated enough anon) |
| -sm row as a decode lever | Impossible | "device ROCm0 does not support split buffers" (gfx1030 VMM: no) |
| GGML_CUDA_GRAPH_OPT=1 | Null | 5.40 t/s, unchanged |
| Speculative decoding via MTP (blk.78) | Unavailable | Loader flags blk.78 TENSOR_SKIP; `--spec-type draft-mtp` cannot work for glm-dsa |
| DFlash speculation | Parked | Merged upstream, but no GLM-5.2 draft model exists; projected ~11–12.5 t/s if one appears; verify-side expert reads erode deep speculation on 8-of-256 MoE |
| Resident experts on ROCm1/2 via -ot (no code change) | Negative result | pp2048 51.74 vs 62.64 all-CPU: 17% worse — resident-weight path ≠ op_offload path |
| -ts 0,1,1,1 to relieve ROCm0 | Self-defeating | op_offload still targets ROCm0; ROCm1–3 OOM on attention layers; context creation fails |
| Round-robin distribution alone fixes prefill | Refuted (pre-registered) | 105.71 vs 104.97 baseline |
| Pipeline parallelism (n_copies > 1) | Dead end | Compute-buffer OOM at n_copies 2 and 4; falls back to sched copies = 1 |
| ZenDNN helps (incl. PR #23414 Q8_0 kernels) | Refuted for this model | Experts are Q4_K/Q5_K on CPU; Q8_0 attention is GPU-resident; ZenDNN+LIBXSMM implicated in the v2 SIGILL/SIGSEGV crashes and removed |

Two genuine bugs were identified in passing: (1) the v2 battery crashes — the 411 GiB `hipHostMalloc` pinned ROCm_Host buffer colliding with the 497.8 GiB GTT pool while ZenDNN/LIBXSMM claimed Q8_0 matmuls (SIGILL on one, SIGSEGV on three), against a same-morning rebuild (b10001); (2) llama-bench's `-d` (depth) KV-restore path crashes at 16,384 cells with `hipMemcpyAsync` illegal access (exit 134) — unreported upstream as of 7/21.

Benchmark-methodology pitfalls worth remembering: `-p` clamps `n_ubatch` (the source of the greatest single distortion in this investigation); llama-bench separates `-ot` rules with semicolons and treats commas as separate benchmark configs (the inverse of llama-server) — this produced the 157,272.74 MiB OOM preserved in newtest.txt; llama-bench installs a null log callback, so GGML_SCHED_DEBUG output only appears with `-v`; `llama-fit-params` is disabled by any of -ngl/-ts/-ot/-ncmoe.

## 6. Production configurations

Two configurations are worth keeping, depending on workload:

**Decode-first (interactive chat):** the fitter configuration — no manual placement flags, let `llama-fit-params` fill VRAM (~106 GiB, 15–16 expert layers resident). Measured 6.01 t/s tg128 / 40.11 t/s pp512. Threads 24–32; never schedule into SMT siblings.

**Prefill-first (long-context ingestion, RAG, agents):** patched build (Edits 1–3), `-ngl 99 -ot exps=CPU -b 8192 -ub 8192 -fa 1 -t 32`. Measured 119.36 t/s pp8192, 86.11 pp16384, 55.76 pp32768, decode 5.51 t/s. The two configurations trade ~0.5 t/s decode for ~3× prefill.

An upstream PR draft for the patch exists (`PR-moe-offload-prefill.md`, drafted 7/21) with six placeholders remaining (hardware table entries, reproduction commands, and comparative numbers on a second model).

## 7. Economics

The build cost $9,050, of which RAM was $5,600 (62%) at $5.47/GB for 1 TB of DDR4 3DS RDIMMs. At July 2026 street prices (researched live during the investigation): DDR4 at ~$5.15/GB and DDR5 at ~$30.94/GB, making the RAM an appreciating asset during the DDR4 supply wind-down and the DDR5-based Genoa counterfactual a $16,050–21,000 machine for an estimated +46% decode (model-based, not measured). The four Radeon Pro V620s at $1,600 total were scored a good purchase: four RTX 3090s would have delivered an estimated 6.3 vs the measured 6.01 t/s decode at several times the cost, and the V620s' 122,816 MiB (~120 GiB) aggregate VRAM is what makes the fitter configuration and the prefill patch worthwhile. All four original purchase justifications (prefill acceleration, KV cache capacity, attention/shared-expert residency, small-models-entirely-in-VRAM) were vindicated by measurement. A planned next step, already purchased: 8 × 256 GB DDR4-2933 3DS RDIMMs for a straight swap to 2 TB (to be run at rated speed, not overclocked).

Context for expectations: the frontier gap was estimated at ~10× (patched Galactus vs hosted Opus-class decode ~58.5 t/s), with 5× of frontier judged the usability bar and ~7–10 t/s the reading-speed threshold that the fitter configuration already meets.

## 8. Open items

The upstream PR needs its six placeholders filled and submission. The llama-bench `-d` crash at 16,384 cells deserves a minimal reproduction and an upstream issue. The THP question (AnonHugePages = 0 in MAIN.log) could be settled from the run's `monitors/meminfo.txt` if the effect is ever worth chasing. The 2 TB RAM swap is pending physical installation. DFlash remains the largest untapped decode lever (~2× projected) and is blocked externally on a GLM-5.2 draft model appearing. A DeepSeek-V4-Flash comparison under the same patched build would test whether the Edit 3 gain generalizes across MoE architectures (its confirmed baseline: pp2048 ~80 t/s, tg128 ~7.16 t/s on build 9942).

---

*Compiled July 24, 2026 from the primary records in `Galactus Testing/`. Every number in this document traces to a benchmark row, log line, or measurement preserved in `04-raw-files/` and indexed in `03-data-extracts/`.*
