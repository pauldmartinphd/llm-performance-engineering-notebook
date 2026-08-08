# Galactus — platform and method

Everything here is **not model-specific**: the machine, the measurement method, the benchmark pitfalls, the platform-general refuted hypotheses, the reusable llama.cpp patch, and the economics. Per-model results are the notes linked at the bottom.

**System:** AMD EPYC 7713 (64C/128T Zen 3), 2 TB DDR4-2933 3DS RDIMM (8-channel), 4 × AMD Radeon Pro V620 (gfx1030, ~120 GiB VRAM total), llama.cpp + ROCm in an LXC container on Proxmox.

> The July GLM-5.2 investigation and the 152 GB/s bandwidth figure are from the earlier 1 TB population (8 × 128 GB). Galactus now has 2 TB (8 × 256 GB), installed August 2026; the STREAM re-baseline on the new DIMMs is an open item.

## 1. Platform characterization (settled facts)

**Memory bandwidth.** STREAM with the RFO correction (Scale ×1.5, Add/Triad ×4/3; Copy needs none — it compiles to non-temporal stores) converges on **~152 GB/s**, 74% of the 204.8 GB/s theoretical ceiling for 8-channel DDR4-2933. Bandwidth saturates at 16 threads and declines beyond. Packed thread binding at 16 threads collapses to ~90 GB/s Copy / ~68 GB/s Triad (each Milan CCD reaches the IO die over one GMI2 link, so 3–4+ CCDs must be active). The LXC container is free: in-container STREAM equals host within 1%.

**Topology.** Single NUMA node (NPS1). CPUs 0–63 are physical cores (CCD *k* = CPUs 8k…8k+7); 64–127 are SMT siblings. ISA: AVX2 + FMA (no AVX-512).

**GPU fabric.** All 12 peer-to-peer directions enabled. Single-pair peer copy 16.2 GB/s; four concurrent pairs 49.4 GB/s. Host→device 22.3 GB/s single stream (≈ PCIe 4.0 ×16), **65.7 GB/s across four concurrent streams** — the headroom the prefill patch exploits.

**Storage.** ZFS on SATA SSD, 1.5 GB/s direct / 2.1 GB/s buffered — never a bottleneck for mmap loads.

## 2. Diagnostic method (transfers to any system)

- **STREAM + RFO correction** to find true DRAM bandwidth. Sanity check: uncorrected-Copy + RFO must not exceed the theoretical ceiling.
- **Split histogram** — `GGML_SCHED_DEBUG=2 … -v`, then `grep '## SPLIT' | sort | uniq -c` — to see whether offload concentrates on one card.
- **Confirm the mechanism before the number.** An equalized histogram and a faster prefill are separate facts; prove the histogram moved before trusting throughput.
- **Decode two-term model:** `t/token ≈ C + bytes_per_token ÷ bandwidth`. On Galactus C ≈ 90 ms; the model predicted three GLM-5.2 configs within a few percent.

## 3. Benchmark pitfalls (properties of llama.cpp, not Galactus)

- **`-p N` clamps `n_ubatch` to N.** The single largest distortion in the GLM-5.2 work — always set `-ub` explicitly.
- **`llama-bench` separates `-ot` rules with semicolons; commas make separate benchmark configs** (the inverse of `llama-server`) → the catch-all is dropped and the GPU OOMs.
- **`llama-bench` installs a null log callback** — `GGML_SCHED_DEBUG` output needs `-v`.
- **`llama-fit-params` is disabled** by any of `-ngl` / `-ts` / `-ot` / `-ncmoe`.
- **`llama-cli` on a 1M-context model** needs explicit `-c` or it fills VRAM with KV and OOMs.
- **`llama-cli` is not a measurement tool** — `| tee` breaks its TTY; `--log-file` drops the timing lines. Use `script -q` or `llama-server` JSON timings.

## 4. The llama.cpp prefill patch (reusable)

Three edits to `ggml/src/ggml-backend.cpp`: distribute offloaded expert matmuls across all GPUs (layer-keyed) instead of always backend 0, and skip the expert-ids device→host read at prefill-sized batches (where every expert is used, so the read only imposes a serializing `synchronize`). On GLM-5.2: **104.97 → 119.36 t/s pp8192 (+13.7%)**, decode unchanged. Distribution alone is a null result; the ids-read bypass is the load-bearing edit. Full write-up and the verbatim edits: the [`patch/`](../patch/README.md) directory.

## 5. Refuted hypotheses and dead ends (platform-general)

| Hypothesis / attempt | Verdict | Evidence |
|---|---|---|
| NUMA imbalance explains slow decode | Refuted | Single node (NPS1); `numactl --hardware` |
| Container overhead | Refuted | In-container STREAM = host within 1%; cgroups unlimited |
| Threadpool wake latency (`--poll`) | Refuted | poll 0 vs 100 identical across the sweep |
| Strict CPU affinity / CCD placement helps llama.cpp | Refuted | ~9% at t=32; −40% at t=16 strict; STREAM's 2× does not transfer |
| CPU_REPACK (Q4_K 8×8) speeds decode | Refuted | Engaged on 62% of expert bytes; no measurable change |
| Transparent hugepages help | Refuted | No change on GLM-5.2 or Qwen; mmap weights are file pages, not anon |
| `-sm row` as a decode lever | Impossible | gfx1030 VMM: no — "device does not support split buffers" |
| Pipeline parallelism (`n_copies` > 1) | Dead end | Compute-buffer OOM at n_copies 2 and 4 |
| ZenDNN helps | Refuted for these models | Experts are Q4_K/Q5_K on CPU; implicated in v2 crashes; removed |
| HIP managed memory | Disaster | 7.2 t/s prefill |
| SMT oversubscription (t=128) | Refuted (harmful) | Decode collapses on every model tested |

## 6. Economics

The original 1 TB build cost $9,050 (RAM $5,600 at $5.47/GB). RAM is now **2 TB** (8 × 256 GB DDR4-2933 3DS RDIMM) at **$7,800** (≈ $3.81/GB), upgraded August 2026. Four V620s $1,600. At July 2026 street prices DDR4 ≈ $5.15/GB vs DDR5 ≈ $30.94/GB; a DDR5/Genoa counterfactual was a $16–21k machine for an estimated +46% decode — evaluated and declined. The V620s' ~120 GiB aggregate VRAM is what makes resident-expert offload and the prefill patch worthwhile.

---

## Model results

Per-model measurements on Galactus:

- [glm-5.2](results-and-takeaways/glm-5.2.md) — 753 B, Q4_K_XL. The main investigation: 119.36 t/s prefill (patched), 7.1 t/s decode (MTP).
- [deepseek-v4-flash](results-and-takeaways/deepseek-v4-flash.md) — 0731, Q8_K_XL. 14.7 t/s decode (DSpark) — the fastest decode on any model here.
- [qwen-3.5-397b](results-and-takeaways/qwen-3.5-397b.md) — 396 B, Q6_K. ~11.7 t/s decode / ~110 t/s prefill (resident offload, 2933).
- [minimax-m2.7](results-and-takeaways/minimax-m2.7.md) — 229 B, Q5_K. 17.37 t/s decode / 154.93 t/s prefill (resident offload).
- [kimi-k2.5](results-and-takeaways/kimi-k2.5.md) — 1.03 T, Q4_K. The largest tested: 6.76 t/s decode / 44 t/s prefill.
