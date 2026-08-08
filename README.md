# LLM Performance Engineering Notebook

Notes on tuning large Mixture-of-Experts (MoE) models on **Galactus**, one lab server:
**AMD EPYC 7713** (64 cores / 128 threads, Zen 3), **2 TB DDR4-2933 3DS RDIMM** (8-channel mode), **4 × AMD Radeon Pro V620** (gfx1030, about 120 GiB VRAM total). It runs llama.cpp and ROCm in an LXC container on Proxmox.

This is one machine, tested in detail. The exact speeds belong to this machine. The method, the llama.cpp patch, and the list of changes that did not help apply to any server that runs MoE models with the routed experts in system RAM and the dense layers on GPUs. If you run large MoE models this way, this repo shows how to find the speed limit and how to raise it.

**Read [docs/what-transfers.md](docs/what-transfers.md) first.** It sorts every result into three groups: method you can copy, models where you supply your own numbers, and numbers that apply only to Galactus.

---

## Headline results (GLM-5.2 UD-Q4_K_XL, 753.86 B parameters, 435 GiB)

| Metric | Start (7/13) | End | Change | How |
|---|---|---|---|---|
| Prefill (pp8192) | 37.63 t/s | **119.36 t/s** | about 3.2× | unclamp `n_ubatch` (2.6×) + 3-edit scheduler patch (+13.7%) |
| Decode (tg128) | 5.15 t/s | **7.1 t/s** | +38% | at the DDR4 bandwidth limit; the +31% is MTP speculative decode (n=2) |

On **DeepSeek-V4-Flash-0731** (UD-Q8_K_XL, MXFP4 routed experts), decode went from 7.16 to 14.7 t/s, about 2×, all from software. The last +45% came from DSpark speculative decoding (n=3). See [docs/dspark-deepseek-v4-flash.md](docs/dspark-deepseek-v4-flash.md).

This project measured both limits; it did not guess them. Memory bandwidth limits decode: STREAM measured 152 GB/s after the RFO correction. A two-term model predicted three separate configurations to within a few percent. Two faults in the llama.cpp scheduler limited prefill. This project fixed both.

Note: the July GLM-5.2 results ran on the earlier 1 TB memory (8 × 128 GB). Galactus now has 2 TB (8 × 256 GB DDR4-2933 3DS RDIMM), installed August 2026. The STREAM re-baseline on the 2 TB DIMMs is still an open item, so the 152 GB/s figure comes from the 1 TB population. See [specs/hardware.md](specs/hardware.md).

## The three changes most likely to help your system

1. **The prefill patch** — [patch/README.md](patch/README.md). Three small edits to `ggml/src/ggml-backend.cpp`. The edits spread the offloaded expert matmuls across all GPUs, instead of sending every one to backend 0. They also skip the expert-ids read from GPU to host during prefill, when every expert is used in any case. On Galactus this raised prefill by 13.7%. The method applies to any CPU-MoE offload setup.
2. **The benchmark faults** — [docs/what-transfers.md](docs/what-transfers.md#benchmark-faults). The largest error in this project was that `-p` limits `n_ubatch`. Many `op_offload` tests had run at ub 512 without our knowledge.
3. **The refuted-hypotheses table** — [docs/results-and-takeaways.md §5](docs/results-and-takeaways.md). NUMA, CPU affinity, `--poll`, CPU_REPACK, THP, ZenDNN, `-sm row`, pipeline parallelism, and managed memory. This project measured each one. Each was null or worse for this workload. The table shows you what to skip.

## Repo map

| Path | Content |
|---|---|
| [docs/what-transfers.md](docs/what-transfers.md) | **The generalization guide. Read this first.** |
| [docs/results-and-takeaways.md](docs/results-and-takeaways.md) | Settled findings: platform measurements, decode and prefill results with the models that explain them, refuted hypotheses, production configurations, and cost. |
| [docs/dspark-deepseek-v4-flash.md](docs/dspark-deepseek-v4-flash.md) | DSpark speculative decode on DeepSeek-V4-Flash, GLM-5.2 MTP, and the 2 TB RAM upgrade. |
| [patch/](patch/) | The llama.cpp scheduler patch (Edits 1–3), and how to apply and check it. |
| [scripts/galactus-diag.sh](scripts/galactus-diag.sh) | The 16-phase diagnostic run. |
| [data/](data/) | CSV files. Every benchmark row. |
| [specs/](specs/) | Hardware sheet and the raw `dmidecode`, STREAM, and kernel-tuning captures. |
| [raw-logs/](raw-logs/) | The `-ot` OOM failure and the raw MTP A/B capture. |
| [REPRODUCE.md](REPRODUCE.md) | How to run the same measurements on your own hardware. |

## Status

Active. The production decode configurations are settled: GLM-5.2 at 7.1 t/s with MTP n=2, and DeepSeek-V4-Flash at 14.7 t/s with DSpark n=3. Open items: a STREAM re-baseline on the new 2 TB DIMMs, an upstream PR for the prefill patch, and a Kimi K3 test. See the open-items sections in the results document and Session 10.

## License

This repo uses two licenses. The code (`patch/`, `scripts/`) is **MIT** — see [LICENSE](LICENSE). The text and data (`docs/`, `data/`, `specs/`, `raw-logs/`, and the root Markdown files) are **CC BY 4.0** — see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt). [LICENSING.md](LICENSING.md) states which license covers which path and how to attribute the text and data.

---

*Not affiliated with AMD, the llama.cpp project, or any model vendor. "Galactus" is the author's name for one machine. All numbers come from that machine unless the text states otherwise.*
