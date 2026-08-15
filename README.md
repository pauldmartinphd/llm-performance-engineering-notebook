# LLM Performance Engineering Notebook

Notes on tuning large Mixture-of-Experts (MoE) models on **Galactus**, one lab server:
**AMD EPYC 7713** (64 cores / 128 threads, Zen 3), **2 TB DDR4-2933 3DS RDIMM** (8-channel mode), **4 × AMD Radeon Pro V620** (gfx1030, about 120 GiB VRAM total). It runs llama.cpp and ROCm in an LXC container on Proxmox.

This is one machine, tested in detail. The exact speeds belong to this machine. The method, the llama.cpp patch, and the list of changes that did not help apply to any server that runs MoE models with the routed experts in system RAM and the dense layers on GPUs. If you run large MoE models this way, this repo shows how to find the speed limit and how to raise it.

**Read [docs/what-transfers.md](docs/what-transfers.md) first.** It sorts every result into three groups: method you can copy, models where you supply your own numbers, and numbers that apply only to Galactus.

---

## Headline results

Best measured decode / prefill per model on Galactus (details in [docs/results-and-takeaways/](docs/results-and-takeaways/)):

| Model | Params / quant | Best decode | Best prefill |
|---|---|---|---|
| MiniMax M2.7 | 229 B / Q5_K | 17.37 t/s (resident offload) | 154.93 t/s |
| DeepSeek-V4-Flash-0731 | Q8_K_XL, MXFP4 | 14.7 t/s (DSpark n=3) | — |
| Qwen 3.5 397B | 396 B / Q6_K | ~11.7 t/s (resident offload) | ~110 t/s |
| GLM-5.2 | 753 B / Q4_K_XL | 7.1 t/s (MTP n=2) | 119.36 t/s (patched) |
| Kimi K2.5 | 1.03 T / Q4_K_XL | 6.76 t/s | 44 t/s |

The GLM-5.2 investigation is the deepest: prefill went from 37.63 to **119.36 t/s** (unclamping `n_ubatch`, 2.6×, plus the 3-edit scheduler patch, +13.7%) and decode from 5.15 to **7.1 t/s** (MTP n=2). DeepSeek-V4-Flash decode roughly doubled to **14.7 t/s** (DSpark n=3), all from software.

This project measured both limits; it did not guess them. Memory bandwidth limits decode: STREAM measured 152 GB/s after the RFO correction. A two-term model predicted three separate configurations to within a few percent. Two faults in the llama.cpp scheduler limited prefill. This project fixed both.

Note: the July GLM-5.2 results ran on the earlier 1 TB memory (8 × 128 GB). Galactus now has 2 TB (8 × 256 GB DDR4-2933 3DS RDIMM), installed August 2026. The STREAM re-baseline on the 2 TB DIMMs is still an open item, so the 152 GB/s figure comes from the 1 TB population. See [hardware/galactus/README.md](hardware/galactus/README.md).

## The three changes most likely to help your system

1. **The prefill patch** — [patch/README.md](patch/README.md). Three small edits to `ggml/src/ggml-backend.cpp`. The edits spread the offloaded expert matmuls across all GPUs, instead of sending every one to backend 0. They also skip the expert-ids read from GPU to host during prefill, when every expert is used in any case. On Galactus this raised prefill by 13.7%. The method applies to any CPU-MoE offload setup.
2. **The benchmark faults** — [docs/what-transfers.md](docs/what-transfers.md#benchmark-faults). The largest error in this project was that `-p` limits `n_ubatch`. Many `op_offload` tests had run at ub 512 without our knowledge.
3. **The refuted-hypotheses table** — [docs/platform-and-method.md §5](docs/platform-and-method.md). NUMA, CPU affinity, `--poll`, CPU_REPACK, THP, ZenDNN, `-sm row`, pipeline parallelism, and managed memory. This project measured each one. Each was null or worse for this workload. The table shows you what to skip.

## Repo map

| Path | Content |
|---|---|
| [docs/what-transfers.md](docs/what-transfers.md) | **The generalization guide. Read this first.** |
| [docs/platform-and-method.md](docs/platform-and-method.md) | The non-model hub: platform, method, benchmark pitfalls, the refuted-hypotheses table, the patch, economics, and the model index. |
| [docs/results-and-takeaways/](docs/results-and-takeaways/) | One note per model: GLM-5.2, DeepSeek-V4-Flash, Qwen 3.5 397B, Kimi K2.5, MiniMax M2.7. |
| [docs/lab-notebook/](docs/lab-notebook/) | The full chronological record and the DSpark / Session-10 addendum. |
| [patch/](patch/) | The llama.cpp scheduler patch (Edits 1–3), and how to apply and check it. |
| [scripts/galactus-diag.sh](scripts/galactus-diag.sh) | The 16-phase diagnostic run. |
| [data/](data/) | CSV files. Every benchmark row. |
| [hardware/](hardware/) | Per-machine hardware notes and raw captures: [galactus/](hardware/galactus/), [borg/](hardware/borg/). |
| [raw-logs/](raw-logs/) | The `-ot` OOM failure, the raw MTP A/B capture, and the per-model benchmark logs. |
| [REPRODUCE.md](REPRODUCE.md) | How to run the same measurements on your own hardware. |

## Status

Active, and now spanning five models. The production decode configurations are settled: GLM-5.2 at 7.1 t/s with MTP n=2, and DeepSeek-V4-Flash at 14.7 t/s with DSpark n=3. Open items: a STREAM re-baseline on the new 2 TB DIMMs, an upstream PR for the prefill patch, and a Kimi K3 test. See the open-items sections in [docs/platform-and-method.md](docs/platform-and-method.md) and the per-model notes.

## License

This repo uses two licenses. The code (`patch/`, `scripts/`) is **MIT** — see [LICENSE](LICENSE). The text and data (`docs/`, `data/`, `hardware/`, `raw-logs/`, and the root Markdown files) are **CC BY 4.0** — see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt). [LICENSING.md](LICENSING.md) states which license covers which path and how to attribute the text and data.

---

*Not affiliated with AMD, the llama.cpp project, or any model vendor. "Galactus" is the author's name for one machine. All numbers come from that machine unless the text states otherwise.*
