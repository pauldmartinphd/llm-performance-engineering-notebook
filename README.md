# LLM Performance Engineering Notebook

A lab notebook of LLM performance engineering: experiments and results in finding and raising the inference speed limits of large Mixture-of-Experts (MoE) models. The measurements come from the lab's own servers — the machine characterized in detail to date is **Galactus** (AMD EPYC 7713, 2 TB DDR4-2933 8-channel, 4 × AMD Radeon Pro V620, llama.cpp + ROCm in an LXC container on Proxmox); **Borg** (Threadripper Pro 3995WX, Radeon AI PRO R9700 + Radeon Pro W6800) is documented and queued. Full specs and raw captures per machine: [hardware/](hardware/).

Every exact speed belongs to the machine it was measured on, and each result states its machine; the measurements published so far are from Galactus. The method, the llama.cpp patch, and the list of changes that did not help apply to any server that runs MoE models with the routed experts in system RAM and the dense layers on GPUs. If you run large MoE models this way, this repo shows how to find the speed limit and how to raise it.

**Read [takeaways/general-principles.md](takeaways/general-principles.md) first.** It sorts every result into three groups: method you can copy, models where you supply your own numbers, and numbers that apply only to Galactus.

---

## Headline results

Best measured decode / prefill per model on Galactus (details in [results/](results/)):

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

1. **The prefill patch** — [patches/prefill/README.md](patches/prefill/README.md). Three small edits to `ggml/src/ggml-backend.cpp`: spread the offloaded expert matmuls across all GPUs instead of sending every one to backend 0, and skip the expert-ids read from GPU to host during prefill, when every expert is used in any case. On Galactus this raised prefill by 13.7%; the method applies to any CPU-MoE offload setup. Being submitted upstream to llama.cpp.
2. **The benchmark faults** — [takeaways/general-principles.md](takeaways/general-principles.md#benchmark-faults). The largest error in this project was that `-p` limits `n_ubatch`. Many `op_offload` tests had run at ub 512 without our knowledge.
3. **The refuted-hypotheses table** — [takeaways/refuted-hypotheses.md](takeaways/refuted-hypotheses.md). NUMA, CPU affinity, `--poll`, CPU_REPACK, THP, ZenDNN, `-sm row`, pipeline parallelism, and managed memory. This project measured each one. Each was null or worse for this workload. The table shows you what to skip.

## Repo map

| Path | Content |
|---|---|
| [takeaways/](takeaways/) | **What carries to your system.** [general-principles.md](takeaways/general-principles.md) (read this first), the [refuted-hypotheses table](takeaways/refuted-hypotheses.md), and [speculative decoding](takeaways/speculative-decoding.md). |
| [results/](results/) | The empirical record: one note per model, the [methodology](results/methodology.md), the [lab notebook](results/lab-notebook/) (Sessions 1–10), [raw logs](results/raw-logs/), and [CSV data](results/data/). |
| [hardware/](hardware/) | Per-machine hardware notes and raw captures: [galactus/](hardware/galactus/), [borg/](hardware/borg/). |
| [patches/](patches/) | The llama.cpp scheduler patch (Edits 1–3), and how to apply and check it. Being submitted upstream. |
| [experiments/](experiments/) | Measurement harnesses: the 16-phase diagnostic run and the Session-10 rerun protocol. |
| [REPRODUCE.md](REPRODUCE.md) | How to run the same measurements on your own hardware. |

## Status

Active, and now spanning five models. The production decode configurations are settled: GLM-5.2 at 7.1 t/s with MTP n=2, and DeepSeek-V4-Flash at 14.7 t/s with DSpark n=3. Open items: a STREAM re-baseline on the new 2 TB DIMMs, an upstream PR for the prefill patch, and a Kimi K3 test. See the open-items sections in the per-model notes under [results/](results/).

## License

This repo uses two licenses. The code (`patches/`, `experiments/`) is **MIT** — see [LICENSE](LICENSE). The text and data (`results/`, `takeaways/`, `hardware/`, and the root Markdown files) are **CC BY 4.0** — see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt). [LICENSING.md](LICENSING.md) states which license covers which path and how to attribute the text and data.

---

*Not affiliated with AMD, the llama.cpp project, or any model vendor. "Galactus" and "Borg" are the author's names for the machines. Every number states which machine it came from.*
