# LLM Performance Engineering Notebook

A lab notebook of LLM performance engineering: experiments and results in finding and raising the inference speed limits of large Mixture-of-Experts (MoE) models. The measurements come from the lab's own servers — the machine characterized in detail to date is **Galactus** (AMD EPYC 7713, 2 TB DDR4-2933 8-channel, 4 × AMD Radeon Pro V620, llama.cpp + ROCm in an LXC container on Proxmox); **Borg** (Threadripper Pro 3995WX, Radeon AI PRO R9700 + Radeon Pro W6800) is documented and queued. Full specs and raw captures per machine: [hardware/](hardware/).

Every exact speed belongs to the machine it was measured on, and each result states its machine; the measurements published so far are from Galactus. The method, the llama.cpp patch, and the list of changes that did not help apply to any server that runs MoE models with the routed experts in system RAM and the dense layers on GPUs. If you run large MoE models this way, this repo shows how to find the speed limit and how to raise it.

**Read [takeaways/general-principles.md](takeaways/general-principles.md) first.** It sorts every result into three groups: method you can copy, models where you supply your own numbers, and numbers that apply only to Galactus.

---

## Headline results — the normalized baseline (2026-08-15/16)

All five models, one build (`3653e6d6d`), one uniform command (`-ngl 99 -ot "exps=CPU" -fa on -b 8192 -ub 8192 -p 8192 -n 128 -r 2`), stock scheduler — the prefill patch is not applied anywhere in this table. One optimization class per number. Full conditions, history, and the speculative sweeps: [Entry 12](results/lab-notebook/12-common-baseline-2tb.md).

| Model | pp8192 (t/s) | tg128 (t/s) | Speculative decode, same build |
|---|---|---|---|
| MiniMax M2.7 † | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| Qwen 3.5 397B | 249.61 ± 19.78 | 9.37 ± 0.16 | — (MTP does not arm on this export) |
| DeepSeek-V4-Flash-0731 | 143.54 ± 1.64 | 10.34 ± 0.10 | **14.1 ± 0.7** (DSpark n=3) |
| GLM-5.2 (t=32) | 95.99 ± 3.36 | 5.30 ± 0.00 | **6.6 ± 0.3** (MTP n=2) |
| Kimi K2.6 | 94.23 ± 4.45 | 5.79 ± 0.01 | — |

† Terminal measurement: MiniMax M2.7 was retired and removed from the machine after this row.

Historical bests under other classes and builds — the July patched prefill (GLM 119.36), the April resident-offload rows (MiniMax 17.37, Qwen ~11.7) — live in the per-model notes under [results/](results/) as history, labeled with their class and build. They are no longer mixed into this table.

The GLM-5.2 investigation is the deepest: prefill went from 37.63 to **119.36 t/s** on the July build (unclamping `n_ubatch`, 2.6×, plus the 3-edit scheduler patch, +13.7%), and decode from 5.15 to 7.1 (July) — **6.6 ± 0.3 on the current build** (MTP n=2). DeepSeek-V4-Flash decode roughly doubled since July, 7.16 → **14.1 ± 0.7** (DSpark n=3), with the stock part of that gain (+44%) coming from upstream llama.cpp churn alone — the argument, in one number, for re-baselining on one build.

This project measured both limits; it did not guess them. Memory bandwidth limits decode: STREAM measured 152 GB/s after the RFO correction. A two-term model predicted three separate configurations to within a few percent. Two faults in the llama.cpp scheduler limited prefill. This project fixed both.

Note: the July GLM-5.2 results ran on the earlier 1 TB memory (8 × 128 GB). Galactus now has 2 TB (8 × 256 GB DDR4-2933 3DS RDIMM), installed August 2026. Re-baselined on the 2 TB DIMMs 2026-08-15, after one failing DIMM in the new population was found and replaced: 148–151 GB/s RFO-corrected, within ~2% of the 1 TB figure — the July numbers stand as measured. See [hardware/galactus/README.md](hardware/galactus/README.md).

## The three changes most likely to help your system

1. **The prefill patch** — [patches/prefill/README.md](patches/prefill/README.md). Three small edits to `ggml/src/ggml-backend.cpp`: spread the offloaded expert matmuls across all GPUs instead of sending every one to backend 0, and skip the expert-ids read from GPU to host during prefill, when every expert is used in any case. On Galactus this raised prefill by 13.7%; the method applies to any CPU-MoE offload setup. Being submitted upstream to llama.cpp.
2. **The benchmark faults** — [takeaways/general-principles.md](takeaways/general-principles.md#benchmark-faults). The largest error in this project was that `-p` limits `n_ubatch`. Many `op_offload` tests had run at ub 512 without our knowledge.
3. **The refuted-hypotheses table** — [takeaways/refuted-hypotheses.md](takeaways/refuted-hypotheses.md). NUMA, CPU affinity, `--poll`, CPU_REPACK, THP, ZenDNN, `-sm row`, pipeline parallelism, and managed memory. This project measured each one. Each was null or worse for this workload. The table shows you what to skip.

## Repo map

| Path | Content |
|---|---|
| [takeaways/](takeaways/) | **What carries to your system.** [general-principles.md](takeaways/general-principles.md) (read this first), the [refuted-hypotheses table](takeaways/refuted-hypotheses.md), and [speculative decoding](takeaways/speculative-decoding.md). |
| [results/](results/) | The empirical record: one note per model, the [methodology](results/methodology.md), the [lab notebook](results/lab-notebook/) (Sessions 1–10 and Entries 11–12), [raw logs](results/raw-logs/), and [CSV data](results/data/). |
| [hardware/](hardware/) | Per-machine hardware notes and raw captures: [galactus/](hardware/galactus/), [borg/](hardware/borg/). |
| [patches/](patches/) | The llama.cpp scheduler patch (Edits 1–3), and how to apply and check it. Being submitted upstream. |
| [experiments/](experiments/) | Measurement harnesses: the 16-phase diagnostic run and the Session-10 rerun protocol (superseded by Entry 12). |
| [REPRODUCE.md](REPRODUCE.md) | How to run the same measurements on your own hardware. |

## Status

Active. All five models were re-baselined 2026-08-15/16 on one build with the stock scheduler ([Entry 12](results/lab-notebook/12-common-baseline-2tb.md)); MiniMax M2.7 was retired after its final row. Production decode configurations on the current build: GLM-5.2 **6.6 ± 0.3 t/s** (MTP n=2) and DeepSeek-V4-Flash **14.1 ± 0.7 t/s** (DSpark n=3). Open items: the prefill-patch A/B across the four kept models, the upstream PR for the patch, the Kimi K2.6 vs K2.7-Code disposition, the speculative-run timing variance, and a Kimi K3 test.

## License

This repo uses two licenses. The code (`patches/`, `experiments/`) is **MIT** — see [LICENSE](LICENSE). The text and data (`results/`, `takeaways/`, `hardware/`, and the root Markdown files) are **CC BY 4.0** — see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt). [LICENSING.md](LICENSING.md) states which license covers which path and how to attribute the text and data.

---

*Not affiliated with AMD, the llama.cpp project, or any model vendor. "Galactus" and "Borg" are the author's names for the machines. Every number states which machine it came from.*
