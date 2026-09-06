# LLM Performance Engineering Notebook

This repository is a laboratory notebook for large language model performance engineering. It records the experiments and results from measuring and raising the inference speed of large Mixture-of-Experts (MoE) models. All measurements come from the laboratory's own machines. **Galactus** (AMD EPYC 7713, 2 TB DDR4-2933 8-channel, 4 × AMD Radeon Pro V620, running llama.cpp and ROCm in an LXC container on Proxmox) is the only machine we have characterized in detail so far. The repository also documents four machines that we have not yet benchmarked, and they use very different memory systems: **Borg** (Threadripper Pro 3995WX, Radeon AI PRO R9700 and Radeon Pro W6800, 8-channel DDR4), **Vision** (a Lenovo ThinkStation P620 with a Threadripper Pro processor and 8-channel DDR4, bought broken and now under repair), **SilverSurfer** (an HP ZBook Ultra G1a with a Ryzen AI Max+ PRO 395 and 128 GB of unified LPDDR5X), and **Magneto** (an Apple M2 Max with 64 GB of unified LPDDR5). The [hardware/](hardware/) directory holds the full specifications and raw captures for each machine.

Every measured speed belongs to the machine that produced it, and each result states its machine. All published measurements so far come from Galactus. The method, the llama.cpp patch, and the list of changes that did not help apply to any server that runs MoE models with the routed experts in system memory and the dense layers on the GPUs. This repository shows how to find and raise the inference speed limit for a system configured in that way.

Read [takeaways/general-principles.md](takeaways/general-principles.md) first. It sorts every result into three groups: methods that transfer directly to other systems, models that require fresh measurements on other hardware, and figures specific to Galactus.

---

## Normalized baseline results (2026-08-15/16)

This table reports all five models under one build (`3653e6d6d`) and one command (`-ngl 99 -ot "exps=CPU" -fa on -b 8192 -ub 8192 -p 8192 -n 128 -r 2`), using the stock scheduler. No figure in this table uses the prefill patch, and each figure isolates a single class of optimization. [Entry 12](results/lab-notebook/12-common-baseline-2tb.md) records the full conditions, the history, and the speculative-decoding sweeps.

| Model | pp8192 (t/s) | tg128 (t/s) | Speculative decode, same build |
|---|---|---|---|
| MiniMax M2.7 † | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| Qwen 3.5 397B | 249.61 ± 19.78 | 9.37 ± 0.16 | — (MTP does not arm on this export) |
| DeepSeek-V4-Flash-0731 | 143.54 ± 1.64 | 10.34 ± 0.10 | **14.1 ± 0.7** (DSpark n=3) |
| GLM-5.2 (t=32) | 95.99 ± 3.36 | 5.30 ± 0.00 | **6.6 ± 0.3** (MTP n=2) |
| Kimi K2.6 | 94.23 ± 4.45 | 5.79 ± 0.01 | — |

† This is the final measurement of MiniMax M2.7. We retired the model and removed it from the machine after this run.

The per-model notes under [results/](results/) hold the best results under other optimization classes and builds; this table does not. Those results include the patched prefill figure from July (GLM-5.2 at 119.36 t/s) and the resident-offload figures from April (MiniMax at 17.37 t/s, Qwen at about 11.7 t/s). Each note labels its figures with their class and build.

The GLM-5.2 investigation is the most detailed in the project. On the July build, prefill rose from 37.63 to **119.36 t/s**: unclamping `n_ubatch` accounted for a factor of 2.6, and the three-edit scheduler patch added a further 13.7%. Decode rose from 5.15 to 7.1 t/s in July and measures **6.6 ± 0.3 t/s on the current build** with MTP speculative decode at n=2. DeepSeek-V4-Flash decode roughly doubled since July, from 7.16 to **14.1 ± 0.7 t/s** with DSpark at n=3. About 44% of that gain came from changes in upstream llama.cpp alone, which is the clearest single reason to re-baseline every model on one build.

This project measured both limits rather than guessing them. Memory bandwidth sets the decode limit: STREAM measured 152 GB/s after the read-for-ownership correction. A two-term model predicted three separate configurations to within a few percent of the measured values. Two faults in the llama.cpp scheduler set the prefill limit, and this project fixed both.

The July GLM-5.2 results ran on the earlier 1 TB memory (8 × 128 GB). Galactus now has 2 TB (8 × 256 GB DDR4-2933 3DS RDIMM), which we installed in August 2026. We re-baselined on the 2 TB modules on 2026-08-15, after finding and replacing one failing module in the new set, and measured 148–151 GB/s after the read-for-ownership correction. That figure is within about 2% of the 1 TB result, so the July numbers stand as measured. See [hardware/galactus/README.md](hardware/galactus/README.md).

## Changes most likely to help a similar system

Three findings from this project are the most likely to transfer to another system.

The first is a patch to the prefill phase; see [patches/prefill/README.md](patches/prefill/README.md). It makes three small edits to `ggml/src/ggml-backend.cpp`. Without the patch, the scheduler sends every offloaded expert matrix multiplication to the first GPU and leaves the other GPUs idle; the patch distributes that work across all of the GPUs. The patch also removes a step that copies the expert-selection indices from each GPU back to the host during prefill. That copy is unnecessary during prefill, because prefill uses every expert in any case. On Galactus, these edits raised prefill throughput by 13.7%. The same approach applies to any system that keeps MoE experts in system memory and offloads their computation. We are preparing the patch for submission to the llama.cpp project.

The second is a set of measurement faults in the benchmark tools; see [takeaways/general-principles.md](takeaways/general-principles.md#benchmark-faults). The most costly error in the project was that the `-p` option silently limits `n_ubatch`. As a result, many `op_offload` tests ran at a micro-batch size of 512 before we noticed.

The third is the table of refuted hypotheses; see [takeaways/refuted-hypotheses.md](takeaways/refuted-hypotheses.md). We measured the effect of NUMA balancing, CPU affinity, the `--poll` option, `CPU_REPACK`, transparent huge pages, ZenDNN, the `-sm row` split mode, pipeline parallelism, and managed memory. Each one produced no improvement, or a regression, for this workload. The table records these results so that others can skip the same tests.

## Repository contents

| Path | Contents |
|---|---|
| [takeaways/](takeaways/) | The findings that transfer to other systems: [general-principles.md](takeaways/general-principles.md) (read this first), the [table of refuted hypotheses](takeaways/refuted-hypotheses.md), and the note on [speculative decoding](takeaways/speculative-decoding.md). |
| [results/](results/) | The record of measurements: one note per model, the [methodology](results/methodology.md), the [lab notebook](results/lab-notebook/) (Sessions 1–10 and Entries 11–12), the [raw logs](results/raw-logs/), and the [CSV data](results/data/). |
| [hardware/](hardware/) | The hardware notes and raw captures for each machine: [galactus/](hardware/galactus/), [borg/](hardware/borg/), [vision/](hardware/vision/), [silversurfer/](hardware/silversurfer/), and [magneto/](hardware/magneto/). |
| [patches/](patches/) | The llama.cpp scheduler patch (Edits 1–3), with instructions to apply and verify it. |
| [experiments/](experiments/) | The measurement harnesses: the 16-phase diagnostic run and the Session-10 rerun protocol, which Entry 12 supersedes. |
| [REPRODUCE.md](REPRODUCE.md) | Instructions for running the same measurements on other hardware. |

## Status

The project is active. We re-baselined all five models on 2026-08-15 and 2026-08-16 on a single build with the stock scheduler; see [Entry 12](results/lab-notebook/12-common-baseline-2tb.md). We retired MiniMax M2.7 after its final measurement. The production decode configurations on the current build are GLM-5.2 at **6.6 ± 0.3 t/s** (MTP at n=2) and DeepSeek-V4-Flash at **14.1 ± 0.7 t/s** (DSpark at n=3). The open items are the prefill-patch comparison across the four remaining models, the upstream pull request for the patch, the choice between Kimi K2.6 and K2.7-Code, the timing variance in the speculative-decoding runs, and a test of Kimi K3.

## License

This repository uses two licenses. The MIT license covers the code in `patches/` and `experiments/`; see [LICENSE](LICENSE). The CC BY 4.0 license covers the text and data (`results/`, `takeaways/`, `hardware/`, and the Markdown files at the repository root); see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt). [LICENSING.md](LICENSING.md) states which license covers each path and how to attribute the text and data.

---

*This project is not affiliated with AMD, the llama.cpp project, or any model vendor. "Galactus", "Borg", "Vision", "SilverSurfer", and "Magneto" are the author's names for the machines. Every number states the machine it came from.*
