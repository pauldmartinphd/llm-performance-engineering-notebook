# LLM Performance Engineering Notebook

*Experiments in hosting large language models on local hardware.*

Your goal is to run the largest models your machines can support, at useful speeds. This notebook covers the work: memory, CPU and GPU execution, placement, transfers, kernels, caching, speculation, storage, power, and serving software. It keeps the failures as well as the improvements.

## Read the notebook

| What you want to read | Start here |
|---|---|
| Every experiment and result | [Lab notebook, in chronological order](results/lab-notebook/README.md) |
| What you tried and what happened | [Investigation chapters](investigations/README.md) |
| The most important demonstrated gains | [Important findings](takeaways/general-principles.md) |
| The project's purpose and current state | Continue below |

## The work

Many large models fit in system RAM but not in GPU memory. Loading them is only the first step. The runtime must divide the work between the CPU and GPUs and move data without spending too much time waiting.

Most tests use mixture-of-experts models. They contain many expert networks but use only a few for each token. System RAM can hold the full pool while the GPUs handle attention, shared work, and cached experts. You measure prompt processing and answer generation separately because a model can write quickly yet take minutes to read a long input.

The main machine is **Galactus**, an EPYC 7713 server with eight-channel DDR4 and four Radeon Pro V620 GPUs. It had 1 TB of RAM in July and 2 TB from August onward. The [hardware notes](hardware/README.md) give the details.

You started with setup, thread counts, placement, and memory-bandwidth measurements. The July GLM-5.2 tests then exposed a batch-size mistake and a scheduler wait. A combined patch improved prefill by 13.7%; distributing the work alone did not help much.

You later combined AMD and model-support changes in the llamda fork. Its measured rates were nearly the same as upstream, so you withdrew the performance announcement and kept the work as an experiment. Maya then gave useful generation rates for GLM-Flash and DeepSeek. The next tests covered long prompts and splitting the GPUs between the two models.

Separate chapters cover oMLX on an M2 Max laptop and Gufo on a Ryzen AI Max+ laptop.

## Current Galactus setup

As of November 10, 2026, you host two Maya services:

| Service | GPUs | Selected configuration |
|---|---|---|
| DeepSeek-V4.1-Flash-1017 | 0–1 | UD-Q8_K_XL weights, DSpark, FP16-output prefill |
| GLM-5.3-Flash | 2–3 | UD-Q4_K_XL weights, native MTP, INT8 streaming KV |

The [deployment entry](results/lab-notebook/2026-11-10-maya-dual-model-production.md) gives the settings. The tests measured each model separately. They did not measure simultaneous generation, quality equivalence for the precision changes, or a full million-token prompt.

## Supporting material

[Results by model](results/README.md) · [Measurement methods](results/methodology.md) · [Recorded procedures](REPRODUCE.md) · [Data](results/data/) · [Raw captures](results/raw-logs/) · [Sources](results/reference-notes/README.md)

The [unsuccessful approaches](takeaways/refuted-hypotheses.md) are kept separately, with the conditions under which they failed or ran slower.

Authored text and data are CC BY 4.0. Code in `patches/` and `experiments/` is MIT-licensed; third-party files keep their own licenses. See [LICENSING.md](LICENSING.md). This project is not affiliated with AMD, llama.cpp, or a model vendor.
