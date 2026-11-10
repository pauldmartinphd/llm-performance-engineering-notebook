# LLM Performance Engineering Notebook

*Experiments in hosting large language models on local hardware.*

I want to run the largest language models my local machines can support, and understand what makes them fast enough to use. That means testing the whole system: memory capacity and bandwidth, CPU work, GPU placement, transfers, computation kernels, caching, speculative decoding, storage, power, and the software that serves requests.

This is a lab notebook of that work. It keeps the experiments and results, including failed attempts. The investigation chapters explain why I ran the tests and how the results changed my decisions.

## Read the notebook

| What you want to read | Start here |
|---|---|
| Every experiment and result | [Lab notebook, in chronological order](results/lab-notebook/README.md) |
| What I did, why I did it, and what followed | [Investigation chapters](investigations/README.md) |
| The most important demonstrated gains | [Important findings](takeaways/general-principles.md) |
| The project's purpose and current state | Continue below |

## The problem

Many large models fit in system RAM but do not fit in GPU memory, or VRAM. Simply loading them is not enough. The engine must read the right weights—the numbers learned during training—move data where it is needed, and keep the CPU and GPUs doing useful work.

Most of this notebook concerns mixture-of-experts models. They hold many expert networks but use only a subset for each token. That makes them attractive for local hosting: system RAM can hold the full model while GPUs handle attention, shared work, and cached experts. It also creates many ways for execution to stall or waste bandwidth.

I measure prompt reading and answer generation separately. A model that writes quickly may still take minutes to read a long input. I also distinguish a model's configured context limit from the prompt lengths I have actually tested.

## How the investigation developed

The main machine is **Galactus**, an EPYC 7713 server with eight-channel DDR4 and four Radeon Pro V620 GPUs. It had 1 TB of RAM during the July work and 2 TB from August onward. The [hardware notes](hardware/README.md) give the configurations and changes.

I began by testing setup, thread counts, and expert placement. Memory measurements established a useful bandwidth reference. A common five-model benchmark then made comparisons easier to interpret.

The July GLM-5.2 investigation found two prefill problems: tests had used a smaller effective batch than intended, and the scheduler concentrated offloaded work while waiting on routing information. A combined patch improved the tested prefill rate by 13.7%. Spreading the work alone did not produce that gain.

I later combined AMD and model-support work in an experimental fork called llamda. Its revision-identified benchmarks were nearly the same as upstream llama.cpp. I kept the work as an experiment and withdrew the performance announcement.

Project Maya then gave useful generation rates for GLM-Flash and DeepSeek. The later tests focused on long prompts, expert-cache memory, precision settings, and dividing the four GPUs between two models. Separate notebook branches cover oMLX on an M2 Max laptop and Gufo on a Ryzen AI Max+ laptop.

## Current Galactus setup

As reported on November 10, 2026, I host two Maya services:

| Service | GPUs | Selected configuration |
|---|---|---|
| DeepSeek-V4.1-Flash-1017 | 0–1 | UD-Q8_K_XL weights, DSpark, FP16-output prefill |
| GLM-5.3-Flash | 2–3 | UD-Q4_K_XL weights, native MTP, INT8 streaming KV |

The [deployment record](results/lab-notebook/2026-11-10-maya-dual-model-production.md) gives the retained settings and evidence. The tests measured each model separately. They do not establish throughput while both generate at once, quality equivalence for every precision setting, or performance at the configured one-million-token ceiling.

## Results and supporting material

The [results index](results/README.md) collects the model measurements. The [measurement guide](results/methodology.md) explains the terms and controls; the [reproduction guide](REPRODUCE.md) links the recorded procedures. [Data tables](results/data/), [raw captures](results/raw-logs/), and [source notes](results/reference-notes/README.md) preserve the details behind the accounts.

The [unsuccessful approaches](takeaways/refuted-hypotheses.md) are kept separately. They show what failed, what was slower, and what remained unresolved on the tested configurations. They are useful limits on the findings, not claims that an approach cannot work elsewhere.

Code in `patches/` and `experiments/` is MIT-licensed. Authored text and data are CC BY 4.0; third-party snapshots retain their own licenses. See [LICENSING.md](LICENSING.md). This project is not affiliated with AMD, llama.cpp, or a model vendor.
