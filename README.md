# LLM Performance Engineering Notebook

This notebook records my measurements of local LLM inference performance. I started with large mixture-of-experts models whose weights exceed GPU memory, placing the routed experts in system RAM while the dense computation runs on GPUs. By measuring where time goes in prompt processing and token generation, I can test changes to the configuration and to llama.cpp itself.

The most detailed investigation so far is GLM-5.2 on Galactus, an EPYC 7713 server with eight-channel DDR4 and four Radeon Pro V620 GPUs. Memory bandwidth explains much of its decode time. Prefill exposed a different problem: llama.cpp concentrated offloaded expert computation on one GPU, and a routing-index readback serialized the work. A scheduler patch raised prefill from **104.97 to 119.36 tokens/s (+13.7%)** in the July comparison. Distributing the work alone produced no meaningful gain; removing the synchronization was also necessary.

The repository includes the measurements, unsuccessful experiments, patch instructions, and chronological notes behind these conclusions. The Galactus investigation now sits alongside the first oMLX measurements on Magneto, a 14-inch M2 Max MacBook Pro. Since the two engines were measured separately, each record gives its platform and conditions rather than treating the results as a common benchmark.

## Results

The latest common baseline in this notebook was measured on **August 15–16, 2026**, after Galactus's upgrade from 1 TB to 2 TB of RAM. All five models used llama.cpp build `3653e6d6d`, the stock scheduler, CPU-resident routed experts, and `-b 8192 -ub 8192 -p 8192 -n 128 -r 2`. GLM used 32 threads; the other models used 64. `pp8192` measures processing an 8,192-token prompt; `tg128` measures generating 128 tokens. Rates are tokens per second.

| Model | pp8192 | tg128 | Speculative decode on the same build |
|---|---|---|---|
| [MiniMax M2.7](results/minimax-m2.7.md) | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| [Qwen 3.5 397B](results/qwen-3.5-397b.md) | 249.61 ± 19.78 | 9.37 ± 0.16 | MTP failed to initialize on this export |
| [DeepSeek-V4-Flash-0731](results/deepseek-v4-flash.md) | 143.54 ± 1.64 | 10.34 ± 0.10 | 14.1 ± 0.7 (DSpark, n=3) |
| [GLM-5.2](results/glm-5.2.md) | 95.99 ± 3.36 | 5.30 ± 0.00 | 6.6 ± 0.3 (MTP, n=2) |
| [Kimi K2.6](results/kimi-k2.5.md) | 94.23 ± 4.45 | 5.79 ± 0.01 | — |

The speculative runs used a ZFS explanation prompt, greedy decoding, and llama-cli rather than llama-bench. Their repeated timings varied by about 9%; the reported spreads are not confidence intervals. MiniMax was retired after this measurement. [Entry 12](results/lab-notebook/12-common-baseline-2tb.md) records the exact model files, conditions, repetitions, and remaining questions.

The July patch result and April resident-expert results used different builds and configurations. They remain in the [model notes](results/README.md), with their original conditions, and are not included in the stock baseline above.

## Apple Silicon measurements (September 30, 2026)

[Qwen3.8-27B on Magneto](results/qwen-3.8-27b.md) records an initial Lightning MTP baseline of **156.1 PP / 16.4 TG tokens/s at 4K**, with tuned ANE + MTP at **170.1 / 13.9** and a later screenshot at **171.0 / 16.9**. The baseline survives as a conversation transcription; the later results have a retained screenshot. Heavy thermal pressure was observed during the investigation. The oMLX leaderboard label omits chassis, so the higher external M2 Max result cannot establish an equivalent-machine comparison. A larger chassis is a hypothesis, not an identified cause. [Entry 13](results/lab-notebook/13-magneto-omlx-qwen38.md) records the ANE tuner/benchmark distinction, DFlash regression, incomplete SpecPrefill experiment, and evidence limitations.

## Reading the notebook

The [findings](takeaways/general-principles.md) collect what the experiments established and what remains uncertain. The [methodology](results/methodology.md) explains how I measured performance; the [reproduction guide](REPRODUCE.md) gives the commands and checks.

| Material | Where to find it |
|---|---|
| Model summaries and common baseline | [Results](results/README.md) |
| Scheduler investigation and exact edits | [Prefill patch](patches/prefill/README.md) |
| Changes that did not help on Galactus | [Negative results](takeaways/refuted-hypotheses.md) |
| MTP and DSpark measurements | [Speculative decoding](takeaways/speculative-decoding.md) |
| Experiments in chronological order | [Lab notebook](results/lab-notebook/00-overview.md) |
| Original captures and extracted measurements | [Raw logs](results/raw-logs/) and [CSV data](results/data/) |
| Machine specifications and memory measurements | [Hardware](hardware/README.md) |
| Diagnostic scripts and historical rerun protocol | [Experiments](experiments/README.md) |

## Work remaining

As of the August 16 entry, the next comparison is the prefill patch against the stock build on the four retained models. An upstream patch submission, investigation of speculative-run timing variance, a choice between Kimi K2.6 and K2.7-Code, and a Kimi K3 evaluation remain open. The August speculative runs supersede the Session 10 rerun protocol.

## License and attribution

Code in `patches/` and `experiments/` is covered by the [MIT license](LICENSE). Text and data are covered by [CC BY 4.0](LICENSE-CC-BY-4.0.txt); [LICENSING.md](LICENSING.md) gives the scope and attribution details.

This project is not affiliated with AMD, llama.cpp, or any model vendor. Galactus and the other machine names are the names I use for the lab hardware.
