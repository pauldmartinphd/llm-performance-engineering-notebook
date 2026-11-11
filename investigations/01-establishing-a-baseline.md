# 1. Establishing a baseline

[Chapter index](README.md) · [Next: the prefill bottleneck](02-following-the-prefill-bottleneck.md)

## The question

Before changing the software, I needed to know what the machine could do. Was inference slow because the model was too large, because the hardware was poorly configured, or because the engine was using it badly?

The main machine is Galactus: an AMD EPYC 7713 server with eight memory channels and four Radeon Pro V620 GPUs. The July investigation used 1 TB of system RAM. I expanded it to 2 TB in August. The cards provide about 120 GB of usable GPU memory in total. Many of the models in this notebook are much larger than that.

A model can fit in system RAM without fitting in GPU memory. That makes it possible to run, but does not say how fast it will run. The main problem is deciding which work stays on the CPU, which weights stay on the GPUs, and how much data must move between them.

## Why mixture-of-experts models were worth testing

A mixture-of-experts model, or MoE, contains many expert networks. For each token, a router selects a small set of them. The model may contain hundreds of billions of parameters while using a much smaller subset for one step.

This creates a useful split. System RAM can hold the full expert pool. The GPUs can hold attention, shared computation, and some frequently used experts. However, the active experts still have to be read and computed for each token. A large RAM capacity solves the storage problem; memory bandwidth and execution time determine much of the speed.

The [March build record](../results/lab-notebook/2026-03-23-galactus-build.md) documents the first installation, memory repairs, and placement trials. Those results belong to an earlier setup. They are part of the notebook, but they are not interchangeable with the later baseline.

## More CPU threads were not always better

The [April tests](../results/lab-notebook/2026-04-07-to-17-thread-and-batch-tests.md) varied thread counts, expert placement, batches, and context. They showed why I could not pick one thread count from the CPU specification and use it for every model.

Processing a prompt can expose a large amount of parallel work. Generating the next token is more dependent on the previous step and can be limited by memory traffic. Extra threads may help the first task while doing little for the second. At high counts, they can make it slower.

The April comparisons also changed memory speed: the early runs used 2666 MT/s, while later runs used 2933 MT/s. That difference matters. Taking the fastest row from each date would mix several changes and hide what each experiment tested.

## Measuring the memory system

In July I used STREAM, a memory benchmark, to measure sustained bandwidth. After accounting for memory traffic that its printed rates omit, the kernels converged near **152 GB/s**. Bandwidth reached its useful limit at about 16 threads. Adding more threads did not make the memory channels proportionally faster.

The [August remeasurement](../results/lab-notebook/2026-08-15-stream-rebaseline-2tb.md) followed the expansion to 2 TB and replacement of a faulty DIMM. The repaired system measured roughly **148–151 GB/s**. Capacity doubled, but bandwidth stayed close to the earlier result. This was useful: I could hold more model data without having to explain a large change in the platform's sustained memory speed.

A first estimate of generation time is:

```text
time per token ≈ memory-read time + the remaining execution time
memory-read time ≈ bytes read for a token / sustained bandwidth
```

For the tested July GLM-5.2 configurations, a model with about 90 ms of additional time predicted 5.5, 6.2, and 3.9 tokens/s. The measured rates were 5.53, 6.01, and 3.87. That agreement made the model useful for those tests. It did not prove that every later model had the same extra cost.

## A common baseline made the model comparison clearer

The earlier records used different builds and settings. In [August I measured five models with one command structure and one stock build](../results/lab-notebook/2026-08-15-to-16-common-baseline-2tb.md). The prompt contained 8192 tokens, generation contained 128 tokens, and each benchmark requested two repetitions. GLM used 32 CPU threads; the other models used 64.

| Model | Prompt processing, tokens/s | Generation, tokens/s |
|---|---:|---:|
| MiniMax M2.7 | 418.83 ± 24.11 | 15.18 ± 0.18 |
| Qwen 3.5 397B | 249.61 ± 19.78 | 9.37 ± 0.16 |
| DeepSeek-V4-Flash-0731 | 143.54 ± 1.64 | 10.34 ± 0.10 |
| GLM-5.2 | 95.99 ± 3.36 | 5.30 ± 0.00 |
| Kimi K2.6 | 94.23 ± 4.45 | 5.79 ± 0.01 |

The spreads are reproduced as llama-bench printed them. They are not confidence intervals. The [dated record](../results/lab-notebook/2026-08-15-to-16-common-baseline-2tb.md) gives the model files, build, and full conditions.

## What this established

I now had two references: the measured bandwidth of the machine and a common inference baseline. They answered different questions. STREAM showed what the memory system could sustain under regular access. The model benchmarks showed what the full engine achieved with routing, arithmetic, transfers, and synchronization included.

The next question was why prompt processing made such poor use of four GPUs. That required following the software's execution path, rather than assuming the hardware was already at its limit.

## Related records

The [May prediction report](../results/lab-notebook/2026-05-06-moe-throughput-predictions.md) and [October bandwidth analysis](../results/lab-notebook/2026-10-21-bandwidth-and-placement.md) show how the estimates evolved. They keep predictions separate from measured results. All early records remain in the [chronological notebook](../results/lab-notebook/README.md).
