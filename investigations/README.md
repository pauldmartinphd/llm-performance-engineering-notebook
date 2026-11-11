# Experiments

We tested hardware limits, prefill scheduling, speculative decoding, inference engines, and dual-model deployment. These accounts connect the results and decisions; the [Lab Notebook](../results/lab-notebook/README.md) contains the dated records.

| Experiment | Question | Result |
|---|---|---|
| [1. Baseline Measurements](01-establishing-a-baseline.md) | What can the hardware do, and which results can we compare? | A measured memory limit and a common five-model baseline |
| [2. Prefill Bottlenecks](02-following-the-prefill-bottleneck.md) | Why were four GPUs so slow at reading a prompt? | A batch-size correction and a measured scheduler improvement |
| [3. Speculative Decoding](03-making-speculation-useful.md) | Can drafting several tokens reduce the cost of generation? | Successful MTP and DSpark tests, with limits on draft length |
| [4. Engine Comparisons](04-from-the-fork-to-maya.md) | Would integrated patches or a different engine improve daily use? | Similar fork/upstream timings, then useful Maya generation rates |
| [5. Long-Prompt Tuning](05-tuning-long-prompts.md) | Which changes reduce the wait before an answer starts? | Model-specific chunk choices and a faster DeepSeek prefill path |
| [6. Dual-Model Deployment](06-sharing-galactus.md) | How much performance must we give up to keep both models available? | Two V620s per model, with GLM-Flash INT8 KV |
| [7. Unified-Memory Systems](07-other-local-machines.md) | What changes when CPU and GPU share memory? | Separate oMLX and Gufo records, with hardware-specific lessons |

The experiments overlap in time. [Findings](../takeaways/general-principles.md), [unsuccessful approaches](../takeaways/refuted-hypotheses.md), and [measurement methods](../results/methodology.md) summarize the supporting evidence.

## Open Questions

The selected service raises three immediate measurement questions: how the precision settings affect answer quality, what happens when both models generate at once, and how performance changes beyond the tested prompt lengths. The current configuration alone cannot answer them.

The older patch work also lacks a matched comparison across the later models and builds. That is unfinished evidence, not a claim that the old patch remains the best next engineering task. Similarly, the alternative Qwen/R9700 layouts and speech/OCR services remain proposals until they have their own runs.

The dated records retain the questions that were open at each stage. A historical plan is not automatically the current work queue.
