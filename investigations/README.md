# Investigation chapters

I want to host large language models on local hardware and understand what limits their speed. The experiments began with basic setup and memory placement. They led to a scheduler patch, tests of speculative decoding, an experimental fork, and a move to Project Maya. Later tests asked how two large models could share the same machine.

These chapters explain that progression. Each starts with a question, follows the tests that mattered, and ends with the decision they supported. The [lab notebook](../results/lab-notebook/README.md) keeps every dated record, including failures, supporting captures, and work that stopped early.

| Chapter | Question | Where the work led |
|---|---|---|
| [1. Establishing a baseline](01-establishing-a-baseline.md) | What can the hardware do, and which results can I compare? | A measured memory limit and a common five-model baseline |
| [2. Following the prefill bottleneck](02-following-the-prefill-bottleneck.md) | Why were four GPUs so slow at reading a prompt? | A batch-size correction and a measured scheduler improvement |
| [3. Making speculation useful](03-making-speculation-useful.md) | Can drafting several tokens reduce the cost of generation? | Successful MTP and DSpark tests, with limits on draft length |
| [4. From the experimental fork to Maya](04-from-the-fork-to-maya.md) | Would integrated patches or a different engine improve daily use? | Similar fork/upstream timings, then useful Maya generation rates |
| [5. Tuning long prompts](05-tuning-long-prompts.md) | Which changes reduce the wait before an answer starts? | Model-specific chunk choices and a faster DeepSeek prefill path |
| [6. Sharing Galactus between two models](06-sharing-galactus.md) | How much performance must I give up to keep both models available? | Two V620s per model, with GLM-Flash INT8 KV |
| [7. Testing other local machines](07-other-local-machines.md) | What changes when CPU and GPU share memory? | Separate oMLX and Gufo records, with hardware-specific lessons |

The chapters are a reading order, not a claim that the work happened in seven clean stages. Several investigations overlap. A failed test can matter to more than one chapter, and later evidence can change how I interpret an earlier result.

For a shorter account, read the [project overview](../README.md) and [important findings](../takeaways/general-principles.md). The [unsuccessful approaches](../takeaways/refuted-hypotheses.md) remain separate. [Measurement terms and methods](../results/methodology.md) explain the metrics used throughout.

## Questions the present evidence does not answer

The selected service raises three immediate measurement questions: how the precision settings affect answer quality, what happens when both models generate at once, and how performance changes beyond the tested prompt lengths. The current configuration alone cannot answer them.

The older patch work also lacks a matched comparison across the later models and builds. That is unfinished evidence, not a claim that the old patch remains the best next engineering task. Similarly, the alternative Qwen/R9700 layouts and speech/OCR services remain proposals until they have their own runs.

The dated records retain the questions that were open at each stage. A historical plan is not automatically the current work queue.
