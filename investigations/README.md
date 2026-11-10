# Investigation chapters

These chapters follow your work from the first hardware tests to the two-model Maya setup. Each explains the tests, results, and decisions. The [dated notebook](../results/lab-notebook/README.md) keeps the full detail.

| Chapter | Focus | Where the work led |
|---|---|---|
| [1. Establishing a baseline](01-establishing-a-baseline.md) | Hardware limits and a common benchmark | A memory-bandwidth reference and five-model baseline |
| [2. Following the prefill bottleneck](02-following-the-prefill-bottleneck.md) | Batches, placement, and waits | A corrected test and a useful scheduler patch |
| [3. Making speculation useful](03-making-speculation-useful.md) | Drafting and verification cost | Working MTP and DSpark configurations |
| [4. From the experimental fork to Maya](04-from-the-fork-to-maya.md) | Engine comparisons | Similar fork/upstream timings, then useful Maya generation |
| [5. Tuning long prompts](05-tuning-long-prompts.md) | Prompt-processing speed | Different chunk choices for DeepSeek and GLM |
| [6. Sharing Galactus between two models](06-sharing-galactus.md) | GPU and memory budgets | Two V620s per model, with INT8 KV for GLM-Flash |
| [7. Testing other local machines](07-other-local-machines.md) | Unified memory | Separate oMLX and Gufo tests |

The work overlapped; it did not happen in seven tidy stages. The chapter order gives you a path through it without having to follow every date.

## Further tests

The current setup leaves three clear gaps: quality comparisons for the precision changes, performance while both models generate, and tests beyond the measured prompt lengths. The older patch also lacks matched tests across the later models and builds. Other Qwen/R9700 layouts and speech/OCR services remain plans.

[Project overview](../README.md) · [Important findings](../takeaways/general-principles.md) · [Unsuccessful approaches](../takeaways/refuted-hypotheses.md) · [Measurement terms](../results/methodology.md)
