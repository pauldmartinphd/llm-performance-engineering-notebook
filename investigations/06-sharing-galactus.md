# 6. Sharing Galactus between two models

[Previous: long prompts](05-tuning-long-prompts.md) · [Chapter index](README.md) · [Next: other machines](07-other-local-machines.md)

## The question

By late October, both GLM-Flash and DeepSeek worked well when they could use all four V620s. I wanted to keep both available without loading a different model each time I switched tasks. How much speed would I lose by giving each model fewer GPUs?

There are two different goals here. Keeping both services resident avoids some switching work. Generating from both at the same time also requires them to share CPU cores and memory bandwidth. My immediate goal was the first. The notebook does not contain a controlled test of overlapping requests to the final two-model setup.

## One GPU per model gave up too much performance

The [November 2 matrix](../results/lab-notebook/2026-11-02-maya-gpu-cache-matrix.md) tested one and four GPUs, with several requested cache budgets. I kept the dense computation on the GPU and changed how much memory could hold routed experts.

The full-allocation comparisons showed a substantial tradeoff:

| Model | GPUs | Built-in mean generation | Fresh prefill at the 64K target |
|---|---:|---:|---:|
| DeepSeek-V4.1-Flash | 4 | 32.1 | 483.4 |
| DeepSeek-V4.1-Flash | 1 | 22.2 | 316.2 |
| GLM-5.3-Flash | 4 | 31.8 | 351.2 |
| GLM-5.3-Flash | 1 | 12.1 | 169.6 |

Rates are tokens/s. The short generation benchmark and long-prefill requests are separate workloads. The full matrix retains all variants and cached repetitions, rather than only these four rows.

A reduced cache budget barely changed fresh prefill in several cases. That did not prove cache size was irrelevant: “reduced” increased the reserved memory rather than enforcing exactly half the cache, and “zero” did not mean zero total GPU memory. The single-GPU reduced GLM case failed while loading, ending the run before its zero-cache case.

The next useful compromise was two GPUs per model.

## Two-GPU DeepSeek remained useful

The November 9 built-in benchmark on GPUs 0 and 1 measured **29.0 tokens/s mean generation**, with 327 tokens/s at 8K prefill. That made the split promising. It preserved useful generation speed while freeing the other pair for GLM.

Requested TP4 did not activate in the later sweep. The engine used TP2 and the runner excluded the rows from TP4 results. I retained the measured two-GPU layout rather than claiming four-way parallelism from a requested flag.

The [later FP16-output prefill test](05-tuning-long-prompts.md#deepseeks-fp16-output-route-was-a-larger-later-improvement) increased the observed prompt-reading rates. It did not change the named Q8 model weights.

## I also tried the full GLM-5.3 model

Before settling on Flash, I tested the much larger full GLM-5.3 UD-Q4_K_XL export on all four cards. The first configuration failed to load. Automatic placement later put almost all of the costly layers on the final two cards, leaving the first pair nearly empty. That run generated at only 3.6–4.3 tokens/s and stalled before completing prefill.

A manual split first failed because it began a part at a layer that shared the previous layer's attention indexer. After moving the boundaries to legal layers, the benchmark completed at **6.7 tokens/s mean generation**, with **47 tokens/s at 8K prefill**.

The [full-model record](../results/lab-notebook/2026-11-09-full-glm53-placement.md) preserves the loading diagnostics and every result. I judged that setup too slow for the intended interactive use and stopped its remaining tuning. This was a decision about the tested implementation and configuration, not a claim that the full model could never run better.

## GLM-Flash improved when I reclaimed reserve memory

Flash on GPUs 2 and 3 began at 18.7 tokens/s mean generation. It used native MTP and a 3072 MB reserve on each card. Reducing the reserve to 1024 MB produced **21.5 tokens/s** in the next built-in run. The 8K prefill rate stayed at 234 tokens/s.

More memory could now hold experts, which is a plausible reason for the gain. With one run per setting, placement and warm-up could also contribute. I kept the smaller reserve because it loaded successfully and gave a favorable result. A speculative-cancellation test was slower, so I did not carry that option forward.

The [reserve and MTP entry](../results/lab-notebook/2026-11-09-glm-flash-reserve-and-speculation.md) gives the three individual answer rates and the log confirming that MTP was active. The [long-prompt tests](05-tuning-long-prompts.md#glm-flash-needed-a-different-chunk-size) then selected 8K effective chunks.

## INT8 KV was the final GLM cache choice

The KV cache stores attention state from earlier tokens. Reducing its precision can save memory. It does not change the precision of the model's stored weights.

I compared FP16 and INT8 KV with the same Q4 model, two GPUs, native MTP, and 8K effective chunks. Across the two repetitions at each prompt length, INT8 showed about **11–13% higher average generation throughput**. Fresh long-prefill rates stayed near **292 tokens/s** at 32K and 64K targets, slightly below the paired FP16 control.

The [KV comparison](../results/lab-notebook/2026-11-09-to-10-glm-flash-kv-cache.md) also shows large variation between short decode samples. I treated the result as support for the operating choice, not a precise universal speedup. I kept an FP16 fallback. No local quality-equivalence test or full-capacity context test was supplied.

## The resulting service

By November 10 I reported both services hosted on Galactus:

| Service | Physical GPUs | Port | Selected settings |
|---|---|---:|---|
| DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL | 0, 1 | 8081 | TP2, DSpark, FP16-output prefill, 32K chunks, 800 MiB ring |
| GLM-5.3-Flash UD-Q4_K_XL | 2, 3 | 8082 | Native MTP, INT8 streaming KV, 8K effective chunks |

Both configs request a 1,048,576-token ceiling and roughly 1 GB of reserve per GPU. That ceiling was not measured end to end. The [deployment record](../results/lab-notebook/2026-11-10-maya-dual-model-production.md) distinguishes the reported operating state, retained experimental configs, and production filenames requested in the conversation.

The result is a practical allocation of the existing hardware. It keeps both large models available and preserves useful individual-model speed. Shared-memory contention under simultaneous generation, very long contexts, and local quality comparisons remain separate experiments.
