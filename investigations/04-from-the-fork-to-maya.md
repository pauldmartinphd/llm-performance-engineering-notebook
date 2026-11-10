# 4. From the experimental fork to Maya

[Previous: speculation](03-making-speculation-useful.md) · [Chapter index](README.md) · [Next: long prompts](05-tuning-long-prompts.md)

## The question

The July patch improved one part of inference. Later models needed more support, and community work offered more AMD optimizations. Could I combine those changes into a useful local engine?

I called the experimental fork **llamda**. It brought model support, V620 patches, GLM MTP work, and my scheduler changes into one place. The goal was practical: run large models well on the hardware I owned. The test was whether that integration improved measured performance and daily use.

## Integration was useful work, but it did not establish a faster engine

An integrated build can load a model that an older build cannot. That is a real capability change. It does not establish that the engine is faster than a current upstream build that supports the same model.

The [October 22 comparison](../results/lab-notebook/2026-10-22-glm53-build-comparison.md) made that distinction clear. Both benchmarks used the same named GLM-5.3-Flash UD-Q4_K_XL export, 32 CPU threads, CPU-resident routed experts, and an 8192-token prompt.

| Printed build | Prefill at 8192 tokens | Generation at 256 tokens |
|---|---:|---:|
| llama.cpp `184f65360` | 284.36 ± 16.33 | 10.16 ± 0.04 |
| llamda `ca3b1fc30` | 284.20 ± 16.42 | 10.09 ± 0.02 |

Rates are tokens/s. The results were essentially the same under these conditions. The builds differed, so this was not a controlled test of my scheduler patch alone. It also did not support announcing the combined fork as a proven performance improvement.

Some CLI commands created a second problem. Both `llama-cli` and `llamda-cli` printed the same upstream build banner. The command name did not establish which implementation or shared libraries actually ran. I could retain their timings, but I could not use those labels as a sound fork-versus-upstream comparison.

## Placement mattered more than a build name

The [earlier GLM-Flash captures](../results/lab-notebook/undated-glm53-flash-placement.md) measured about 19.1–19.4 generated tokens/s with automatic placement. Forcing the routed experts into system RAM produced about 13.7–14.4 without an explicit cache allocation. The effective tensor placement was not captured, so I could not identify the exact reason for the difference.

Larger requested expert caches were not consistently faster. Some configurations failed to allocate memory. The [80 GiB cache record](../results/lab-notebook/2026-10-29-glm53-expert-cache.md) also lacked a matching cache-disabled control. Those tests preserved useful questions about residency and execution, but did not establish a general cache-size rule.

The broader lesson was about the complete forward pass. Faster isolated kernels and more cached experts do not automatically remove waits between CPU work, GPU work, transfers, and draft verification. The [fork retrospective](../results/lab-notebook/2026-10-29-llamda-experimental-fork-retrospective.md) records that analysis, the source credits, and the work that remained untested.

## Maya changed the practical result

On October 23 I tested Project Maya v1.1.24 on Galactus. It produced useful interactive generation rates with models that exceeded GPU memory:

| Model | Recorded generation rate | Conditions |
|---|---:|---|
| GLM-5.3-Flash UD-Q4_K_XL | 31.6 tokens/s | Four V620s; native speculative path; three-answer benchmark |
| DeepSeek-V4.1-Flash-1017, identified as UD-Q8_K_XL | 29.3 tokens/s | Four V620s; DSpark calibration |

The [GLM A/B test and DeepSeek calibration](../results/lab-notebook/2026-10-23-maya-glm53-speculation.md) established the benefit of speculation within Maya. They also gave me a working path for daily use.

This did not mean Maya won every comparison. Its dedicated GLM 8K prefill measured about 240–242 tokens/s, below the roughly 284 in the earlier llama-bench comparison. The engines also used different prompts, context allocations, placement, and speculative paths. Dividing one generation rate by another would hide those differences.

The practical reason to move was that Maya's tested configurations generated answers at useful rates. It was not a claim that every component of Maya was faster or that the full engine difference had been isolated.

## The decision

I withdrew the performance announcement for llamda and kept it as an experimental record. The integration work, failed tests, and July patch still belong in the notebook. Their value does not depend on presenting the combined fork as a success it had not demonstrated.

I then prioritized Maya for the large-model service. The [October 28 status review](../results/lab-notebook/2026-10-28-maya-priority-and-upstream-status.md) separates that choice from unfinished upstream work and proposed model alternatives. From there, the main questions became how to reduce long-prompt waits and how to allocate four GPUs between two models.

## Daily use added another kind of evidence

The October 24 dashboard showed 90 requests and 86,075 output tokens, with a reported aggregate generation rate of 28.0 tokens/s. Its last request measured 31.6 tokens/s. That supported a picture of ongoing use beyond a short calibration.

A dashboard combines requests with different prompts and cache histories. It is therefore useful operational evidence, but not a replacement for a controlled benchmark. The [snapshot record](../results/deepseek-v4.1-flash.md#october-24-dashboard-from-ongoing-use) keeps the two kinds of evidence separate.
