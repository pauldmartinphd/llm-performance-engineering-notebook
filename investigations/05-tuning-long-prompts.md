# 5. Long-Prompt Tuning

[Previous: the fork and Maya](04-from-the-fork-to-maya.md) · [Experiments](README.md) · [Next: sharing Galactus](06-sharing-galactus.md)

## Objective

Fast generation does not make a long document fast to read. A model may produce an answer at 30 tokens/s while spending minutes processing the input first. We wanted to reduce that wait without assuming that a setting from one engine would work in another.

The main variables were prompt chunks, transfer buffers, temporary GPU workspace, and numerical precision. A **chunk** is a group of prompt tokens processed together. A transfer **ring** holds data while it moves between memory and the GPU. Larger buffers can help reuse and overlap, but they also consume memory that could hold model data.

## DeepSeek Chunk and Ring Sizes

After recovering from a later-release stall, we used Maya v1.1.30 for the [October 28 sweep](../results/lab-notebook/2026-10-28-maya-prefill-and-regression.md). It tested three chunk sizes against three ring sizes, plus a repeated baseline. Ten starts produced 60 valid measurements across 8K, 32K, and 64K prompt targets.

| Configuration | 8K prefill | 32K prefill | 64K prefill |
|---|---:|---:|---:|
| Initial baseline: 8K chunk, 400 MiB ring | 271.2 | 406.1 | 434.0 |
| Selected: 32K chunk, 800 MiB ring | 266.0 | 438.8 | 480.0 |
| Closing baseline: 8K chunk, 400 MiB ring | 270.6 | 405.4 | 433.7 |

Rates are tokens/s; each cell is the median of two requests. The selected combination improved 32K prefill by about 8% and 64K by about 11%, at a small cost on the 8K prompt. At 64K, the rounded rates imply about 14 seconds less prefill time. The closing baseline stayed near the opening one, which made the comparison more useful.

The gain depended on the combination. A larger ring alone did not make the 8K chunk faster at 64K. We carried forward the 32K chunk and 800 MiB ring, then tested parallel layout and workspace separately.

## Additional Settings

The second October sweep stopped after 15 completed cases. TP1 was slower. Requested TP4 fell back to TP2, so those rows did not measure TP4. Workspace changes produced differences too small to justify replacing the automatic choice.

Later production requests confirmed roughly the same long-prefill range. A fresh 65,568-token request measured 488.6 tokens/s on October 29. Two October 30 passes measured 469.6 and 485.6. A November 2 replay measured 480.9. These observations support the working configuration, but changing releases and service state prevent a precise release-to-release speed comparison.

The [November 5 ten-variant sweep](../results/lab-notebook/2026-11-05-maya-v1154-prefill-and-peer-ar.md) tested automatic chunks, flow options, expert MMQ, and peer all-reduce. Defaults reached about 479.4 tokens/s at 64K. Automatic chunk selection was slower. Neither requested peer path activated in the completed runs; both used host transport.

A separate larger-reserve peer test failed during startup. It completed no transport samples and produced no throughput result. That is different from a successful host fallback. The [complete sweep](../results/lab-notebook/2026-11-05-maya-v1154-prefill-and-peer-ar.md#decode-results-from-the-same-sweep) includes all 60 prefill measurements, 20 decode cases, configs, and logs.

## GLM-Flash Chunk Size

On November 9 we tried the large-chunk approach with GLM-Flash on two V620s. The first tests requested 16K or 32K chunks, but the logs showed an effective size of 8K. The 4096 MB workspace budget could not support the requested larger allocation.

Increasing the budget to 8192 MB allowed a common 16K chunk across the two devices. That made the larger-chunk experiment real, but did not make it faster:

| Target prompt | 8K effective chunks | 16K effective chunks |
|---|---:|---:|
| 8K | 225.1 | 225.7 |
| 32K | 297.5 | 233.9 |
| 64K | 294.0 | 231.9 |

Rates are medians in tokens/s, rounded to one decimal. The larger chunks lost about 21% at the two longer prompts. The [full record](../results/lab-notebook/2026-11-09-glm-flash-prefill-chunks.md) retains both repetitions, effective allocations, and the failed first benchmark runner.

We kept 8K effective chunks for GLM. The tests did not isolate which internal cost caused the loss, but they were enough to reject copying DeepSeek's chunk size directly.

## DeepSeek FP16-Output Prefill

The November 9 DeepSeek test enabled a faster RDNA 2 route for dense prompt projections. It keeps the inputs and outputs of those matrix operations in FP16, then widens the outputs to FP32. This changes rounding. It is separate from the model's Q8 weight quantization and from its attention cache.

The optimized two-GPU run completed 12 valid measurements:

| Prompt target | Optimized median prefill, tokens/s |
|---|---:|
| 8K | 389.55 |
| 16K | 501.95 |
| 32K | 589.70 |

The original baseline used v1.1.54. The optimized run used v1.1.57 with a different binary and corpus. An intervening TP4-request run used the same later binary and corpus, but actually executed TP2 without FP16 output. Against those actual TP2 observations, the optimized medians were about 23%, 27%, and 30% higher.

That is useful timing evidence, with an imperfect control: the same-version comparison comes from a fallback configuration and the runs were not interleaved. The log reports zero nonfinite FP16 outputs. That checks overflow, not answer quality. No local held-out quality evaluation accompanied the speed test. The [dated entry](../results/lab-notebook/2026-11-09-deepseek-fp16-prefill.md) preserves these limits and every measurement.

## Measurement Limits

A repeated prompt can reuse nearly all of its stored state. In the November 2 matrix, some “32K” repeats processed only one new token. Their low printed prefill rates measured that tiny suffix and its overhead, not a fresh 32K read. We therefore track new tokens and cached tokens separately.

A configured context ceiling is also different from a measured prompt. Setting one million tokens does not test a one-million-token request. Most controlled long tests in this investigation stopped at 64K; the optimized DeepSeek sweep stopped at 32K.

The practical result was two different prefill choices: 32K chunks and an 800 MiB ring for DeepSeek, and 8K effective chunks for GLM-Flash. The next chapter explains how those choices fit into a two-model service.
