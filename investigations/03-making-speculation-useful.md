# 3. Making speculation useful

[Previous: the prefill bottleneck](02-following-the-prefill-bottleneck.md) · [Chapter index](README.md) · [Next: the fork and Maya](04-from-the-fork-to-maya.md)

## The question

The scheduler work made prompts faster to read, but generation remained slow. Each new token normally depends on the previous one. Could a cheaper predictor draft tokens ahead, allowing the main model to verify several positions together?

This is **speculative decoding**. A draft path proposes tokens. The target model checks them and accepts the usable part. The potential gain comes from doing fewer expensive target steps per output token. Drafting and verification also take time, so accepted tokens are useful only if they save more work than they add.

The notebook tests two approaches: MTP, a model's multi-token prediction path, and DSpark drafting for DeepSeek. Model support, draft loading, and a measured speedup are separate milestones.

## The first practical barrier was getting the draft to run

Some exports and builds could not initialize the required path. Qwen's MTP failed to initialize in the August common-baseline record. An earlier DeepSeek-V4.1 attempt completed target-only generation at 11.2 tokens/s, then failed to allocate about 2464 MiB while loading the drafter.

Those failures do not measure a speculative speedup or slowdown. They show that the tested configuration did not reach the comparison. The [V4.1 record](../results/lab-notebook/undated-deepseek41-dspark.md) keeps that distinction explicit.

## Draft acceptance alone did not predict speed

The [August 7 DeepSeek-V4-Flash-0731 captures](../results/lab-notebook/2026-08-07-deepseek-v4-dspark-acceptance.md) show why I had to measure time as well as acceptance:

| Maximum draft length | Draft acceptance | Generation, tokens/s |
|---|---:|---:|
| 1 | 78.9% | 12.83 |
| 2 | 67.6% | 14.45 |
| 5 | 40.0% | 12.22 |

Each timing covers 256 generated tokens. The logs do not establish an exact build or identical settings outside draft length, so they are not a fully controlled sweep. Even so, they show a useful pattern: the highest acceptance fraction was not the fastest run, and the longest draft was not the fastest either.

A longer draft can yield more accepted tokens per cycle while making that cycle more expensive. The same set also contains a second length-five capture at 11.35 tokens/s with the same acceptance counts. Counts cannot replace elapsed time or repeated trials.

## The common-build tests established useful gains

In [August I repeated speculation on the same build used for the five-model baseline](../results/lab-notebook/2026-08-15-to-16-common-baseline-2tb.md). The ordinary baseline used llama-bench. The speculative runs used a ZFS explanation prompt in llama-cli with greedy decoding. They are separate workloads, which limits the precision of the comparison.

| Model | Ordinary baseline | Speculative generation | Approximate gain over baseline |
|---|---:|---:|---:|
| GLM-5.2 | 5.30 | 6.6 ± 0.3 | 25% |
| DeepSeek-V4-Flash-0731 | 10.34 | 14.1 ± 0.7 | 36% |

Rates are tokens/s. The speculative spreads summarize two repetitions; they are not confidence intervals. Timing varied by about 9%, so close settings could not be ranked from one run. [The recovered commands and repetitions](../results/lab-notebook/2026-08-16-speculation-and-rebench-records.md) preserve the underlying evidence.

These results made speculation worth carrying into later testing. They did not establish a universal draft length or a gain on every kind of answer.

## Maya later provided a clearer within-engine comparison

The October 23 Maya tests used a different engine and newer models. They should not be merged into the August baseline, but their controls answered the same question more directly.

| Model and test | Without speculation | With speculation | Within-test gain |
|---|---:|---:|---:|
| GLM-5.3-Flash, three-answer benchmark | 19.9 | 31.6 | 58.8% |
| DeepSeek-V4.1-Flash-1017, DSpark calibration | 21.2 | 29.3 | 38% |

The GLM run generated three distinct answers of up to 256 tokens. The DeepSeek calibration measured 1152 tokens per mode after warm-up and reported 69% draft acceptance. These are separate tests, not a model-to-model ranking.

GLM's default run also measured 29.4 tokens/s on an earlier pass. Changing its speculative head from two GPUs to one or three reduced the observed rate. Several draft-expert settings were close to the default, but none established a better choice. I kept the working defaults rather than treating one favorable run as a precise optimum.

## What I carried forward

Speculation gave useful generation gains in the successful GLM and DeepSeek tests. I retained it where loading and throughput had been verified. I also kept three checks separate: whether the draft loaded, whether it was used, and whether it improved the full request's timing.

The later [two-GPU GLM-Flash tests](../results/lab-notebook/2026-11-09-glm-flash-reserve-and-speculation.md) confirmed active native MTP. They did not include an MTP-off control, so they add evidence of operation rather than another independent speedup measurement.

The [technical speculation note](../takeaways/speculative-decoding.md) retains the cost model and detailed historical results. The next chapter explains why the work moved from integrating llama.cpp patches to using Maya.
