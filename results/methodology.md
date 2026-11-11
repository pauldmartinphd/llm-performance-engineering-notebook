# How we measure local inference

A useful benchmark answers a specific question. We separate the work of reading a prompt from the work of generating an answer, record what the engine actually ran, and keep the conditions beside the result. This lets a later reader tell whether two numbers describe a real comparison.

The [lab notebook](lab-notebook/README.md) contains the experiments. The [chapters](../investigations/README.md) explain the decisions. This page defines the terms and checks used to interpret them.

## What the measurements mean

| Term | Meaning | What to check |
|---|---|---|
| Prefill, or prompt processing | Reading input and building the state needed to answer | Prompt length, new versus cached tokens, chunk size, and effective batch |
| Decode, or generation | Producing output tokens after the prompt | Prompt content, context already present, output length, sampling, and speculation |
| Time to first token | The wait before output begins | Includes work beyond the engine's prefill time; do not substitute total request time |
| Single-request rate | Speed for one request | Does not establish aggregate throughput across simultaneous requests |
| Aggregate rate | Total output across requests | Does not establish each user's latency |
| Configured context | The maximum history the setup permits | Is not evidence that a prompt of that length ran |

For bespoke short CLI prompts, we report generation throughput only. Prefill comparisons use dedicated tests with an explicit prompt length.

A token is a piece of text, often a word fragment, word, or punctuation. `pp8192` labels an 8192-token prefill test; `tg128` labels a 128-token generation test. Chat wrappers and model tokenizers can change the actual count, so target lengths and observed counts are recorded separately.

## Memory and precision are separate controls

**System RAM** holds model data on the CPU side. **VRAM** is memory on a discrete GPU. **Unified memory** is shared by CPU and GPU, so the reported pools cannot automatically be added together.

A **weight quantization** such as Q4 or Q8 changes how model weights are stored. A **KV cache** holds attention state from previous tokens; changing its precision is a different experiment. A **computation precision** setting can change intermediate rounding without changing either the weights or the cache format.

A **batch** groups work so the engine can reuse weights and run larger matrix operations. A **microbatch** is a smaller execution group within it. A Maya prompt **chunk** is another engine-specific grouping and is not interchangeable with every llama.cpp batch control. **Workspace** is temporary memory used during computation.

**Tensor parallelism**, or TP, divides calculations across devices. Layer partitioning assigns different parts of the model to different devices. Both can use several GPUs, but they do not create the same execution schedule. We record the observed layout rather than infer it from a requested flag.

## The comparison loop

We start with a question and a baseline, then change the setting intended to answer it. Before interpreting the rate, we check that the requested path activated. If the engine silently changes a batch, chunk, or parallel layout, the row describes the effective setting.

We keep individual repetitions and any closing baseline. A small difference is less persuasive when the same setup varies by more than that difference. Where a test changes several things at once, we describe the combined result and leave their separate effects unresolved.

We also keep distinct outcomes distinct. Loading successfully is not a throughput result. A failed startup is not a slow completed run. An inactive option is not a benchmark of that feature. A speed test without an accuracy evaluation does not establish quality equivalence.

## Fresh prompts and reused state

A repeated long request may reuse nearly its whole prompt. The engine then processes only a small suffix. We do not compare that suffix rate with a fresh full-prompt rate. The November 2 Maya matrix illustrates this directly: repeated long requests sometimes processed only one new token.

Some historical runners excluded any cached row; others allowed a small amount of reuse. Each entry states its rule. A printed `ok` status means the runner's checks passed, not that an independent audit proved zero reuse or numerical accuracy.

## Make the experimental account explicit

Each experiment entry states its purpose and reasoning, experimental design, results, and conclusions before the detailed record. This is the scientific method expressed as an account of the work, rather than a set of literal questions.

The purpose explains the tested mechanism where it is documented. The design identifies what changed and what serves as the control. The results report observations without turning them into explanations. The conclusions then connect those observations to the original purpose and state the remaining uncertainty.

Exploratory sweeps and baseline measurements need not have a directional prediction. If the source does not preserve one, we do not create one retrospectively. Several measurements can answer one experimental purpose, while a failed startup may leave that purpose unresolved.

## What belongs with a result

A useful record identifies the model/export, machine, engine revision, effective configuration, workload, cache state, repetitions, metric, and source. Where the source lacks one of these, we state the resulting limit on the comparison rather than fill the gap from a filename or a later configuration.

Reported spreads retain their original meaning. Printed llama-bench deviations and spreads from two CLI runs are not automatically confidence intervals. Different prompts, versions, quantizations, and memory populations remain separate comparisons.

## Worked method: July GLM-5.2

The following sections preserve the detailed method developed during the July Galactus investigation. The [August common baseline](lab-notebook/2026-08-15-to-16-common-baseline-2tb.md) uses the later build and 2 TB memory population. These historical commands and fitted constants are examples to check, not universal settings.

A companion article presents the patch investigation as a narrative: [A Scientific Method for Measuring the Limits of Local LLM Inference Speed](https://technicomplabs.io/posts/2026/08/measuring-local-llm-inference-limits/).

## 1. Establish the memory-bandwidth limit (STREAM and the RFO correction)

For CPU-resident MoE, memory bandwidth bounds decode: the speed depends on how fast the active experts stream from DRAM. So the first number is not a model benchmark. It is the memory bandwidth.

We run a STREAM thread sweep and apply the **read-for-ownership (RFO) correction**. STREAM undercounts write traffic, because an ordinary store first reads the cache line it will overwrite. We multiply the Scale result by 1.5 and the Add and Triad results by 4/3. Copy needs no correction if it compiled to non-temporal stores.

The correction depends on the generated stores. A corrected Copy result above the theoretical peak indicates that the assumed traffic model needs checking; it is consistent with non-temporal stores, but is not by itself proof of them. On Galactus, all four kernels converge on about **152 GB/s** after correction, which is 81% of the 187.7 GB/s theoretical peak for 8-channel DDR4-2933. Bandwidth reaches its maximum at 16 threads and declines beyond that count.

## 2. Compute the theoretical peak and predict decode

Compute the theoretical DRAM peak (`channels × transfers/s × 8 bytes`) and find what fraction of it we reach. Galactus's measured 152 GB/s is **81% of the 187.7 GB/s peak**, which suggests that the platform is delivering a substantial fraction of its nominal bandwidth. A result near 50% would warrant checking memory topology, clocks, and benchmark behavior before attributing the gap to inference software.

Predict decode before measuring it, with a two-term model:

```
time_per_token ≈ C + (bytes_read_per_token ÷ bandwidth)
```

`C` approximates the remaining per-token cost for a particular model, build, and placement; it must be fitted and checked when those conditions change. `bytes_read_per_token` is the active-expert footprint at our quantization. On Galactus, `C ≈ 90 ms`, and the model predicted three GLM-5.2 configurations at **5.5 / 6.2 / 3.9 t/s** against measured values of **5.53 / 6.01 / 3.87**. Agreement supports the model over the configurations tested. A discrepancy can reveal a missing cost or a configuration error; neither agreement nor disagreement alone establishes the cause.

## 3. The iterative test loop

Change one variable at a time, and treat each change as an experiment with a predicted outcome.

1. We set the baseline: the current configuration, measured correctly (see §5). We record it before changing anything.
2. Change one variable and sweep it across a sensible range. Read the shape of the curve, not only the peak.
3. Move to the next variable. Carry forward the best setting only after we understand why it won.

Two Galactus sweeps show the shape:

| Threads | GLM-5.2 decode (t/s) |
|---|---|
| 24–32 | ~5.5 (peak) |
| 96 | 2.76 |
| 128 | 1.29 (SMT collapse) |

| n_ubatch | GLM-5.2 pp8192 (t/s) |
|---|---|
| 512 | 25.90 |
| 2048 | 62.64 |
| 8192 | 104.97 |

The thread sweep peaks at half the physical cores and then collapses onto the SMT siblings. The ubatch ladder rises monotonically and is worth a factor of 4. The sweeps establish where those trends hold and where a setting stops helping.

## 4. Check interactions between settings

Some settings change the conditions under which another setting is tested. We check those interactions before interpreting a result.

The `-p` option silently clamps `n_ubatch`. Before that interaction was caught, the op_offload tests on Galactus had run at ub 512 rather than the intended ub 8192. Those results described the small-batch regime, where streaming dominated and op_offload reduced throughput. At ub 8192, it produced a substantial gain. The earlier runs could not answer the question they were intended to test.

Placement also changes the execution path. Resident experts placed with `-ot` do not follow the same path as op_offload; on GLM-5.2, the resident path ran 17% slower at pp2048. The scheduler patch affected prefill and left measured decode unchanged. These distinctions matter when choosing a comparator or investigating a regression.

We record these dependencies with the configuration, especially when a requested value differs from the effective value.

## 5. Measurement hygiene

We encountered several tool behaviors that complicated the measurements:

- The `-p N` option clamps `n_ubatch` to N. Set `-ub` explicitly, use a sufficiently large `-p`, and verify the effective micro-batch size.
- In the tested builds, llama-cli required terminal-aware capture: piping it through `tee` breaks its terminal display, and `--log-file` drops the timing lines. Capture with `script -q`, or read the JSON timings from `llama-server`.
- `GGML_SCHED_DEBUG` output appears only with `-v`, because `llama-bench` otherwise installs a null log callback.
- `llama-fit-params` turns off if we pass any of `-ngl`, `-ts`, `-ot`, or `-ncmoe`.
- Check placement and throughput separately. An equalized split histogram and faster prefill are separate facts; `GGML_SCHED_DEBUG=2 … -v | grep '## SPLIT' | sort | uniq -c` confirms whether the histogram moved.

## 6. Record results and the refuted hypotheses

We record every run with its full configuration. Without the flags and build, a throughput figure cannot support a reproducible comparison. A workable schema is:

```
date, experiment, configuration (exact flags + build), metric, value, source
```

We record the failures as well. The [negative-results table](../takeaways/refuted-hypotheses.md) records what was tested and why an approach was set aside. The Galactus investigation tested or ruled out the following for this workload: NUMA imbalance, container overhead, `--poll`, strict CPU affinity, `CPU_REPACK`, transparent huge pages, `-sm row`, pipeline parallelism, ZenDNN, and HIP managed memory. These results help prioritize further testing; their scope is this workload, hardware, and set of builds.

---

## Worked example: the scheduler patch

The baseline was GLM-5.2 prefill at 104.97 t/s (ub 8192). The split histogram exposed the placement problem: it showed 731 of 1,186 GPU splits on one card (ROCm0), concentrating the expert offload there. We tested distributing the offload across all four GPUs. Before the run, the prediction was a null result: distribution alone would not help, because the expert copies were already asynchronous, and only a per-split synchronize, needed to read the routing ids, serialized them. The run confirmed that prediction: 105.71 versus 104.97 t/s, no change. The combined patch also skipped the ids read at prefill-sized batches. It treated all experts as used and copied their weights without waiting for the routing ids, removing that synchronization. The result was 119.36 t/s, a gain of 13.7%, and the histogram equalized to 285/300/294/292. See [the patch note](../patches/prefill/README.md).

The distribution-only result mattered because it separated a placement change from a throughput change. The combined patch provided the measured improvement; its behavior on the later common build and other models remains to be tested.
