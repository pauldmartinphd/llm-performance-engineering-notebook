# A performance-testing methodology for LLM inference

This is the procedure I used to find the system's inference limits and test changes. It was developed for hybrid Mixture-of-Experts (MoE) inference, in which the routed experts sit in system RAM and the dense path runs on the GPUs, but the test loop also applies to other systems. Every step below uses numbers from **Galactus** (EPYC 7713, DDR4-2933, 4 × Radeon Pro V620); see the [per-model results](README.md) and the [hardware note](../hardware/galactus/README.md).

The investigation starts with memory bandwidth and an estimate of time per token. Controlled sweeps test that estimate and identify the settings or code paths to inspect. This page uses the July GLM-5.2 investigation as its worked example; the [August common baseline](lab-notebook/12-common-baseline-2tb.md) records the later build and memory population.

> A companion article on the Technicomp Labs blog presents this method as a narrative, with the full patch investigation: [A Scientific Method for Measuring the Limits of Local LLM Inference Speed](https://technicomplabs.io/posts/2026/08/measuring-local-llm-inference-limits/).

---

## 1. Establish the memory-bandwidth limit (STREAM and the RFO correction)

For CPU-resident MoE, memory bandwidth bounds decode: the speed depends on how fast the active experts stream from DRAM. So the first number is not a model benchmark. It is the memory bandwidth.

Run a STREAM thread sweep and apply the **read-for-ownership (RFO) correction**. STREAM undercounts write traffic, because an ordinary store first reads the cache line it will overwrite. Multiply the Scale result by 1.5 and the Add and Triad results by 4/3. Copy needs no correction if it compiled to non-temporal stores.

The correction depends on the generated stores. A corrected Copy result above the theoretical peak indicates that the assumed traffic model needs checking; it is consistent with non-temporal stores, but is not by itself proof of them. On Galactus, all four kernels converge on about **152 GB/s** after correction, which is 81% of the 187.7 GB/s theoretical peak for 8-channel DDR4-2933. Bandwidth reaches its maximum at 16 threads and declines beyond that count.

## 2. Compute the theoretical peak and predict decode

Compute the theoretical DRAM peak (`channels × transfers/s × 8 bytes`) and find what fraction of it you reach. Galactus's measured 152 GB/s is **81% of the 187.7 GB/s peak**, which suggests that the platform is delivering a substantial fraction of its nominal bandwidth. A result near 50% would warrant checking memory topology, clocks, and benchmark behavior before attributing the gap to inference software.

Predict decode before measuring it, with a two-term model:

```
time_per_token ≈ C + (bytes_read_per_token ÷ bandwidth)
```

`C` approximates the remaining per-token cost for a particular model, build, and placement; it must be fitted and checked when those conditions change. `bytes_read_per_token` is the active-expert footprint at your quantization. On Galactus, `C ≈ 90 ms`, and the model predicted three GLM-5.2 configurations at **5.5 / 6.2 / 3.9 t/s** against measured values of **5.53 / 6.01 / 3.87**. Agreement supports the model over the configurations tested. A discrepancy can reveal a missing cost or a configuration error; neither agreement nor disagreement alone establishes the cause.

## 3. The iterative test loop

Change one variable at a time, and treat each change as an experiment with a predicted outcome.

1. Set the baseline: the current configuration, measured correctly (see §5). Record it before changing anything.
2. Change one variable and sweep it across a sensible range. Read the shape of the curve, not only the peak.
3. Move to the next variable. Carry forward the best setting only after you understand why it won.

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

Some settings change the conditions under which another setting is tested. Check those interactions before interpreting a result.

The `-p` option silently clamps `n_ubatch`. Before that interaction was caught, the op_offload tests on Galactus had run at ub 512 rather than the intended ub 8192. Those results described the small-batch regime, where streaming dominated and op_offload reduced throughput. At ub 8192, it produced a substantial gain. The earlier runs could not answer the question they were intended to test.

Placement also changes the execution path. Resident experts placed with `-ot` do not follow the same path as op_offload; on GLM-5.2, the resident path ran 17% slower at pp2048. The scheduler patch affected prefill and left measured decode unchanged. These distinctions matter when choosing a comparator or investigating a regression.

Record these dependencies with the configuration, especially when a requested value differs from the effective value.

## 5. Measurement hygiene

I encountered several tool behaviors that complicated the measurements:

- The `-p N` option clamps `n_ubatch` to N. Set `-ub` explicitly, use a sufficiently large `-p`, and verify the effective micro-batch size.
- In the tested builds, llama-cli required terminal-aware capture: piping it through `tee` breaks its terminal display, and `--log-file` drops the timing lines. Capture with `script -q`, or read the JSON timings from `llama-server`.
- `GGML_SCHED_DEBUG` output appears only with `-v`, because `llama-bench` otherwise installs a null log callback.
- `llama-fit-params` turns off if you pass any of `-ngl`, `-ts`, `-ot`, or `-ncmoe`.
- Check placement and throughput separately. An equalized split histogram and faster prefill are separate facts; `GGML_SCHED_DEBUG=2 … -v | grep '## SPLIT' | sort | uniq -c` confirms whether the histogram moved.

## 6. Record results and the refuted hypotheses

Record every run with its full configuration. Without the flags and build, a throughput figure cannot support a reproducible comparison. A workable schema is:

```
date, experiment, configuration (exact flags + build), metric, value, source
```

Record the failures as well. The [negative-results table](../takeaways/refuted-hypotheses.md) records what was tested and why an approach was set aside. The Galactus investigation tested or ruled out the following for this workload: NUMA imbalance, container overhead, `--poll`, strict CPU affinity, `CPU_REPACK`, transparent huge pages, `-sm row`, pipeline parallelism, ZenDNN, and HIP managed memory. These results help prioritize further testing; their scope is this workload, hardware, and set of builds.

---

## Worked example: the scheduler patch

The baseline was GLM-5.2 prefill at 104.97 t/s (ub 8192). The split histogram exposed the placement problem: it showed 731 of 1,186 GPU splits on one card (ROCm0), concentrating the expert offload there. I tested distributing the offload across all four GPUs. Before the run, the prediction was a null result: distribution alone would not help, because the expert copies were already asynchronous, and only a per-split synchronize, needed to read the routing ids, serialized them. The run confirmed that prediction: 105.71 versus 104.97 t/s, no change. The combined patch also skipped the ids read at prefill-sized batches. It treated all experts as used and copied their weights without waiting for the routing ids, removing that synchronization. The result was 119.36 t/s, a gain of 13.7%, and the histogram equalized to 285/300/294/292. See [the patch note](../patches/prefill/README.md).

The distribution-only result mattered because it separated a placement change from a throughput change. The combined patch provided the measured improvement; its behavior on the later common build and other models remains to be tested.
