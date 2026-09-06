# A performance-testing methodology for LLM inference

This is a repeatable procedure for finding a system's real inference limits and improving them. It is written for hybrid Mixture-of-Experts (MoE) inference, in which the routed experts sit in system RAM and the dense path runs on the GPUs, but the loop generalizes to other systems. Every step below uses real numbers from **Galactus** (EPYC 7713, DDR4-2933, 4 × Radeon Pro V620); see the [per-model results](README.md) and the [hardware note](../hardware/galactus/README.md).

The method in one sentence: establish the physical limit, predict what the software should reach, change one variable at a time, account for every number, and keep a record of the refuted hypotheses.

> A companion article on the Technicomp Labs blog presents this method as a narrative, with the full patch investigation: [A Scientific Method for Measuring the Limits of Local LLM Inference Speed](https://technicomplabs.io/posts/2026/08/measuring-local-llm-inference-limits/).

---

## 1. Establish the memory-bandwidth limit (STREAM and the RFO correction)

For CPU-resident MoE, memory bandwidth bounds decode: the speed depends on how fast the active experts stream from DRAM. So the first number is not a model benchmark. It is the memory bandwidth.

Run a STREAM thread sweep and apply the **read-for-ownership (RFO) correction**. STREAM undercounts write traffic, because an ordinary store first reads the cache line it will overwrite. Multiply the Scale result by 1.5 and the Add and Triad results by 4/3. Copy needs no correction if it compiled to non-temporal stores.

To check the correction, confirm that the uncorrected Copy rate plus the RFO correction does not exceed the theoretical limit; if it does, Copy used non-temporal stores and needs no correction. On Galactus, all four kernels converge on about **152 GB/s** after correction, which is 81% of the 187.7 GB/s theoretical peak for 8-channel DDR4-2933. Bandwidth reaches its maximum at 16 threads and declines beyond that count.

## 2. Compute the theoretical peak and predict decode

Compute the theoretical DRAM peak (`channels × transfers/s × 8 bytes`) and find what fraction of it you reach. Galactus's measured 152 GB/s is **81% of the 187.7 GB/s peak**, which is a healthy platform and the number that sets the limit. A result near 50% of theoretical would mean the memory topology needs fixing, not the software.

Predict decode before measuring it, with a two-term model:

```
time_per_token ≈ C + (bytes_read_per_token ÷ bandwidth)
```

`C` is a GPU-side constant; measure it once. `bytes_read_per_token` is the active-expert footprint at your quantization. On Galactus, `C ≈ 90 ms`, and the model predicted three GLM-5.2 configurations at **5.5 / 6.2 / 3.9 t/s** against measured values of **5.53 / 6.01 / 3.87**. When a prediction and a measurement agree, you understand the system. When they diverge, the gap marks something worth investigating.

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

The thread sweep peaks at half the physical cores and then collapses onto the SMT siblings. The ubatch ladder rises monotonically and is worth a factor of 4. Neither result is predictable, so measure both by sweeping.

## 4. The dependency tree of changes

Changes are not independent. Before trusting a result, find where its variable sits in the tree.

Some changes corrupt every result that depends on them. The `-p` option silently clamps `n_ubatch`: every op_offload result on Galactus, before we caught this, had run at ub 512 rather than the ub 8192 intended, and one bad default invalidated a day of numbers.

Some changes help only in one regime. `op_offload` is a net loss at small ubatch, where the streaming term dominates, and a large win at ub 8192. Test it where it can win.

Some changes affect only one phase. The scheduler patch below raises prefill by 13.7% and leaves decode unchanged, so a decode regression after it would mean a mistake, not a trade-off.

Some paths look equivalent but are not. Resident experts placed with `-ot` and `op_offload` use different code paths; on GLM-5.2 the resident path ran 17% slower at pp2048.

Draw the tree so that an earlier setting cannot silently corrupt a later result.

## 5. Measurement hygiene

The tools have several traps, each of which cost us time:

- The `-p N` option clamps `n_ubatch` to N, so always set `-ub` explicitly.
- `llama-cli` is not a measurement tool: piping it through `tee` breaks its terminal display, and `--log-file` drops the timing lines. Capture with `script -q`, or read the JSON timings from `llama-server`.
- `GGML_SCHED_DEBUG` output appears only with `-v`, because `llama-bench` otherwise installs a null log callback.
- `llama-fit-params` turns off if you pass any of `-ngl`, `-ts`, `-ot`, or `-ncmoe`.
- Confirm the mechanism before you trust the number. An equalized split histogram and a faster prefill are separate facts; prove that the histogram moved (`GGML_SCHED_DEBUG=2 … -v | grep '## SPLIT' | sort | uniq -c`) before you believe the throughput.

## 6. Record results and the refuted hypotheses

Record every run with its full configuration. A t/s figure without its flags and build is noise. A workable schema is:

```
date, experiment, configuration (exact flags + build), metric, value, source
```

Record the failures as well. The refuted-hypotheses table is the most useful record a performance investigation produces. On Galactus, we measured and refuted all of the following for this workload: NUMA imbalance, container overhead, `--poll`, strict CPU affinity, `CPU_REPACK`, transparent huge pages, `-sm row`, pipeline parallelism, ZenDNN, and HIP managed memory. Each negative result is a path the next engineer need not take.

---

## Worked example: the scheduler patch

This is the whole loop in one finding. The baseline was GLM-5.2 prefill at 104.97 t/s (ub 8192). We found the bottleneck by mechanism: the split histogram showed 731 of 1,186 GPU offload splits on one card (ROCm0), with three cards idle. The hypothesis was to distribute the offload across all four GPUs. We pre-registered the null result: distribution alone would not help, because the expert copies were already asynchronous, and only a per-split synchronize, needed to read the routing ids, serialized them. We predicted this before the run and confirmed it: 105.71 versus 104.97 t/s, no change. The real fix was to skip the ids read at prefill-sized batches, where every expert is used in any case, which removes the serializing synchronize. The result was 119.36 t/s, a gain of 13.7%, and the histogram equalized to 285/300/294/292. See [the patch note](../patches/prefill/README.md).

That is the method at work: predict, measure, account for the gap, and let a negative result point to the real cause.
