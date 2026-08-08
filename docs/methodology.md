# A performance-testing methodology for LLM inference

A repeatable procedure for finding a system's real inference limits and moving them, without fooling yourself. It is written for hybrid Mixture-of-Experts (MoE) inference — routed experts in system RAM, dense path on GPUs — but the loop generalizes. Every step is demonstrated with real numbers from **Galactus** (EPYC 7713, DDR4-2933, 4 × Radeon Pro V620); see the [per-model results](results-and-takeaways/) and the [platform note](platform-and-method.md).

The one-sentence version: **establish the physical ceiling, predict what the software should reach, then change one variable at a time and explain every number — keeping the refuted hypotheses.**

> A companion article on the Technicomp Labs blog presents this method as a narrative, with the full patch investigation: [A Scientific Method for Measuring the Limits of Local LLM Inference Speed](https://technicomplabs.io/posts/2026/08/measuring-local-llm-inference-limits/).

---

## 1. Establish the memory-bandwidth ceiling (STREAM + RFO)

For CPU-resident MoE, decode is bounded by how fast the active experts stream from DRAM. So the first number is not a model benchmark — it is the memory bandwidth.

Run a STREAM thread sweep and apply the **read-for-ownership (RFO) correction**: STREAM undercounts write traffic because ordinary stores first read the cache line they overwrite. Correct Scale ×1.5 and Add/Triad ×4/3; Copy needs no correction *if* it compiled to non-temporal stores.

- **Sanity check:** uncorrected-Copy + RFO must not exceed the theoretical ceiling. If it does, Copy used non-temporal stores and needs no correction.
- **Galactus:** all four kernels converge on **~152 GB/s** after correction — 81% of the 187.7 GB/s theoretical peak for 8-channel DDR4-2933. Bandwidth saturates at 16 threads and *declines* beyond.

## 2. Compute the theoretical peak, then predict decode

Compute the theoretical DRAM peak (`channels × transfers/s × 8 bytes`) and check what fraction you reach. Galactus's measured ~152 GB/s is **81% of a ~187.7 GB/s ceiling** — a healthy platform, and the number that sets the roof. Getting only ~50% of theoretical would mean "fix the memory topology," not "tune the software."

Now predict decode *before* measuring it, with a two-term model:

```
time_per_token ≈ C + (bytes_read_per_token ÷ bandwidth)
```

`C` is a GPU-side constant (measure it once); `bytes_read_per_token` is the active-expert footprint at your quant. On Galactus, `C ≈ 90 ms`, and the model predicted three independent GLM-5.2 configurations at **5.5 / 6.2 / 3.9 t/s** against measured **5.53 / 6.01 / 3.87**. When a prediction and a measurement agree, you understand the system; when they diverge, you have found something worth chasing.

## 3. The iterative test loop

One variable at a time, each an experiment with a predicted outcome:

1. **Baseline** — the current config, measured *correctly* (see §5). Record it before touching anything.
2. **Change one variable** and **sweep it** across a sensible range. Read the shape, not just the peak.
3. **Move to the next variable.** Carry forward the best setting only when you understand *why* it won.

Two Galactus sweeps that show the shape:

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

The thread sweep peaks at half the physical cores and collapses into SMT siblings; the ubatch ladder is monotonic and worth 4×. Neither is guessable — both must be swept.

## 4. The dependency tree of changes

Changes are not independent. Before trusting a result, know where the variable sits in the tree:

- **Some changes poison everything upstream of them.** `-p` silently clamps `n_ubatch`: every "op_offload" result on Galactus before this was caught had secretly run at ub 512, not the ub 8192 intended. One bad default invalidated a day of numbers.
- **Some changes only help in a regime.** `op_offload` is a *net loss* at small ubatch (the streaming term dominates) and a large win at ub 8192. Test it where it can win.
- **Some changes touch only one phase.** The scheduler patch (below) lifts prefill +13.7% and leaves decode unchanged — so decode regressions after it would mean a mistake, not a tradeoff.
- **Some paths look equivalent but are not.** Resident experts via `-ot` placement and `op_offload` are different code paths; on GLM-5.2 the resident path was 17% *slower* at pp2048.

Draw the tree so a later result cannot be silently confounded by an earlier setting.

## 5. Measurement hygiene (how not to fool yourself)

Properties of the tools, learned the hard way:

- **`-p N` clamps `n_ubatch` to N.** Set `-ub` explicitly, always.
- **`llama-cli` is not a measurement tool.** `| tee` breaks its TTY; `--log-file` drops the timing lines. Use `script -q` or `llama-server`'s JSON timings.
- **`GGML_SCHED_DEBUG` needs `-v`** — `llama-bench` installs a null log callback otherwise.
- **`llama-fit-params` is disabled** by any of `-ngl` / `-ts` / `-ot` / `-ncmoe`.
- **Confirm the mechanism before the number.** An equalized split histogram and a faster prefill are separate facts — prove the histogram moved (`GGML_SCHED_DEBUG=2 … -v | grep '## SPLIT' | sort | uniq -c`) before you believe the throughput.

## 6. Record results — and the refuted hypotheses

Record every run with its full configuration; a t/s figure without its flags and build is noise. A workable schema:

```
date, experiment, configuration (exact flags + build), metric, value, source
```

And record the **dead ends**. The refuted-hypotheses table is the most time-saving artifact a performance investigation produces. On Galactus, all of these were measured and refuted for this workload: NUMA imbalance, container overhead, `--poll`, strict CPU affinity, `CPU_REPACK`, transparent hugepages, `-sm row`, pipeline parallelism, ZenDNN, and HIP managed memory. Each "no" is a road someone else does not have to drive down.

---

## Worked example: the scheduler patch

The whole loop in one finding. **Baseline:** GLM-5.2 prefill at 104.97 t/s (ub 8192). **Bottleneck, found by mechanism:** the split histogram showed 731 of 1,186 GPU offload splits piled onto one card (ROCm0), three cards idle. **Hypothesis:** distribute the offload across all four GPUs. **Pre-registered null:** distribution *alone* would not help, because the expert copies were already asynchronous and only a per-split `synchronize` (to read the routing ids) serialized them — predicted before the run, and confirmed: 105.71 vs 104.97, no change. **The real fix:** skip the ids read at prefill-sized batches (where every expert is used anyway), removing the serializing sync. **Result:** 119.36 t/s (+13.7%), the histogram equalized to 285/300/294/292. See [patch/](../patch/README.md).

That is the method working: predict, measure, explain the gap, and let a *negative* result point at the real cause.
