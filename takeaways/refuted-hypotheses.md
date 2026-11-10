# Unsuccessful approaches and unresolved tests

I tested these approaches on Galactus for the hybrid CPU-MoE workload (routed experts in system RAM, dense path on GPUs). Some produced no improvement or regressed; others failed to run or did not apply to the supported code path. I've kept those outcomes distinct. For platform context, see [../hardware/galactus/README.md](../hardware/galactus/README.md); for how we tested each item, see [../results/methodology.md](../results/methodology.md) and the [lab notebook](../results/lab-notebook/).

## Earlier Galactus tests

| Hypothesis / attempt | Verdict | Evidence |
|---|---|---|
| NUMA imbalance explains slow decode | Refuted | Single node (NPS1); `numactl --hardware` |
| Container overhead | Refuted | In-container STREAM = host within 1%; cgroups unlimited |
| Threadpool wake latency (`--poll`) | Refuted | poll 0 vs 100 identical across the sweep |
| Strict CPU affinity / CCD placement helps llama.cpp | Refuted | ~9% at t=32; −40% at t=16 strict; STREAM's 2× does not transfer |
| CPU_REPACK (Q4_K 8×8) speeds decode | Refuted | Engaged on 62% of expert bytes; no measurable change |
| Transparent hugepages help | Refuted | No change on GLM-5.2 or Qwen; mmap weights are file pages, not anon |
| `-sm row` as a decode lever | Impossible | gfx1030 VMM: no — "device does not support split buffers" |
| Pipeline parallelism (`n_copies` > 1) | Allocation failure | Compute-buffer OOM at n_copies 2 and 4 |
| ZenDNN helps | Refuted for these models | Experts are Q4_K/Q5_K on CPU; implicated in v2 crashes; removed |
| HIP managed memory | Regression | 7.2 t/s prefill |
| SMT oversubscription (t=128) | Refuted (harmful) | Decode collapses on every model tested |

These results apply to this workload on one class of machine. NUMA imbalance is real on multi-socket boards, transparent huge pages help anonymous-page workloads, and ZenDNN helps Q8_0-on-CPU paths. None of that contradicts the table. The table records what happened here and gives me a basis for choosing tests on another system. See [findings and limits](general-principles.md) for the scope of the evidence.

## Later engine and allocation tests

A failed run, an inactive option, and a measured slowdown are different outcomes. These later records keep them separate.

| Attempt | Observed outcome | Scope and evidence |
|---|---|---|
| Combined llamda fork as a performance improvement | No useful gain established over upstream in the revision-identified GLM benchmark | [October 22 comparison](../results/lab-notebook/2026-10-22-glm53-build-comparison.md); not a patch-isolated test |
| GLM expert-cache increases | No consistent gain established; some allocation failures | [Placement record](../results/lab-notebook/undated-glm53-flash-placement.md) and [80 GiB capture](../results/lab-notebook/2026-10-29-glm53-expert-cache.md); effective allocations and controls incomplete |
| Requested TP4 on the tested DeepSeek packs/configs | Effective TP2; excluded as TP4 evidence | [October sweep](../results/lab-notebook/2026-10-28-maya-prefill-and-regression.md) and [November layout test](../results/lab-notebook/2026-11-09-deepseek-tp2-and-tp4.md) |
| Automatic DeepSeek prompt chunks in the November 5 sweep | Slower than the tested defaults | [Ten-variant sweep](../results/lab-notebook/2026-11-05-maya-v1154-prefill-and-peer-ar.md); not a result for every release |
| Peer all-reduce in that sweep | Completed runs retained host transport; larger-reserve retest failed before readiness | [Transport logs and retest](../results/lab-notebook/2026-11-05-maya-v1154-prefill-and-peer-ar.md); no successful peer throughput result |
| Full GLM-5.3 Q4 on four V620s | Corrected split ran, but was too slow for the intended use | [Full-model test](../results/lab-notebook/2026-11-09-full-glm53-placement.md); 6.7 tokens/s mean generation |
| GLM-Flash speculative cancellation | Lower observed generation than the retained 1 GiB reserve run | [November reserve/cancellation tests](../results/lab-notebook/2026-11-09-glm-flash-reserve-and-speculation.md); one run per setting |
| Larger effective GLM-Flash prefill chunks | About 21% slower at 32K and 64K targets | [Chunk comparison](../results/lab-notebook/2026-11-09-glm-flash-prefill-chunks.md); 16K versus 8K effective chunks |
| DFlash with requested ANE on Magneto | Slower trial; ANE did no recorded MLP/GDN work | [oMLX record](../results/qwen-3.8-27b.md); does not measure a correctly running ANE-plus-DFlash combination |

## Workarounds belong to their dates

Disabling stream priority allowed a recorded Maya v1.1.36 benchmark to complete after the reported priority-on deadlock. By October 30, that workaround was no longer in use. It remains part of the failure investigation, not a current requirement for every configuration. [Regression record](../results/lab-notebook/2026-10-28-maya-prefill-and-regression.md) · [Later production tests](../results/lab-notebook/2026-10-30-maya-v1141-benchmark.md).

The [important findings](general-principles.md) give the successful results. The [chapters](../investigations/README.md) show how both useful and unsuccessful tests shaped the next decision.
