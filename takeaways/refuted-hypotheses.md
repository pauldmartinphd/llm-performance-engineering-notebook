# Refuted hypotheses

We measured every entry below on Galactus, and each one produced no improvement, or a regression, for the hybrid CPU-MoE workload (routed experts in system RAM, dense path on GPUs). Each negative result is a path you need not take. For platform context, see [../hardware/galactus/README.md](../hardware/galactus/README.md); for how we tested each item, see [../results/methodology.md](../results/methodology.md) and the [lab notebook](../results/lab-notebook/).

| Hypothesis / attempt | Verdict | Evidence |
|---|---|---|
| NUMA imbalance explains slow decode | Refuted | Single node (NPS1); `numactl --hardware` |
| Container overhead | Refuted | In-container STREAM = host within 1%; cgroups unlimited |
| Threadpool wake latency (`--poll`) | Refuted | poll 0 vs 100 identical across the sweep |
| Strict CPU affinity / CCD placement helps llama.cpp | Refuted | ~9% at t=32; −40% at t=16 strict; STREAM's 2× does not transfer |
| CPU_REPACK (Q4_K 8×8) speeds decode | Refuted | Engaged on 62% of expert bytes; no measurable change |
| Transparent hugepages help | Refuted | No change on GLM-5.2 or Qwen; mmap weights are file pages, not anon |
| `-sm row` as a decode lever | Impossible | gfx1030 VMM: no — "device does not support split buffers" |
| Pipeline parallelism (`n_copies` > 1) | Dead end | Compute-buffer OOM at n_copies 2 and 4 |
| ZenDNN helps | Refuted for these models | Experts are Q4_K/Q5_K on CPU; implicated in v2 crashes; removed |
| HIP managed memory | Disaster | 7.2 t/s prefill |
| SMT oversubscription (t=128) | Refuted (harmful) | Decode collapses on every model tested |

One distinction matters when reading this table: these are refutations for this workload on one class of machine. NUMA imbalance is real on multi-socket boards, transparent huge pages help anonymous-page workloads, and ZenDNN helps Q8_0-on-CPU paths. None of that contradicts the table. The table states that these changes did nothing here, by measurement, and that the burden of proof for your system is one benchmark. See [general-principles.md](general-principles.md) for the tier of each claim.
