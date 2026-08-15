# Refuted hypotheses and dead ends

Every entry below was measured on Galactus and came back null or worse for the hybrid CPU-MoE workload (routed experts in system RAM, dense path on GPUs). Each "no" is a road you do not have to drive down. Platform context: [../hardware/galactus/README.md](../hardware/galactus/README.md); how each was tested: [../results/methodology.md](../results/methodology.md) and the [lab notebook](../results/lab-notebook/).

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

The distinction that matters when reading this table: these are refutations **for this workload shape** on one machine class. NUMA imbalance is real on multi-socket boards; THP helps anonymous-page workloads; ZenDNN helps Q8_0-on-CPU paths. None of that contradicts the table — the table says they did nothing *here*, measured, and the burden of proof for your system is one benchmark away. See [what-transfers.md](what-transfers.md) for which tier each claim lives in.
