# Qwen3.8-27B — oMLX on Magneto

I ran the first Apple Silicon benchmarks in this notebook with `Qwen3.8-27B-oQ4e-fp16-mtp` on [Magneto](../hardware/magneto/README.md), a **14-inch MacBook Pro, M2 Max, 38 GPU cores, 64 GB unified memory**. The September 30 session covered Lightning MTP, ANE tuning, DFlash, and the gap between my results and the oMLX leaderboard. The Galactus common baseline uses llama-bench; these are oMLX single-request measurements.

## Results and evidence

PP is prompt-processing throughput; TG is generation throughput, both in tokens/s. TTFT is time to first token; TPOT is time per output token. The baseline and intermediate rows come from the conversation PDF; the original exports are missing. [Entry 13](lab-notebook/13-magneto-omlx-qwen38.md) gives the source pages. The last two rows are readable in the retained [benchmark screenshot](raw-logs/omlx-magneto/2026-09-30-throughput.png). I have no repeatability estimate for these runs.

| Run | Context | PP (t/s) | TG (t/s) | TTFT (s) | TPOT (ms) |
|---|---:|---:|---:|---:|---:|
| Initial baseline | 1K | 143.8 | 15.7 | 7.12 | 64.3 |
| Initial baseline | 4K | 156.1 | 16.4 | 26.23 | 61.6 |
| Initial baseline | 8K | 153.2 | 16.7 | 53.48 | 60.4 |
| Initial baseline | 16K | 152.9 | 19.1 | 107.18 | 52.6 |
| Initial baseline | 32K | 138.8 | 17.3 | 236.01 | 58.1 |
| Initial baseline | 64K | 117.3 | 13.9 | 558.50 | 72.7 |
| Tuned ANE + MTP | 4K | 170.1 | 13.9 | 24.08 | — |
| Tuned ANE + MTP | 16K | 165.6 | 15.7 | 98.95 | — |
| DFlash; ANE requested, inactive | 4K | 153.7 | 15.5 | 26.66 | — |
| DFlash; ANE requested, inactive | 16K | — | 11.2 | — | — |
| Later run, recipe incomplete | 4K | 165.5 | 15.9 | — | — |
| Later run, recipe incomplete | 16K | 171.1 | 18.2 | — | — |
| Latest screenshot, recipe incomplete | 4K | 171.0 | 16.9 | 23.9575 | 59.6 |
| Latest screenshot, recipe incomplete | 16K | 173.0 | 19.4 | 94.7002 | 52.0 |

The initial run was plugged in and started cold, with Nominal thermal pressure and the full 2,048-token warm-up selected. Lightning MTP was on; ANE Prefill, SpecPrefill, DFlash, VLM MTP, and TurboQuant KV were off. Here “cold” means the laptop started cool. The benchmark still ran its warm-up. Since the record does not include cache state or a thermal trace covering every row, the initial Nominal observation cannot establish the conditions of the complete sweep.

The later screenshot shows Code (Python), full warm-up, ANE-aligned prompts off, and 128 generated tokens. Its batch-size-2 selection applies to the separate continuous-batching phase; the table above contains single-request results. The conversation was inconsistent about SpecPrefill: it was enabled during setup, then recommended off because no draft had been selected, then described as enabled again. The screenshot does not resolve that inconsistency. Although the saved settings below have SpecPrefill on, no draft model is recorded and the benchmark record does not show token selection. My [measurement summary](raw-logs/omlx-magneto/user-measurement-summary.txt) identifies **oMLX 0.6.4 build 2529**; the external result used 0.6.3rc2. The exact model revision, macOS build, and thermal state of each row remain unrecorded.

## ANE Prefill: tuning versus benchmark results

The tuner report (PDF pp. 71–72, 87) gives **155.7 PP t/s GPU-only**, **182.5 predicted optimum (+17.2%)**, and **196.3 profile-refined optimum (+26.1%)**. Its proposed override was approximately 60% MLP on ANE, 38% GDN on ANE, CPU fractions 10%/0%/8%, and tail padding at ≥16,224. The fields visible earlier showed 0.53 MLP / 0.5 GDN, and my final saved settings also contain those fractions. Since the source does not show how much of the proposed override was applied, the tuner output cannot serve as the record of my final configuration. The intermediate 69.35, 90.17, and 23.92 ms timings cover individual operations, so I do not use them as full-model throughput.

The subsequent ANE + Lightning MTP benchmark measured **170.1 PP at 4K (+9.0%)** and **165.6 at 16K (+8.3%)** relative to the initial GPU-prefill rows. TTFT fell 8.2% and 7.7%. TG simultaneously fell 15.2% and 17.8%; reported end-to-end time nevertheless fell from 34.1 to 33.3 seconds at 4K and 113.9 to 107.1 at 16K (PDF pp. 88–89). The improvement in PP and corresponding reduction in TTFT indicate that prefill was faster in this comparison. However, since these were single runs with unmatched thermal state, I cannot assign the whole difference to ANE. The 26% improvement reported by the tuner did not carry through to the benchmark.

The quoted 16K log records **441/448 expected MLP operations and 336/336 GDN operations** (PDF p. 94). These operation counts establish that ANE was doing work at 16K, even though the warm-up warning suggested otherwise. These counts do not establish that every layer or prompt shape followed the intended route; the saved toggle alone was a poor guide to what actually ran. I also had trouble getting settings to persist until I saved them in profiles; the saved state is recorded below.

## Lightning MTP: active, but no measured MTP-off speedup

The baseline log recap records the following for 128 generated tokens (PDF pp. 64–65):

| Context | Backbone cycles | Tokens/cycle | Draft acceptance |
|---|---:|---:|---:|
| 4K | 57 | 2.25 | 81.4% |
| 8K | 54 | 2.37 | 84.1% |
| 16K | 45 | 2.84 | 94.3% |
| 32K | 35 | 3.66 | 98.9% |
| 64K | 47 | 2.72 | 91.9% |

The ANE + MTP 16K run later reports 43 cycles, 2.98 tokens/cycle, 91.5% acceptance, and 7,605.5 ms of backbone time: **176.9 ms per cycle** (PDF p. 94). At 4K, that run initially reports 66.7% acceptance and 1.61 tokens/cycle before parking MTP (p. 95). These results establish that MTP was running and that its behavior varied with the workload. They also show why high acceptance alone could not resolve the throughput gap: the backbone cycles were still expensive. A speculative cycle does more work than an ordinary decode pass, so tokens/cycle cannot stand in for an MTP-off speed comparison. I did not complete that comparison.

Baseline aggregate generation was reported as 15.7 / 22.3 / 38.3 t/s at batch sizes 1 / 2 / 4 (PDF pp. 63–67). The same log recap says MTP was disabled at batch size ≥2. The 2.44× gain at batch 4 is aggregate throughput across requests. Individual latency and batched MTP scaling were not measured by that comparison.

## DFlash: slower trial with ANE inactive

The experiment requested tuned ANE plus DFlash, with Lightning MTP off, and selected `z-lab/Qwen3.8-27B-DFlash2` as the intended draft (PDF pp. 73–80; exact loaded revision not preserved). At 4K the UI reported **153.7 PP / 15.5 TG**, versus baseline 156.1 / 16.4; at 16K it reported **11.2 TG**, versus 19.1. Those TG differences are **−5.5%** and **−41.4%**. The 16K PP value is not recoverable from the text and is left blank.

The quoted logs report ANE enabled but inactive and zero MLP/GDN operations at both contexts. They also identify a DFlash engine and a slower tiled SDPA256 attention path without a guard-headroom provider (PDF pp. 82–84). The logs indicate that the requested ANE + DFlash combination did not run as configured. The attention fallback may have contributed to the slowdown, but its contribution cannot be determined from this trial. This result applies to the runtime I tested. [Upstream issue 2986](https://github.com/jundot/omlx/issues/2986) describes a similar ANE bypass on a different machine/build; it is corroborating context, not this machine's raw log.

| Context | UI TG (t/s) | Internal generation (t/s) | Acceptance | Draft timing as quoted | Verify timing as quoted |
|---|---:|---:|---:|---:|---:|
| 4K | 15.5 | 3.7 | 69.5% | 1.55 s | 679 ms |
| 16K | 11.2 | 1.0 | 79.7% | 833 ms | 10.27 s |

The source does not preserve the scope of the internal timings, so I keep them separate from UI TG and end-to-end time. Verification became much more expensive at 16K even though acceptance improved. ANE did zero operations, but I had requested it on; this was not a separately controlled DFlash-only test.

## SpecPrefill: setup explored, benefit not demonstrated

The setup showed a 20% keep rate and an 8,192-token threshold, but no draft model selected (PDF p. 90). Under those settings 4K is below threshold. The conversation then recommended turning it off after raising compatibility concerns for this target (pp. 90–91). Its cited [issue 2902](https://github.com/jundot/omlx/issues/2902) is a user report, not proof that every current implementation is incompatible.

The record contains no local result showing a loaded draft or actual token selection, and no quality evaluation. As a result, the later 16K numbers cannot establish a SpecPrefill gain. The toggle is enabled in the saved configuration, but its benefit remains unmeasured.

## Thermal observations and the leaderboard

The [powermetrics screenshot](raw-logs/omlx-magneto/2026-09-30-powermetrics.png) directly shows **Heavy** thermal pressure, a GPU active frequency of **942 MHz**, **84.40%** active residency, and GPU power of **21.572 W** in its GPU section. The package summary reports **21.742 W** GPU and **25.643 W** combined CPU + GPU + ANE. Those are separate readings in the same capture. The conversation also reports earlier Nominal samples around 27–29 W and 96–100% GPU activity, but their original captures are missing.

**Interpretation:** the laptop did reach Heavy thermal pressure. I cannot quantify its effect on throughput from these snapshots. The later 444 MHz / 178 mW sample appears mostly idle, with no verified workload phase, so it does not establish throttling under load. Nominal pressure also does not establish equal clocks, power, or cooling across machines. The observed 28 W is an operating point, not a measured chassis power limit.

The [external submission `f4f6h86x`](https://omlx.ai/benchmarks/performance/f4f6h86x), checked September 30, reports the same model display name and M2 Max / 38 GPU cores / 64 GB label, **201.8 PP / 25.7 TG** at 4K Code (Python), oMLX **0.6.3rc2**, and macOS **26.5.2**. Lightning MTP is on; ANE, SpecPrefill, DFlash, VLM MTP, and TurboQuant KV are off. It reports Nominal thermal state and 98% average GPU use. Against the initial 4K row, its PP is 29.3% higher and TG 56.7% higher. Matching display names and main toggles does not establish identical weights, software, or execution conditions.

**Confirmed metadata limitation:** the public hardware label does not distinguish a 14-inch MacBook Pro, 16-inch MacBook Pro, or Mac Studio. The inspected [oMLX upload implementation](https://github.com/jundot/omlx/blob/main/omlx/admin/benchmark.py) sends chip name/variant, memory, and GPU cores, but no chassis or Mac model identifier in `_upload_to_omlx_ai`. That is what the implementation and public record contained when checked.

**Hypothesis:** the high external result may come from a larger chassis with more sustained cooling capacity. I do not know the submitter’s chassis or how much cooling contributed to the gap. [Apple's power-mode documentation](https://support.apple.com/en-us/101613) supports the general mechanism that additional fan cooling can improve sustained performance; it does not establish a numerical power limit or an LLM speedup for these machines. GPU watts from unrelated workloads cannot predict decode throughput proportionally.

Other checked 38-core/64-GB M2 Max submissions report [124.3 PP / 20.2 TG](https://omlx.ai/benchmarks/performance/1c99hg1r) and [116.6 PP / 19.9 TG](https://omlx.ai/benchmarks/performance/yg88uk4v) at 4K using ordinary Qwen3.8-27B-4bit, with different software versions. A [DFlash-tagged session](https://omlx.ai/benchmarks/performance/r98fth4g) reports 20.9 TG at 1K and 20.2 at 4K. These support treating 25.7 as a high comparison point, not a statistically established outlier or a controlled target for Magneto. They also do not establish that DFlash executed correctly on those external machines. Comparing the local 19.4 at **16K** with 25.7 at **4K** also changes context length. Software/model revisions, prompt and generation settings, thermal history, and measurement variance remain alternatives to the chassis explanation.

## Final settings for now (September 30, 2026)

I’m stopping the measurement work here for now. This is the configuration saved for the model and its active `profile-1-copy` profile. I checked the saved files directly; these settings are not inferred from the tuner’s proposed override.

| Setting | Saved value |
|---|---|
| Engine | oMLX 0.6.4 build 2529 |
| Model | `Jundot/Qwen3.8-27B-oQ4e-fp16-mtp` |
| Model size | 17.50 GB |
| Context limit | 131,072 tokens |
| Maximum output | 32,768 tokens |
| Lightning MTP | On |
| Qwen ANE Prefill | On |
| ANE prompt block | 2,048 tokens |
| MLP on ANE / layer limit | 0.53 / 64 |
| Use both ANEs | On |
| GDN acceleration | On |
| GDN on ANE / layer limit | 0.5 / 48 |
| Tail padding minimum | 0 tokens |
| Fused down projection | Off |
| CPU prefill sharing | Off |
| SpecPrefill | On; keep fraction 0.2; threshold 8,192 tokens; no draft model recorded |
| DFlash | Off |
| VLM MTP | Off |
| TurboQuant KV | Off |
| Thinking | On; thinking-budget limit disabled |
| Sampling | Temperature 1; top-p 0.95; top-k 20; force sampling off |

The [saved settings snapshot](data/magneto-omlx-final-settings.json) retains the remaining parameters, including values for disabled features. SpecPrefill being enabled is a saved setting, not evidence that it ran. CPU sharing is off even though CPU fractions remain in the file. This records where I left the system; it does not assign these settings retroactively to every benchmark row.
