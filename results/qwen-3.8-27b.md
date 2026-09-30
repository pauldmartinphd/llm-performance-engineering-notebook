# Qwen3.8-27B — oMLX on Magneto

The first Apple Silicon measurements in this notebook use `Qwen3.8-27B-oQ4e-fp16-mtp` on [Magneto](../hardware/magneto/README.md), a **14-inch MacBook Pro, M2 Max, 38 GPU cores, 64 GB unified memory**. The September 30 record covers a Lightning MTP baseline, tuned ANE prefill, a DFlash trial, later measurements, and a thermal and leaderboard investigation. These oMLX single-request results are separate from the Galactus llama-bench common baseline.

## Results and evidence

PP is prompt-processing throughput; TG is generation throughput, both in tokens/s. TTFT is time to first token; TPOT is time per output token. The baseline and intermediate rows survive in the full conversation PDF, not in recovered raw exports (page references in the [source ledger](lab-notebook/13-magneto-omlx-qwen38.md)). The two later rows were independently transcribed from the [benchmark screenshot](raw-logs/omlx-magneto/2026-09-30-throughput.png). No repeated-run uncertainty is available.

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

The initial run was reported as plugged in and starting cold, with Nominal thermal pressure and the full 2,048-token warm-up selected. Lightning MTP was on; ANE Prefill, SpecPrefill, DFlash, VLM MTP, and TurboQuant KV were off. “Cold” describes the machine's thermal starting condition, not an absence of benchmark warm-up or a verified cache state. A starting thermal observation does not establish the state throughout the context sweep.

The later screenshot shows Code (Python), full warm-up, ANE-aligned prompts off, and 128 generated tokens. Its batch-size-2 selection applies to the separate continuous-batching phase; the table above contains single-request results. The conversation describes ANE Prefill and Lightning MTP, but gives conflicting accounts of SpecPrefill: it was enabled during setup, then explicitly abandoned without a selected draft, then casually described as enabled again. The latest screenshot does not show those settings. These later rows are not verified SpecPrefill results. Paul’s subsequently supplied [measurement summary](raw-logs/omlx-magneto/user-measurement-summary.txt) identifies the local software as **oMLX 0.6.4 build 2529**, resolving the conversation’s version uncertainty. This differs from the external 0.6.3rc2 result. Exact model revision, complete applied recipe, macOS build, and per-row thermal traces remain unrecorded.

## ANE Prefill: tuning versus benchmark results

The tuner report (PDF pp. 71–72, 87) gives **155.7 PP t/s GPU-only**, **182.5 predicted optimum (+17.2%)**, and **196.3 profile-refined optimum (+26.1%)**. Its proposed override was approximately 60% MLP on ANE, 38% GDN on ANE, CPU fractions 10%/0%/8%, and tail padding at ≥16,224. These are reported tuner outputs, not an exact exported serving recipe. The earlier visible 0.53 MLP / 0.5 GDN values preceded applying the result. Intermediate operation timings (69.35, 90.17, and 23.92 ms) do not establish full-model throughput.

The subsequent ANE + Lightning MTP benchmark measured **170.1 PP at 4K (+9.0%)** and **165.6 at 16K (+8.3%)** relative to the initial GPU-prefill rows. TTFT fell 8.2% and 7.7%. TG simultaneously fell 15.2% and 17.8%; reported end-to-end time nevertheless fell from 34.1 to 33.3 seconds at 4K and 113.9 to 107.1 at 16K (PDF pp. 88–89). These are single-run comparisons without thermal matching or repeatability estimates. They support a useful prefill improvement in this trial, not a guaranteed isolated ANE effect or a 26% production speedup.

The quoted 16K log records **441/448 expected MLP operations and 336/336 GDN operations** (PDF p. 94). This is stronger evidence that ANE executed than its GUI toggle or a warm-up warning. It does not prove every layer or prompt shape used the intended route. The final recipe still needs an export, particularly because the conversation reports settings failing to persist until profiles were saved.

## Lightning MTP: active, but no measured MTP-off speedup

The baseline log recap records the following for 128 generated tokens (PDF pp. 64–65):

| Context | Backbone cycles | Tokens/cycle | Draft acceptance |
|---|---:|---:|---:|
| 4K | 57 | 2.25 | 81.4% |
| 8K | 54 | 2.37 | 84.1% |
| 16K | 45 | 2.84 | 94.3% |
| 32K | 35 | 3.66 | 98.9% |
| 64K | 47 | 2.72 | 91.9% |

The ANE + MTP 16K run later reports 43 cycles, 2.98 tokens/cycle, 91.5% acceptance, and 7,605.5 ms of backbone time: **176.9 ms per cycle** (PDF p. 94). At 4K, that run initially reports 66.7% acceptance and 1.61 tokens/cycle before parking MTP (p. 95). This establishes runtime activity and variation by workload; high acceptance alone does not guarantee high wall-clock throughput. Tokens/cycle is not a measured speedup over an MTP-off run, because a speculative cycle has different work. The proposed MTP-off comparison was deferred and has no recorded result.

Baseline aggregate generation was reported as 15.7 / 22.3 / 38.3 t/s at batch sizes 1 / 2 / 4 (PDF pp. 63–67). The same log recap says MTP was disabled at batch size ≥2. Thus 2.44× aggregate throughput at batch 4 is not a 2.44× single-user latency improvement or a measurement of batched MTP scaling.

## DFlash: slower trial with ANE inactive

The experiment requested tuned ANE plus DFlash, with Lightning MTP off, and selected `z-lab/Qwen3.8-27B-DFlash2` as the intended draft (PDF pp. 73–80; exact loaded revision not preserved). At 4K the UI reported **153.7 PP / 15.5 TG**, versus baseline 156.1 / 16.4; at 16K it reported **11.2 TG**, versus 19.1. Those TG differences are **−5.5%** and **−41.4%**. The 16K PP value is not recoverable from the text and is left blank.

The quoted logs report ANE enabled but inactive and zero MLP/GDN operations at both contexts. They also identify a DFlash engine and a slower tiled SDPA256 attention path without a guard-headroom provider (PDF pp. 82–84). This supports a failure of the requested ANE combination in the tested runtime, and an attention fallback worth investigating. It does not isolate that fallback as the entire cause of the slowdown or establish a limitation of every DFlash version. [Upstream issue 2986](https://github.com/jundot/omlx/issues/2986) describes a similar ANE bypass on a different machine/build; it is corroborating context, not this machine's raw log.

| Context | UI TG (t/s) | Internal generation (t/s) | Acceptance | Draft timing as quoted | Verify timing as quoted |
|---|---:|---:|---:|---:|---:|
| 4K | 15.5 | 3.7 | 69.5% | 1.55 s | 679 ms |
| 16K | 11.2 | 1.0 | 79.7% | 833 ms | 10.27 s |

The scopes and aggregation of the internal timings are not preserved. Do not substitute internal generation rates for the UI metric or combine these timings into an end-to-end duration. The larger verification timing despite higher acceptance is a diagnostic lead. The requested ANE-on run is also not a separately controlled ANE-off DFlash run merely because no ANE operations were observed.

## SpecPrefill: setup explored, benefit not demonstrated

The setup showed a 20% keep rate and an 8,192-token threshold, but no draft model selected (PDF p. 90). Under those settings 4K is below threshold. The conversation then recommended turning it off after raising compatibility concerns for this target (pp. 90–91). Its cited [issue 2902](https://github.com/jundot/omlx/issues/2902) is a user report, not proof that every current implementation is incompatible.

No recovered local result verifies draft loading and actual token selection. Later references to SpecPrefill being enabled conflict with the earlier decision to abandon it. Record this as an incomplete experiment, with no measured speedup or quality result. A faster 16K row by itself cannot establish sparse prefill. Any future test must verify the runtime path and retained-token count and evaluate quality as well as TTFT.

## Thermal observations and the leaderboard

The [powermetrics screenshot](raw-logs/omlx-magneto/2026-09-30-powermetrics.png) directly shows **Heavy** thermal pressure, a GPU active frequency of **942 MHz**, **84.40%** active residency, and GPU power of **21.572 W** in its GPU section. The package summary reports **21.742 W** GPU and **25.643 W** combined CPU + GPU + ANE. Preserve these as separate reported readings rather than substituting one for the other. The conversation also reports earlier Nominal samples around 27–29 W and 96–100% GPU activity, but their original captures are missing.

**Interpretation:** thermal state is a real uncontrolled variable. Heavy pressure is consistent with thermal constraints; these snapshots do not measure how much of any throughput difference they caused. The conversation's later 444 MHz / 178 mW observation has no verified workload phase and could be idle. Nominal pressure does not establish equal power, clock, or cooling conditions across machines. Conversely, a 28 W observation is not proof of a fixed 14-inch power ceiling.

The [external submission `f4f6h86x`](https://omlx.ai/benchmarks/performance/f4f6h86x), checked September 30, reports the same model display name and M2 Max / 38 GPU cores / 64 GB label, **201.8 PP / 25.7 TG** at 4K Code (Python), oMLX **0.6.3rc2**, and macOS **26.5.2**. Lightning MTP is on; ANE, SpecPrefill, DFlash, VLM MTP, and TurboQuant KV are off. It reports Nominal thermal state and 98% average GPU use. Against the initial 4K row, its PP is 29.3% higher and TG 56.7% higher. Matching display names and main toggles does not establish identical weights, software, or execution conditions.

**Confirmed metadata limitation:** the public hardware label does not distinguish a 14-inch MacBook Pro, 16-inch MacBook Pro, or Mac Studio. The inspected [oMLX upload implementation](https://github.com/jundot/omlx/blob/main/omlx/admin/benchmark.py) sends chip name/variant, memory, and GPU cores, but no chassis or Mac model identifier in `_upload_to_omlx_ai`. This describes the inspected implementation and public record, not a claim about every past or future version.

**Hypothesis:** the high external result may come from a larger chassis with more sustained cooling capacity. Neither the submitter's chassis nor the contribution of cooling to the gap has been established. [Apple's power-mode documentation](https://support.apple.com/en-us/101613) supports the general mechanism that additional fan cooling can improve sustained performance; it does not establish a numerical power limit or an LLM speedup for these machines. GPU watts from unrelated workloads cannot predict decode throughput proportionally.

Other checked 38-core/64-GB M2 Max submissions report [124.3 PP / 20.2 TG](https://omlx.ai/benchmarks/performance/1c99hg1r) and [116.6 PP / 19.9 TG](https://omlx.ai/benchmarks/performance/yg88uk4v) at 4K using ordinary Qwen3.8-27B-4bit, with different software versions. A [DFlash-tagged session](https://omlx.ai/benchmarks/performance/r98fth4g) reports 20.9 TG at 1K and 20.2 at 4K. These support treating 25.7 as a high comparison point, not a statistically established outlier or a controlled target for Magneto. They also do not establish that DFlash executed correctly on those external machines. Comparing the local 19.4 at **16K** with 25.7 at **4K** also changes context length. Software/model revisions, prompt and generation settings, thermal history, and measurement variance remain alternatives to the chassis explanation.

## Next measurements

[Entry 13](lab-notebook/13-magneto-omlx-qwen38.md) records provenance, missing evidence, and corrections to the conversation's interpretation. The next useful comparison is a repeated, single-context 4K test with an exported recipe, pinned software and model revisions, and time-aligned thermal telemetry. Change one acceleration setting at a time; test interactions separately. Record output quality as well as speed before selecting an approximate prefill configuration. Tuned ANE + Lightning MTP is the working candidate supported by this investigation, with DFlash and SpecPrefill off; the present record does not establish a final optimal recipe or an isolated MTP gain.
