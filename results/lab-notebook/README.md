# Lab notebook: experiments and results

This is the complete index of the lab records. It includes successful tests, failed attempts, incomplete runs, supporting captures, and plans. Each row says what kind of record it is, so a proposal or recovered log does not look like another experiment.

Dates refer to the work when known. A later recovery date is recorded inside the entry. Undated captures stay labeled as undated; I place them near the related work without inventing a run date. A recovered copy of a run is evidence for that run, not an extra repetition.

For a connected account, read the [investigation chapters](../../investigations/README.md). For the shorter conclusions, read the [important findings](../../takeaways/general-principles.md). [How to read a record](overview.md) explains the terms, evidence, and navigation.

## March–May 2026

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Mar 23 | [Galactus build record — March 23, 2026](2026-03-23-galactus-build.md) | Build record | **Can Galactus host these models?** Initial ROCm/LXC setup, DIMM repairs, and placement trials; an earlier configuration. |
| Apr 7–17 | [April: thread counts, resident experts, and long-context Qwen tests](2026-04-07-to-17-thread-and-batch-tests.md) | Experiments | **How do threads, resident experts, and batches affect speed?** Thread optima differed by workload; expert placement changed the comparison. |
| May 6 analysis | [Throughput predictions for eight MoE models: speculation and layer offload](2026-05-06-moe-throughput-predictions.md) | Prediction report | **What rates might eight MoE models reach?** Preserves measured inputs and untested predictions; no new run. |

## July 2026

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Jul 13 | [July 13, 2026 (evening) — Background and plan](2026-07-13-background-and-plan.md) | Plan | **Which parts of GLM should run on the CPU and GPUs?** Defined the offload question and initial estimates; external claims stay separate. |
| Jul 14 | [July 14, 2026 (morning) — Baselines, memory bandwidth, and the first diagnostic script](2026-07-14-0700-baselines-and-memory-bandwidth.md) | Experiments | **Is the memory system limiting generation?** Established a July baseline and measured memory bandwidth near 152 GB/s. |
| Jul 14 | [July 14, 2026 (mid-day) — v2 crash forensics and the v3 relaunch](2026-07-14-1022-crash-forensics-and-relaunch.md) | Failure investigation | **Why did the second diagnostic attempt crash?** Recorded failed phases, loading issues, and the revised diagnostic launch. |
| Jul 14 | [July 14, 2026, 11:08–14:25 — The v3 diagnostic run (machine log)](2026-07-14-1108-v3-diagnostic-run.md) | Diagnostic run | **Which settings and execution paths explain the slow rate?** Retains the full diagnostic sequence; later work found the batch-size trap. |
| Jul 14 | [July 14, 2026 (afternoon) — Interpreting v3; the untested ubatch regime](2026-07-14-1213-interpreting-v3.md) | Analysis and plan | **Did the first tests exercise large GPU batches?** Identified the untested larger-microbatch regime and planned a corrected ladder. |
| Jul 14 | [July 14, 2026 (late afternoon) — The ubatch ladder and the economics of the build](2026-07-14-1529-ubatch-ladder-and-economics.md) | Experiments and analysis | **Does larger-batch offload justify the GPUs?** Prompt throughput rose with effective microbatch size; examined hardware costs. |
| Jul 14 | [July 14, 2026 (evening) — Multi-card placement experiments and the split histogram](2026-07-14-1813-multi-card-placement.md) | Placement tests | **Where does the scheduler send the work?** Found concentrated offload and tested device placement; some attempts failed. |
| Jul 14 | [July: reconstructing the prefill investigation from the diagnostic logs](2026-07-14-diagnostic-reconstruction.md) | Recovered evidence | **What do the full v3/v4 logs establish?** Reconstructs the ladder, monitors, completed rows, and interrupted tests; no extra repetitions. |
| Jul 14 | [GLM-5.2 scheduler traces: distribution without a throughput gain](2026-07-14-glm52-scheduler-traces.md) | Supporting traces | **Did work spread across the cards?** The traces show distribution; they do not by themselves show a speedup. |
| Jul 14–16 | [evening, July 14 – July 16, 2026 — The H2D result and the patch, interrupted](2026-07-14-to-16-h2d-and-patch-investigation.md) | Experiments and analysis | **Are transfers or synchronization the remaining limit?** Measured 65.7 GB/s across four host-to-device streams and developed the patch hypothesis. |
| Jul 21 | [July 21, 2026 — Building and benchmarking the patch](2026-07-21-prefill-patch.md) | Patch experiment | **Does removing the routing-index wait improve prefill?** Distribution alone barely helped; the combined patch reached 119.36 tokens/s, a 13.7% gain. |

## August 2026

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Aug 7 | [DeepSeek-V4-Flash-0731: DSpark acceptance and generation speed](2026-08-07-deepseek-v4-dspark-acceptance.md) | Server tests | **Does a longer draft make generation faster?** Length two beat length one and five in these captures; acceptance alone did not rank speed. |
| Compiled Aug 8; runs undated | [DeepSeek-V4 DSpark investigation and interval notes](undated-deepseek-v4-dspark-investigation.md) | Compiled investigation | **Which DSpark settings and memory changes are worth testing?** Retains early results, unresolved timing conflicts, and later-superseded plans; compiled August 8. |
| Aug 15 | [STREAM re-baseline on the 2 TB population](2026-08-15-stream-rebaseline-2tb.md) | Hardware remeasurement | **Did the 2 TB upgrade change memory bandwidth?** After DIMM replacement, bandwidth was about 148–151 GB/s; capacity rose with little bandwidth change. |
| Aug 15–16 | [the 2 TB common baseline (stock build)](2026-08-15-to-16-common-baseline-2tb.md) | Common benchmark | **How do five models perform under comparable conditions?** One stock build and command structure; also records separate MTP/DSpark tests. |
| Aug 16 | [August 16: speculation and rebenchmark captures](2026-08-16-speculation-and-rebench-records.md) | Recovered evidence | **Do the saved commands support the speculation results?** Preserves individual CLI runs and resolves model/build ambiguity in benchmark filenames. |

## September 2026

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Compiled Sep 30 | [Qwen3.8-27B, oMLX, and chassis comparisons](undated-magneto-omlx-qwen38.md) | Compiled experiments | **What explains Qwen performance on the M2 Max?** ANE, MTP, DFlash, and thermal observations; compiled September 30, with some run dates unknown. |

## October 2026

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Undated; recorded Oct 21 | [GLM-5.3-Flash: placement mattered more than the build label](undated-glm53-flash-placement.md) | Undated experiments | **Does placement explain GLM-Flash speed?** Automatic placement reached 19.1–19.4 tokens/s; effective tensor placement was not captured. |
| Undated; recorded Oct 21 | [DeepSeek-V4.1: target inference worked; DSpark ran out of VRAM](undated-deepseek41-dspark.md) | Undated experiments | **Can the V4.1 drafter load and help?** Target-only generation completed at 11.2 tokens/s; draft loading ran out of VRAM. |
| Oct 21 | [llamda: the experimental AMD integration](2026-10-21-llamda-amd-fork.md) | Integration record | **What did the experimental AMD fork contain?** Records combined model support and patches; separates integration from proven performance. |
| Oct 21 | [Why STREAM bandwidth is not a decode prediction](2026-10-21-bandwidth-and-placement.md) | Analysis | **Why is bandwidth divided by weights not the observed rate?** Explains missing execution costs and the measurements needed to test the model. |
| Oct 22 | [GLM-5.3-Flash: revision-identified builds and placement tests](2026-10-22-glm53-build-comparison.md) | Build comparison | **Did the fork improve on upstream?** Dedicated benchmarks were nearly identical; CLI banners exposed build ambiguity. |
| Oct 23 | [DeepSeek-V4.1: expert-cache capacity and Maya DSpark results](2026-10-23-deepseek41-cache-and-maya.md) | Experiments and follow-up | **Does caching or Maya improve DeepSeek?** Retains one full cache benchmark, Maya calibration, RAM checks, and October 24 service observations. |
| Oct 23 | [Two large MoE models at roughly 30 tokens/s on four V620s](2026-10-23-maya-glm53-speculation.md) | Maya experiments | **Can Maya generate useful answers from both large models?** Separate runs reached roughly 30 tokens/s; GLM speculation A/B and head/expert sweeps retained. |
| Oct 28–29 | [Maya v1.1.30 recovery, prefill tuning, and the later-release regression](2026-10-28-maya-prefill-and-regression.md) | Sweeps and failure investigation | **Can long prompts run faster without losing a working service?** Selected 32K chunks and an 800 MiB ring; partial layout sweep and stream-priority trigger retained. |
| Oct 28 | [Galactus: prioritizing Maya and reassessing upstream performance work](2026-10-28-maya-priority-and-upstream-status.md) | Decision and source review | **Which engine and upstream work should I prioritize?** Prioritized Maya; records dated upstream status and model choices without claiming new tests. |
| Undated run; recorded Oct 29 | [GLM-5.3-Flash: 80 GiB expert-cache benchmark and investigation](2026-10-29-glm53-expert-cache.md) | Recovered benchmark and analysis | **Did an 80 GiB GLM expert cache help?** The benchmark is retained, but no matching cache-off control establishes its effect; run date unknown. |
| Oct 29 | [Maya v1.1.36: production configuration and uncached 64K confirmation](2026-10-29-maya-production-64k.md) | Production check | **Does the selected tuning work on a fresh 64K request?** 65,568 prompt tokens, zero cache hits, 488.6 tokens/s prefill. |
| Oct 29 | [Open WebUI: working configuration, Code Interpreter regression, and remaining work](2026-10-29-openwebui-settings-and-next-steps.md) | Service discussion | **Which application settings worked and which work remained?** Records restored responses and pending integration work; not an inference benchmark. |
| Oct 29 | [Research questions: learned memory, sparse attention sharing, and active experts](2026-10-29-sparse-architecture-research.md) | Research plan | **Could sparse architecture or learned memory change the design?** Four proposed experiments; none run in this record. |
| Oct 29 | [llamda: what the experimental fork established, and what it did not fix](2026-10-29-llamda-experimental-fork-retrospective.md) | Retrospective | **What did the fork actually demonstrate?** Withdraws the performance announcement; preserves code history, failures, and unfinished comparisons. |
| Oct 30 | [Maya v1.1.41: short benchmark and uncached 32K/64K production tests](2026-10-30-maya-v1141-benchmark.md) | Production tests | **Did the update retain short and long-prompt performance?** Two fresh 32K/64K passes; 64K prefill 469.6 and 485.6 tokens/s; original responses now retained. |

## November 2026

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Nov 2 | [Maya v1.1.49: production replay and the two-pair utilization question](2026-11-02-maya-v1149-production-recheck.md) | Production tests | **Does the newer installation retain long-prefill speed?** Fresh 32K/64K replays at 461.1 and 480.9 tokens/s. |
| Nov 2 | [Maya v1.1.49: one versus four GPUs and reduced expert-cache budgets](2026-11-02-maya-gpu-cache-matrix.md) | Allocation matrix | **What happens with one or four GPUs and different cache budgets?** Nine variants completed; 54 long observations and nine short reports; next GLM case failed loading. |
| Nov 2–8 discussion | [Maya releases, memory budgets, and alternative deployment plans](2026-11-02-maya-release-and-capacity-planning.md) | Planning, November 2–8 | **Which release features and alternate deployments deserve tests?** Keeps release claims, memory proposals, and operational guidance separate from measurements. |
| Nov 1–3 | [November 1–3: Gufo on SilverSurfer, memory allocation, and Jan](2026-11-02-silversurfer-gufo.md) | Experiments, November 1–3 | **Can the shared-memory laptop host Flash-Next?** Resolved successive loading obstacles; retained a 318-token answer at 29.85 tokens/s on battery. |
| Nov 5 | [DeepSeek-V4.1-Flash: style optimization and optional refusal reduction](2026-11-05-deepseek-style-and-heretic-plan.md) | Evaluation plan | **How could I compare writing style and refusal behavior?** Proposed accuracy and style checks; no model modification or evaluation run. |
| Nov 5 | [Maya v1.1.54: prefill sweep and HIP peer all-reduce failure](2026-11-05-maya-v1154-prefill-and-peer-ar.md) | Prefill sweep and startup failure | **Do flow, kernel, or peer-transfer switches help?** Default 64K prefill 479.4 tokens/s; no useful gain; peer retest failed before readiness. |
| Nov 5 | [Maya v1.1.54: recovered decode measurements and complete sweep evidence](2026-11-05-maya-v1154-recovered-decode.md) | Recovered evidence | **What do the complete November 5 artifacts add?** All 60 prefill rows, 20 decode cases, and configs recovered; no additional experiment. |
| Nov 9 | [Throughput bounds and alternative Qwen deployments](2026-11-09-throughput-models-and-alternative-hardware.md) | Planning and estimates | **Should I use other models or hardware?** Keeps bandwidth estimates, Qwen alternatives, and external quality claims distinct from local tests. |
| Nov 9 | [DeepSeek on two V620s: baseline and requested TP4 fallback](2026-11-09-deepseek-tp2-and-tp4.md) | Layout tests | **Can two GPUs preserve useful DeepSeek speed?** 29.0 tokens/s built-in mean generation; requested TP4 later executed TP2. |
| Nov 9 | [DeepSeek RDNA 2 FP16-output prefill: measured speedup and fidelity limits](2026-11-09-deepseek-fp16-prefill.md) | Precision-path test | **Does FP16 output speed dense prefill operations?** Optimized medians 389.55/501.95/589.70 tokens/s at 8K/16K/32K; control and quality limits retained. |
| Nov 9 | [Full GLM-5.3 Q4: allocation failures, legal layer boundaries, and a slow working split](2026-11-09-full-glm53-placement.md) | Model and placement tests | **Is full GLM-5.3 practical here?** A corrected four-card split completed at 6.7 generation tokens/s; returned to Flash for daily use. |
| Nov 9 | [GLM-5.3-Flash on two V620s: MTP, reserve reduction, and cancellation](2026-11-09-glm-flash-reserve-and-speculation.md) | Memory and speculation tests | **Can Flash keep useful speed on the other pair?** Lower reserve raised observed mean generation to 21.5 tokens/s; cancellation was slower. |
| Nov 9 | [GLM-Flash prefill: requested versus effective chunks and the failed first harness](2026-11-09-glm-flash-prefill-chunks.md) | Workspace and chunk tests | **Does the DeepSeek large-chunk approach help GLM?** Verified 16K chunks were about 21% slower on long prompts; kept 8K effective chunks. |
| Nov 9 run; reported Nov 10 | [GLM-Flash FP16 versus INT8 KV: repeated prefill and decode observations](2026-11-09-to-10-glm-flash-kv-cache.md) | KV comparison | **Should Flash use FP16 or INT8 attention state?** INT8 had 11–13% higher mean generation across paired samples; no local quality-equivalence test. |
| Nov 10 | [Galactus: GLM-Flash INT8 KV and DeepSeek on separate GPU pairs](2026-11-10-maya-dual-model-production.md) | Deployment record | **What was finally hosted?** DeepSeek on GPUs 0–1 and GLM-Flash INT8 KV on 2–3; overlapping-load throughput unmeasured. |

## Reference captures with mixed or unknown dates

| Date or period | Record | Kind | Question and outcome |
|---|---|---|---|
| Mixed and unknown dates | [Maya archive: upstream reference tests and unassigned live logs](undated-maya-reference-benchmarks-and-live-logs.md) | Reference evidence | **Which supplied artifacts belong to other hosts or unknown dates?** Separates upstream V100 tests and loose logs from dated Galactus measurements. |

## Supporting material

[Model results](../README.md) collect measurements by model. [Data tables](../data/) retain individual observations. [Raw captures](../raw-logs/) and [source notes](../reference-notes/README.md) preserve the evidence and its origin. The [old filename map](filename-map.md) helps resolve earlier links.
