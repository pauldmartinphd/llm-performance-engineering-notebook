# Lab notebook: experiments and results

This is the complete index of the 47 substantive lab records, including 33 experimental accounts and 14 plans, analyses, or service records. It includes successful tests, failed attempts, incomplete runs, supporting captures, and plans. Each row says what kind of record it is, so a proposal does not look like a completed experiment.

Dates refer to the work when known. Source and recording dates remain with the entry. Undated captures stay labeled as undated; I place them near the related work without inventing a run date. Source files and individual measurements are incorporated into the relevant experiment. Collecting a log does not create another run.

For a connected account, read the [investigation chapters](../../investigations/README.md). For the shorter conclusions, read the [important findings](../../takeaways/general-principles.md). [How to read a record](overview.md) explains the terms, evidence, and navigation.

## March–May 2026

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Mar 23 | [Galactus build record — March 23, 2026](2026-03-23-galactus-build.md) | Build record | Establish a working large-model host, then test whether memory population and expert placement limit its speed. Initial ROCm/LXC setup, DIMM repairs, and placement trials; an earlier configuration. |
| Apr 7–17 | [April: thread counts, resident experts, and long-context Qwen tests](2026-04-07-to-17-thread-and-batch-tests.md) | Experiments | Find useful CPU thread counts and test how resident experts, batch size, and context change performance. Thread optima differed by workload; expert placement changed the comparison. |
| May 6 analysis | [Throughput predictions for eight MoE models: speculation and layer offload](2026-05-06-moe-throughput-predictions.md) | Prediction report | Preserves measured inputs and untested predictions; no new run. |

## July 2026

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Jul 13 | [July 13, 2026 (evening) — Background and plan](2026-07-13-background-and-plan.md) | Plan | Defined the offload question and initial estimates; external claims stay separate. |
| Jul 14 | [July 14, 2026 (morning) — Baselines, memory bandwidth, and the first diagnostic script](2026-07-14-0700-baselines-and-memory-bandwidth.md) | Experiments | Establish a GLM-5.2 baseline and compare it with the memory system's sustained bandwidth. Established a July baseline and measured memory bandwidth near 152 GB/s. |
| Jul 14 | [July 14, 2026 (mid-day) — v2 crash forensics and the v3 relaunch](2026-07-14-1022-crash-forensics-and-relaunch.md) | Failure investigation | Find why most v2 diagnostic phases crashed and choose a configuration that could complete the next investigation. Recorded failed phases, loading issues, and the revised diagnostic launch. |
| Jul 14 | [July 14, 2026 — Diagnostic runs: batches, placement, and hardware monitoring](2026-07-14-1108-v3-diagnostic-run.md) | Diagnostic run | The diagnostic series contains 94 v3 rows, 50 completed v4 rows, and their hardware monitors; larger effective batches changed the offload result. |
| Jul 14 | [July 14, 2026 (afternoon) — Interpreting v3; the untested ubatch regime](2026-07-14-1213-interpreting-v3.md) | Analysis and plan | Identified the untested larger-microbatch regime and planned a corrected ladder. |
| Jul 14 | [July 14, 2026 (late afternoon) — The ubatch ladder and the economics of the build](2026-07-14-1529-ubatch-ladder-and-economics.md) | Experiments and analysis | Test whether larger effective microbatches let GPU expert offload repay its transfer cost. Prompt throughput rose with effective microbatch size; examined hardware costs. |
| Jul 14 | [July 14, 2026 (evening) — Multi-card placement experiments and the split histogram](2026-07-14-1813-multi-card-placement.md) | Placement tests | Test whether the expert-offload work is concentrated on one GPU and whether distributing it changes throughput. Found concentrated offload and tested device placement; some attempts failed. |
| Jul 14–16 | [evening, July 14 – July 16, 2026 — The H2D result and the patch, interrupted](2026-07-14-to-16-h2d-and-patch-investigation.md) | Experiments and analysis | Compare the available transfer bandwidth with the engine's observed behavior, then determine whether a scheduler change could use that headroom. Measured 65.7 GB/s across four host-to-device streams and developed the patch hypothesis. |
| Jul 21 | [July 21, 2026 — Building and benchmarking the patch](2026-07-21-prefill-patch.md) | Patch experiment | Measure whether distributing offload and removing the routing-index readback wait improve GLM-5.2 prefill. Distribution alone barely helped; the combined patch reached 119.36 tokens/s, a 13.7% gain. |

## August 2026

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Aug 7 | [DeepSeek-V4-Flash-0731: DSpark acceptance and generation speed](2026-08-07-deepseek-v4-dspark-acceptance.md) | Server tests | Compare draft length, accepted tokens, and actual generation time for DeepSeek-V4-Flash-0731. Length two beat length one and five in these captures; acceptance alone did not rank speed. |
| Compiled Aug 8; runs undated | [DeepSeek-V4 DSpark investigation and interval notes](undated-deepseek-v4-dspark-investigation.md) | Compiled investigation | Investigate DSpark settings and preserve the interval between the July patch work and the August common benchmark. Retains early results, unresolved timing conflicts, and later-superseded plans; compiled August 8. |
| Aug 15 | [STREAM re-baseline on the 2 TB population](2026-08-15-stream-rebaseline-2tb.md) | Hardware remeasurement | Check whether expanding Galactus from 1 TB to 2 TB changed the sustained memory bandwidth used in inference estimates. After DIMM replacement, bandwidth was about 148–151 GB/s; capacity rose with little bandwidth change. |
| Aug 15–16 | [the 2 TB common baseline (stock build)](2026-08-15-to-16-common-baseline-2tb.md) | Common benchmark | Establish a comparable five-model reference after the RAM upgrade, then check speculative generation on the same build. One stock build and command structure; also records separate MTP/DSpark tests. |

## September 2026

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Compiled Sep 30 | [Qwen3.8-27B, oMLX, and chassis comparisons](undated-magneto-omlx-qwen38.md) | Compiled experiments | Test Qwen3.8-27B on the M2 Max and examine ANE prefill, MTP, DFlash, and the difference from an external leaderboard. ANE, MTP, DFlash, and thermal observations; compiled September 30, with some run dates unknown. |

## October 2026

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Undated; recorded Oct 21 | [GLM-5.3-Flash: placement mattered more than the build label](undated-glm53-flash-placement.md) | Undated experiments | Compare GLM-5.3-Flash expert placement, cache budgets, and draft settings on Galactus. Automatic placement reached 19.1–19.4 tokens/s; effective tensor placement was not captured. |
| Undated; recorded Oct 21 | [DeepSeek-V4.1: target inference worked; DSpark ran out of VRAM](undated-deepseek41-dspark.md) | Undated experiments | Establish target-only DeepSeek-V4.1 inference and then test whether its DSpark drafter can load and improve generation. Target-only generation completed at 11.2 tokens/s; draft loading ran out of VRAM. |
| Oct 21 | [llamda: the experimental AMD integration](2026-10-21-llamda-amd-fork.md) | Integration record | Records combined model support and patches; separates integration from proven performance. |
| Oct 21 | [Why STREAM bandwidth is not a decode prediction](2026-10-21-bandwidth-and-placement.md) | Analysis | Explains missing execution costs and the measurements needed to test the model. |
| Oct 22 | [GLM-5.3-Flash: revision-identified builds and placement tests](2026-10-22-glm53-build-comparison.md) | Build comparison | Compare the integrated AMD fork with upstream on the same named GLM-Flash export, then test CLI placement and draft length. Dedicated benchmarks were nearly identical; CLI banners exposed build ambiguity. |
| Oct 23 | [DeepSeek-V4.1: expert-cache capacity and Maya DSpark results](2026-10-23-deepseek41-cache-and-maya.md) | Experiments and follow-up | Compare DeepSeek cache behavior and establish a useful Maya configuration, including whether DSpark improves sustained generation. Retains one full cache benchmark, Maya calibration, RAM checks, and October 24 service observations. |
| Oct 23 | [Two large MoE models at roughly 30 tokens/s on four V620s](2026-10-23-maya-glm53-speculation.md) | Maya experiments | Test Maya with the existing GLM-Flash export, measure the effect of its speculative path, and check whether head grouping or draft-expert settings improve on the defaults. Separate runs reached roughly 30 tokens/s; GLM speculation A/B and head/expert sweeps retained. |
| Oct 28–29 | [Maya v1.1.30 recovery, prefill tuning, and the later-release regression](2026-10-28-maya-prefill-and-regression.md) | Sweeps and failure investigation | Restore a working Maya service, tune DeepSeek long-prefill throughput, and investigate the later-release stall. Selected 32K chunks and an 800 MiB ring; partial layout sweep and stream-priority trigger retained. |
| Oct 28 | [Galactus: prioritizing Maya and reassessing upstream performance work](2026-10-28-maya-priority-and-upstream-status.md) | Decision and source review | Prioritized Maya; records dated upstream status and model choices without claiming new tests. |
| Undated run; recorded Oct 29 | [GLM-5.3-Flash: 80 GiB expert-cache benchmark and investigation](2026-10-29-glm53-expert-cache.md) | Benchmark and analysis | Document GLM-Flash performance with an 80 GiB requested expert cache and determine what evidence would establish whether that cache helps. The benchmark is retained, but no matching cache-off control establishes its effect; run date unknown. |
| Oct 29 | [Maya v1.1.36: production configuration and uncached 64K confirmation](2026-10-29-maya-production-64k.md) | Production check | Confirm that the selected DeepSeek production settings complete a fresh long request at the expected prefill rate. 65,568 prompt tokens, zero cache hits, 488.6 tokens/s prefill. |
| Oct 29 | [Open WebUI: working configuration, Code Interpreter regression, and remaining work](2026-10-29-openwebui-settings-and-next-steps.md) | Service discussion | Records restored responses and pending integration work; not an inference benchmark. |
| Oct 29 | [Research questions: learned memory, sparse attention sharing, and active experts](2026-10-29-sparse-architecture-research.md) | Research plan | Four proposed experiments; none run in this record. |
| Oct 29 | [llamda: what the experimental fork established, and what it did not fix](2026-10-29-llamda-experimental-fork-retrospective.md) | Retrospective | Withdraws the performance announcement; preserves code history, failures, and unfinished comparisons. |
| Oct 30 | [Maya v1.1.41: short benchmark and uncached 32K/64K production tests](2026-10-30-maya-v1141-benchmark.md) | Production tests | Check that Maya v1.1.41 completes both its built-in benchmark and the saved long requests without a meaningful loss of the working prefill performance. Two fresh 32K/64K passes; 64K prefill 469.6 and 485.6 tokens/s; original responses now retained. |

## November 2026

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Nov 2 | [Maya v1.1.49: production replay and the two-pair utilization question](2026-11-02-maya-v1149-production-recheck.md) | Production tests | Check whether the updated DeepSeek installation retains its long-prefill performance, and put the observation of alternating GPU pairs in context. Fresh 32K/64K replays at 461.1 and 480.9 tokens/s. |
| Nov 2 | [Maya v1.1.49: one versus four GPUs and reduced expert-cache budgets](2026-11-02-maya-gpu-cache-matrix.md) | Allocation matrix | Measure the performance cost of giving DeepSeek and GLM-Flash fewer GPUs and smaller requested expert-cache budgets. Nine variants completed; 54 long observations and nine short reports; next GLM case failed loading. |
| Nov 2–8 discussion | [Maya releases, memory budgets, and alternative deployment plans](2026-11-02-maya-release-and-capacity-planning.md) | Planning, November 2–8 | Keeps release claims, memory proposals, and operational guidance separate from measurements. |
| Nov 1–3 | [November 1–3: Gufo on SilverSurfer, memory allocation, and Jan](2026-11-02-silversurfer-gufo.md) | Experiments, November 1–3 | Bring Qwen3.8-Flash-Next up on SilverSurfer and test whether the shared-memory laptop can serve it usefully on battery. Resolved successive loading obstacles; retained a 318-token answer at 29.85 tokens/s on battery. |
| Nov 5 | [DeepSeek-V4.1-Flash: style optimization and optional refusal reduction](2026-11-05-deepseek-style-and-heretic-plan.md) | Evaluation plan | Proposed accuracy and style checks; no model modification or evaluation run. |
| Nov 5 | [Maya v1.1.54: prefill sweep and HIP peer all-reduce failure](2026-11-05-maya-v1154-prefill-and-peer-ar.md) | Prefill sweep and startup failure | Test whether new prefill, kernel, and transport switches improve DeepSeek on four V620s, then isolate the HIP peer-probe limitation with more reserve memory. Default 64K prefill 479.4 tokens/s; no useful gain; peer retest failed before readiness. |
| Nov 9 | [Throughput bounds and alternative Qwen deployments](2026-11-09-throughput-models-and-alternative-hardware.md) | Planning and estimates | Keeps bandwidth estimates, Qwen alternatives, and external quality claims distinct from local tests. |
| Nov 9 | [DeepSeek on two V620s: baseline and requested TP4 fallback](2026-11-09-deepseek-tp2-and-tp4.md) | Layout tests | Test whether DeepSeek can keep useful speed on GPUs 0 and 1, freeing the other pair, and check whether a requested TP4 layout actually activates. 29.0 tokens/s built-in mean generation; requested TP4 later executed TP2. |
| Nov 9 | [DeepSeek RDNA 2 FP16-output prefill: measured speedup and fidelity limits](2026-11-09-deepseek-fp16-prefill.md) | Precision-path test | Measure whether the RDNA 2 FP16-output path speeds DeepSeek dense prefill operations. Optimized medians 389.55/501.95/589.70 tokens/s at 8K/16K/32K; control and quality limits retained. |
| Nov 9 | [Full GLM-5.3 Q4: allocation failures, legal layer boundaries, and a slow working split](2026-11-09-full-glm53-placement.md) | Model and placement tests | Test the full GLM-5.3 Q4 model on Galactus and determine whether correcting the layer split makes it practical for the intended interactive use. A corrected four-card split completed at 6.7 generation tokens/s; returned to Flash for daily use. |
| Nov 9 | [GLM-5.3-Flash on two V620s: MTP, reserve reduction, and cancellation](2026-11-09-glm-flash-reserve-and-speculation.md) | Memory and speculation tests | Establish GLM-Flash on the second GPU pair, reclaim reserve memory for experts, and test speculative cancellation. Lower reserve raised observed mean generation to 21.5 tokens/s; cancellation was slower. |
| Nov 9 | [GLM-Flash prefill: requested versus effective chunks and the failed first harness](2026-11-09-glm-flash-prefill-chunks.md) | Workspace and chunk tests | Test whether larger effective chunks improve GLM-Flash long-prefill throughput, as they had for DeepSeek. Verified 16K chunks were about 21% slower on long prompts; kept 8K effective chunks. |
| Nov 9 run; reported Nov 10 | [GLM-Flash FP16 versus INT8 KV: repeated prefill and decode observations](2026-11-09-to-10-glm-flash-kv-cache.md) | KV comparison | Compare FP16 and INT8 attention KV caches for GLM-Flash after selecting 8K effective chunks. INT8 had 11–13% higher mean generation across paired samples; no local quality-equivalence test. |
| Nov 10 | [Galactus: GLM-Flash INT8 KV and DeepSeek on separate GPU pairs](2026-11-10-maya-dual-model-production.md) | Deployment record | DeepSeek on GPUs 0–1 and GLM-Flash INT8 KV on 2–3; overlapping-load throughput unmeasured. |

## Reference captures with mixed or unknown dates

| Date or period | Record | Kind | Purpose and outcome |
|---|---|---|---|
| Mixed and unknown dates | [Maya archive: upstream reference tests and unassigned live logs](undated-maya-reference-benchmarks-and-live-logs.md) | Reference evidence | Separates upstream V100 tests and loose logs from dated Galactus measurements. |

## Supporting material

[Model results](../README.md) collect measurements by model. [Data tables](../data/) retain individual observations. [Raw captures](../raw-logs/) and [source notes](../reference-notes/README.md) preserve the evidence and its origin. The [old filename map](filename-map.md) helps resolve earlier links.
