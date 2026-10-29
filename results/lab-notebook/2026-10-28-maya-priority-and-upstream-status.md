# Galactus: prioritizing Maya and reassessing upstream performance work

**Recorded:** October 28, 2026 (America/New_York). **Status checked:** approximately 23:09 EDT, October 28 / 03:09 UTC, October 29. **Machine:** Galactus, EPYC 7713, 2 TiB RAM, four Radeon Pro V620 GPUs, Proxmox LXC. This is a synthesis and engineering decision, not a new benchmark run.

I am prioritizing Project Maya for Galactus. The October measurements make it the most promising working platform for my large models, and its existing Qwen3.8 and MiniMax M3.1 execution paths substantially reduce the model-coverage argument for continuing to invest primarily in llAMDa/llama.cpp. I will retain llama.cpp as a compatibility backend and a place to evaluate upstream improvements, but I do not have evidence that the pending patches will collectively reproduce Maya's performance on this machine.

For Qwen, I will use **base Qwen3.8-Flash-Next, not CYBER-FROST**. Maya already implements the model family through its packaged CYBER-FROST configuration, which makes base-checkpoint compatibility a reasonable expectation. Loading, conversion, template handling, and output correctness with the base checkpoint still need validation; this entry does not claim that I have run that configuration successfully.

## Evidence and scope

This entry memorializes the October 27–28 discussion, [VLLM RAM Offload Support](https://chatgpt.com/c/6ae151e4-b7cc-83e9-9dbc-c2bcf2b10600), together with my subsequent clarification about using base Qwen. The accessible conversation retrieval was partial and supplied no older-page cursor. I therefore use the notebook's retained benchmark captures for measurements and the upstream sources below for integration and review status. Assistant statements in the conversation are leads to verify, not primary experimental evidence.

Dates on upstream events below are UTC unless otherwise stated. The status snapshot uses my local October 28 date, so an update shortly after midnight UTC on October 29 can still belong to this entry. PR descriptions and discussions can change after this record; the head revisions at the end identify the versions examined. Review activity and GitHub's mergeability flag do not establish that every required check passed or that a maintainer will merge a change.

## What Galactus actually measured

All rates are tokens/s. The linked entries retain commands, logs, and the distinction between printed measurements and conversation-only summaries.

| Model and execution path | Prefill | Generation | Evidence and conditions |
|---|---:|---:|---|
| GLM-5.3-Flash UD-Q4_K_XL, llama-bench `184f65360` | pp8192: 284.36 ± 16.33 | tg128: 10.06 ± 0.01; tg256: 10.16 ± 0.04 | CPU expert override, 32 threads, batch/microbatch 8192, two repetitions; no speculative options in command. [October 22](2026-10-22-glm53-build-comparison.md). |
| Same GLM export, llamda-bench `ca3b1fc30` | pp8192: 284.20 ± 16.42 | tg128: 10.07 ± 0.10; tg256: 10.09 ± 0.02 | Same requested benchmark configuration. [October 22](2026-10-22-glm53-build-comparison.md). |
| GLM, CLI automatic placement, requested MTP | Not compared here | 17.8–18.1 | Short ZFS prompt, context 8192, greedy, output limit 256, draft maximum 2. Every completed CLI capture prints `b11829-184f65360`, even commands named `llamda-cli`. Executable names do not establish separate engine results. [October 22](2026-10-22-glm53-build-comparison.md). |
| GLM, Maya v1.1.24, default speculation | 8192-token test: 240–242 | 29.4 and 31.6 | Two benchmark runs; means of three different 256-token answers, thinking off. [October 23 GLM](2026-10-23-maya-glm53-speculation.md). |
| GLM, Maya, speculation disabled | 8192-token test: 241 | 19.9 | Same benchmark suite with `MAYA_GLM_NO_SPEC=1`. [October 23 GLM](2026-10-23-maya-glm53-speculation.md). |
| DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL, llAMDa, requested 32 GiB expert cache | pp8192: 132.66 ± 0.30 | tg1024: 9.51 ± 0.07 | Full command and printed output retained; executable revision absent. [October 23 DeepSeek](2026-10-23-deepseek41-cache-and-maya.md). |
| DeepSeek, llAMDa cache sweep, disabled / 64 GiB | pp8192: 132.54 / 132.82 | tg1024: 10.20 / 11.54 | **Conversation-summary evidence only**, weaker than the 32 GiB capture; 80 GiB reportedly OOM. [Cache record](2026-10-23-deepseek41-cache-and-maya.md). |
| DeepSeek, Maya v1.1.24 benchmark | 2048: 135; 8192: 282 | 20.5 | Three 256-token answers. Surrounding discussion identifies an undrafted baseline; printed benchmark does not itself expose that setting. [DeepSeek record](2026-10-23-deepseek41-cache-and-maya.md). |
| DeepSeek, Maya DSpark calibration, off / on | Not a prefill comparison | 21.2 / 29.3 | 1152 measured output tokens per mode after warm-up; 69% draft acceptance with DSpark. [Calibration](2026-10-23-deepseek41-cache-and-maya.md). |

The strongest acceleration evidence is within Maya: the GLM comparison is approximately **58.8%** faster at 31.6 versus 19.9, and the DeepSeek calibration is approximately **38.2%** faster at 29.3 versus 21.2. These percentages are arithmetic comparisons of the recorded rates, not confidence intervals. GLM's two default runs differ by about 7.5%, which is a reason to retain both measurements rather than promote the faster one to a universal rate.

The cross-engine numbers describe useful operating configurations, but they do not isolate an engine-only effect. Prompts, generated lengths, context allocations, effective placement, cache warm-up, speculation, and timing methods differ. Separate llama-bench pp and tg rows do not measure generation following that pp prompt. The DeepSeek export name describes mixed quantization, and the actual Maya model path and effective residency settings still need capture. Even for GLM's same named export, file identity and effective runtime settings should be recorded for a controlled comparison. The dedicated GLM llama-bench prefill numbers are numerically higher than Maya's; the claim is therefore not that Maya wins every metric.

The memlock investigation also changed several settings together. Unlimited shell limits and a requested locked load do not prove how much memory the inference process locked. Neither SSD bottlenecks nor memlock alone can be assigned responsibility for the earlier poor performance without process residency and I/O evidence. These limits are preserved in the [DeepSeek record](2026-10-23-deepseek41-cache-and-maya.md).

## Upstream model support is distinct from the performance work

**GLM-5.3-Flash is upstream.** [PR #27773](https://github.com/ggml-org/llama.cpp/pull/27773) merged September 30 at 06:20:33 UTC as `649dcb1036fd47d01a83b56c5e35913e65c277ba`. It adds GLM5-Next support and preserves NextN tensors, but explicitly separates MTP into [#27917](https://github.com/ggml-org/llama.cpp/pull/27917). That MTP PR remains open and draft at the snapshot. Local builds with additional MTP patches must not be described as evidence that this particular upstream MTP PR merged.

**DeepSeek V4.1 is also upstream.** [PR #31086, “model: DSV4.1 flash”](https://github.com/ggml-org/llama.cpp/pull/31086), merged October 25 at 01:43:51 UTC (October 24 at 21:43 EDT) as `74bca3950054eed1795f525cb9cf06bfe796492d`. Its [changed files](https://github.com/ggml-org/llama.cpp/pull/31086/files) include a DeepSeek V4.1 converter and model implementation, V4.1 DSpark conversion handling, and parser/vision changes. This corrects any earlier characterization of V4.1 integration as still pending. It does not establish that the October 22–23 local builds contained this later merge, or that current upstream matches Maya's throughput.

## Core performance PRs and merge readiness

These are source-verified states followed by my assessment of their relevance to Galactus. No row predicts a measured V620 speedup.

| PR | State at snapshot and readiness evidence | Expected relevance; still a hypothesis on Galactus |
|---|---|---|
| [#31188: fused MoE gate/up matmuls and GLU](https://github.com/ggml-org/llama.cpp/pull/31188) | **Merged October 28, 13:04:29 UTC**, `f5222f25c18da062ab2b8a2dbbfbf4feae398566`. The earlier “closest to merge” description is obsolete. | CUDA MMQ prompt-processing optimization. Author reports roughly 5.7–17.2% gains on RTX 5090/GB10 configurations; these are neither V620 measurements nor a forecast of decode gains. |
| [#30881: batched tensor-copy API](https://github.com/ggml-org/llama.cpp/pull/30881) | Open, non-draft, mergeable in the API snapshot. On October 28, `ggerganov` [planned to merge the following day if CI and further review permitted](https://github.com/ggml-org/llama.cpp/pull/30881#issuecomment-6471922611). Strongest explicit near-term merge signal among the remaining open items. | Useful transfer infrastructure, with backend hooks and generic fallback. It does not itself provide adaptive expert placement or establish an inference speedup. |
| [#30966: CPU execution of missed experts and tuning](https://github.com/ggml-org/llama.cpp/pull/30966) | Open, non-draft; API reports not mergeable. `pwilkin` [approved October 22](https://github.com/ggml-org/llama.cpp/pull/30966#pullrequestreview-5615263261). Subsequent fixes and testing continued; one Vulkan reporter [confirmed a clean rebuild resolved corruption](https://github.com/ggml-org/llama.cpp/pull/30966#issuecomment-6454256553). Approval does not make the current branch ready to merge. | Directly relevant to deciding whether to upload a missed expert or compute it on the CPU. Cache size and CPU share require hardware-specific measurement; the fastest setting can be CPU share zero. |
| [#31139: maximum allocation size and cache splitting](https://github.com/ggml-org/llama.cpp/pull/31139) | Open, non-draft, API mergeable. `am17an` [accepted the cache changes in a COMMENTED review](https://github.com/ggml-org/llama.cpp/pull/31139#pullrequestreview-5647943211); the backend-interface discussion remains. This is not a blanket approval or a scheduled merge. | Enables caches otherwise blocked by per-allocation limits, particularly the reported RADV/Vulkan case. No basis to assign a routine HIP throughput gain or identify it as the cause of Galactus's 80 GiB OOM. |
| [#30971: adaptive single-graph MoE offload](https://github.com/ggml-org/llama.cpp/pull/30971) | Open **draft**, API not mergeable. Author presents it as an architectural discussion with a large change set. | Relevant exploration of CPU/GPU scheduling; not a near-term drop-in performance promise. |
| [#30594: hybrid expert cache and profile prefilling](https://github.com/ggml-org/llama.cpp/pull/30594) | Open, non-draft, API not mergeable. | Relevant residency/cache work, but the description limits its hybrid path to one-token batches and warns that speculative multi-token verification does not benefit and can run slower. |
| [#29963: pipeline parallelism with host-resident experts](https://github.com/ggml-org/llama.cpp/pull/29963) | Open, non-draft, API mergeable; no merge commitment established here. | Author reports substantial long-prompt gains on three RTX PRO 4000 GPUs, with generation approximately unchanged. Worth separating from a single-stream decode strategy. |
| [#31156: per-size graphs and schedulers](https://github.com/ggml-org/llama.cpp/pull/31156) | Open **draft**, API mergeable; retained as draft for feedback. | Addresses repeated graph/scheduler work when verification batch sizes vary, at additional memory cost. Strong mechanism to investigate, but CUDA results do not establish HIP/V620 benefit. |
| [#31233: ClassLRU expert replacement](https://github.com/ggml-org/llama.cpp/pull/31233) | Open **draft**, API mergeable. | Author reports warm-cache improvements of 6–7% and fewer uploads, while noting small-cache limitations. Useful incremental evidence on the author's workloads, not a Galactus result. |

The remaining mergeable flags mean GitHub could construct a merge at inspection, not that reviews, correctness, or all required checks were complete. I did not perform a full CI/review-thread audit. The CPU/cache proposals are also not a validated additive patch stack: they address related execution paths, and integration and performance interactions need testing. Summing their reported percentages would be unjustified.

## Why speculation can still lose

Draft acceptance is only one term in the cost. A useful accounting model is time per committed token approximately equal to draft work, target verification, and scheduling/transfer overhead divided by the number of committed tokens. It is a model for investigation, not a fitted explanation of the Galactus timings. More accepted tokens help only when the extra work per window is sufficiently small.

[#31156](https://github.com/ggml-org/llama.cpp/pull/31156) specifically retains graphs for small batch sizes and gives them separate schedulers to avoid repeated capture, construction, and splitting. Its description reports about 400 MB additional VRAM for one configuration. The [October 27 comparison](https://github.com/ggml-org/llama.cpp/pull/31156#issuecomment-6451382035) reports different benefits across NVIDIA devices and raises scheduler-memory concerns. This supports a concrete overhead hypothesis, not a claim that graph rebuilding accounts for a measured fraction of my V620 runtime.

Locally, increasing GLM's requested CLI draft maximum from two to four reduced the observed rate from roughly 18 to 16.2 tokens/s. Maya's balanced two-GPU speculative head group gave 31.6, compared with 24.6 for one head GPU and 22.9 for three. These observations show that more speculative resources need not improve throughput, but neither experiment separately measured drafting, verification, transfer, and synchronization costs. The [GLM record](2026-10-23-maya-glm53-speculation.md) also preserves the MTP expert sweep and its missing final summary rather than implying a universal optimum.

The next useful comparison must capture those component times, accepted tokens per window, graph rebuilds, expert-cache occupancy/hits, CPU work, and transfer volume. It must also preserve output correctness. A faster engine producing degraded output is not an improvement for the notebook's purposes.

## Maya coverage and the resulting engineering priority

The [Maya source at `f031a23b9682551e744e29d0e9f8d527df49f282` (v1.1.36)](https://github.com/mw00/project-maya/blob/f031a23b9682551e744e29d0e9f8d527df49f282/README.md), committed October 28, documents GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017, MiniMax-M3.1-Flash, and a Qwen3.8 path. This is newer than the **v1.1.24** used in the retained benchmarks; current documentation is not evidence that those benchmarks used later optimizations.

The [pinned settings](https://github.com/mw00/project-maya/blob/f031a23b9682551e744e29d0e9f8d527df49f282/docs/SETTINGS.md) describe MiniMax setup, CPU experts, residency controls, and DSpark calibration, as well as Qwen's native path and MTP packaging. The README still describes AMD as experimental and lists V620/W6800 as not yet validated upstream. My local GLM and DeepSeek results are evidence for those configurations; they do not validate every backend, quantization, or model on V620.

I will concentrate on validating **base Qwen3.8-Flash-Next** and MiniMax M3.1 on Maya, recording exact checkpoint revisions and file hashes, conversion steps, templates, effective settings, and correctness checks. The packaged CYBER-FROST path establishes architecture coverage but does not prove that swapping checkpoints is sufficient. I will then compare the engines with identical prompts, token budgets, context, sampling, and quantization where possible, and with speculation both enabled and disabled. Repeated interleaved trials should report prefill, time to first token, sustained generation, and total request time separately.

This is an allocation-of-effort decision based on a working platform and observed rates. It is not a conclusion that upstream performance work is unimportant. The CPU/GPU expert split and speculative scheduling changes are particularly relevant candidates for later controlled tests.

## CYBER-FROST: evidence considered, checkpoint rejected

The [publisher's model card](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-BF16) identifies a Qwen3.8-Flash-Next derivative with security fine-tuning and a behavioral modification stage. It does not establish a standardized cybersecurity capability result for the exact BF16 checkpoint. Its refusal results are described as development evidence from an earlier suite subsequently corrected; I do not carry the conversation's exact refusal percentages forward as independently verified results.

The card's harness experiment used the NVFP4 sibling, with unchanged weights: verified solves rose from one to five, with ten valid attempts and two infrastructure-invalid tasks. That supports a harness comparison, not superiority of fine-tuned weights over base Qwen, and does not automatically transfer to BF16.

My skepticism is therefore an evidence requirement, not a finding that all fine-tuning is ineffective. I found no matched base-versus-fine-tune capability comparison in the material checked. A persuasive result would hold prompts, tools, inference settings, and scoring constant, distinguish refusals from incorrect answers, and assess regressions. I will use base Qwen; CYBER-FROST is not part of the deployment plan.

## Source revision ledger

The upstream PR links above were read through GitHub metadata and, where cited, discussions or changed files. The following short hashes identify the **head revisions** observed for open PRs; a merge commit reported by GitHub for an open PR is not evidence of an actual merge.

| PR | Inspected head |
|---|---|
| #27917 | `5b8593b5451e` |
| #30881 | `4d18e0b1d536` |
| #30966 | `ae36884b7d08` |
| #31139 | `9f3968b66b59` |
| #30971 | `f8c7d3092252` |
| #30594 | `9177f038969a` |
| #29963 | `5547b255c2a5` |
| #31156 | `7d229e47955e` |
| #31233 | `9ac703c38c28` |

The benchmark sources are the existing October 22–23 entries linked above and their retained raw logs. No new Galactus measurements were collected for this entry. The CYBER-FROST card was checked as a live publisher source on the recording date; it is not an immutable evaluation artifact or independent replication.
