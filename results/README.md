# Results

These notes collect the measurements, the best configuration I found for each model, and what I could conclude. Each note names the machine: Galactus for llama.cpp, llAMDa and Project Maya; Magneto for oMLX. For machine specifications, see [Hardware](../hardware/README.md). For the method behind the numbers, see [methodology.md](methodology.md). For what transfers to other systems, see [../takeaways/](../takeaways/).

For the fork's outcome, see [the llamda retrospective](lab-notebook/2026-10-29-llamda-experimental-fork-retrospective.md). I no longer present it as an established AMD performance improvement. The historical measurements remain part of the record.

## October 29: DeepSeek production prefill confirmation

Project Maya v1.1.36 on Galactus completed **65,568 prompt tokens with zero cache hits at 488.6 tokens/s prefill**, followed by **24.4 tokens/s decode**. The working configuration retains TP2, a 32768-token chunk, an 800 MiB ring, and `DS4_STREAM_PRIO=0` on four V620s. The [production follow-up](lab-notebook/2026-10-29-maya-production-64k.md) preserves the result and configuration; the approximately 1.8% difference from the earlier ~480 baseline is not an established release speedup.

## October 23: approximately 30 tokens/s with Maya

| Model and quantization | Generation with speculation | Without speculation | Test |
|---|---:|---:|---|
| GLM-5.3-Flash UD-Q4_K_XL | **31.6 tokens/s** | 19.9 tokens/s | Three 256-token answers, greedy, thinking off |
| DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL, as identified in the conversation | **29.3 tokens/s** | 21.2 tokens/s | DSpark calibration, 1152 measured tokens per mode |

Maya v1.1.24 on Galactus reached roughly 30 tokens/s with each model in separate October 23 runs. Maya’s actual DeepSeek model path and effective configuration remain uncaptured; the calibration output establishes the reported throughput and draft acceptance. GLM's speculation A/B gives a 58.8% gain; DeepSeek's calibration gives 38% with 69% draft acceptance. The [October 23 record](lab-notebook/2026-10-23-maya-glm53-speculation.md) includes the initial GLM result of 29.4, all head/KEEP variants, configurations, and limits. The [October 22 record](lab-notebook/2026-10-22-glm53-build-comparison.md) adds dedicated llama.cpp/llAMDa benchmarks with printed revisions and generation-only CLI results.

## DeepSeek-V4.1-Flash-1017 cache record

The [October 23 entry](lab-notebook/2026-10-23-deepseek41-cache-and-maya.md) records **132.66 ± 0.30 pp8192 / 9.51 ± 0.07 tg1024 tokens/s** from `llamda-bench` with UD-Q8_K_XL and a requested 32 GiB expert cache. It preserves the exact command and separately labels summary-only cache-sweep results from the captured Maya benchmark and DSpark calibration. The source revision and individual run date remain unknown.

## DeepSeek-V4.1-Flash and GLM-5.3-Flash results

I ran these V620 tests with the V620 patches, older DeepSeek implementation, GLM-5.3 MTP support, and my round-robin prefill patch merged together. I clarified the build combination on October 22, 2026; the captures do not establish the run dates or the exact V620 source revision.

| Experiment | Result | Full record |
|---|---|---|
| [GLM-5.3-Flash](glm-5.3-flash.md), V620 build, MTP on, automatic placement | 19.1–19.4 generated tokens/s | [Configuration and all 15 launches](lab-notebook/undated-glm53-flash-placement.md) |
| [GLM-5.3-Flash](glm-5.3-flash.md), V620 build, MTP on, forced-RAM experts, no explicit cache allocation | 13.7–14.4 generated tokens/s | [Placement and cache comparisons](lab-notebook/undated-glm53-flash-placement.md) |
| [DeepSeek-V4.1-Flash](deepseek-v4.1-flash.md) Q4_K_M, target only | 11.2 generated tokens/s | [Target run and failed DSpark launch](lab-notebook/undated-deepseek41-dspark.md) |
| [DeepSeek-V4.1-Flash](deepseek-v4.1-flash.md) with DSpark | Draft model failed to load after a 2464.20 MiB allocation failed on ROCm0 | [Failure and proposed test matrix](lab-notebook/undated-deepseek41-dspark.md) |

I did not run a round-robin-off control, so these results do not isolate my patch's contribution. The GLM record also includes comparisons with a separate PATH binary. These short-prompt CLI tests predate the newer Unsloth integration and use a different workload from the normalized baseline below. I retained the [original logs and hashes](raw-logs/2026-10-21-flash-experiments/) and [all GLM launch records](data/glm53-flash-historical.csv).

## The normalized baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

This table uses a common command structure and one build over August 15–16. Thread counts and model files are recorded in the August common baseline; the speculative column uses separate llama-cli runs. [The August common baseline](lab-notebook/2026-08-15-to-16-common-baseline-2tb.md) has the full conditions ledger. The rows are sorted by baseline decode.

| Model | pp8192 (t/s) | tg128 (t/s) | Speculative decode, same build |
|---|---|---|---|
| [MiniMax M2.7](minimax-m2.7.md) † | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| [DeepSeek-V4-Flash-0731](deepseek-v4-flash.md) | 143.54 ± 1.64 | 10.34 ± 0.10 | **14.1 ± 0.7** (DSpark n=3) |
| [Qwen 3.5 397B](qwen-3.5-397b.md) | 249.61 ± 19.78 | 9.37 ± 0.16 | — (MTP does not arm) |
| [Kimi K2.6](kimi-k2.5.md) | 94.23 ± 4.45 | 5.79 ± 0.01 | — |
| [GLM-5.2](glm-5.2.md) (t=32) | 95.99 ± 3.36 | 5.30 ± 0.00 | **6.6 ± 0.3** (MTP n=2) |

† This is the final MiniMax measurement; the model was removed afterward.

Baseline ± values are reproduced from llama-bench. The speculative ± values summarize two llama-cli repetitions, not confidence intervals; those runs used a ZFS prompt, greedy decoding, and 256 generated tokens rather than the baseline’s tg128 test. The August common baseline records the cross-tool comparison and about 9% speculative timing variation.

The model notes retain earlier results with their builds and configurations. They include the July patched prefill (GLM-5.2 at 119.36), the April resident-offload rows (MiniMax at 17.37 t/s, Qwen at about 11.7 t/s), and the April K2.5 numbers.

Memory bandwidth and the active-expert footprint explain part of decode time; the GPU-side cost also differs by model. Prefill is fastest for MiniMax (A10B), followed by Qwen (A17B), V4-Flash, GLM (A40B), and Kimi (A32B). This is not a simple ordering by active parameter count: quantization and architecture also differ. MTP and DSpark improve decode on the two models tested successfully, but their timings vary more than the ordinary decode benchmarks.

## Apple Silicon: oMLX on Magneto

[Qwen3.8-27B](qwen-3.8-27b.md) records a six-context Lightning MTP baseline and later 4K/16K measurements on the 14-inch M2 Max (38 GPU cores, 64 GB). Tuned ANE improved measured prefill about 8–9% in the recorded comparison. DFlash was slower with ANE inactive at runtime; SpecPrefill setup never established a successful local run. These observations apply to the configurations I tested. The thermal investigation and the leaderboard's missing chassis field are recorded separately from throughput measurements in [the Magneto measurements](lab-notebook/undated-magneto-omlx-qwen38.md). I measured these separately from the Galactus normalized baseline.

## The full record

| Path | Content |
|---|---|
| [methodology.md](methodology.md) | The measurement method these results were produced with. |
| [lab-notebook/overview.md](lab-notebook/overview.md) | Hardware and source background; the [chronological index](lab-notebook/README.md) links dated experiments and clearly labeled undated captures. |
| [raw-logs/](raw-logs/) | The primary captures: the diagnostic run, the MTP A/B test, the per-model benchmark logs, and the failure logs. |
| [data/](data/) | CSV and JSONL extracts, plus a readable ledger of the recovered benchmark rows. |
| [lab-notebook/2026-03-23-galactus-build.md](lab-notebook/2026-03-23-galactus-build.md) | Earlier build, device setup, DIMM replacement, and placement trials, recorded separately from the normalized baseline. |
| [reference-notes/](reference-notes/README.md) | Retained authored notes: dated surveys, architecture comparisons, and an unmeasured residency proposal, with a source manifest. |

## April–August experiment archive

I added 46 original files from `tests.tar.gz`, including 270 printed benchmark rows, 11 completed CLI timings, and four DSpark server captures with acceptance counts. The first July diagnostic repeats the 94 rows already extracted here; I have not counted them as additional runs. See the [archive inventory](raw-logs/2026-10-22-galactus-archive/README.md), [complete benchmark ledger](data/galactus-archive-benchmarks.md), and [April](lab-notebook/2026-04-07-to-17-thread-and-batch-tests.md), [July](lab-notebook/2026-07-14-diagnostic-reconstruction.md), and [August/speculation](lab-notebook/2026-08-16-speculation-and-rebench-records.md) entries.

Undated supporting records: [scheduler traces](lab-notebook/2026-07-14-glm52-scheduler-traces.md) and [V4 DSpark acceptance logs](lab-notebook/2026-08-07-deepseek-v4-dspark-acceptance.md).
