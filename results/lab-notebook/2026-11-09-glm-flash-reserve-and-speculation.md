# GLM-5.3-Flash on two V620s: MTP, reserve reduction, and cancellation

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

## Purpose and reasoning

Establish GLM-Flash on the second GPU pair, reclaim reserve memory for experts, and test speculative cancellation. Check that native MTP is active before attributing performance to it.

## Experimental design

I used GPUs 2 and 3, Q4 weights, FP16 streaming KV, a 1M configured ceiling, and the built-in greedy three-answer benchmark with up to 256 tokens per answer. I compared 3072 and 1024 MB reserve, then the cancellation option. I read draft-loading/acceptance logs. Automatic CUDA-lane calibration declines HIP tuning.

## Results

The baseline mean generation was 18.7 tokens/s. The smaller-reserve run measured 21.5, while the cancellation run returned to 18.7. All three measured 234 tokens/s at 8K prefill. The log confirmed block-45 MTP loading and active drafting.

## Conclusions

I retained the smaller reserve and default speculation, and did not carry cancellation forward. One run per setting supports an operating choice more strongly than a precise causal speedup. An active drafter is not an MTP-off comparison, and the declined calibration did not create a tuned HIP profile.

## Detailed record

**Experiments: November 9, 2026.**

I configured GLM-5.3-Flash UD-Q4_K_XL on physical GPUs 2 and 3, leaving GPUs 0 and 1 for DeepSeek. The Flash engine used `build-hip/maya`, 1,048,576 configured context, FP16 KV with streaming on, 32,768 resident KV positions, automatic expert caching and layer splitting, and native MTP. That context setting establishes capacity configuration; the built-in benchmark measured only 2K and 8K prompts.

## Baseline, automatic calibration, and MTP

The 21:11 report loaded in 98 seconds and measured 18.7 tokens/s mean decode. The log explicitly records `MAYA_GLM_KV_INT8=0`, `MAYA_GLM_NO_MTP=0`, and `MAYA_GLM_NO_SPEC=0`, loading the block-45 NextN drafter on engine-local CUDA1. It reports 824 speculative positions and 89.3% acceptance. Engine-local device numbering is remapped from the physical pair [2,3]; it does not imply use of physical GPU1.

The supplied explanation describes one-token-ahead native drafting. No multi-token drafting comparison was executed. The calibration command at 21:16 again declined HIP tuning and kept the engine's own settings, just as on November 2. A saved pre-calibration JSON therefore is not evidence that calibration changed a profile.

## Reserve and speculative cancellation

All rates are tokens/s; each built-in run uses three distinct 256-token answers, greedy with thinking off.

| Configuration | Load s | CSV | Prose | HTML | Printed mean TG | PP 2K | PP 8K |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3072 MB reserve, baseline | 98 | 16.2 | 16.2 | 23.6 | 18.7 | 177 | 234 |
| 1024 MB reserve | 96 | 19.3 | 19.4 | 25.7 | 21.5 | 179 | 234 |
| 1024 MB, speculative cancellation | 95 | 16.8 | 17.8 | 21.3 | 18.7 | 177 | 234 |

Reducing reserve increased the observed mean decode by about 15.0%, while 8K prefill was unchanged at the printed precision. The mechanism is plausible: more VRAM is available for experts. One run at each setting does not isolate placement and warm-up variation, so this is a successful configuration change with a favorable measurement, not a precise causal estimate.

The cancellation configuration is retained with `MAYA_GLM_SPEC_CANCEL=1`. It reduced measured mean decode back to 18.7, approximately 13.0% below the 1 GiB reserve run, without a prefill gain. I did not carry it into the selected configuration. The archive confirms DeepSeek's two-GPU FP16 profile already requests `--reserve-mib 1024`, so it did not need another 3-to-1 GiB change at that point.

## What carries forward

I retained the 1024 MB reserve, native MTP and automatic expert placement, then investigated [larger prefill chunks](2026-11-09-glm-flash-prefill-chunks.md). The benchmark comparison itself ran one model at a time. My contemporaneous correction was that I was optimizing residency and model switching, not asking for requests to execute on both models at once. This matters when describing the later cohosted service: two resident endpoints do not imply a measured aggregate concurrent-load throughput.

## Sources and recording dates

**Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
