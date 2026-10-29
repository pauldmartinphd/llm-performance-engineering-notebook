# Two large MoE models at roughly 30 tokens/s on four V620s

**Measured:** October 23, 2026. **Machine:** Galactus, EPYC 7713, 2 TiB RAM, four Radeon Pro V620 GPUs in Proxmox LXC. **Engine:** Project Maya v1.1.24, HIP. [GLM source record](../raw-logs/2026-10-23-maya-glm-and-builds/README.md) · [DeepSeek benchmark and calibration](2026-10-23-deepseek41-cache-and-maya.md).

This is a substantial milestone for Galactus. I now have **GLM-5.3-Flash UD-Q4_K_XL at 31.6 generated tokens/s** and **DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL at 29.3 tokens/s with DSpark** on the same four V620s. Both models exceed the available GPU memory, yet the recorded configurations reach approximately 30 tokens/s using the server's GPU and system memory together. I am using Maya for this stage of the investigation.

The GLM result uses the existing Unsloth Q4_K_XL export. The DeepSeek conversation identifies UD-Q8_K_XL as the intended export; the retained Maya output does not print its actual model path. That filename describes a mixed-format model, not uniform eight-bit weights. The measurements are separate runs, not evidence that both models sustain these rates concurrently. The GLM result is for Flash, not the larger GLM-5.3 flagship.

| Model and quantization | Speculative generation | Without speculation | Measured gain | Workload |
|---|---:|---:|---:|---|
| GLM-5.3-Flash UD-Q4_K_XL | 31.6 tokens/s | 19.9 tokens/s | +58.8% | Maya benchmark, three 256-token answers |
| DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL | 29.3 tokens/s | 21.2 tokens/s | +38% | DSpark calibration, 1152 measured tokens per mode |

Each gain compares modes within its own test. DeepSeek's calibration reports 69% draft acceptance; the GLM benchmark does not print acceptance counts. The GLM and DeepSeek rates therefore describe two successful configurations, rather than a controlled comparison between the models.

## GLM configuration and benchmark

The captured configuration selects GPUs `[0, 1, 2, 3]`, context **32768**, and the pack at `/models/GLM-5.3-Flash/UD-Q4_K_XL/pack`. Its printed engine arguments are:

```text
--glm-pack /models/GLM-5.3-Flash/UD-Q4_K_XL/pack
--max-context 32768
--expert-cache auto
--prefill auto
--kv-streaming on
--expert-profile /opt/project-maya/data/expert-profile-glm.bin
--vram-reserve-mib 3072
```

The printed environment contains `MAYA_GLM_RESERVE_MB=3072`, `MAYA_GLM_RAM_HEADROOM_GB=16`, `MAYA_GLM_PREFILL_SUB=1024`, and `MAYA_GLM_PREFILL_MB=4096`. It does not contain `MAYA_GLM_RAM_RESIDENT`, so the earlier suggestion to enable strict GLM RAM residency is not an established setting for this result.

The benchmark generates a Python CSV parser, a photosynthesis explanation, and an HTML canvas animation, each limited to 256 tokens with greedy decoding and thinking off. It measures prefill separately at 2048 and 8192 tokens; those are dedicated prefill tests, not processing rates inferred from the short generation prompts. The engine reports 2004 GB RAM and all four V620s.

```sh
./maya.sh --backend hip --bench --config maya-ud-q4_k_xl-hip.json
./maya.sh --backend hip --bench --config maya-glm-no-spec-hip.json
```

The second configuration was prepared with `MAYA_GLM_NO_SPEC=1`. The source records the original settings and the two benchmark outputs, although it does not include a full dump of every variant JSON.

| Printed time | Configuration | Load, seconds | CSV / science / HTML generation, tokens/s | Printed mean | Prefill 2048 | Prefill 8192 |
|---|---|---:|---|---:|---:|---:|
| 16:24 | Initial default run | 77 | 29.6 / 27.0 / 31.7 | 29.4 | 223 | 240 |
| 16:29 | Default speculative path | 78 | 30.7 / 31.1 / 33.1 | 31.6 | 238 | 242 |
| 16:32 | `MAYA_GLM_NO_SPEC=1` | 78 | 19.7 / 20.2 / 19.8 | 19.9 | 239 | 241 |

Speculation improves generation substantially in this comparison while the dedicated prefill rates remain nearly unchanged. The two default runs also differ by about 7.5%, so I retain both rather than present 31.6 as an invariant rate. These are benchmark means across different prompts, not repeated measurements of one identical request, and I did not perform a long-context quality evaluation or sustained service-load test.

## Speculative-head sweep

Maya's [v1.1.24 settings](https://github.com/mw00/project-maya/blob/v1.1.24/docs/SETTINGS.md) define `MAYA_GLM_SPEC_HEAD` as the number of GPUs in the speculative head group, defaulting to half the GPUs. On four cards the default is two. The head works on the draft position while the remaining GPUs finish the current token.

| Printed time | Head GPUs | Generation, tokens/s | Prefill 8192, tokens/s |
|---|---:|---:|---:|
| 16:36 | 1 | 24.6 | 242 |
| 16:29, reused default baseline | 2 | 31.6 | 242 |
| 16:40 | 3 | 22.9 | 242 |

The balanced 2+2 grouping gave the highest observed generation rate. Neither alternative changed the measured 8192-token prefill rate. This supports keeping the default on this configuration; the timings alone do not diagnose the precise bottleneck within either group.

## MTP expert sweep

The same version's documentation gives `MAYA_GLM_MTP_KEEP` a default of two. It limits which high-ranked experts missing from VRAM the draft block computes. Changing that work can affect both draft cost and acceptance, but the logs here measure throughput rather than those components separately.

| Printed time | MTP_KEEP | CSV / science / HTML generation, tokens/s | Mean generation | Prefill 8192 |
|---|---:|---|---:|---:|
| 16:44 | 0 | 29.1 / 29.4 / 32.6 | 30.4 | 238 |
| 16:47 | 1 | 26.2 / 27.8 / 30.7 | 28.2 | 240 |
| 16:29, reused default baseline | 2 | 30.7 / 31.1 / 33.1 | 31.6 | 242 |
| 16:50 | 3 | 29.7 / 29.9 / 31.9 | 30.5 | 235 |
| Timestamp/summary absent | 4 | 28.0 / 27.8 / 29.9 | ~28.6, calculated | 240 |

The final KEEP=4 capture stops before the summary; 28.6 is the arithmetic mean of its three printed rates, not a printed benchmark mean. The default row is reused from the A/B test, not an additional repetition. All variant loads report 78 seconds.

The defaults gave the highest observed result in both sweeps, so I retained them. KEEP=0 and KEEP=3 were close enough to the default that repeated interleaved trials would be needed to rank them reliably. The evidence supports retaining a working configuration more strongly than it supports calling that configuration a universal optimum.

## Operational findings

An initial benchmark refused to run because port 8080 was occupied. The process listing identified the listener as **Open WebUI**, and the GLM benchmark completed after the configuration was moved to port 8100. That failure was not evidence of another Maya engine consuming the GPUs.

The attempted HIP `--calibrate --no-start` returned that the tuning measures the CUDA engine's CPU lane and that this PC was **not tuned**. The reported GLM results therefore come from the retained HIP settings and manual sweeps, not a completed automatic hardware calibration.

The conversation ends with requests to rename the configurations to `maya-glm5.3-flash-hip.json` and `maya-ds4.1-flash-hip.json`, using ports 8082 and 8081 while Open WebUI remains on 8080. Those are the requested deployment names and ports; the source has no confirming output for the final changes. Benchmark provenance uses the filenames actually printed during the runs. Deleting the experimental configurations and benchmark files was also proposed without confirming output. The later [October 24 dashboard](../deepseek-v4.1-flash.md#october-24-dashboard-from-ongoing-use) shows DeepSeek on port 8081, but does not establish either JSON rename or GLM’s move to 8082.

## How this changes the earlier comparison

The [October 22 dedicated benchmarks](2026-10-22-glm53-build-comparison.md) measured approximately 284 tokens/s at pp8192 and 10 tokens/s at tg128/tg256 for both llama.cpp and llAMDa. Their CLI runs reached 17.8–18.1 generated tokens/s with automatic placement and requested MTP. Maya now delivers roughly 30 tokens/s from the same named GLM export, with a speculation-off control that explains a substantial part of its own generation improvement.

The engines still differ in context allocation, prompts, speculation paths, and placement. I therefore do not treat the ratio between Maya generation and a llama-bench tg row as an isolated engine speedup. Nor does Maya lead every metric: the dedicated llama-bench pp8192 rates are numerically higher than Maya's 240–242. The practical milestone is that both large models now generate at roughly 30 tokens/s in the recorded Maya configurations, with the GLM A/B result and DeepSeek calibration showing why speculative execution is worth retaining.
