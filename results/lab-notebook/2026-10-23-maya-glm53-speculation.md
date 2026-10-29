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

The conversation ends with requests to rename the configurations to `maya-glm5.3-flash-hip.json` and `maya-ds4.1-flash-hip.json`, using ports 8082 and 8081 while Open WebUI remains on 8080. Those are the requested deployment names and ports; the original conversation has no confirming output for the final changes. The later [recovered shell history](2026-10-23-deepseek41-cache-and-maya.md#changes-after-the-measured-baseline) records the DeepSeek rename and port edit, while the [operational continuation](2026-10-23-deepseek41-cache-and-maya.md#october-24-continuation-live-context-cooling-and-hermes) separates that evidence from the still-unconfirmed GLM port change and Hermes setup. Benchmark provenance uses the filenames actually printed during the runs. Deleting the experimental configurations and benchmark files was also proposed without confirming output. The later [October 24 dashboard](../deepseek-v4.1-flash.md#october-24-dashboard-from-ongoing-use) shows DeepSeek on port 8081, but does not establish either JSON rename or GLM’s move to 8082.

## How this changes the earlier comparison

The [October 22 dedicated benchmarks](2026-10-22-glm53-build-comparison.md) measured approximately 284 tokens/s at pp8192 and 10 tokens/s at tg128/tg256 for both llama.cpp and llAMDa. Their CLI runs reached 17.8–18.1 generated tokens/s with automatic placement and requested MTP. Maya now delivers roughly 30 tokens/s from the same named GLM export, with a speculation-off control that explains a substantial part of its own generation improvement.

The engines still differ in context allocation, prompts, speculation paths, and placement. I therefore do not treat the ratio between Maya generation and a llama-bench tg row as an isolated engine speedup. Nor does Maya lead every metric: the dedicated llama-bench pp8192 rates are numerically higher than Maya's 240–242. The practical milestone is that both large models now generate at roughly 30 tokens/s in the recorded Maya configurations, with the GLM A/B result and DeepSeek calibration showing why speculative execution is worth retaining.


## Setup choices and proposals retained from the source discussion

**Consolidated October 29:** The original AI dialogue has been removed from the current notebook tree. Its terminal evidence is retained in the [extracted captures](../raw-logs/2026-10-23-maya-glm-and-builds/2026-10-23-evidence.md). The following records preserve the setup choices and superseded proposals without treating them as additional experiments.

The initial plan was to compare Flash across llAMDa and Maya using the same weights, then test the full GLM-5.3 flagship separately. The discussion described Flash as approximately 320–321B total / 18B active (`glm5-next`) and the flagship as approximately 754B (`glm-dsa`), with an approximately 467 GB, eleven-shard flagship UD-Q4_K_XL export. Those flagship sizing and support statements were planning claims, not a completed flagship run in this record. Maya support for the flagship remained a separate question.

Before reusing the existing six-shard Flash GGUF, the discussion proposed two Maya-native quantizations:

| Candidate | Size quoted in the discussion | Quoted accuracy relative to FP8 | Quoted next-token agreement with FP8 | Status |
|---|---:|---:|---:|---|
| Maya-M | Approximately 116 GB | 97.9% | 86% | Superseded proposal; no local run here |
| Maya-L | Approximately 156–156.3 GB | 99.2% | 90% | Superseded proposal; no local run here |
| Unsloth UD-Q4_K_XL | Initially ~190 GB; later 199.7 GB | Not supplied | 92.2% against **BF16**, as quoted | Selected export; the agreement figure uses a different reference |

These quoted publisher figures and sizing estimates lacked usable source URLs in the export and were not independently checked during consolidation. They explain the original choice discussion, not a verified quality ranking. The discussion described Maya-M as primarily 2–3-bit routed experts and Maya-L as primarily 3–4-bit, with IQ3_S gate/up experts, IQ4_XS down experts, Q5_K sensitive layers, and Q6_K dense weights. The roughly 28% size difference between 199.7 and 156.3 GB is not a measurement of quality loss. The two agreement percentages cannot isolate a quantization effect because their references differ.

The proposed native setup used `--model Maya-M` or `--model Maya-L`, HIP, GPUs `0,1,2,3`, context 32768, `--no-vision`, and `--no-start`; expected filenames were `maya-maya-m-hip.json` and `maya-maya-l-hip.json`. A later Maya-L proposal added `--vram-reserve-mib 3072` and `--env MAYA_GLM_RAM_RESIDENT=1`. Neither native quantization was needed for the recorded run. The selected setup recipe instead reused the local GGUF directory:

```sh
cd /opt/project-maya
./maya.sh --backend hip --setup \
  --gguf-dir /models/GLM-5.3-Flash/UD-Q4_K_XL \
  --gpus 0,1,2,3 --context 32768 --no-vision \
  --vram-reserve-mib 3072 --no-start
```

The source describes a writable model directory for pack creation and treats non-Maya quantizations as experimental. The subsequent inventory and benchmark verify this particular pack/configuration path; they do not establish support for every GGUF. Using `tmux` during a long setup and stopping the other inference engine before benchmarks were operational suggestions, not additional measured settings.

For the A/B test, the proposed preparation copied the baseline to `maya-glm-mtp-original.json`, then set `env.MAYA_GLM_NO_SPEC` to `"1"` in `maya-glm-no-spec-hip.json`. Benchmark copies were named `maya-bench-mtp.txt` and `maya-bench-no-spec.txt`. `NO_SPEC=1` tests the default speculative execution path against decoding without speculation; it need not mean that the MTP block is removed. The baseline environment printed neither `MAYA_GLM_NO_SPEC` nor `MAYA_GLM_NO_MTP`.

The head variants set `env.MAYA_GLM_SPEC_HEAD` to `"1"` or `"3"` in `maya-glm-head1-hip.json` and `maya-glm-head3-hip.json`. KEEP variants were named `maya-glm-mtp-keep{0,1,3,4}-hip.json`, with benchmark copies `maya-bench-mtp-keep{0,1,3,4}.txt`; the source includes the run loop but not a complete dump of every variant. Head=1 reduced the observed mean by 22.2% and head=3 by 27.5% relative to 31.6 tokens/s. A group imbalance is a hypothesis, not a profiled cause. KEEP changes trade draft work against potential acceptance, but no acceptance counts were captured.

The final cleanup request proposed deleting those experimental JSONs, the original backup, and the named benchmark copies, while retaining the two model configurations. It has no confirming execution output. Renaming configurations also requires updating any startup scripts or services that reference their old names. The recorded route uses localhost; binding Maya to `0.0.0.0` with authentication or using SSH forwarding were alternative access proposals, not confirmed changes. GPU layer distribution, expert-cache allocation, and HIP prefill tuning remained later experiments rather than results of this sweep.
