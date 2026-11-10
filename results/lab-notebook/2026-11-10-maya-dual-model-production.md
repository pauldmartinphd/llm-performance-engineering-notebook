# Galactus: GLM-Flash INT8 KV and DeepSeek on separate GPU pairs

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

**Record type:** Reported deployment state; measurements are linked to their experiments.

**Deployment status reported November 10, 2026.**

The testing led me to host **GLM-5.3-Flash UD-Q4_K_XL with INT8 KV on GPUs 2 and 3**, alongside **DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL on GPUs 0 and 1**, using separate Project Maya services on Galactus. This operating state is confirmed by my November 10 notebook-update request. The supplied files establish each pair's configuration and individual performance; they do not contain a synchronized two-process readback or an overlapping-request benchmark.

## Selected layout

| Setting | DeepSeek service | GLM service |
|---|---|---|
| Model | DeepSeek-V4.1-Flash-1017 | GLM-5.3-Flash |
| Weight quantization | UD-Q8_K_XL | UD-Q4_K_XL |
| Physical GPUs | 0, 1 | 2, 3 |
| Port | 8081 | 8082 |
| Engine | `build-hip/maya-ds4` | `build-hip/maya` |
| Configured context | 1,048,576 | 1,048,576 |
| GPU reserve | 1024 MiB per GPU | 1024 MB per GPU in GLM's setting |
| Speculation | DSpark on | Native block-45 NextN MTP |
| Prefill | 32K chunks, 800 MiB ring | 8K effective chunks, 4096 MB budget, 1024 sub-batches |
| Precision setting tested | FP16-output dense prefill with nonfinite checks | INT8 KV; FP16 KV fallback retained |
| KV residency | Engine-specific compressed representation | Streaming on, 32,768 positions resident |

“DeepSeek 16-bit” here refers to the selected FP16 prefill computation path in the supplied configuration. The model weights are still Q8, and the supplied DeepSeek config has no analogous `--kv fp16` flag. It would be inaccurate to describe the whole DeepSeek model as a 16-bit weight deployment. GLM's INT8 setting specifically quantizes its attention cache.

## Why this arrangement

The [November 2 matrix](2026-11-02-maya-gpu-cache-matrix.md) showed the substantial loss from reducing each model to a single GPU. The [DeepSeek TP2 run](2026-11-09-deepseek-tp2-and-tp4.md) then retained 29.0 tokens/s built-in mean decode with two GPUs. Its [FP16-output prefill test](2026-11-09-deepseek-fp16-prefill.md) reached median 389.55 / 501.95 / 589.70 tokens/s at 8K / 16K / 32K targets.

The [full GLM investigation](2026-11-09-full-glm53-placement.md) completed but remained too slow for the intended use. GLM-Flash instead reached 21.5 tokens/s built-in mean decode after reducing reserve, and the [long-prompt comparison](2026-11-09-to-10-glm-flash-kv-cache.md) favored INT8 KV for decode while keeping approximately 292 tokens/s at 32K and 64K targets. Those figures belong to different workload families and are not a combined-system benchmark.

## Production names and retained evidence

The conversation requested `maya-ds4-tp2-production.json` and `maya-ds4-tp2-fp16-production.json`, followed by `maya-glm-5.3-flash-fp16-production.json` and `maya-glm-5.3-flash-int8-production.json`. The source archive retains the corresponding experimental configs, but not these final four filenames. The intended GLM variants both use port 8082, `MAYA_GLM_KV_INT8=0/1`, and `--kv fp16/int8` respectively. The retained INT8 config already prints the selected pair, port and reserve.

The recorded historical launch forms are:

```sh
./maya.sh --backend hip --config maya-ds4-tp2-fp16-production.json
./maya.sh --backend hip --config maya-glm-5.3-flash-int8-production.json --port 8082
```

These are requested production names from the discussion, not newly fabricated config files in the notebook. Both engines must be built separately. The November 10 export ends with build and tmux questions, without a final combined launch transcript.

## What remains unmeasured

Hosting both services does not establish throughput while both generate at once. My November 9 instruction prioritized switching between resident models; shared CPU and DDR4 contention under overlapping requests remains unmeasured. Neither model was tested at its configured 1M ceiling in these experiments. Local numerical-quality checks for INT8 KV and FP16-output prefill are absent. The DeepSeek prompt-skip log message alone does not identify the excluded aggressive mode; the [layout entry](2026-11-09-deepseek-tp2-and-tp4.md#prompt-skip-diagnostics) explains that separate issue. I retain the operating choice and those measurement limits together.

## Sources and recording dates

**Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
