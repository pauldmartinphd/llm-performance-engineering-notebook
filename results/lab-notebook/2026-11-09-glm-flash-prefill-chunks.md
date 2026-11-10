# GLM-Flash prefill: requested versus effective chunks and the failed first harness

**Experiments: November 9, 2026.** **Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.

I tried to transfer the successful DeepSeek large-chunk approach to GLM-Flash, but the effective allocation and long-prompt measurements favored different settings. Asking for 16K or 32K with a 4096 MB workspace still produced 8K chunks. Increasing the workspace to 8192 MB produced a common 16K chunk, but that setting was substantially slower on long prompts.

## Short benchmarks and allocation readback

| Requested chunk / budget | Effective common chunk | Load s | CSV / prose / HTML TG | Mean TG | PP 2K / 8K |
|---|---:|---:|---|---:|---|
| 16K / 4096 MB | 8192 | 95 | 19.4 / 19.4 / 27.1 | 22.0 | 179 / 235 |
| 32K / 4096 MB | 8192 | 95 | 17.5 / 17.0 / 22.2 | 18.9 | 175 / 234 |
| 32K / 8192 MB | 16384 | 100 | 17.6 / 17.7 / 25.7 | 20.4 | 179 / 235 |

Rates are tokens/s. The first two logs report 8192 on both local devices, borrowing about 3791 and 3767 MB. The larger budget permits 32768 on local CUDA0 and 16384 on CUDA1; the engine then reduces CUDA0 to the split's smallest chunk. Treating the requested 32K as an actual 32K run would mislabel this experiment. The unchanged 8K benchmark cannot resolve the value of chunks larger than that prompt.

## First harness failed before measurement

`bench-glm-long.py` first failed parsing the selected tokenizer JSON (`ModelUntagged`, line 2 column 16). After changes, it reached server startup but exited before readiness, then cleanup raised `ProcessLookupError` while signaling an already-exited process. The pasted server traceback establishes an engine exit; it does not supply the underlying engine failure. No throughput was collected by those failed attempts. I deleted that attempt and requested a replacement based on the earlier DeepSeek sweep.

The replacement [GLM prefill runner](../raw-logs/2026-11-10-maya-collection/extracted/maya-glm-prefill-sweep.py) uses Maya's token endpoints and engine timing fields, one fresh server per configuration, a warm-up, two repetitions per length, and 128 generated tokens. It obtains expected chat token counts through `/api/tokens` and excludes rows with more than 1% prefix reuse. Thus `ok` is not an independent assertion of exactly zero cached tokens. The terminal transcript lacks individual cache counters.

## Completed long-prompt sweep

`glm-prefill-20261109-220014` is named and printed in the conversation but absent from the supplied tar. The retained source is the pasted twelve-row terminal output; all rows printed `ok`. The [transcribed individual observations](../data/maya-glm-november-long-observations.csv) retain both repetitions. Actual reported chat counts were 8076, 32652 and 65420 for the 8K, 32K and 64K targets.

| Target | 8K chunks PP pair | 16K chunks PP pair | Median change, 16K vs 8K | 8K chunks TG pair | 16K chunks TG pair |
|---|---|---|---:|---|---|
| 8K | 225.3 / 224.9 | 225.4 / 225.9 | +0.2% | 19.2 / 25.6 | 18.8 / 22.1 |
| 32K | 297.7 / 297.3 | 233.8 / 234.0 | −21.4% | 19.5 / 24.0 | 20.4 / 24.0 |
| 64K | 293.1 / 294.9 | 231.9 / 231.8 | −21.1% | 23.5 / 23.9 | 22.5 / 24.0 |

The larger chunk does not improve the small prompt and loses about one fifth of sustained prefill at 32K/64K. The transcript does not identify whether workspace, expert borrowing, staging or another kernel effect accounts for the loss, so I do not assign a measured cause. The result is sufficient to retain **8K effective chunks with the 4096 MB budget** for the next experiment. DeepSeek's 32K chunk choice does not transfer to this engine.

I explicitly limited the sweep to 64K; no 128K or 1M request was tested. The [KV comparison](2026-11-09-to-10-glm-flash-kv-cache.md) follows with that 8K chunk baseline.
