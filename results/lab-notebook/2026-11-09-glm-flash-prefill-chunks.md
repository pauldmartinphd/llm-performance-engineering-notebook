# GLM-Flash prefill: requested versus effective chunks and the failed first harness

[All lab records](README.md) · [Related investigation: chapter 5](../../investigations/05-tuning-long-prompts.md)

**Experiments: November 9, 2026.**

You compared larger GLM-Flash prefill chunks after they had helped DeepSeek.

## Setup

You checked the chunks actually used with 4096 MB and 8192 MB workspaces. The long sweep compared effective 8K and 16K chunks, with one fresh start per setting, two repeats at 8K/32K/64K, and 128 output tokens. Rows with more than 1% prefix reuse were excluded.

## Results

With 4096 MB of workspace, both larger chunk requests still ran at 8K. An 8192 MB workspace allowed 16K chunks, but those were about 21% slower on the longer prompts.


## Short benchmarks and allocation readback

| Requested chunk / budget | Effective common chunk | Load s | CSV / prose / HTML TG | Mean TG | PP 2K / 8K |
|---|---:|---:|---|---:|---|
| 16K / 4096 MB | 8192 | 95 | 19.4 / 19.4 / 27.1 | 22.0 | 179 / 235 |
| 32K / 4096 MB | 8192 | 95 | 17.5 / 17.0 / 22.2 | 18.9 | 175 / 234 |
| 32K / 8192 MB | 16384 | 100 | 17.6 / 17.7 / 25.7 | 20.4 | 179 / 235 |

Rates are tokens/s. With the smaller budget, both devices used 8192-token chunks and borrowed about 3791 and 3767 MB. With the larger budget, local CUDA0 could use 32768 tokens and CUDA1 16384. The engine reduced both to the smaller size. The requested 32K run was therefore a 16K run. The built-in 8K prompt was too short to compare these larger chunks.

## First harness failed before measurement

The first `bench-glm-long.py` attempt failed to parse the tokenizer JSON (`ModelUntagged`, line 2 column 16). After changes, the engine exited during startup. Cleanup then raised `ProcessLookupError` because the process had already stopped. The traceback did not explain the engine failure. You replaced this runner; it collected no throughput results.

The replacement [runner](../raw-logs/2026-11-10-maya-collection/extracted/maya-glm-prefill-sweep.py) read Maya's timing fields and counted chat tokens through `/api/tokens`. It used a fresh server for each setting, a warm-up, two repeats per length, and 128 output tokens. It excluded more than 1% prefix reuse, so `ok` allowed a small amount of caching. The printed output does not give the cache count for each row.

## Completed long-prompt sweep

The saved result is the twelve-row terminal output. The named directory, `glm-prefill-20261109-220014`, is missing from the archive. All rows printed `ok`; the [CSV transcription](../data/maya-glm-november-long-observations.csv) includes both repeats. The actual chat counts were 8076, 32652, and 65420 tokens.

| Target | 8K chunks PP pair | 16K chunks PP pair | Median change, 16K vs 8K | 8K chunks TG pair | 16K chunks TG pair |
|---|---|---|---:|---|---|
| 8K | 225.3 / 224.9 | 225.4 / 225.9 | +0.2% | 19.2 / 25.6 | 18.8 / 22.1 |
| 32K | 297.7 / 297.3 | 233.8 / 234.0 | −21.4% | 19.5 / 24.0 | 20.4 / 24.0 |
| 64K | 293.1 / 294.9 | 231.9 / 231.8 | −21.1% | 23.5 / 23.9 | 22.5 / 24.0 |

The 16K chunks did not help the short prompt and lost about one fifth of prefill throughput at 32K and 64K. The tests did not identify which internal cost caused the loss.

You stopped at 64K; there was no 128K or 1M test. The [KV comparison](2026-11-09-glm-flash-kv-cache.md) follows with that 8K chunk baseline.

## Sources

[Original files and terminal output](../raw-logs/2026-11-10-maya-collection/README.md).

## Conclusion

You kept 8K chunks and the 4096 MB budget. Larger chunks were slower here; the test did not identify the internal cause. The runner allowed a small amount of cache reuse, so its ok label did not mean exactly zero reuse.
