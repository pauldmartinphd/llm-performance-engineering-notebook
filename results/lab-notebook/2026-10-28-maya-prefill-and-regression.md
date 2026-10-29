# Maya v1.1.30 recovery, prefill tuning, and the later-release regression

**Experiments:** October 28, 2026; written October 29. **Machine:** Galactus, EPYC 7713, 2 TiB RAM (Maya reports 2004 GB), four Radeon PRO V620 GPUs, HIP/ROCm 7.2.0. **Model:** DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL. **Evidence:** commands, engine excerpts, dashboard screenshots, and sweep output supplied in the conversation. The [October 29 archive](../raw-logs/2026-10-29-maya-collection/README.md) now retains the original CSVs, configs, metadata, and logs. The first-sweep table was initially transcribed from the conversation and has now been checked against the collected summary. Exact unrounded medians remain in that CSV.

## Restoring a usable installation

After updating to v1.1.35, I encountered an engine stall during Maya's built-in benchmark. The model loaded in 312 seconds and generated the first 256-token answer at 19.38 tokens/s, then stopped making progress. The engine reported a 600-second watchdog timeout. Its final markers showed the main thread reading a prompt and all four GPU workers reporting no run yet. Those markers do not establish that no GPU work had been submitted: the upstream report includes a stack inside the serial forward/layer path.

I restored `project-maya-old` to `/opt/project-maya`, checked out v1.1.30, and confirmed the tag with `git describe --tags --exact-match`. The old build was moved to `outputs/before-v1.1.30/build-hip`. I recovered the configuration from `outputs/before-v1.1.30/configs/maya-ds4-hip.json.before-dspark-retest` and rebuilt. The recovered JSON explicitly requests TP=2, GPUs 0–3, 3072 MiB reserve, context 32768, vision off, DSpark off, and `--mmap-experts off`. Its environment contains `DS4_ENGRAM_RAM=1`, `DS4_REQUIRE_ALL_RAM=1`, and `MAYA_DS4_Q8_BATCH=1`; its original port is 8099. This supplies settings missing from the earlier October 23 record. Its old 1.31/3.14 tokens/s calibration metadata precedes the successful RAM-residency retest and is not the performance of the restored installation.

For daily use I requested context **1048576**, vision enabled, DSpark enabled, and port **8081**. Tests use **8082**. These are distinct from the historical 32K text-only baseline. The later dashboard confirms v1.1.30, 1024K context, and port 8081; the collected sweep configs and server logs confirm 1048576 context, DSpark enabled, and images on for these tests. Model weights remain under `/models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL`, with the engine pack in `maya-ds4`.

The supplied v1.1.35 engine log confirms 202.758 GB of locked Engram tables and all 10,938 CPU-lane experts copied into RAM, approximately 205.6 GB, with pinned expert storage. It reports 64 CPU threads and two TP=2 GPU groups. These are direct runtime observations for that run, not a retrospective verification of every October 23 allocation.

On October 29 the directory listing contains both `/opt/project-maya` and `/opt/project-maya-production`. Collection confirms v1.1.30 in `project-maya-production` and v1.1.36 in `project-maya`. However, configs in the production directory still name the executable and working directory under `/opt/project-maya`; that path mismatch must be resolved before treating a launch from the production directory as a v1.1.30 run.

## Performance before the sweep

The v1.1.30 daily-use dashboard showed approximately 434–437 tokens/s for several fresh 100K–110K prompts, and around 30–33 generated tokens/s on several requests. It showed 40 requests over roughly 2.5 hours, no SSD-only experts, and both full rereads and substantial prefix reuse. A cached-prefix request is a different workload from a fresh prefill and cannot be compared using total prompt length alone.

A separate built-in benchmark launched with `DS4_CHUNK=16384` loaded in 300 seconds and reported 123 tokens/s at 2048 tokens, 266 at 8192, and a decode mean of 32.7 over three 256-token answers (32.0, 27.7, 38.3). The collected benchmark engine lines confirm chunks of 16384. These short prompts cannot establish long-prompt throughput or a 1000 tokens/s ceiling.

## Completed chunk and ring sweep

The [first sweep script](../../experiments/maya-prefill-sweep.py) was run as `/opt/project-maya/.venv/bin/python /opt/maya-prefill-sweep.py`. Results were written to `/opt/maya-test-results/20261028-200032`. It tested chunks of 8192, 16384, and 32768 tokens against `DS4_RING_MB` values of 128, 400, and 800 MiB. The 8192/400 baseline was repeated at the end, giving ten engine starts. Each start measured two requests at each of 8192, 32768, and 65536 target prompt tokens after warm-up. The script randomizes intermediate cases and request order, retains engine/server logs and responses, and excludes failed or substantially cached requests from its summary.

All **60 measurements** were reported valid, with **zero failed cases**. The following rates are medians of the two reported repetitions for each case, rounded to one decimal. Baseline start and finish remain separate so drift is visible.

| Chunk tokens | Ring MiB | 8K prefill tokens/s | 32K prefill tokens/s | 64K prefill tokens/s |
|---:|---:|---:|---:|---:|
| 8192 | 400, initial | 271.2 | 406.1 | 434.0 |
| 32768 | 400 | 253.2 | 430.2 | 471.8 |
| 32768 | 128 | 219.0 | 393.5 | 439.1 |
| 32768 | 800 | 266.0 | 438.8 | 480.0 |
| 16384 | 128 | 228.2 | 384.8 | 417.3 |
| 8192 | 800 | 273.1 | 403.3 | 426.6 |
| 16384 | 800 | 271.7 | 432.3 | 464.2 |
| 16384 | 400 | 267.1 | 419.6 | 455.5 |
| 8192 | 128 | 232.5 | 359.9 | 387.3 |
| 8192 | 400, final | 270.6 | 405.4 | 433.7 |

I selected **32768-token chunks and an 800 MiB ring** for subsequent tuning. Against the initial baseline this is approximately 8.1% faster at 32K and 10.6% faster at 64K, with a 1.9% reduction at 8K. At 65536 tokens, dividing prompt length by the measured rates gives approximately 151 seconds versus 137 seconds, or about 14 seconds saved in prefill. This calculation excludes generation and other request overhead.

The nearly identical first and final baselines support the comparison, although two repetitions per length within one engine start do not establish a broad confidence interval. A larger ring alone did not help the 8192-token chunk at 64K. The observed benefit depends on the combination of settings. No result here approaches 1000 tokens/s, and no evidence establishes that rate as attainable on this workload. The collected `maya-ds4-production.json` now contains `DS4_CHUNK=32768`, `DS4_RING_MB=800`, and `--prefill-chunk 32768`; this confirms the saved setting, without proving which config a currently running process uses.

## Internal batching and the next sweep

Inspection of [v1.1.30's model implementation](https://github.com/mw00/project-maya/blob/v1.1.30/src/ds4/ds4_model.cu) shows that chunk length is not a direct equivalent of all llama.cpp batch and ubatch controls. `DS4_ATTN_SLAB_MB` controls the attention workspace and resulting row subdivision; `DS4_W16_MB` controls FP16 weight scratch space used in matrix multiplication. They are memory budgets, not general token microbatch sizes. The indexer separately uses a bounded row count; at the configured 1M maximum context its formula selects 32 rows. I did not identify an environment override for that indexer limit. Changing it would require source modification and a rebuild.

The routing slab constant is 8192 rows and the maximum chunk is 32768. The busy-expert FP16 path associated with `DS4_FAT_ROWS` and `DS4_FAT_PIECE` is disabled under HIP, so it is not a useful tuning axis for this machine.

The [second sweep script](../../experiments/maya-prefill-tp-sweep.py) keeps the ring at 800 MiB and tests chunks 16384/32768, TP 1/2/4, and attention-slab/weight-scratch profiles `auto`, `32:64`, `64:128`, and `128:256` (MiB). That gives 24 combinations plus a repeated baseline, or 150 timed requests at the same three prompt lengths and two repetitions. It records actual TP layout and workspace diagnostics and excludes missing or mismatched layouts from its summary. Local mock tests exercised the harness; they are not GPU performance validation. **The sweep was intentionally stopped early after the trend was clear.** The archive contains 15 completed cases and an interrupted sixteenth, not a completed 25-case experiment. Of 90 timed rows, 54 have verified layouts and 36 are excluded because TP=4 fell back to TP=2. The closing baseline was not reached. The partial results are analyzed below.

Both scripts isolate test configurations and output, use port 8082, and guard against a concurrent daily-use server. They use existing model files and the installed binary. Their source is retained as delivered in `experiments/`, and exact server copies are retained in the archive. They differ only by a trailing blank line; each server copy matches the hash recorded in its sweep metadata.

## Quality constraint and additional candidates

I will not trade answer quality for speed. `MAYA_DS4_PROMPT_SKIP` stays at **1**, the default exact-skip mode. Mode **2** performs more aggressive skipping and is excluded from future tests and production changes. This is an explicit constraint, not merely a benchmark preference.

| Setting | Verified behavior in v1.1.30 | Decision |
|---|---|---|
| `MAYA_DS4_Q8_BATCH=1` | Batches activation rounding; documented as bit-for-bit equivalent | Retain the existing setting |
| `DS4_LT=0` | Disables hipBLASLt and falls back to hipBLAS | Unmeasured candidate; backend changes may change floating-point rounding |
| `MAYA_DS4_PROMPT_SKIP=2` | Aggressive prompt skipping; source warns that answers shift | Excluded because it changes computation and can affect quality |
| `MAYA_DS4_DSPARK_CUM=0.3` | Lowers the cumulative draft-confidence threshold from 0.5; target verification remains | Unmeasured decode candidate, not a prefill optimization |

These interpretations come from the [settings documentation](https://github.com/mw00/project-maya/blob/v1.1.30/docs/SETTINGS.md), [Lt implementation](https://github.com/mw00/project-maya/blob/v1.1.30/src/ds4/ds4_lt.cu), and [drafting implementation](https://github.com/mw00/project-maya/blob/v1.1.30/src/ds4/ds4_main.cpp). Scheduling and TP changes are not intended quality reductions, but floating-point ordering can change generated wording; that is not proof of bitwise equivalence. None of the newly proposed settings is recorded as applied to production.

## Regression investigation: hypothesis, not root cause

[Upstream issue #621](https://github.com/mw00/project-maya/issues/621) reports short-prompt hangs starting in v1.1.31 on two V620s, while v1.1.30 works. The reporter's RAM capacity, quantization, and effective TP differ from mine. The matching GPU family and release boundary make it relevant, but do not establish an identical cause. My pasted local failure is from v1.1.35; a controlled local v1.1.31 capture remains to be collected.

The [v1.1.30 to v1.1.31 comparison](https://github.com/mw00/project-maya/compare/v1.1.30...v1.1.31) includes [commit 6227775](https://github.com/mw00/project-maya/commit/6227775e41a189c49b54f01375ae8045277368a4), which changes layer streams to the highest available priority by default and also changes CPU-lane batching and response handling. The stream-priority change is a plausible first isolation target: a waiting kernel and the work that satisfies its dependency might interact differently under the new scheduling. This is a hypothesis from code inspection, not a demonstrated deadlock mechanism.

A controlled test would compare default v1.1.31 with the same configuration and only `DS4_STREAM_PRIO=0`, alternating fresh starts. `DS4_STREAM=0` is a different switch and does not test this hypothesis. If priority does not explain the failure, the CPU-lane changes in the same commit remain candidates; a parent-versus-commit test can narrow the boundary. Programmatic dependent launch is disabled under HIP in this implementation, so it is a weaker direct explanation for V620. No commit bisect is documented. The later evidence now includes a successful v1.1.36 benchmark with priority disabled, and my October 29 confirmation that priority enabled causes a deadlock. That identifies a reproduced trigger and workaround; it does not yet establish the exact synchronization mechanism.

## Collected partial TP/workspace results

I stopped the overnight sweep because the trend was already clear. The verified cases below use an 800 MiB ring. Workspace profiles are attention-slab MiB : FP16 scratch MiB; `auto` retains the engine's choice. Values are medians of two requests at each target length, derived from `status=ok` rows only.

| Case | Chunk | TP | Workspace profile | 8K prefill | 32K prefill | 64K prefill |
|---|---:|---:|---|---:|---:|---:|
| 01 | 32768 | 2 | auto | 262.10 | 440.50 | 479.80 |
| 03 | 16384 | 2 | 32-64 | 273.25 | 434.65 | 466.80 |
| 04 | 32768 | 2 | 128-256 | 252.60 | 436.30 | 475.50 |
| 06 | 16384 | 1 | 64-128 | 165.65 | 319.45 | 388.70 |
| 07 | 16384 | 2 | 64-128 | 269.95 | 432.65 | 463.90 |
| 09 | 32768 | 1 | auto | 165.25 | 261.10 | 322.00 |
| 13 | 32768 | 2 | 32-64 | 265.55 | 442.00 | 482.65 |
| 14 | 16384 | 1 | 128-256 | 165.95 | 320.50 | 390.25 |
| 15 | 32768 | 1 | 32-64 | 164.95 | 259.70 | 323.15 |

TP=1 was consistently slower than the tested TP=2 configurations. The best verified 64K result was 482.65 tokens/s at chunk 32768, TP=2, and profile 32:64, compared with 479.80 for the opening auto-workspace baseline: about **0.6%**. At 32K the corresponding difference was about 0.3%. With two repetitions per length and no closing baseline, I do not consider this sufficient evidence to replace the automatic workspace selection. The paired workspace profiles also do not isolate which of their two budgets caused a difference.

The TP=4 rows are especially important: the engine explicitly reports `tensor parallel 4 -> 2`. Similar performance in those rows is not evidence that TP=4 and TP=2 are equivalent. The run did not measure a genuine TP=4 layout. The notebook therefore retains TP=2, the selected 32K chunk and 800 MiB ring, and automatic workspaces as the supported baseline.

The logs also report a missing `TensileLibrary_lazy_gfx1030.dat` file in hipBLASLt. Successful requests follow those messages, so they do not by themselves identify the deadlock. They make `DS4_LT=0` a relevant backend-isolation candidate, but no controlled Lt-off comparison is included here.

## Stream-priority bug and the v1.1.36 capture

On October 29 I confirmed that `DS4_STREAM_PRIO=1` causes a deadlock. The captured v1.1.36 benchmark uses `maya-ds4-production-no-stream-prio.json`, which sets **`DS4_STREAM_PRIO=0`**, and completes all three decode answers and both prefill tests. It reports **31.1 tokens/s** mean decode, **195 tokens/s** at 2048-token prefill, and **334 tokens/s** at 8192. The archive supports the successful workaround run; the priority-on reproduction is my direct report, rather than a paired failure capture positively identified in this bundle.

This upgrades stream priority from a code-inspection hypothesis to a reproduced trigger on my setup. The exact deadlock mechanism and introducing commit remain unresolved. The v1.1.36 short benchmark is not a controlled release-performance comparison with the earlier v1.1.30 benchmark: chunk size, layer placement, and other conditions differ. It also does not establish indefinite production stability.

The [October 29 production follow-up](2026-10-29-maya-production-64k.md) records the subsequent 65,568-token request: **488.6 tokens/s prefill**, **24.4 tokens/s decode**, and **zero cache hits**, using the same named no-stream-priority production configuration. It retains TP2, 32768-token chunks, an 800 MiB ring, and port 8081. This is conversation-derived evidence separate from the archive and preserves approximately the earlier ~480 tokens/s prefill performance.

## Reproducibility and remaining limits

The [evidence ledger](../raw-logs/2026-10-29-maya-collection/README.md) records archive validation, per-file hashes, exact script provenance, checkout and binary identities, and the derived partial summary. Both sweeps record v1.1.30 commit `8621e33159f426da2d376be4ec012b0ce8fe2948` and the same engine binary hash. Prompt targets differ from actual API token counts: for example, a target 32768 produces 32800 prompt tokens in the retained rows. Raw counts and timings are preserved.

The metadata's purported llama.cpp revision repeats the parent Maya revision and must not be treated as an independently verified dependency commit. Model shard hashes and a full model inventory remain missing. The archive excludes builds and weights; it is not a complete installation backup. No `PROMPT_SKIP=2` override appears in the collected JSON configs, and the policy against quality-reducing optimizations remains in force.

The archive contains a point-in-time snapshot from 14:15 Eastern on October 29. Cleanup may remove completed, archived sweep directories, but must preserve the archive, checkouts, current configs, scripts, weights, and any newer logs. An Open WebUI URL-fetch stall discussed during this session occurred while Maya was idle; it is not evidence of the engine regression or a prefill measurement.
