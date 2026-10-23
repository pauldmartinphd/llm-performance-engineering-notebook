# August 16: speculation and rebenchmark captures

**Experiments:** August 16, 2026 for the CLI transcripts; original file modification times place the adjacent benchmark captures on August 15–16. **Added:** October 22, 2026. [Original files](../raw-logs/2026-10-22-galactus-archive/README.md) · [Extracted CLI results](../data/galactus-archive-cli.jsonl).

The archive supplies direct evidence for much of the speculation work already summarized in the notebook. It also contains additional results and filename traps. I used the commands, loaded model paths, and final timing lines to identify each run instead of assuming that a filename described everything inside it.

## August 16: recover the command for every repetition

The file named `dsv4-dspark-n3.txt` contains six completed runs:

| Start time (EDT) | Requested n-max | Prompt t/s | Generation t/s |
|---|---:|---:|---:|
| 12:25:42 | 3 | 12.2 | 14.7 |
| 12:34:55 | 3 | 13.0 | 13.4 |
| 12:35:43 | 2 | 13.2 | 14.3 |
| 12:36:31 | 4 | 13.0 | 13.6 |
| 12:37:19 | 8 | 12.8 | 12.5 |
| 12:38:13 | 1 | 12.8 | 12.9 |

The commands specify the V4-Flash-0731 target and DSpark sidecar, 64 threads, CPU-MoE placement, flash attention, context/batch/microbatch 8192, greedy decoding, and the 256-token ZFS prompt test. The table records the requested draft length; the earlier notebook explains the block-size clamp. These transcripts confirm the run order behind the August common baseline, rather than adding six new repetitions to it.

The GLM file similarly changes n-max despite its `n2` filename:

| Start time (EDT) | Requested n-max | Prompt t/s | Generation t/s |
|---|---:|---:|---:|
| 12:13:27 | 2 | 2.5 | 6.3 |
| 12:17:50 | 2 | 1.9 | 6.9 |
| 12:19:44 | 3 | 2.5 | 6.3 |
| 12:21:35 | 1 | 2.5 | 5.9 |

These commands use GLM-5.2, 32 threads, CPU expert overrides, and the same 8192 context/batch/microbatch and 256-token greedy prompt. The separate target-only GLM transcript reports 1.9 prompt and 5.3 generation t/s. The two n-max 2 runs average 6.6 t/s, matching the earlier summary. I retain the individual rates because two observations do not establish a tight uncertainty estimate.

The Qwen3.5 MTP transcript ends with exit code 1 before the server becomes ready. It supplies no speculation timing. Likewise, the Qwen resident-expert benchmark fails during context creation. Neither failure should enter a throughput average as zero.

## Reconcile the benchmark files with the existing baseline

The MiniMax file exactly matches the existing 418.83 ± 24.11 pp8192 and 15.18 ± 0.18 tg128 rows. The Qwen file matches 249.61 ± 19.78 and 9.37 ± 0.16. The build capture identifies `3653e6d6d (10326)`, compiled with GNU 14.2.0 for Linux x86_64.

The GLM benchmark file contains another pair: **96.17 ± 3.49 pp8192 and 5.21 ± 0.03 tg128**, rather than the existing common-baseline pair of 95.99 ± 3.36 and 5.30 ± 0.00. I preserved both records. Without a run timestamp or explanation tying the two together, I cannot silently replace one with the other or treat them as repetitions from a controlled series.

The file named `deepseek-v4-flash.md` prints `deepseek2 671B BF16`, **553.71 GiB**, and **1026.41 B parameters**, with rates **94.23 ± 4.45** and **5.79 ± 0.01**. Those values match the Kimi K2.6 rows in the existing baseline, not its DeepSeek-V4-Flash rows. The capture does not include a model path that independently resolves the identity. I renamed the preserved file to describe that discrepancy and kept the earlier Kimi attribution as prior notebook context rather than claiming that the filename proves a DeepSeek measurement.

I did not change the normalized baseline or infer a new model ranking from these files. They strengthen the evidence behind some existing rows, add a distinct GLM pair, and make the filename mismatch explicit. The archive also preserves the commands and failures needed to reconstruct how I reached the later configurations.

The separate [V4-Flash-0731 DSpark acceptance logs](2026-08-07-deepseek-v4-dspark-acceptance.md) contain no wall-clock date, but their original modification times place them on August 7. They are not August 16 runs.

The benchmark files were last written August 15 at 13:07:13 EDT (`glm-5.2.md`) and 21:58:56 EDT (`deepseek-v4-flash.md`), then August 16 at 09:01:23 (`qwen-3.5-397b.md`), 09:44:29 (`qwen-3.5-397b-offload.md`), and 09:54:48 (`minimax-m2.7.md`). These file dates do not resolve the model-identity discrepancy above. [Timestamp evidence](../raw-logs/2026-10-22-galactus-archive/source-timestamps.md).
