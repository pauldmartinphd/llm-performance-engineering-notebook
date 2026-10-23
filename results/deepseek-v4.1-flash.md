# DeepSeek-V4.1-Flash on Galactus

## October 23: 29.3 tokens/s with Maya and DSpark

DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL reached **29.3 sustained generated tokens/s** in Project Maya v1.1.24 on four V620s, compared with **21.2 without drafting** in the same calibration. The measured gain was **38%**, with **69% draft acceptance**. Each mode generated 1152 measured tokens after warm-up. The separate undrafted benchmark reported 20.5 generation tokens/s and 282 tokens/s for an 8192-token prefill test.

The [calibration and memory record](lab-notebook/2026-10-23-deepseek41-cache-and-maya.md) preserves the outputs and configuration limits. Together with [GLM-5.3-Flash at 31.6 tokens/s](lab-notebook/2026-10-23-maya-glm53-speculation.md), this gives Galactus two large MoE models running at approximately 30 tokens/s in separate Maya tests.

## Flash-1017 cache sweep, recorded October 23

The newer UD-Q8_K_XL `llamda-bench` capture reports **132.66 ± 0.30 pp8192** and **9.51 ± 0.07 tg1024 tokens/s** with a requested 32 GiB expert cache, 32 threads, layer splitting, and CPU placement for Engram embeddings and routed experts. The source revision is not printed. The accompanying conversation summary reports generation at 10.20 tokens/s without the cache and 11.54 at 64 GiB, plus an OOM at 80 GiB; those entries lack their original command/output captures.

The [cache and Maya comparison record](lab-notebook/2026-10-23-deepseek41-cache-and-maya.md) includes the exact available command, captured Maya benchmark and DSpark calibration (+38% sustained generation), evidence limits. These measurements use a different export and workload from the earlier tests below.

## Earlier Q4_K_M and DSpark tests

I tested the eleven-shard Q4_K_M export from `vcruz305/DeepSeek-V4.1-Flash-GGUF` with the DSpark export from `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF`. My V620 build combined the V620 patches, older DeepSeek implementation, GLM-5.3 MTP support, and my round-robin prefill patch. It reports `b0-unknown`; the captures do not establish the exact commit or run dates.

| Configuration | Generation t/s | Outcome |
| --- | ---: | --- |
| Target only | 11.2 | Completed the 256-token generation test |
| DSpark, n-max 2, draft GPU offload | — | Failed to allocate 2464.20 MiB on ROCm0 while loading the draft |

I used 32 threads, context 8192, flash attention, `--load-mode none`, and greedy decoding for the ZFS explanation prompt. The target-only command did not explicitly specify CPU-MoE placement or tensor splitting. The output contained unfinished reasoning at the token limit; I did not score answer quality.

The log ends with commands for a four-way comparison of DSpark off/on and automatic/forced-RAM placement, but it contains no output from those commands. I therefore have no successful V4.1 DSpark timing in this capture and no round-robin-off comparison.

These results precede the replacement with Unsloth's V4.1-Flash-1017 implementation. They also differ from the earlier [DeepSeek-V4-Flash-0731 results](deepseek-v4-flash.md); those successful DSpark runs cannot substitute for a V4.1 measurement.

[Full experiment record](lab-notebook/undated-deepseek41-dspark.md) · [Original log](raw-logs/2026-10-21-flash-experiments/DeepSeekV4Flash%20Experiments.txt) · [Chronological notebook](lab-notebook/README.md)
