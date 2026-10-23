# DeepSeek-V4.1-Flash on Galactus

I tested the eleven-shard Q4_K_M export from `vcruz305/DeepSeek-V4.1-Flash-GGUF` with the DSpark export from `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF`. My V620 build combined the V620 patches, older DeepSeek implementation, GLM-5.3 MTP support, and my round-robin prefill patch. It reports `b0-unknown`; the captures do not establish the exact commit or run dates.

| Configuration | Prompt t/s | Generation t/s | Outcome |
|---|---:|---:|---|
| Target only | 13.5 | 11.2 | Completed the 256-token generation test |
| DSpark, n-max 2, draft GPU offload | — | — | Failed to allocate 2464.20 MiB on ROCm0 while loading the draft |

I used 32 threads, context 8192, flash attention, `--load-mode none`, and greedy decoding for the ZFS explanation prompt. The target-only command did not explicitly specify CPU-MoE placement or tensor splitting. The output contained unfinished reasoning at the token limit; I did not score answer quality.

The log ends with commands for a four-way comparison of DSpark off/on and automatic/forced-RAM placement, but it contains no output from those commands. I therefore have no successful V4.1 DSpark timing in this capture and no round-robin-off comparison.

These results precede the replacement with Unsloth's V4.1-Flash-1017 implementation. They also differ from the earlier [DeepSeek-V4-Flash-0731 results](deepseek-v4-flash.md); those successful DSpark runs cannot substitute for a V4.1 measurement.

[Full experiment record](lab-notebook/undated-deepseek41-dspark.md) · [Original log](raw-logs/2026-10-21-flash-experiments/DeepSeekV4Flash%20Experiments.txt) · [Chronological notebook](lab-notebook/README.md)
