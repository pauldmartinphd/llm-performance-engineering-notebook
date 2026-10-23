# DeepSeek-V4-Flash-0731: DSpark acceptance and generation speed

**Run date:** Not recorded. **Added:** October 22, 2026. The server logs report elapsed times only. [Archive inventory](../raw-logs/2026-10-22-galactus-archive/README.md).

## DeepSeek-V4 DSpark: accepted tokens versus generation speed

Four server logs explicitly load `DeepSeek-V4-Flash-0731-UD-Q8_K_XL` and the matching `DeepseekV4-Flash-20260731-DSpark.gguf`. They report `draft-dspark`, block size 5, and p-min 0. These are **V4-Flash-0731**, not V4.1, and therefore do not fill the missing V4.1 DSpark measurement in the DeepSeek-V4.1 experiments.

| Capture | n-max | Final generation t/s | Accepted / drafted | Acceptance | Mean length |
|---|---:|---:|---:|---:|---:|
| Armed test | 5 | 11.35 | 169 / 423 | 39.953% | 2.99 |
| nmax1 | 1 | 12.83 | 112 / 142 | 78.873% | 1.79 |
| nmax2 | 2 | 14.45 | 146 / 216 | 67.593% | 2.35 |
| nmax5 | 5 | 12.22 | 169 / 423 | 39.953% | 2.99 |

Each final timing covers 256 generated tokens after a 28-token prompt. I used the final `eval time` rate, not the rolling `tg` or `tg_3s` values printed during generation. The files do not retain the full launch command or an exact build revision, so I cannot establish that every other setting stayed fixed.

Increasing n-max from 1 to 2 reduced the acceptance fraction but increased throughput. Increasing it to 5 produced a longer accepted sequence per step while reducing throughput. Acceptance percentage and mean length therefore do not measure the performance benefit on their own: drafting and verification also consume time. The two n-max 5 captures have identical acceptance counts but different elapsed times, which further shows why acceptance counts cannot substitute for repeated timing.

`dspark-baseline.log` is empty. I cannot calculate a target-only speedup from this group of server files. The raw 12.22 t/s result adds an n-max 5 observation to the earlier 11.35 t/s armed test; it does not invalidate the earlier measurement.

Related dated results: [August 16 speculation tests](2026-08-16-speculation-and-rebench-records.md).
