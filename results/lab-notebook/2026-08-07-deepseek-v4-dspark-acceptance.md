# DeepSeek-V4-Flash-0731: DSpark acceptance and generation speed

[All lab records](README.md) · [Related investigation: chapter 3](../../investigations/03-making-speculation-useful.md)

You compared DSpark draft lengths using both acceptance counts and generation time.

## Setup

The saved server runs used maximum draft lengths 1, 2, and 5. Each final timing covers 256 output tokens. Full launch commands and the exact build are missing, and the baseline log is empty.

## Results

The length-one, length-two, and length-five captures measured 12.83, 14.45, and 12.22 tokens/s. Their acceptance fractions were about 78.9%, 67.6%, and 40.0%. A second length-five capture had the same acceptance counts but measured 11.35 tokens/s.

**Capture date:** August 7, 2026, inferred from the original files' modification times. The logs contain elapsed times only, but the saved source files place nmax1, nmax2, and nmax5 at 19:47:02, 19:47:46, and 19:48:34 EDT; the armed-test file was last written at 19:55:26 EDT. You recovered these timestamps on October 23. They establish a file-write sequence, not necessarily the order in which you ran the tests. [Timestamp evidence](../raw-logs/2026-10-22-galactus-archive/source-timestamps.md).

## DeepSeek-V4 DSpark: accepted tokens versus generation speed

Four server logs explicitly load `DeepSeek-V4-Flash-0731-UD-Q8_K_XL` and the matching `DeepseekV4-Flash-20260731-DSpark.gguf`. They report `draft-dspark`, block size 5, and p-min 0. These are **V4-Flash-0731**, not V4.1, and therefore do not fill the missing V4.1 DSpark measurement in the DeepSeek-V4.1 experiments.

| Capture | n-max | Final generation t/s | Accepted / drafted | Acceptance | Mean length |
|---|---:|---:|---:|---:|---:|
| Armed test | 5 | 11.35 | 169 / 423 | 39.953% | 2.99 |
| nmax1 | 1 | 12.83 | 112 / 142 | 78.873% | 1.79 |
| nmax2 | 2 | 14.45 | 146 / 216 | 67.593% | 2.35 |
| nmax5 | 5 | 12.22 | 169 / 423 | 39.953% | 2.99 |

Each final timing covers 256 generated tokens after a 28-token prompt. You used the final `eval time` rate, not the rolling `tg` or `tg_3s` values printed during generation. The files do not retain the full launch command or an exact build revision, so you cannot tell that every other setting stayed fixed.

Increasing n-max from 1 to 2 reduced the acceptance fraction but increased throughput. Increasing it to 5 produced a longer accepted sequence per step while reducing throughput. Acceptance percentage and mean length therefore do not measure the performance benefit on their own: drafting and verification also consume time. The two n-max 5 captures have identical acceptance counts but different elapsed times, which further shows why acceptance counts cannot substitute for repeated timing.

`dspark-baseline.log` is empty. You cannot calculate a target-only speedup from this group of server files. The raw 12.22 t/s result adds an n-max 5 observation to the earlier 11.35 t/s armed test; it does not invalidate the earlier measurement.

Related dated results: [August 16 speculation tests](2026-08-15-to-16-common-baseline-2tb.md#individual-runs-and-benchmark-file-checks).

## Conclusion

Length two was fastest in these runs. Acceptance alone did not rank speed. The missing control and incomplete settings prevent a target-only speedup calculation or a general rule for draft length.
