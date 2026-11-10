# Maya v1.1.49: one versus four GPUs and reduced expert-cache budgets

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

**Experiments: November 2, 2026.**

I ran DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL and GLM-5.3-Flash UD-Q4_K_XL through the same orchestration, first with four GPUs and then one, while varying expert-cache controls. The retained CSV contains **54 measurements: 18 uncached rows and 36 cached repetitions**. GLM's single-GPU reduced-budget case failed during loading and ended the run. The single-GPU zero-cache case was therefore never reached.

## Method and allocation controls

The [historical runner](../raw-logs/2026-11-10-maya-collection/extracted/maya_matrix_all_in_one.py) generated configurations, ran the built-in 2K/8K benchmark, and then posted the saved 32K/64K chat requests three times per variant. The engine used HIP on the EPYC 7713, 2 TiB RAM and V620s. The harness used port 18081, a 1024 MiB reserve, and an additional 8192 MiB reserve for `reduced`. Those labels do not mean precisely half of the expert cache or zero total GPU memory. Shared experts, attention, dense weights, KV state and temporary buffers remain separate allocations.

GLM's `--expert-cache 0` controls only the first GPU, so the four-GPU zero-cache case was explicitly skipped. DeepSeek's effective expert allocations must be read from its engine logs; a `zero` label alone does not prove a GPU had no routed experts. I retain the labels as experiment identifiers rather than replacing them with stronger claims.

## Uncached results

All rates are tokens/s; the two decode columns belong only to the first, uncached request at each length. DeepSeek processed 32,800 / 65,568 prompt tokens; GLM processed 32,017 / 63,733 because the same text tokenizes differently. Every row generated 128 tokens.

| Variant | PP 32K target | PP 64K target | TG after 32K | TG after 64K |
|---|---|---|---|---|
| deepseek-4gpu-full | 460.9 | 483.4 | 25.0 | 40.4 |
| deepseek-4gpu-reduced | 463.1 | 481.7 | 24.2 | 32.6 |
| deepseek-4gpu-zero | 461.5 | 483.9 | 31.2 | 34.9 |
| deepseek-1gpu-full | 297.7 | 316.2 | 16.5 | 20.3 |
| deepseek-1gpu-reduced | 297.7 | 316.4 | 17.0 | 19.7 |
| deepseek-1gpu-zero | 297.7 | 316.3 | 17.1 | 19.5 |
| glm-4gpu-full | 348.7 | 351.2 | 28.9 | 31.4 |
| glm-4gpu-reduced | 355.2 | 356.6 | 25.2 | 27.0 |
| glm-1gpu-full | 175.2 | 169.6 | 10.2 | 14.0 |


With the full allocations, four versus one GPU increased DeepSeek prefill from 297.7 to 460.9 tokens/s at 32K and 316.2 to 483.4 at 64K. GLM increased from 175.2 to 348.7 and 169.6 to 351.2. These results supported keeping more than one GPU per model; they did not yet measure the two-GPU compromise chosen later.

## Built-in short benchmarks

Each completed variant also retains a v1.1.49 report with three distinct answers of up to 256 tokens and separate 2K/8K prefill. All rates are tokens/s. These are not the 128-token continuations from the long-prompt table.

| Variant | CSV TG | Prose TG | HTML TG | Mean TG | PP 2K | PP 8K |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-1gpu-full | 22.1 | 18.7 | 25.7 | 22.2 | 127 | 235 |
| deepseek-1gpu-reduced | 21.9 | 18.5 | 21.2 | 20.5 | 127 | 235 |
| deepseek-1gpu-zero | 21.4 | 17.8 | 25.6 | 21.6 | 127 | 235 |
| deepseek-4gpu-full | 31.7 | 27.2 | 37.5 | 32.1 | 195 | 335 |
| deepseek-4gpu-reduced | 28.6 | 24.0 | 32.0 | 28.2 | 195 | 338 |
| deepseek-4gpu-zero | 30.0 | 23.9 | 34.6 | 29.5 | 196 | 337 |
| glm-1gpu-full | 10.3 | 11.3 | 14.8 | 12.1 | 142 | 171 |
| glm-4gpu-full | 32.2 | 28.5 | 34.5 | 31.8 | 200 | 232 |
| glm-4gpu-reduced | 26.5 | 26.9 | 28.0 | 27.2 | 187 | 231 |

The full-cache short reports measured 32.1 mean decode on four GPUs versus 22.2 on one for DeepSeek, and 31.8 versus 12.1 for GLM. These short workloads support the GPU-allocation tradeoff without requiring an interpretation of the long-prompt decode variation. All individual short-answer rates are also included in the [short-report CSV](../data/maya-november-short-benchmarks.csv).

## Cached repetitions and the interrupted matrix

Repeated DeepSeek requests reused 32,799 / 65,567 tokens and processed one new token. GLM reused 32,016 / 63,732 and likewise processed one. Their printed 7.5–19.6 prefill tokens/s describe a tiny uncached suffix and its overhead, not a fresh long-prompt throughput collapse. The original CSV labels these `cached-not-comparable`; I exclude them from the table while retaining every repetition and its decode rate in the [raw matrix](../raw-logs/2026-11-10-maya-collection/extracted/20261102-124651-maya-matrix/results.csv).

Decode varied with prompt content, speculation and cache history. In particular, the 40.4 tokens/s DeepSeek result after the 64K first request does not show that longer context intrinsically accelerates decoding. The three 128-token continuations are not independent full-prefill repetitions.

`glm-1gpu-reduced/short.log` reports that the engine stopped while loading. Its `ERROR.txt` records exit 1 and points to that log, which in turn points to the shared benchmark log. The directory lacks a matching engine traceback, so OOM is a hypothesis rather than the established cause. The archive has no `glm-1gpu-zero` directory. No result should be extrapolated into those missing cases.

## Decision

Reduced reserve budgets barely changed uncached prefill in these runs, but they did not establish cache-size independence. I kept the single-GPU tradeoff visible and moved toward testing two GPUs per model. The planned separate decode experiment was not completed by this matrix. [November 9](2026-11-09-deepseek-tp2-and-tp4.md) supplies the later two-GPU evidence.

## Sources and recording dates

**Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
