# DeepSeek on two V620s: baseline and requested TP4 fallback

**Experiments: November 9, 2026.** **Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.

I tested whether DeepSeek could retain useful interactive performance on GPUs 0 and 1, freeing GPUs 2 and 3 for GLM-Flash. The first built-in v1.1.54 report measured **29.0 tokens/s mean decode**, 187 tokens/s at 2K prefill, and 327 at 8K, with 1,048,576 context configured and a 298-second load. Its three 256-token answer rates were 29.1, 24.2 and 33.6. This made the two-GPU split worth pursuing; it did not demonstrate a full-context workload or simultaneous request performance.

## Long-context baseline

The recovered sweep ran 32K chunks, an 800 MiB ring, automatic workspaces, DSpark and 128 generated tokens, at document targets of 8K, 16K and 32K. Effective layout reports TP2 on both devices and 4096 attention rows. The [baseline directory](../raw-logs/2026-11-10-maya-collection/extracted/20261109-175610/) retains one completed start with six zero-cache rows; the command planned two starts, but the second is not present.

| Target | Actual prompt tokens in captured rows | Prefill pair, tokens/s | Median | Decode pair, tokens/s |
|---|---|---|---:|---|
| 8K | 8,224 | 313.6 / 315.1 | 314.35 | 29.3 / 33.6 |
| 16K | 16,416 | 395.2 / 392.3 | 393.75 | 27.2 / 28.3 |
| 32K | 32,799–32,800 | 453.8 / 452.3 | 453.05 | 27.4 / 27.4 |

Metadata identifies v1.1.54 commit `c1670044bbc09310f35f9d00f7596c50ae8a38f1`, binary SHA256 `9602726904061a1885234efaaa38b9737eab872ef89b77847f0a086523135820`. The TP4-request and FP16 runs subsequently use v1.1.57 and a different binary and corpus. That change prevents treating the original baseline as a same-binary FP16 control.

## Requested TP4 did not activate

The next command requested `--tps 4` with the same two-device source config. Both starts observed TP2 and marked all 12 rows `layout-mismatch-exclude`. Their medians were 315.8, 395.65 and 454.85 tokens/s at the three lengths. These are retained observations of the fallback, not TP4 results. No engine case failed, but zero samples qualified as a verified requested layout.

The archive's `20261109-181250` directory contains corpus, requests and source config only; it has no result CSV. `20261109-181257` contains the two completed fallback starts. I do not count the preparation directory as another run.

I had already encountered the fallback in October and reported that `MAYA_DS4_TP_GGUF_UNEVEN=1` had performed poorly. The supplied November material adds no successful uneven-TP4 measurement. Nor does a TP4 request through a two-device config establish a valid all-four-device TP4 experiment. I kept two-GPU TP2 as the working layout.

## Quality-related log detail

The baseline engine log reports `prompt skip: layers 20-39 ... early rows KV-only`, including approximately 29,620 skipped early rows for a 32K target. Thus zero prefix-cache hits and verified TP2 do not mean every prompt row received full computation at every layer. This log behavior also appears in the optimized run. The harness's `ok` label checks timings, prefix reuse and requested layout; it is not a numerical-fidelity verdict. I preserve this distinction because the earlier notebook excluded quality-reducing prompt skipping.
