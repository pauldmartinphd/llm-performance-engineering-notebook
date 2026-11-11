# DeepSeek on two V620s: baseline and requested TP4 fallback

[All lab records](README.md) · [Related experiment: 6](../../investigations/06-sharing-galactus.md)

## Purpose and reasoning

We test whether DeepSeek can keep useful speed on GPUs 0 and 1, freeing the other pair, and check whether a requested TP4 layout actually activates.

## Experimental design

We ran the v1.1.54 built-in benchmark on two GPUs, then the 32K-chunk/800 MiB-ring sweep at 8K/16K/32K targets with 128 output tokens. We retained effective layout and cache counters. The later requested-TP4 run uses v1.1.57 and still the two-device source config, so requested and observed parallelism must be distinguished.

## Results

The built-in generation mean was 29.0 tokens/s, with 327 at 8K prefill. The retained TP2 long-sweep start contains six zero-cache rows. Both later TP4-request starts observed TP2 and excluded all 12 rows as layout mismatches.

## Conclusions

Two-GPU TP2 was a useful allocation candidate. The TP4-request rows describe fallback, not TP4 performance or a successful four-device experiment. The original long baseline and later runs also differ in release, binary, and corpus.

## Detailed record

**Experiments: November 9, 2026.**

We tested whether DeepSeek could retain useful interactive performance on GPUs 0 and 1, freeing GPUs 2 and 3 for GLM-Flash. The first built-in v1.1.54 report measured **29.0 tokens/s mean decode**, 187 tokens/s at 2K prefill, and 327 at 8K, with 1,048,576 context configured and a 298-second load. Its three 256-token answer rates were 29.1, 24.2 and 33.6. This made the two-GPU split worth pursuing; it did not demonstrate a full-context workload or simultaneous request performance.

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

The archive's `20261109-181250` directory contains corpus, requests and source config only; it has no result CSV. `20261109-181257` contains the two completed fallback starts. We do not count the preparation directory as another run.

We had already encountered the fallback in October and reported that `MAYA_DS4_TP_GGUF_UNEVEN=1` had performed poorly. The supplied November material adds no successful uneven-TP4 measurement. Nor does a TP4 request through a two-device config establish a valid all-four-device TP4 experiment. We kept two-GPU TP2 as the working layout.

## Prompt-skip diagnostics

The baseline log reports `prompt skip: layers 20-39 ... early rows KV-only`, including about 29,620 early rows for a 32K target. That line records an execution choice; it does not by itself identify a quality-reducing mode.

The [October investigation](2026-10-28-maya-prefill-and-regression.md#quality-constraint-and-additional-candidates) distinguishes the documented default exact-skip mode (`MAYA_DS4_PROMPT_SKIP=1`) from the more aggressive mode 2, which we excluded. The retained November configs do not request mode 2. The log message alone is therefore insufficient evidence that this test violated that constraint. Nor is a prefix-cache count a test of internal computation or answer quality.

The runner's `ok` label checks timing, prefix reuse, and layout. It does not replace a numerical-quality evaluation. The FP16-output change has a separate rounding effect, discussed in its own entry.

## Sources and recording dates

**Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
