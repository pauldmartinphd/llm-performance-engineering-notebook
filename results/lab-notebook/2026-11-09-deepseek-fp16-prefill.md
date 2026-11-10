# DeepSeek RDNA 2 FP16-output prefill: measured speedup and fidelity limits

**Experiments: November 9, 2026.** **Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.

I enabled `MAYA_DS4_PROMPT_F16=1` with `MAYA_DS4_PROMPT_F16_CHECK=1` on DeepSeek TP2. The dense prompt projections produce FP16 output and widen it to FP32, taking a faster RDNA 2 path while changing rounding. This is a prefill computation setting, separate from DeepSeek's Q8 model weights and its attention-cache representation.

## Matched configuration and provenance

The [optimized run](../raw-logs/2026-11-10-maya-collection/extracted/20261109-184435/) completed two starts with two repetitions per length: **12 valid zero-cache measurements**, effective TP2, 32K chunks and an 800 MiB ring. It used GPUs 0 and 1, 1024 MiB reserve, DSpark, and 128 output tokens. The source pack is DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL. The optimized run records v1.1.57 commit `00655481ce2b0689ab1c2e8d67034dc5701d6511` and binary SHA256 `548558f8bf789a022002495dc4fb45506b31ac753076145ed97817f7870c4ed2`. The original TP2 baseline is v1.1.54 with a different binary and corpus; the intervening TP4-request run uses the optimized run's binary and corpus but actually executes TP2 with FP16 output disabled.

The baseline supplies one completed start; the optimized setting supplies two. They were not interleaved. The engine logs positively identify the FP16-output route on both cards and report zero nonfinite FP16-out outputs. The metadata's llama.cpp query returns Maya's own commit, so it does not independently identify a dependency revision.

## Long-prefill results

All rates are tokens/s. Baseline medians use two measurements per length; optimized medians use four. Gains are calculated from those medians.

| Target | Baseline PP median | FP16-output PP range | FP16-output PP median | Change | Optimized TG median |
|---|---:|---|---:|---:|---:|
| 8K | 314.35 | 384.6–390.7 | 389.55 | +23.9% | 30.05 |
| 16K | 393.75 | 497.2–503.1 | 501.95 | +27.5% | 31.85 |
| 32K | 453.05 | 588.7–590.7 | 589.70 | +30.2% | 28.60 |

These baseline-to-optimized gains include a release and corpus change, so they do not isolate the FP16 setting. The intervening requested-TP4 run provides a closer control: its actual layout is TP2, using the same v1.1.57 binary and corpus, with prefill medians 315.8 / 395.65 / 454.85. Relative to those fallback observations, optimized prefill is 23.4% / 26.9% / 29.6% faster. Those rows remain excluded from TP4 results; using their actual TP2 behavior for context does not turn them into successful TP4 measurements. No interleaved, explicitly requested TP2 off/on control was run on v1.1.57.

The improvement is substantial within the captured workloads. It is not a twofold gain, and the run stops at a 32K target. It supplies no optimized 64K measurement and does not measure 1M-token prefill. Comparing 589.7 at 32K with an older four-GPU 64K result would mix prompt lengths and releases.

## Numerical and workload limits

The conversation quotes community PR #869's different-host evidence: approximately 18% faster 2K and 29% faster 8K prefill, mean KL 0.0257, 91.2% top-1 agreement, and perplexity 16.773 versus 16.874 on its specified short corpus. These are the contributor's measurements, not local Galactus quality results. The copied PR text is retained in the terminal-evidence file; its assertions are historical source claims.

Zero nonfinite outputs checks overflow, not semantic accuracy, perplexity or equality with the baseline. Both local logs also report early-row prompt skipping, as described in the [baseline entry](2026-11-09-deepseek-tp2-and-tp4.md). Because that behavior is shared, it does not by itself erase the observed relative timing difference; it does prevent presenting these measurements as proof of an exact, full-computation optimization. No local held-out quality comparison was supplied.

## Configuration decision

I requested two production profiles, one with this prefill route and one without, keeping TP2 on GPUs 0 and 1. The archive retains `maya-ds4-tp2-fp16.json` and `maya-ds4-tp2-baseline.json`; the final production-named copies are requested in the conversation but absent from the collection. The [deployment record](2026-11-10-maya-dual-model-production.md) distinguishes those retained settings from the later reported operating state.
