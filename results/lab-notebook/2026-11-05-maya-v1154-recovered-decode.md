# Maya v1.1.54: recovered decode measurements and complete sweep evidence

**Experiments: November 5, 2026; archive recovered November 10.** **Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.

The new archive fills gaps in the [original November 5 record](2026-11-05-maya-v1154-prefill-and-peer-ar.md). Its full CSV contains 80 rows: 60 prefill measurements and 20 decode measurements. Earlier, only 47 individual prefill rows and the 30 printed prefill means were available. The full CSV supports the original prefill conclusion and adds the decode observations below.

## Decode results

Each variant has two distinct 256-token decode cases after a warm-up; these are different workloads, not repetitions of one prompt. Rates are tokens/s. The arithmetic mean here is derived from the two raw observations, not Maya's separate three-answer built-in benchmark.

| Variant | Case 1 | Case 2 | Derived mean |
|---|---|---|---|
| default | 31.31 | 27.08 | 29.20 |
| tp_even | 30.86 | 27.11 | 28.98 |
| p2p_generic | 31.24 | 26.19 | 28.71 |
| p2p_hip_auto | 27.66 | 26.89 | 27.27 |
| flow_dma | 31.78 | 26.97 | 29.38 |
| flow_ai | 31.77 | 26.36 | 29.06 |
| flow_chunk | 32.16 | 27.62 | 29.89 |
| flow_yoco | 31.09 | 26.25 | 28.67 |
| flow_combined | 32.14 | 27.36 | 29.75 |
| expert_mmq | 32.06 | 26.19 | 29.12 |


The results do not establish a reliable decode gain from any switch. Case content, engine warm-up and fixed run order matter, and two cases per variant do not measure run-to-run variance. The benchmark's primary finding remains that the default 32K chunk reaches about 479.4 tokens/s at 64K while automatic chunk selection is slower.

## Recovered provenance

All ten config copies, engine logs, the experiment plan, skipped-case notes, summary and update logs are now retained under [the completed run](../raw-logs/2026-11-10-maya-collection/extracted/maya-bench-1154-20261105-142331/). They complement the supplied script, rather than replacing the earlier conversation transcription. Three preceding directories retain the failed setup attempts. The separate `201458` directory retains the focused peer retest config and log, so the requested 3072 MiB reserve is now directly inspectable.

The original [P2P runner](../raw-logs/2026-11-10-maya-collection/extracted/maya_benchmark_p2p_test.py) is also recovered. The retest CSV has a header and no completed measurements; its startup probe ended without a ready engine. The earlier distinction between successful host fallback in the main sweep and failure before readiness in the retest still holds.

This recovery closes missing-artifact work. It does not constitute another experiment, a numerical-quality test, or a successful HIP peer-transport result.
