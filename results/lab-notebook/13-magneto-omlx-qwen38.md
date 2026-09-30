# Entry 13 — Qwen3.8-27B, oMLX, and chassis comparisons

[Notebook index](00-overview.md) · [Model summary](../qwen-3.8-27b.md)

**Date:** September 30, 2026 (record compiled; exact timestamps for the earlier baseline are unavailable). **Machine:** Magneto, 14-inch M2 Max MacBook Pro, 38 GPU cores, 64 GB.
**Object:** Record local oMLX results and determine what can explain their difference from a high M2 Max leaderboard submission.

## Source ledger

The full source is my **Bootstrap Model Harnesses.pdf**, 123 pages, printed September 30 at 09:53 ET from [the shared conversation](https://chatgpt.com/share/6abd142f-eb54-83ea-877e-17842567811a). SHA-256: `f9cf4dc18ffbb46f918fbc58c780da865666137c307d81199ff58cfce50fce1c`. The original chat ID is `6abc725e-fb88-83ea-9f1d-4e402a4e0c13`. Page numbers below are PDF page numbers.

The PDF preserves conversation prose and selected quoted log lines, but original image and document attachments appear as placeholders. Consequently its numerical summaries remain conversation transcriptions. Two original screenshots were separately recovered and retained. No new inference benchmarks were run for this entry. The unrelated harness discussion and full conversation are not copied into the repository.

| Evidence | Location | What it supports |
|---|---|---|
| [User-supplied measurement summary](../raw-logs/omlx-magneto/user-measurement-summary.txt) | Supplied September 30 after PDF review | Local oMLX **0.6.4 build 2529**, model size 17.50 GB, benchmark and telemetry transcriptions |
| Initial throughput/context sweep | PDF pp. 58–65, 114–115 | Six PP/TG/TTFT rows, memory growth, baseline controls, batching |
| Initial MTP/ANE log recap | PDF pp. 64–67 | MTP cycles/acceptance; ANE zero as expected with toggle off; batching disables MTP |
| ANE tuning | PDF pp. 71–72, 87 | Tuner baseline, prediction and refined result; proposed fractions, not an exported recipe |
| DFlash trial | PDF pp. 73–84 | Intended draft, UI results, ANE inactivity, attention fallback and internal timing recap |
| ANE + MTP benchmark | PDF pp. 88–89, 94–95 | PP/TG/TTFT/E2E changes, observed ANE operations, MTP cycles and parking |
| SpecPrefill setup and reversal | PDF pp. 90–91 | Keep rate/threshold; draft missing; no successful experiment established |
| Later 4K/16K pair | PDF p. 106 | 165.5/15.9 and 171.1/18.2 PP/TG; complete recipe unavailable |
| [09:44:54 throughput screenshot](../raw-logs/omlx-magneto/2026-09-30-throughput.png) | Original image, also discussed PDF pp. 110–112 | 171.0/16.9 and 173.0/19.4; visible benchmark controls; no acceleration toggles |
| [09:44:59 powermetrics screenshot](../raw-logs/omlx-magneto/2026-09-30-powermetrics.png) | Original image | Heavy pressure; 942 MHz / 84.40% active GPU; power readings |
| Earlier telemetry summaries | PDF pp. 98–107 | Nominal intervals, active-load readings, idle-like Heavy sample; full traces unavailable |
| [External f4f6h86x](https://omlx.ai/benchmarks/performance/f4f6h86x) and [upload implementation](https://github.com/jundot/omlx/blob/main/omlx/admin/benchmark.py) | Checked September 30 | External performance/settings; public hardware fields omit chassis |

The [CSV extract](../data/magneto-omlx-qwen38.csv) preserves missing fields as blanks and distinguishes PDF transcriptions from directly inspected screenshots. The [evidence notes](../raw-logs/omlx-magneto/README.md) record capture limitations. “Trials: 2” in the screenshot accompanies two context rows and is not evidence of two repetitions per condition.

## Sequence and corrections

1. **Reported baseline:** Lightning MTP on, ANE Prefill and SpecPrefill off, DFlash off, full warm-up, Code (Python), generation 128. I confirmed AC power and an initially cold run. MTP activity is supported by quoted log statistics; no MTP-off timing followed.
2. **Tuner result:** 155.7 → 196.3 PP t/s is the tuner comparison. **Correction:** it is not the serving benchmark's improvement. The proposed fractions must not be presented as a recovered complete recipe.
3. **DFlash trial:** ANE was requested but recorded zero operations. UI TG was 15.5 at 4K and 11.2 at 16K. **Correction:** do not call this successful ANE + DFlash or a controlled DFlash-only test; do not compare internal engine rates with UI TG.
4. **ANE + Lightning MTP:** 170.1/13.9 at 4K and 165.6/15.7 at 16K; quoted nonzero ANE operations confirm work at 16K. PP improved about 8–9% while TG fell. Thermal/runtime state was not controlled well enough to assign the entire difference to ANE.
5. **SpecPrefill:** enabled during setup, no selected draft, then recommended off. Later statements that it was enabled conflicted with that decision. The final saved settings have it enabled with no draft recorded. **Correction:** none of the later throughput rows demonstrates SpecPrefill acceleration.
6. **Thermal investigation:** an initial recap describes 61 Nominal samples with low/bursty GPU activity; later active-load recaps describe ~0.92–1.01 GHz, 88–100% residency, 23–29 W, still Nominal. The later Heavy screenshots establish changed thermal state, not a phase-aligned throughput penalty. The 444 MHz / 178 mW sample was mostly idle. **Correction:** neither excluding chassis because pressure was Nominal nor declaring a fixed 28 W ceiling follows from these data.
7. **Later results:** 165.5/15.9 and 171.1/18.2, then 171.0/16.9 and 173.0/19.4. The complete recipes and thermal states per row are unknown. **Correction:** the final response's description of 19.4 as the “coldest/best” result is unsupported; a nearby screenshot shows Heavy pressure.
8. **Leaderboard investigation:** chip/core/RAM metadata omits chassis. **Hypothesis:** a larger chassis contributed to the high 25.7 TG result. Software, exact weights, prompt/sampling settings and variance also remain uncontrolled. No inference identifies its chassis.

## Memory comparison

The later local screenshot reports 25.9 GB peak memory at 4K and 28.1 GB at 16K. The conversation recalls approximately 23.3 GB for the original 4K baseline, but the original capture is not available (PDF p. 63 also gives 19.2/23.3/24.1/25.5/26.5/28.8 GB across 1K–64K). The external submission separately reports MLX active peak 20.54 GB, MLX cache peak 2.92 GB, and process peak footprint 21.33 GB. These are different metrics and possibly different allocation lifetimes. A mismatch is a reason to record definitions, versions, and model state; it does not prove different model weights or a broken installation.

## Where I left it

I’m done with further measurements for now. I checked the saved model settings and the active `profile-1-copy` profile: Lightning MTP and ANE Prefill are on, DFlash/VLM MTP/TurboQuant are off, and CPU prefill sharing is off. SpecPrefill is saved as on at a 20% keep rate and an 8,192-token threshold, with no draft model recorded. Its runtime benefit remains unmeasured.

The [model note](../qwen-3.8-27b.md#final-settings-for-now-september-30-2026) lists the final settings; the [JSON snapshot](../data/magneto-omlx-final-settings.json) records the values read from the saved configuration on September 30. The source ledger still has gaps in original logs, exact model revision, and macOS/MLX versions. Those are limits of this record, rather than a list of more experiments to run.
