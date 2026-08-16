# Entry 11 — STREAM re-baseline on the 2 TB population

**Date:** Friday, August 15, 2026. **Machine:** Galactus, bare-metal host, `~/STREAM`.
**Object:** close the open item standing since the August RAM upgrade — re-measure platform DRAM bandwidth on the new population (8 × 256 GB DDR4-2933 3DS RDIMM at rated speed) with the identical July sweep, so every decode budget in this notebook has a current denominator.

## Provenance — one DIMM replaced, sweep re-run

The first 2 TB sweep of the day ran with a DIMM that was found to be failing. The module was replaced the same day and the sweep re-run on the repaired population. The figures below are from the re-run; the pre-replacement capture is superseded and removed. For the record: the two captures agreed on best rates to within 0.3% per kernel at t=16, so the fault was not costing steady-state bandwidth — and a dropped channel would have read near 7/8 of these figures, so both captures had all eight channels active. Two defects of the first capture are absent from the re-run: the t=56 rung printed no rows, and t=128 showed ~3× max-time outliers (0.34–0.69 s against 0.23 s minimums). Whether those were symptoms of the failing module cannot be established now that it is out of the machine; they are noted, not attributed.

## Command

```bash
for t in 16 24 32 40 48 56 64 80 96 112 128; do   echo "===== THREADS=$t =====";   OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores ./stream_c |     awk -v t="$t" '/Copy:|Scale:|Add:|Triad:/{print "Threads=" t, $0}';   echo; done | tee stream_sweep_2tb.log
```

Raw capture: `galactus_triad_2tb.txt` (repo `hardware/galactus/`) / `stream-triad-host-sweep-2tb.txt` (vault raw logs). CSV extract with RFO-corrected column: `results/data/stream-host-sweep-2tb.csv`. The capture is complete at all eleven thread counts.

## Result — best rates at saturation (t=16), RFO-corrected

| Kernel | 2 TB (MB/s raw) | 2 TB corrected (GB/s) | 1 TB July corrected (GB/s) | Delta |
|---|---|---|---|---|
| Copy (×1.0) | 145,877 | 145.9 | 151.8 | −3.9% |
| Scale (×1.5) | 100,850 | 151.3 | 155.1 | −2.5% |
| Add (×4/3) | 110,596 | 147.5 | 149.3 | −1.2% |
| Triad (×4/3) | 111,152 | 148.2 | 150.0 | −1.2% |

Convergence: **148–151 GB/s**, ≈ 79% of the 187.7 GB/s theoretical limit (July: ~152, 81%).

## Observations

- **The shape reproduces exactly.** Saturation at t=16, t=32 second-best (Triad 108.2k), the same dips at the non-CCD-aligned counts t=40 and t=80 that the July sweep showed, and a flat SMT plateau (Triad 102–104k at 96–128). The placement physics of the platform did not change with the DIMMs.
- **The new population is ~2% slower on average** (per-kernel −1.2% to −3.9% against the July values). Different rank organization at the same 2933 MT/s; direction and size are plausible for 256 GB 3DS parts, and nothing suggests a fault.
- **The re-run is clean where the superseded capture was not.** All eleven rungs present, including t=56 (138.7–142.3 corrected, in line with its t=48 and t=64 neighbours), and max times within 4% of min times at every rung — including t=128, where the superseded capture had shown 3× outliers.
- **Consequence for every decode budget: multiply by ≈ 0.98.** That is smaller than benchmark run-to-run variance (±2–4% across this notebook), so no production configuration changes and no prior number needs restating. The platform figure is now quoted as **~150 GB/s** (152 measured on 1 TB, 148–151 on 2 TB).
- **Decision:** open item closed. The 2 TB population — with the replaced DIMM — is accepted at rated 2933 with no further tuning.
