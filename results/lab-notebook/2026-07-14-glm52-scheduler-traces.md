# GLM-5.2 scheduler traces: distribution without a throughput gain

**Capture date:** July 14, 2026, inferred from the original files' modification times. **Added:** October 22, 2026. I recovered the timestamps on October 23: `sched_full.txt` was last written at 19:07:54 EDT and `sched_patched.txt` at 19:58:26 EDT. These are file-write times, not run-start times. [Timestamp evidence](../raw-logs/2026-10-22-galactus-archive/source-timestamps.md).

## The scheduler traces show distribution, not a speedup

The two scheduler captures make the placement change visible:

| Backend | Original split count | Distributed split count |
|---|---:|---:|
| CPU | 308 | 308 |
| ROCm0 | 731 | 285 |
| ROCm1 | 133 | 300 |
| ROCm2 | 175 | 294 |
| ROCm3 | 147 | 292 |

The GPU split totals differ slightly because the scheduler partitions the graphs differently. These counts describe graph splits, not bytes transferred or time spent on each GPU. They establish that I spread the offloaded work much more evenly.

The pp512 rows, however, report **28.59 ± 0.00 t/s before and 28.50 ± 0.00 after**. Redistributing this workload did not improve its throughput. The allocation records show the cost: the original trace reserves roughly 4260 MiB on ROCm0 but only 205, 205, and 339 MiB on the other GPUs; the distributed trace reserves roughly 4260, 4260, 3852, and 3852 MiB. Using more GPUs required more scratch space without producing a gain in this small-prompt test.

These traces complement the later [pp8192 patch experiment](2026-07-21-prefill-patch.md), where distribution alone again made little difference and combining it with the routing readback bypass raised throughput to 119.36 t/s. The 28.59/28.50 rows are not the 104.97/105.71 comparison and should not replace it.
