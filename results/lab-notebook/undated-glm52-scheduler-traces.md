# GLM-5.2 scheduler traces: distribution without a throughput gain

**Run date:** Not recorded. **Added:** October 22, 2026. These captures belong with the July GLM-5.2 scheduler investigation, but they do not identify a day. [Archive inventory](../raw-logs/2026-10-22-galactus-archive/README.md).

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
