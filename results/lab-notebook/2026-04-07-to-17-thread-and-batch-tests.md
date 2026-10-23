# April: thread counts, resident experts, and long-context Qwen tests

**Experiments:** April 7–17, 2026, where the captures include timestamps. **Added:** October 22, 2026. I recovered these logs from `tests.tar.gz`; they predate the July scheduler patch and the later V620/model integration. [Original files and provenance](../raw-logs/2026-10-22-galactus-archive/README.md) · [Complete benchmark ledger](../data/galactus-archive-benchmarks.md).

I tested how CPU thread count affected large MoE models whose routed experts exceeded GPU memory, then tried keeping some Qwen expert layers on the four V620s. The logs distinguish three variables that a single headline rate would conceal: CPU thread count, expert placement, and the amount of context already in the model. I cannot treat the fastest row from each configuration as a common benchmark.

## Thread sweeps: more threads helped prefill but eventually hurt decode

The captures identify four V620s with 30,704 MiB each. The early build reports `58190cc84 (8671)` and lists both ROCm and Vulkan backends; the later build reports `0893f50f2 (8746)` and ROCm. A backend list does not establish how much work each backend performed. Most of these sweeps use `pp512`, `tg128`, batch and microbatch 4096, Q8_0 attention caches, flash attention, and `nopo=1`. The full rows retain the settings, including expert-placement expressions.

| Capture | Model label and placement | Best measured tg128 | tg128 at 128 threads | Best measured pp512 |
|---|---|---:|---:|---:|
| April 7 | `deepseek2`, 1026.41 B parameters, Q4_K Medium; experts in RAM | 6.76 ± 0.02 at 32 threads | 3.25 ± 0.57 | 44.41 ± 0.03 at 96 threads |
| April 9 | Same printed model size/type and build; experts in RAM | 7.64 ± 0.00 at 32 threads | No completed 128-thread row | 44.05 ± 0.04 at 96 threads |
| Qwen capture, date not in filename | Qwen3.5 397B Q6_K, 319.21 GiB; experts in RAM | 9.69 ± 0.02 at 32 threads | 4.57 ± 0.64 | 87.54 ± 0.41 at 96 threads |
| April 10 | Qwen3.5 397B Q6_K; five expert layers per GPU, remaining experts in RAM | 11.67 ± 0.07 at 32 threads | 5.78 ± 0.69 | 110.23 ± 0.25 at 96 threads |
| April 17 | MiniMax M2.7 Q5_K Medium; experts in RAM | 14.76 ± 0.01 at 32 threads | 8.96 ± 2.84 | 101.71 ± 0.13 at 96 threads |
| April 17 | Qwen3.5 397B Q4_K Medium; experts in RAM | 11.43 ± 0.02 at 32 threads | 5.69 ± 0.43 | 100.58 ± 0.04 at 96 threads |

I got the best decode rate at 32 threads in each of these sweeps, while prefill generally peaked at 96. Moving to 128 threads roughly halved decode throughput in the Qwen captures and reduced it substantially in the other models. That shows why one thread count does not optimize both phases. The measurements establish the regression; they do not, by themselves, distinguish SMT contention, scheduling overhead, and memory-access effects.

The April 8 file contains two full sweeps, not one. At 32 threads they both reached 6.76 t/s; at 128 threads they fell to 3.11 and 3.69 t/s with larger spreads. I kept both sequences rather than averaging them into a single experiment. The April 9 run reached 7.64 t/s at 32 threads despite the same printed build and model summary. The captures do not identify the changed condition, so I cannot attribute that improvement to a code change.

The early files print `deepseek2 671B` in the model label but **1026.41 B** in the parameter column. I retained both fields. An architecture label is not enough to identify the model export, and these are not DeepSeek-V4.1 measurements.

## Resident experts changed the Qwen comparison

For the April 10 Q6_K sweep, I explicitly placed expert layers 0–4 on ROCm0, 5–9 on ROCm1, 10–14 on ROCm2, and 15–19 on ROCm3, with the remaining experts in RAM. At 32 threads this configuration reached 11.67 t/s, compared with 9.69 t/s in the earlier Q6_K CPU-expert capture. Prefill also increased, from a peak of 87.54 to 110.23 t/s.

Those are different builds as well as different placements. I therefore cannot assign the entire difference to VRAM residency. They do establish that the April resident-expert configuration completed both prompt processing and generation, which matters when comparing it with the later allocation failures under larger graphs.

The first MiniMax attempt on April 17 failed to load `root/models/...`, a path without a leading slash. The follow-up capture starts roughly half a minute later and contains the successful sweep. I preserved the failed attempt as a path/load failure, not a zero-throughput result.

## Batch sweeps exposed the limits of the configuration

The April 11 morning files include an invalid `-c` argument, a ROCm failure, and a partially successful batch sweep. In the successful file, a banner says `-p 65536`, but the result row prints `pp16384`. I report the result as printed: 103.27 ± 0.11 prompt tokens/s and 11.49 ± 0.05 generated tokens/s, with batch 512 and microbatch 4096. Another completed prompt row reaches 103.72 ± 0.10. Later attempts fail in ROCm or during context creation. The banner discrepancy prevents me from labeling the successful result a 65,536-token prompt test.

The evening sweep has a clearer context marker: `@ d131072`. I used four expert layers per GPU, Q8_0 attention caches, and build `0893f50f2`. Across the completed `pp512 @ d131072` rows, prompt throughput stayed between 70.41 and 71.13 t/s and decode between 6.46 and 6.52 t/s. Longer prompt tests reached 70.39–74.13 t/s for pp4096 and 69.39–69.40 t/s for pp16384 before later ROCm failures.

The accompanying [sweep script](../raw-logs/2026-10-22-galactus-archive/scripts/qwen-batch-sweep.sh.txt) specifies 64 threads, two repetitions, and depth 131072. That makes this a different workload from the short-context thread sweeps. It also explains why quoting the 6.5 t/s result without its context would misrepresent the machine's shorter-context performance.

These files supply the missing details behind the April comparisons in the model notes. I retain the August common baseline separately because the builds, model exports, memory population, and prompt sizes changed between those experiments.
