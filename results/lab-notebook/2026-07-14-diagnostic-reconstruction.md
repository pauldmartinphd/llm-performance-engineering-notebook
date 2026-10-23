# July: reconstructing the prefill investigation from the diagnostic logs

**Experiments:** July 14, 2026. The supplementary placement captures lack independent run timestamps. **Added:** October 22, 2026. [Archive inventory](../raw-logs/2026-10-22-galactus-archive/README.md) · [All printed benchmark rows](../data/galactus-archive-benchmarks.md).

The archive contains the full v3 and v4 diagnostic logs, their hardware monitors, and related placement captures. I can now follow the investigation from thread and placement tests to the larger microbatch experiment, then compare the completed placement and fitter tests. These captures concern GLM-5.2 on the 1 TB Galactus configuration, not the later Flash models or the 2 TB rebaseline.

## The first diagnostic run matches the existing notebook

The v3 capture contains 94 benchmark rows. Their test labels and rates match the existing `diag-run-llama-bench-results.csv` in order, so I have not counted them as additional repetitions. The larger original log adds detailed allocation and scheduler output; its line numbers differ from the earlier extraction. I kept both sources and generated a separate ledger for this archive.

The monitor timestamps cover 11:09:23–14:25:06 on July 14. Across 1,175 vmstat samples, the reported swap usage, swap-in, swap-out, I/O wait, and steal columns remain zero. The CPU-frequency capture spans 1,171 samples and reports average frequency between 2038 and 3100 MHz. These observations narrow the investigation: they show no swapping or reported CPU steal during the sampled interval, but an average clock reading cannot prove that every core sustained a particular frequency.

## The real microbatch ladder

The v4 run began later that afternoon. For an 8,192-token prompt and batch 8192, the completed rows show:

| Microbatch | Operation offload on (`nopo=0`) | Operation offload off (`nopo=1`) |
|---:|---:|---:|
| 512 | 25.90 ± 0.36 | 31.88 ± 0.03 |
| 1024 | 41.68 ± 0.12 | 32.89 ± 0.02 |
| 2048 | 62.64 ± 0.13 | 33.31 ± 0.01 |
| 4096 | 84.62 ± 3.58 | 33.54 ± 0.01 |
| 8192 | 104.97 ± 0.53 | 33.77 ± 0.00 |

Rates are tokens/s. At microbatch 512, operation offload was slower than the CPU-expert control. At microbatch 8192, it was about 3.11 times as fast. Increasing the microbatch gave the GPU enough work to amortize the offload cost; disabling offload left throughput almost flat. This is why the earlier small-prompt tests did not expose the same opportunity.

The improvement did not continue indefinitely. With a 16,384-token prompt and batch 16384, microbatches 8192 and 16384 reached 75.15 ± 1.55 and 80.48 ± 0.19 t/s. Those are different prompt workloads, so I do not treat their lower rates as a reversal of the pp8192 ladder. They do show that increasing the configured sizes alone did not produce another throughput gain.

At microbatch 4096, the v4 prefill thread sweep stayed around 84.69–87.31 t/s from 8 through 96 threads. Once the GPU offload path carried the substantial matrix work, additional CPU threads had little effect in this configuration.

## Placement and fitter experiments

I also tried reserving ROCm0 for offload scratch space with the phase labeled `-ts 0,1,1,1`. The first three rows in that phase reached 62.70, 87.28, and 104.95 t/s at microbatches 2048, 4096, and 8192. The largest result essentially matched the ordinary ladder's 104.97 t/s. The table abbreviates the split vector, so the phase heading and original command context matter when interpreting these rows.

The fitter experiments changed both placement and available workspace. With pp8192 and microbatch 4096, fitter margins of 2048, 8192, and 16384 MiB produced 91.85 ± 3.62, 89.48 ± 3.75, and 70.96 ± 18.83 t/s. The largest margin did not help, and its spread was much larger.

For decode with fitter margin 2048 MiB and fit context 16384, 24 and 32 threads reached 6.12 and 6.11 t/s. Disabling operation offload gave 6.16 and 6.15 t/s at those thread counts. A later 512 MiB margin test reached 6.19 ± 0.01 t/s, while the 1024 MiB test reached 5.86 ± 0.23. I would not infer a smooth margin-to-throughput relationship from these individual configurations.

The v4 capture contains 50 completed benchmark rows and then ends partway through another run, at `llama-bench: be`. The plan printed at the top lists more work than the surviving output completes. I kept the completed rows and did not turn planned phases into results.

## Hardware monitoring and failed placements

The v4 vmstat capture spans 15:01:17–19:54:11 and contains 1,758 samples, again with zero reported swap usage, swap traffic, I/O wait, and steal. ROCm monitoring records each GPU reaching 99% utilization at some point. Peak sampled package powers were 223/230/226/169 W in v3 and 252/251/251/254 W in v4, ordered ROCm0–3. Those are whole-run maxima, not simultaneous readings or a phase-specific load balance measurement. I cannot use them to claim a thermal cause or balanced execution throughout the run.

The smaller placement captures also matter. `ot3` fails during model loading; `ot6` and `ts` fail while creating the context. `ts2` contains only the four-device list. The successful `ot7` configuration explicitly places some expert layers on ROCm1 and ROCm2 and reports pp2048 at 51.74 ± 0.00 t/s. I kept these outcomes separate because fitting the weights, allocating the compute graph, and completing inference are different stages of the experiment.

I discuss the undated [scheduler traces](2026-07-14-glm52-scheduler-traces.md) alongside the July patch investigation.
