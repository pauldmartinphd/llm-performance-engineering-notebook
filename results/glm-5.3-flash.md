# GLM-5.3-Flash on Galactus

## October 29 record: 80 GiB expert-cache benchmark

A recovered `llamda-bench` capture for build `754aa4151 (11811)` reports **279.51 ± 15.37 pp8192** and **8.63 ± 2.84 tg256** with `--moe-cache-mib 81920`, forced CPU expert placement, 32 threads, flash attention, and batch/microbatch 8192 on four V620s. The run date is not printed. Two repetitions and no matching cache-disabled run leave the cache's effect unresolved; the earlier build comparison uses different revisions.

The [complete cache investigation record](lab-notebook/2026-10-29-glm53-expert-cache.md) preserves the command, hardware report, historical untested 96 GiB patch proposal, implementation claims awaiting verification, and proposed controlled comparisons. The October 29 continuation could not access `/root/llama.cpp`; no cache counters, new timings, or validated fixes were obtained.

## October 23: 31.6 tokens/s with Project Maya

Using the existing Unsloth UD-Q4_K_XL export, Project Maya v1.1.24 reached **31.6 generated tokens/s** on Galactus's four V620s. Setting `MAYA_GLM_NO_SPEC=1` reduced the same three-prompt benchmark to **19.9 tokens/s**, a **58.8%** gain with speculation. Both runs used context 32768, greedy decoding, thinking off, and three 256-token answers. Dedicated prefill at 8192 tokens was 242 with speculation and 241 without it.

The initial default run measured 29.4 tokens/s. Head settings of one and three GPUs produced 24.6 and 22.9, while MTP_KEEP variants produced 28.2–30.5 tokens/s. The default two-GPU head and KEEP=2 gave the highest observed rate; small differences among KEEP settings need repeated trials. Automatic GLM hardware calibration declined to run on HIP in this version.

[Full A/B test, tuning tables, and captured configuration](lab-notebook/2026-10-23-maya-glm53-speculation.md).

## October 22: llama.cpp and llAMDa baselines

The [revision-identified benchmarks](lab-notebook/2026-10-22-glm53-build-comparison.md) measured 284.36 ± 16.33 pp8192 and 10.16 ± 0.04 tg256 for llama.cpp `184f65360`, versus 284.20 ± 16.42 and 10.09 ± 0.02 for llAMDa `ca3b1fc30`. Their generation-only CLI record reached 17.8–18.1 tokens/s with automatic placement. The CLI banners all print `b11829-184f65360`, including commands named `llamda-cli`, so those CLI names cannot establish an implementation comparison.

## Earlier V620 tests

I tested the six-shard Unsloth UD-Q4_K_XL export on Galactus with 2 TB of eight-channel DDR4 and four V620 GPUs. My V620 build combined the V620 patches, older DeepSeek support, GLM-5.3 MTP support, and my round-robin prefill patch. It reports `b0-unknown`; I clarified the patch combination on October 22, 2026, but the captures do not establish the exact source revision or run dates.

## Measured configurations

All rows below use the V620 binary, the ZFS explanation prompt, 32 threads, context 8192, a 256-token generation limit, greedy decoding, flash attention, and MTP. Cache values are requested MiB, not measured allocations.

| Expert placement | Requested cache MiB | MTP n-max | Generation t/s |
| --- | ---: | --- | ---: |
| Automatic | Omitted | Omitted | 19.1 |
| Automatic | Omitted | 2 | 19.4 |
| Forced RAM | 0 | 2 | 14.4 |
| Forced RAM | Omitted | Omitted | 13.7 / 13.8 |
| Forced RAM | 64536 | 2 | 12.0 |
| Forced RAM | 98304 | 2 | 16.0 |
| Forced RAM | 98304 | 3 | 15.1 |

Automatic placement gave the highest generation rates in these tests. I did not record actual tensor placement, so I cannot assign the difference to a specific kernel or transfer mechanism. The cache results did not improve consistently as I increased the requested budget. Every successful GLM run enabled MTP, and I did not disable the round-robin patch for a control, so these tests do not isolate either feature's contribution.

I also tested a separate PATH binary reporting `b11580-b3b711e4f`, and the larger GLM-5.3 non-Flash model. The [complete experiment record](lab-notebook/undated-glm53-flash-placement.md) includes all 15 launches, including a cache-related allocation failure. I keep these short-prompt CLI results separate from the August pp8192/tg128 baseline and from the newer Unsloth integration.

[Original log](raw-logs/2026-10-21-flash-experiments/GLM-Flash%20Experiments.txt) · [All launch records](data/glm53-flash-historical.csv) · [Chronological notebook](lab-notebook/README.md)
