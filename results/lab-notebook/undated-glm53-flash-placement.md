# GLM-5.3-Flash: placement mattered more than the build label

**Recorded:** October 21, 2026, from an undated historical capture. **Machine:** Galactus, EPYC 7713P, 2 TB eight-channel DDR4, four V620 GPUs. [Raw log](../raw-logs/2026-10-21-flash-experiments/GLM-Flash%20Experiments.txt) · [All launch records as CSV](../data/glm53-flash-historical.csv).

**Build clarification, October 22, 2026:** I ran the V620 tests with the V620 patches, the older DeepSeek implementation, GLM-5.3 MTP support, and my round-robin prefill patch merged together. These results describe that combined build, before I replaced the DeepSeek implementation with Unsloth's newer port. The executable reports `b0-unknown`, so the log does not identify the exact commit.

I measured 19.1–19.4 generated tokens/s with GLM-5.3-Flash under automatic placement in the V620 build. Forcing routed experts into system RAM produced approximately 13.7–14.4 t/s in the recorded runs without an explicit cache allocation. I did not capture which tensors the automatic fitter placed on each device, so I cannot attribute that improvement to a particular kernel or transfer mechanism.

The rows labeled V620 use that combined build. I also ran the separate `llama-cli` executable on PATH, which reports `b11580-b3b711e4f`; I have not identified it as the same patch combination. I did not run a round-robin-off control, so this comparison does not measure the individual effect of my scheduler patch.

## Model and test configuration

I used the six-shard `GLM-5.3-Flash-UD-Q4_K_XL` export. For each successful run, I used the same ZFS explanation prompt, 32 CPU threads, an 8,192-token context, a 256-token generation limit, greedy decoding, flash attention, and `--load-mode none`. I enabled `--spec-type draft-mtp` in every successful GLM run, so I do not have an MTP-off control in these results.

The V620 binary reports `b0-unknown`; the executable found on PATH reports `b11580-b3b711e4f`. I did not explicitly set split mode or tensor split in these commands, so I cannot confirm that they produced the equal four-GPU placement I intended. I marked omitted cache and draft-length options as omitted in the table because the commands do not establish their effective values. The model mostly produced unfinished reasoning within the token limit; I did not score answer quality.

## Results

Rates are tokens/s. RAM means the command explicitly requested `-ngl all -cmoe`; auto means those options were absent. Cache values are requested MiB, not measured allocations or per-device occupancy.

| Command line | Binary | Model | Placement | Cache | MTP n-max | Generation |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 194 | V620 | Flash | RAM | 0 | 2 | 14.4 |
| 255 | V620 | non-Flash | RAM | 0 | 2 | 6.3 |
| 311 | V620 | non-Flash | RAM | 0 | 3 | 6.1 |
| 367 | V620 | Flash | RAM | 64536 | 2 | 12.0 |
| 428 | V620 | Flash | RAM | 98304 | 2 | 16.0 |
| 490 | V620 | Flash | RAM | 98304 | 3 | 15.1 |
| 552 | PATH | Flash | RAM | 98304 | 3 | Allocation failure |
| 565 | PATH | Flash | RAM | omitted | 3 | 14.0 |
| 626 | PATH | Flash | RAM | omitted | omitted | 14.0 |
| 688 | V620 | Flash | RAM | omitted | omitted | 13.7 |
| 749 | V620 | Flash | RAM | omitted | omitted | 13.8 |
| 811 | V620 | Flash | auto | omitted | omitted | 19.1 |
| 874 | PATH | Flash | auto | omitted | omitted | 18.3 |
| 933 | V620 | Flash | auto | omitted | 2 | 19.4 |
| 998 | PATH | Flash | RAM | omitted | omitted | 13.9 |

## Placement, cache size, and draft length

For the V620 runs with draft length omitted, automatic placement at 19.1 t/s is about 39% above the mean of the two forced-RAM results, 13.75 t/s. The PATH results show the same direction: 18.3 t/s with automatic placement versus 13.9–14.0 t/s with forced RAM. Both builds ran faster with automatic placement, but I need repeated trials and tensor-allocation records to determine which placement changes produced the gain.

The build comparison is weaker. Automatic placement gave 19.1 t/s for V620 and 18.3 t/s for PATH, a 4.4% difference in these individual runs. Under forced RAM, the V620 results of 13.7–13.8 t/s did not exceed the PATH results of 13.9–14.0 t/s. I did not find a consistent decode advantage from the V620 build in these tests.

Cache behavior was not monotonic. With n-max 2, requesting 64536 MiB reduced generation from 14.4 to 12.0 t/s; requesting 98304 MiB raised it to 16.0 t/s. I ran each of these settings once and did not record cache-hit statistics or control the warm state, so I cannot identify an optimal cache budget from this comparison. The PATH attempt at 98304 MiB failed while allocating a 3483.11 MiB compute buffer on ROCm3 and produced no timing result.

Increasing n-max from 2 to 3 also failed to improve the two directly recorded pairs: Flash at cache 98304 fell from 16.0 to 15.1 t/s, and the larger non-Flash model fell from 6.3 to 6.1 t/s. Drafting more tokens creates additional work; without acceptance counts I cannot separate lower acceptance from draft or verification cost. Since every successful run has MTP enabled, none measures MTP's benefit over ordinary decoding.

To measure the effects separately, I need to compare speculation off and on under both automatic and forced-RAM placement, using identical split settings and explicit cache budgets. Recording actual placement, accepted draft tokens, and repeated timings would let me distinguish the configuration effects from run-to-run variation. I cannot compare these short-prompt CLI rates directly with the normalized pp8192/tg128 baseline because the workloads differ.
