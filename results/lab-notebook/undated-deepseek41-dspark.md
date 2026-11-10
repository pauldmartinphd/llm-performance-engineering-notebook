# DeepSeek-V4.1: target inference worked; DSpark ran out of VRAM

[All lab records](README.md) · [Related investigation: chapter 3](../../investigations/03-making-speculation-useful.md)

You ran the DeepSeek-V4.1 target, then tried to load its DSpark drafter.

## Setup

The target used Q4_K_M, the ZFS prompt, 32 threads, 8192 context, greedy decoding, and 256 output tokens. The later four-way comparison commands have no saved output.

## Results

Target-only generation completed at 11.2 tokens/s. The DSpark attempt failed while allocating 2464.20 MiB on ROCm0 during draft loading. The captured answer ended at the token limit and was not scored for correctness.

**Date:** Unknown. [Raw record](../raw-logs/2026-10-21-flash-experiments/DeepSeekV4Flash%20Experiments.txt).

**Build clarification, October 22, 2026:** you ran the V620 tests with the V620 patches, the older DeepSeek implementation, GLM-5.3 MTP support, and your round-robin prefill patch merged together. These results describe that combined build, before you replaced the DeepSeek implementation with Unsloth's newer port. The executable reports `b0-unknown`, so the log does not identify the exact commit.

You completed one target-only generation run, then ran out of VRAM while loading the DSpark draft model. You also saved commands for the four-way comparison, but the log contains no output from those commands. You therefore have a target-only timing and a draft-loading failure, not a DSpark performance result.

The target was the eleven-shard Q4_K_M export from `vcruz305/DeepSeek-V4.1-Flash-GGUF`; the draft download came from `JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF`. The historical V620 executable reported `b0-unknown`. With 32 threads, an 8,192-token context, flash attention, `--load-mode none`, greedy decoding, and a 256-token limit, the ZFS prompt produced **11.2 generated tokens/s**. Its command did not explicitly request CPU-MoE placement or a tensor split. The output was recognizable but unfinished reasoning, not a complete answer or a correctness check.

The DSpark command requested draft GPU offload and n-max 2. After about four minutes fifty seconds, ROCm0 could not allocate another **2464.20 MiB**, and the draft model failed to load. The log labels the earlier `ctx_other` message as normal during fitting. The allocation failure stopped the test before you could measure generation speed or draft acceptance.

The remaining commands specify equal layer-split weights across four GPUs, cache zero, and n-max 5 for DSpark, with speculation off/on crossed against automatic/forced-RAM expert placement. There is no corresponding output. You cannot fill those four cells from the earlier two launches, which used different settings.

You did not run a round-robin-off control in this capture, so the target-only timing does not isolate the scheduler patch's effect.

## Replacing the older port with Unsloth

You replaced the older V4.1 port with Daniel Han and Unsloth's implementation, as described in [the AMD fork notes](2026-10-21-llamda-amd-fork.md). You documented the pairing of the Unsloth V4.1-Flash-1017 target export and matching DSpark sidecar. You have not benchmarked that pairing. The vcruz305/JigSawPT run used different exports and an older implementation, so you need a new run with the source revision, GGUF identity, launch settings, and speculation statistics to evaluate the replacement.

You also documented the reported prompt-prefix reuse issue and the `cache_prompt: false` workaround. That issue affects the newer integration; it does not explain the VRAM allocation failure in this earlier test.

You measured the earlier [DeepSeek-V4-Flash-0731 DSpark result](2026-08-15-to-16-common-baseline-2tb.md) with another model revision and build. You cannot use it to fill the missing V4.1 DSpark result.

## Conclusion

The target worked, but the drafter failed to load. This run gives you no V4.1 DSpark timing. Results from the older V4 model cannot fill that gap.
