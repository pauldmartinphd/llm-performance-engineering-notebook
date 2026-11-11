# Full GLM-5.3 Q4: allocation failures, legal layer boundaries, and a slow working split

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

## Purpose and reasoning

Test the full GLM-5.3 Q4 model on Galactus and determine whether correcting the layer split makes it practical for the intended interactive use.

## Experimental design

I attempted the recorded pack load, inspected physical/process memory and GPU allocation, then used all four V620s with 128K configured context. I diagnosed automatic placement, moved manual boundaries to legal full-indexer layers, and ran the three-answer benchmark with separate 2K/8K prefill. MTP is checked in the log.

## Results

The first config failed loading. The automatic split loaded but generated only 3.6–4.3 tokens/s and stalled before completing prefill. The first manual boundaries were illegal. The corrected 18/38/58 split completed at 6.7 tokens/s mean generation, with 60 at 2K prefill and 47 at 8K.

## Conclusions

The corrected split made the benchmark complete, but the measured setup was too slow for the intended use. I stopped the remaining full-model matrix and returned to Flash. The generic initial memory error and later watchdog do not independently establish every underlying failure mechanism.

## Detailed record

**Experiments: November 9, 2026.**

I tested the full GLM-5.3 UD-Q4_K_XL already stored as eleven GGUF shards, separately from GLM-5.3-Flash. The first configuration failed to load; automatic four-card placement then loaded but generated only 3.6–4.3 tokens/s and stalled before completing prefill. A legal manual split completed at **6.7 tokens/s mean decode**, with **60 tokens/s at 2K prefill and 47 at 8K**. I stopped the full-model investigation and returned to Flash for the two-model service.

## Loading and memory diagnosis

The initial setup used `build-hip/maya`, `MAYA_GLM_SPLIT=0`, a 4096 MB prefill budget, 1024-token mixer sub-batches and 3072 MB reserve. A port-8080 guard initially blocked the benchmark; changing a requested port did not isolate GPU memory, and the guard still triggered. The next attempt ended with `maya generate: pack: out of memory`.

Physical-memory checks showed about 2 TiB available, unlimited memlock and address space, no swap, no cgroup memory ceiling and no recorded cgroup OOM kill. This weakens the explanation that system RAM capacity was exhausted. The generic pack error did not identify the allocation; the conversation's GPU/KV explanation remained a hypothesis. I explicitly corrected the intended context to 128K and requested all four GPUs. The initial failed log's 1M context line is preserved rather than retroactively changed.

## Automatic split and prefill failure

The four-GPU, 131,072-context run loaded in 441 seconds. Its three 256-token decode cases measured 4.3, 3.6 and 4.3 tokens/s. The engine reported the split `[0,1)`, `[1,2)`, `[2,42)`, `[42,78)`, leaving almost 29.1 GB free on each of the first two cards while the last two had about 3 GB free. Its own cost model predicted approximately 407.4 ms/token, with almost all cost on the last two parts.

The critical prefill line says the CUDA2 pool could not lend 3210 MB and fell back to token-by-token execution. The common split chunk became one token. The benchmark later stopped after the server's 180-second no-progress watchdog. This is a recorded workspace fallback followed by a stall; a slow path alone does not prove the watchdog's underlying cause. The retained log reports zero disk reads per token during the decode observations and hundreds of CPU-lane expert operations per token, so it does not support a disk-capacity explanation for those slow answers.

Native MTP was active: the log loaded block 78 and printed draft acceptance of 85.4–98.0% across the three measured answers. The low throughput was therefore not simply an absent drafter.

## Invalid and corrected manual boundaries

The first manual split `20,40,60` failed immediately. Layer 20 shares the previous layer's indexer top-k and cannot begin a new part. The diagnostic lists legal full-indexer boundaries, including 18 and 22. The corrected retained config uses `18,38,58`; it requests all four cards, 128K context and INT8 KV. These are layer partitions, not four-way tensor parallelism.

| Working manual split benchmark | Result |
|---|---:|
| Load | 416 seconds |
| CSV / prose / HTML decode | 6.6 / 5.5 / 8.1 tokens/s |
| Printed mean decode | 6.7 tokens/s |
| 2,048-token prefill | 60 tokens/s |
| 8,192-token prefill | 47 tokens/s |

The benchmark prints 19:51; the paste is timestamped 20:51. Both identify November 9. The [retained config](../raw-logs/2026-11-10-maya-collection/extracted/maya-glm-5.3-q4-4gpu-balanced.json) records the final settings, but `maya-bench.txt` and `maya-bench.log` are shared filenames overwritten by later Flash tests. The full-model timings and failure diagnostics survive in the extracted conversation captures.

## Interpretation and decision

The conversation compared the result with earlier approximately 6.1–6.3 tokens/s MTP results. Without matched model hashes, prompts, contexts and builds, that is historical context rather than a measured Maya-versus-llama.cpp improvement. Likewise, short prompts may fail to amortize setup, but the supplied evidence does not prove that shortness explains the 47 tokens/s 8K prefill result.

I judged this full-model setup unsuitable for my intended interactive use and ended its remaining KV/MTP matrix. That decision applies to this configuration, not every possible implementation of full GLM. I then moved to [GLM-Flash on GPUs 2 and 3](2026-11-09-glm-flash-reserve-and-speculation.md).

## Sources and recording dates

**Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
