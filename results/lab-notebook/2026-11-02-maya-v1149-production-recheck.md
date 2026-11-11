# Maya v1.1.49: production replay and the two-pair utilization question

[All lab records](README.md) · [Related investigation: chapter 5](../../investigations/05-tuning-long-prompts.md)

## Purpose and reasoning

Check whether the updated DeepSeek installation retains its long-prefill performance, and put the observation of alternating GPU pairs in context.

## Experimental design

I replayed the original saved 32K/64K request payloads against the production endpoint. I read engine timing and cache counters, keeping 128 output tokens per request. The client reads connection details from the config but does not independently prove the running engine's full settings.

## Results

The 32,800-token request measured 461.1 prefill and 24.7 generation tokens/s. The 65,568-token request measured 480.9 and 30.9. Both reported zero cached tokens.

## Conclusions

The production replay retained the established long-prefill range. Alternating activity between GPU groups did not by itself establish a new scheduler defect. One pass per length does not isolate the release's performance effect.

## Detailed record

**Experiments: November 2, 2026.**

I updated the detached Maya checkout to v1.1.49 and replayed the original long requests against DeepSeek on Galactus. Both requests completed with zero cached tokens. The 64K result remained near the approximately 480 tokens/s range established in the October tuning, so the observation that only one pair of V620s appeared active at a time did not identify a new regression.

## Production replay

| Saved request | Actual prompt tokens | Cached | Output | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| `r1-32768.json` | 32,800 | 0 | 128 | 461.1 | 24.7 |
| `r1-65536.json` | 65,568 | 0 | 128 | 480.9 | 30.9 |

The client read `maya-ds4-production.json` for connection details and posted the unchanged requests to port 8081. The [retained responses](../raw-logs/2026-11-10-maya-collection/extracted/20261102-083737-production-recheck/) include both payloads and responses. The directory is named `083737`; the conversation paste is timestamped 09:44. I preserve both clocks rather than forcing them to agree. Reading a configuration for the port does not independently prove the running server's configuration.

## GPU utilization and the next question

I asked why prefill appeared to use two cards instead of all four. The conversation proposed a dependency between two TP2 groups. That explains why utilization alone is insufficient to diagnose a scheduler defect: sequential layer groups still hold different weights and expert caches, while cards within a TP group can work together. The supplied screenshot is referenced by the export but its image bytes are absent, so its utilization cannot be remeasured here.

The subsequent question was whether keeping DeepSeek and GLM-Flash resident could avoid model-switching delays. I considered two instances with bounded expert pools, hot-swapping GPU caches, and assigning fewer GPUs to each model. No cache-suspend/resume implementation was demonstrated. This motivated the [GPU and cache matrix](2026-11-02-maya-gpu-cache-matrix.md), rather than establishing that simultaneous residency was already working.

The [October 30 entry](2026-10-30-maya-v1141-benchmark.md) now also has its original four long-response files in this collection. Today's replay is another production observation, not a controlled v1.1.41/v1.1.49 A/B test.

## Sources and recording dates

**Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
