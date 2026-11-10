# Throughput bounds and alternative Qwen deployments

**Discussion: November 9, 2026; no Galactus Qwen experiment in this entry.** **Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.

I began with a bandwidth estimate for Qwen3.6-35B-A3B, then considered full GLM-5.3 and Qwen3.8-Flash-Next as possible uses of the available hardware. The practical question was whether four GPUs should serve one large model or several services. The subsequent DeepSeek and GLM tests answered parts of that question; the calculations and external reports remained candidate-selection material.

## Bounds versus end-to-end throughput

The initial prompt said 150 MB/s, while the discussion used approximately 150 GB/s as Galactus's system-memory bandwidth assumption. A bandwidth-only decode ceiling takes the form `bandwidth / bytes read per generated token`. It does not include compute, synchronization, routing, attention state or speculation, and moving all weights into VRAM changes which bandwidth is relevant. The four cards' nominal bandwidth cannot simply be added when layer groups execute sequentially.

The conversation's full GLM estimate of 15–30 tokens/s generation and approximately 2,000 tokens/s prefill was unvalidated. The later full-model measurement was only 6.7 decode and 47 prefill tokens/s at 8K. That disagreement is useful evidence against treating the simple bound as a prediction. The [full-model entry](2026-11-09-full-glm53-placement.md) explains the actual allocation and prefill failures.

## External reference material

The conversation cited single-V620 Vulkan Qwen3.6 UD-Q5_K_XL results around 78 tokens/s without speculation and 92–137 with MTP, plus context-dependent prefill results; it also cited R9700 Qwen and EPYC/R9700 Strata reports. These are assistant-retrieved claims preserved in the supplied export, not independently reproduced Galactus measurements. No original external benchmark datasets were included for those claims, so I have not promoted them into the notebook's measured comparison tables.

The Qwen3.8-Flash-Next proposal assumed CPU-resident routed experts with GPU attention, dense/shared work, prefill and a hot-expert cache. Its quoted 1.50 GB of routed-expert reads per token and assumed 150 GB/s give a memory-only 100 tokens/s ceiling before other work; cache hits and speculative execution alter the relationship. The proposed 65–105 generation and 1,100–1,800 tokens/s 32K prefill ranges were engineering targets derived from another host, not observations on my EPYC 7713.

## Decision boundary

I considered a fast 35B model on one card, Flash-Next on an R9700, and separate speech/OCR services. None of those alternatives was deployed or benchmarked in the supplied evidence. I instead tested DeepSeek TP2 and GLM-Flash on the four existing V620s. The archive also contains upstream V100 reports, recorded [separately](undated-maya-reference-benchmarks-and-live-logs.md), which must not be mistaken for Galactus results.

## Weight-quantization discussion retained as unverified planning evidence

The exported assistant response reported GLM-Flash Q4/Q6 sizes of 199.71 / 291.83 GB, mean KLD versus BF16 of 0.049294 / 0.019007, and top-token agreement of 92.22% / 95.23%. It attributed these to external quantization reports, not my tests. It also reported Q4 perplexity 3.3038 ± 0.042 without a matched Q6 value. No underlying evaluation data is supplied here, so these remain source claims and cannot establish a local quality difference. The suggested 15–35% Q6 decode penalty was an estimate; no Q6 deployment or speed test followed.

EXL3 Q4 was discussed as a separate weight format, not an alias for the GGUF UD-Q4_K_XL pack tested here. Neither this discussion nor the external quality figures measures FP16-versus-INT8 KV behavior. I kept the existing Q4 weights during the local cache experiment.
