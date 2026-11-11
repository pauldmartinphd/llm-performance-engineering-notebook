# Throughput predictions for eight MoE models: speculation and layer offload

[All lab records](README.md) · [Related experiment: 1](../../investigations/01-establishing-a-baseline.md)

**Record type:** Analysis of predictions; no new benchmark.

**Date:** May 6, 2026. **Source:** *Speculative Decoding and Layer Offload on Galactus: Throughput Analysis Across Eight MoE Models*, a report written by Claude; its document metadata gives the creation date as May 6, 2026. **Status:** AI-summarized analysis and predictions; no new benchmark. The original document is kept outside this repository.

The report combined our April measurements with a structural model of speculative decoding and expert-layer offload, and predicted decode throughput for eight MoE models on Galactus with 1 TB of DDR4, before the August 2 TB upgrade. This entry keeps its measured inputs, model, predictions, and proposed test order, and compares them with what we measured later. The report's predictions are not measurements. Its literature and engine-status claims (the GLM-5 paper's QAT description, arXiv 2602.15763; the Qwen 3.6-35B-A3B speculation benchmarks; llama.cpp issue #22319; the antirez DeepSeek V4 fork; SGLang and vLLM support for MiMo V2.5) are leads we have not verified here.

## Measured inputs

| Report value | Where the notebook records it | Note |
|---|---|---|
| Kimi K2.5 UD-Q4_K_XL, 6.76 t/s at 32 threads, no offload | [Kimi K2.5/K2.6](../kimi-k2.5.md), [April tests](2026-04-07-to-17-thread-and-batch-tests.md) | Offload not measured. |
| Qwen 397B Q6_K_L, 9.69 t/s, no offload; 10.81 (16 layers) and 11.10 (20 layers) | [Qwen 3.5 397B](../qwen-3.5-397b.md) | 16 and 20 layers are four and five expert layers per card. |
| Qwen 397B, 11.67 t/s with 20 layers at DDR4-2933 | [Qwen 3.5 397B](../qwen-3.5-397b.md), April 10 re-run, build 8746 | |
| MiniMax M2.7 Q5_K_M, 14.76 t/s, no offload; 17.37 with 42 layers | [MiniMax M2.7](../minimax-m2.7.md) | `-ngl 42` across four GPUs. |
| STREAM: DDR4-2666 Triad 104.5, Copy 140.6 GB/s; DDR4-2933 Triad 109.5, Copy 150.7 GB/s | Not retained elsewhere | The [March build entry](2026-03-23-galactus-build.md) records 141 GB/s Copy and 107 GB/s Triad for the matched DDR4-2933 DIMMs. |

The report places the Kimi and Qwen no-offload baselines at DDR4-2666. That is correct: the 1 TB memory ran at 2666 MT/s until April 10, and the April 10 Qwen run was the first after we overclocked it to 2933 MT/s. The 2 TB population that replaced it in August runs at its rated 2933 MT/s. The model notes' System lines were corrected on November 2.

## The report's model

**Expert union in the verify pass.** Each candidate token routes to its own experts, so verifying K candidates loads the union of their experts. Under uniform random routing, the expected unique count for k of N experts is N × (1 − ((N − k)/N)^K). For K = 3:

| Model | Routing | Unique experts | Expansion |
|---|---|---|---|
| Kimi K2.5 | 8 of 384 | 23.5 | 2.94× |
| GLM 5.1 | 8 of 256 | 23.3 | 2.91× |
| Qwen 397B | 8 of 128 | 22.6 | 2.83× |
| DeepSeek V4 Pro | 6 of 384 | 17.9 of 18 | 2.98× |

The report put the speedup ceiling at acceptance ÷ expansion: 2.7 ÷ ~2.9 ≈ 0.93× before amortizing fixed costs. It estimated that trained-router correlation reduces the unique count by 5–10%.

**Hybrid placement.** Putting attention, dense components, and the KV cache on the GPUs raises baseline throughput, but the report argued that it leaves only routed-expert loading on the CPU side, which speculation cannot amortize under sparse routing.

**Layer offload.** Verify-pass work for MoE layers resident in VRAM was treated as free of the K penalty: effective scaling = f_GPU × 1.0 + f_RAM × K expansion, where f_GPU is the fraction of MoE layers in VRAM.

**Inferences from the measurements.** Qwen at 31% of layers offloaded gained 14–21%, not the 45% that bandwidth scaling alone predicts. The report inferred that CPU memory bandwidth accounts for about 25–30% of single-token decode time. It also inferred that offload gains plateau near 1.20× in llama.cpp (MiniMax 1.18×, Qwen 1.14×), that the 2666-to-2933 change gives about 5% throughput, and that decode peaks at 32–64 threads and fails at 128. It attributed the decline above 96 threads to NUMA-boundary contention.

## Predictions

Decode in tokens per second. Bold values were measured; the others are the report's predictions.

| Model | Active | Quantization | Baseline | + Offload | + Speculation |
|---|---|---|---|---|---|
| DeepSeek V4 Flash, 284B | 13B | FP4 native | 13–16 | 15–19 | 18–23 |
| MiMo V2.5, 310B | 15B | Q6 (FP8 native) | 9–11 | 11–13 | 13–17 |
| MiniMax M2.7, 230B | 10B | Q5 | **14.76** | **17.37** | 19–22 |
| Qwen 397B A17B | 17B | Q6 | **9.69** | **11.67** | 13–15; 16 good |
| Kimi K2.5, 1.00T | 32B | INT4 native (QAT) | **6.76** | 7–7.8 | 7–8.5 |
| GLM 5.1, 754B | 40B | INT4 native (QAT) | 5.4 | 6.2–6.5 | 6.8–7.5 |
| MiMo V2.5 Pro, 1.02T | 42B | Q6 (FP8 native) | 3.5–4.8 | 4–5.8 | 4.5–6.5 |
| DeepSeek V4 Pro, 1.6T | 49B | FP4 native | 4–5 | 4.5–6 | 5–6.5 |

The GLM 5.1 baseline was scaled from Kimi: 6.76 × 32/40 = 5.4. The report estimated about 30 of 43 DeepSeek V4 Flash layers fitting in VRAM at FP4 (about 3.3 GB per layer) and 7 of 61 for V4 Pro (about 13 GB per layer). It classed DeepSeek V4 Flash, MiMo V2.5, MiniMax M2.7, and Qwen 397B as throughput choices above 10 t/s, and the other four as capability choices at 4–9 t/s. It expected a 33–50% penalty for running a model at Q6 rather than a Q4-class native format, and a ±10% byte-size difference between FP4 native and Q4 GGUF.

Proposed test order: Qwen 397B with 20-layer offload plus speculation; MiniMax M2.7 with speculation; DeepSeek V4 Flash once ROCm support existed; Kimi K2.5 offload before speculation; the capability models once each. For Qwen it set decision thresholds: above 16 t/s would mean the model underestimates speculation on Galactus; below 13 would mean batched verify has more overhead than modeled, and Kimi and GLM speculation should not be pursued.

## Later results

- **DeepSeek V4 Flash:** the [August common baseline](2026-08-15-to-16-common-baseline-2tb.md) measured V4-Flash-0731 UD-Q8_K_XL with MXFP4 experts at 10.34 t/s and 14.1 ± 0.7 with DSpark n=3, below the predicted 13–16 and 18–23. That run used the 2 TB population and a different build. With Maya on October 23, DeepSeek-V4.1-Flash-1017 reached 29.3 t/s with speculation and 21.2 without ([record](2026-10-23-maya-glm53-speculation.md)); that is a different model revision and engine.
- **Qwen 397B:** the report's first test did not run as planned. In August, Qwen MTP did not arm, and decode was 9.37 t/s with all routed experts in RAM. The April 16- and 20-layer placements no longer fit beside the ub-8192 compute buffers, so we abandoned the offload reruns.
- **Speculation ceiling:** measured gains exceed the report's 1.10–1.40× range: GLM-5.3-Flash at +58.8% with Maya. The earlier llama.cpp gains were +31% for GLM-5.2 MTP and +45% for DeepSeek V4 Flash DSpark. The [speculative decoding takeaways](../../takeaways/speculative-decoding.md) use a fitted C + S cost model for those llama.cpp results instead of the acceptance ÷ expansion ceiling.
- **GLM:** the report predicted GLM 5.1. We measured GLM-5.2 in August at 5.30 t/s and 6.6 ± 0.3 with MTP n=2.
- **Kimi:** K2.6 measured 5.79 t/s in August. Kimi K2.5 offload was not measured.
- **MiniMax M2.7:** August decode was 15.18 t/s with all routed experts in RAM. Speculation was not tested before we retired the model on August 16.
- **Threads:** decode collapse at 128 threads matches the SMT oversubscription row in [Negative results](../../takeaways/refuted-hypotheses.md). The NUMA explanation is refuted: Galactus runs as a single NUMA node (NPS1).
- **Not measured:** MiMo V2.5, MiMo V2.5 Pro, DeepSeek V4 Pro.
