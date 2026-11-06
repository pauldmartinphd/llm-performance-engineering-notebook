# Fast Card Prefill Acceleration

**User:** Anonymous  
**Created:** 11/5/2026 8:44:58  
**Updated:** 11/6/2026 15:45:18  
**Exported:** 11/6/2026 15:45:21  
**Link:** [https://chatgpt.com/c/6aec885a-8ba4-83e9-8dea-6de84e1bb971](https://chatgpt.com/c/6aec885a-8ba4-83e9-8dea-6de84e1bb971)  

## Prompt:
11/5/2026, 8:45:10 AM

Can a single fast card help with the acceleration of prefill on a huge model?

## Response:
11/5/2026, 8:45:24 AM · gpt-6-thinking

> **Searched 7 websites**
>
> 
>
> Worked for 14s

**Yes. A single high-performance GPU can substantially accelerate prefill for a model much larger than its VRAM.** However, the benefit depends on how the inference engine partitions the model, where the weights reside, and how quickly the GPU can access them.

For your Project Maya configuration, with a very large MoE model and multiple Radeon Pro V620s, adding a faster GPU could improve prefill even if that GPU cannot hold most of the model.

The important distinction is that **prefill and token generation have different computational characteristics.**

## Why a fast GPU can help with prefill

During prefill, the model processes many prompt tokens simultaneously. This lets the engine turn much of the work into large matrix multiplications.

During decoding, the engine typically processes one new token per sequence at a time. Large MoE models often become limited by memory bandwidth, expert loading, and communication rather than raw arithmetic throughput.

Consequently\:

- **Prefill\:** High compute throughput, efficient matrix multiplication, and sufficient memory bandwidth matter.
- **Decode\:** Weight-access latency, memory bandwidth, expert placement, and synchronization become particularly important.

A newer GPU with much higher matrix multiplication throughput could therefore offer a disproportionate improvement to prefill without delivering a comparable increase in decoding speed.

## The architecture I would consider

Suppose you add one significantly faster GPU, such as a Radeon AI PRO R9700, to your four V620s.

Conceptual heterogeneous inference architecture

**Fast GPU - R9700**

Accelerates prefill computation, with weights streamed or partitioned across devices

Activations, weights or KV-cache transfers

V620 1

32 GB

V620 2

32 GB

V620 3

32 GB

V620 4

32 GB

128 GB aggregate V620 VRAM for expert storage and distributed execution

Conceptual design, not a currently verified Project Maya capability.

There are several ways to organize this\:

| Approach | Potential | Main limitation |
|--------|--------|--------|
| Fast GPU handles selected layers | Moderate | Other layers remain bottlenecks |
| Fast GPU handles attention, other GPUs handle experts | Potentially substantial | Cross-GPU communication and scheduling |
| Fast GPU performs complete prefill, V620s decode | Potentially substantial | Fast GPU must access all required weights |
| Fast GPU receives a larger share of model layers | Straightforward | Speed depends on partitioning and weight movement |

The third approach uses *disaggregated prefill and decode*, an established inference architecture. \[[How NVIDIA GB200 NVL72 and NVIDIA Dynamo Boost Inference Performance for MoE Models](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models?utm_source=chatgpt.com); [Designing distributed AI inference: Core concepts and scaling dimensions](https://developers.redhat.com/articles/2026/06/22/designing-distributed-ai-inference-core-concepts-and-scaling-dimensions?utm_source=chatgpt.com)\]

However, one fast GPU cannot perform an entire model's prefill merely by accessing the other GPUs' VRAM as though it were local memory. Moving weights or intermediate results introduces bandwidth and latency costs.

## What matters for Project Maya specifically

Maya's documented approach partitions layers across GPUs. Its prefill implementation also streams experts and adjusts chunk sizes according to available memory. The documentation notes that, for a split model, GPUs may operate sequentially on a single prompt chunk, with overlapping work becoming possible across multiple chunks. \[[project-maya/docs/SETTINGS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md?utm_source=chatgpt.com); [mw00/project-maya v1.0.12 on GitHub](https://newreleases.io/project/github/mw00/project-maya/release/v1.0.12?utm_source=chatgpt.com)\]

That means adding an R9700 would not automatically turn it into a dedicated prefill accelerator. Maya would need a scheduling strategy that exploits the faster device.

There is also an important difference between your configuration and a model that fits entirely in VRAM: **expert transfers could dominate enough of the runtime to negate much of the faster GPU's advantage.**

For example, if transferring weights and synchronizing GPUs takes 70% of prefill time, even an infinitely fast compute device could only deliver a theoretical 1.43× acceleration without reducing that overhead.

## What I would test on your machine

Your previous DeepSeek measurements reached roughly 460-480 prefill tokens/s on long prompts with four V620s. That provides a useful baseline.

I would measure per-stage prefill time, GPU utilization, effective weight-transfer bandwidth, and whether the GPU kernels are actually compute-bound. Then compare the existing V620 configuration against heterogeneous layer partitioning.

**My main question is whether you want to add one fast GPU to your four V620s, or use the fast GPU exclusively for prefill while keeping the entire model on the V620s for decoding.** The latter is a more ambitious architecture, but potentially more interesting if prefill speed is your primary objective.

## Prompt:
11/5/2026, 8:46:14 AM

How would a single R9700 do

## Response:
11/5/2026, 8:46:24 AM · gpt-6-thinking

> **Searched 3 websites**
>
> 
>
> Worked for 11s

A single AMD Radeon AI PRO R9700 could be a useful prefill accelerator for your four V620s, but **I would not expect it to dramatically accelerate DeepSeek V4.1 Flash without substantial changes to Project Maya's execution strategy.**

The R9700 has considerably stronger matrix hardware than the V620, although its memory bandwidth is only modestly higher.

| Specification | R9700 | V620 |
|--------|--------|--------|
| Architecture | RDNA 4 | RDNA 2 |
| VRAM | 32 GB | 32 GB |
| Memory bandwidth | 640 GB/s | 512 GB/s |
| FP16 vector throughput | 47.8 TFLOPS | 40.55 TFLOPS |
| FP16 matrix throughput | 191 TFLOPS | Not directly comparable |
| PCIe | 5.0 x16 | 4.0 x16 |

AMD's specifications: \[[AMD Radeon™ AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html?utm_source=chatgpt.com); [AMD Radeon™ PRO V620](https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html?utm_source=chatgpt.com)\]

## Actual Project Maya results

The Maya developer has published measurements with an R9700 on GLM-5.3-Flash\:

Reported prefill throughput

Upper end of the reported R9700 range shown; results depend on configuration. \[[project-maya/docs/AMD\_MAYA.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md?utm_source=chatgpt.com); [mw00/project-maya v1.0.14 on GitHub](https://newreleases.io/project/github/mw00/project-maya/release/v1.0.14?utm_source=chatgpt.com)\]

Your four V620s achieved approximately **480 tokens/s** on DeepSeek long-context prefill.

The R9700's published 490-560 tokens/s results look comparable, but they involve a different model, quantization, workload, and memory configuration. They do not establish how the R9700 would perform on your DeepSeek model.

## The main obstacle: moving model weights

If the R9700 performs all prefill computation, it needs access to the entire model's weights.

With only 32 GB of VRAM, most weights must come from system RAM or the V620s. Your EPYC 7713 platform also limits the R9700 to PCIe 4.0 speeds, regardless of its PCIe 5.0 capability.

That makes transferring weights a potentially severe bottleneck. For a model with hundreds of gigabytes of weights, transferring each weight only once can already be expensive. Repeated transfers would be worse.

A more promising approach would be to leave the MoE experts on the V620s and use the R9700 for selected computationally intensive stages, particularly attention and dense projections. However, that requires specialized heterogeneous scheduling.

## My assessment

| Configuration | Expected benefit |
|--------|--------|
| R9700 alone, model mostly in system RAM | Uncertain; PCIe bottleneck |
| R9700 + four V620s, ordinary layer splitting | Likely limited prefill improvement |
| R9700 for dense computation, V620s for experts | Potentially substantial, but requires engine development |
| R9700 exclusively for prefill with fully optimized weight streaming | Interesting research direction; benefit unproven |

**I would not purchase an R9700 solely to accelerate your existing four-V620 setup yet.** Its compute capabilities make it a reasonable candidate for heterogeneous prefill, but Maya would need to exploit them specifically.

The most interesting possibility is assigning the R9700 the attention and dense layers while the V620s process MoE experts in parallel. That would preserve most of your existing VRAM capacity while exploiting RDNA 4's faster matrix hardware.

## Prompt:
11/5/2026, 8:47:20 AM

So here's the thing.  I have a PC with an R9700 and two V620s in it.  I am figuring out whether to get another R9700 or not

## Response:
11/5/2026, 8:47:23 AM · gpt-6-thinking

I’ll compare the value of a second R9700 against the existing R9700 + two V620s, focusing on prefill, memory placement, and what Project Maya can actually exploit.

> **Searched 9 websites**
>
> 
>
> Worked for 17s

That changes the decision considerably. You already have a heterogeneous three-GPU system, so the question is whether moving from **1× R9700 + 2× V620 to 2× R9700 + 2× V620** produces enough additional inference performance to justify the purchase.

My preliminary recommendation is to **keep the existing three cards unless you can establish that the second R9700 will accelerate prefill rather than simply add VRAM and another layer partition.**

## What the second R9700 adds

| Configuration | Current | With upgrade |
|--------|--------|--------|
| R9700 | 1 | 2 |
| V620 | 2 | 2 |
| Total VRAM | 96 GB | 128 GB |
| Aggregate theoretical VRAM bandwidth | 1.66 TB/s | 2.30 TB/s |
| RDNA 4 FP16 matrix throughput | 191 TFLOPS | 382 TFLOPS |

The aggregate bandwidth and compute figures are hardware sums, not bandwidth or compute available to any arbitrary model operation. AMD specifies 640 GB/s memory bandwidth and 191 TFLOPS dense FP16 matrix throughput per R9700. \[[AMD Radeon™ AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html?utm_source=chatgpt.com); [AMD Radeon™ PRO V620](https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html?utm_source=chatgpt.com)\]

The second card therefore adds considerable matrix compute capacity. But accessing that capacity during prefill requires the engine to distribute useful work to both R9700s.

## Why the RDNA 4 versus RDNA 2 distinction matters

Project Maya's current AMD documentation explicitly notes that the V620 lacks the matrix hardware used by its RDNA 4 path. Instead, Maya uses portable kernels on RDNA 2, including an FP32 attention path. \[[project-maya/docs/AMD\_MAYA.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md?utm_source=chatgpt.com)\]

That suggests a second R9700 may be particularly useful if it takes layers currently executing on the V620s. Merely adding another card while retaining an equal layer split may leave substantial prefill work on the slower devices.

There is an architectural complication: Maya's AMD documentation describes heterogeneous two-GPU layer splitting, but I could not verify a benchmark for your exact three- or four-GPU mixture. Its broader multi-GPU implementation supports more than two GPUs, but that does not establish efficient RDNA 2/RDNA 4 scheduling. \[[project-maya/docs/AMD\_MAYA.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md?utm_source=chatgpt.com); [mw00/project-maya v1.0.12 on GitHub](https://newreleases.io/project/github/mw00/project-maya/release/v1.0.12?utm_source=chatgpt.com)\]

## How I would evaluate the purchase

Compare these configurations on the same quantization, context lengths, and prompts\:

| Configuration | Purpose |
|--------|--------|
| 1× R9700 | Measure fast-GPU prefill with weight offloading |
| 1× R9700 + 1× V620 | Measure heterogeneous layer splitting |
| 1× R9700 + 2× V620 | Establish your existing baseline |
| 2× R9700 + 2× V620 | Evaluate the proposed upgrade |

The fourth configuration cannot be tested before buying, but the first three can reveal whether distributing work onto V620s improves prefill or simply introduces additional bottlenecks.

I would also examine whether a **2× R9700 configuration without the V620s** might outperform all four cards on prefill. That is plausible for workloads that benefit substantially from RDNA 4 matrix acceleration, but model residency and expert caching could reverse the result.

One especially important consideration is that your machine would have 64 GB of RDNA 4 VRAM after the upgrade. This might let you place more computationally intensive layers on the R9700s while using the V620s primarily for expert storage.

**My current inclination is that a second R9700 could be a more meaningful upgrade than replacing either V620 with another similarly performing card.** I cannot yet assign a credible percentage improvement.

What CPU and memory configuration does this particular PC have? Specifically, is it your 64-core, 512 GB system, and does it provide PCIe 5.0 connectivity to the R9700s? Those details materially affect whether a second R9700 can operate efficiently alongside the V620s.

## Prompt:
11/5/2026, 8:48:17 AM

I'm confused why specific RDNA2 kernels cannot be used to map the matrix operations on the the hardware that the V620 does have including dot products to accelerate matrix math

## Response:
11/5/2026, 8:48:25 AM · gpt-6-thinking

> **Searched 12 websites**
>
> 
>
> Worked for 26s

**They absolutely can. Your objection is correct.** RDNA 2 lacks dedicated matrix instructions such as WMMA and MFMA, but that does not mean it lacks hardware capable of accelerating matrix multiplication.

The V620 has native packed dot-product instructions, substantial FP16 arithmetic throughput, and 72 compute units. A properly designed GEMM kernel can exploit those capabilities.

My earlier discussion overstated the distinction between RDNA 2 and RDNA 4. In particular, treating the V620's lack of WMMA as though it necessarily requires inefficient general-purpose FP32 matrix multiplication was incorrect.

## What RDNA 2 actually supports

The V620's `gfx1030` instruction set includes\:

| Instruction | Operation | Matrix multiplication application |
|--------|--------|--------|
| `v_dot2_f32_f16` | 2 FP16 products, FP32 accumulation | FP16 GEMM |
| `v_dot4_i32_i8` | 4 INT8 products, INT32 accumulation | Quantized GEMM |
| `v_dot8_i32_i4` | 8 INT4 products, INT32 accumulation | Quantized GEMM |
| `v_pk_fma_f16` | 2 packed FP16 FMA operations | FP16 GEMM |
| `v_fma_f32` | FP32 fused multiply-add | FP32 GEMM |

AMD's RDNA 2 ISA and ROCm documentation confirm these operations. \[["RDNA 2" Instruction Set Architecture](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf?utm_source=chatgpt.com); [Floating-point dot product builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins.html?utm_source=chatgpt.com); [Syntax of GFX10 RDNA2 Instructions](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.14.1/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html?utm_source=chatgpt.com)\]

The important instruction for standard half-precision model inference is `v_dot2_f32_f16`\:

$$
d = a_0b_0+a_1b_1+c
$$

It handles two FP16 multiplications and accumulates their result into FP32.

For quantized weights, `v_dot4_i32_i8` can perform four integer products per instruction. That offers another efficient route for matrix multiplication, provided the kernel handles quantization scales and accumulation correctly.

Neither operation requires dedicated matrix cores.

## Why specialized kernels matter

There is already evidence that developers can exploit these instructions for LLM prefill.

The [RDNA 2 vLLM optimization project](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/CHANGES.md?utm_source=chatgpt.com) describes a W4A8 MoE prefill kernel that uses INT8 dot products on the V620.

Its approach is particularly relevant\:

1. Quantize activations to INT8.
2. Unpack INT4 weights into INT8 values.
3. Use `v_dot4_i32_i8` to calculate matrix products with INT32 accumulation.
4. Apply quantization scales.

The author reports that an earlier INT4 kernel achieved only about 4 TFLOPS because unpacking and converting weights consumed substantial execution time. The newer implementation avoids FP16 conversion in the inner loop. \[[vllm-rdna2-qwen/docs/rdna2/CHANGES.md at rdna2/qwen38-flash-next · leapdragon/vllm-rdna2-qwen](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/CHANGES.md?utm_source=chatgpt.com)\]

This illustrates why kernel design can matter much more than the presence or absence of WMMA.

There are still constraints. Register pressure, tile sizes, LDS usage, operand layouts, and quantization overhead all affect attainable performance. Moreover, converting FP16 or quantized computations to integer dot products is not automatically numerically equivalent to the original calculation.

## What Project Maya is doing

I checked [Maya's AMD documentation](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md?utm_source=chatgpt.com)\.

It states that RDNA 2 uses portable kernels, FP32 prompt attention, and hipBLAS for GEMMs. RDNA 2 remains experimental.

The documentation also identifies hipBLASLt tuning for prompt FP16 projections on supported architectures, with ordinary hipBLAS as the fallback. \[[project-maya/docs/AMD\_MAYA.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md?utm_source=chatgpt.com)\]

This raises an important concern: **Maya may be leaving substantial RDNA 2 prefill performance unused.**

The related Strata project documents a concrete problem: its ROCm library configuration lacks optimized gfx1030 kernels for certain FP16-input/FP32-output GEMMs. Its measurements show generic kernels running much slower than alternative implementations. \[[Strata/docs/AMD\_HIP.md at main · Niko1221/Strata](https://github.cnmirror.space/Niko1221/Strata/blob/main/docs/AMD_HIP.md?utm_source=chatgpt.com); [Strata/src/prefill/gemm.cu at main · Niko1221/Strata](https://github.com/Niko1221/Strata/blob/main/src/prefill/gemm.cu?utm_source=chatgpt.com)\]

That does not establish the same slowdown for your current Maya build, but it identifies a specific problem to investigate.

## Implications for your R9700 purchase

I would now prioritize RDNA 2 kernel optimization before purchasing another R9700.

Your V620s have\:

- 40.55 TFLOPS peak FP16 vector arithmetic per card.
- Native FP16 and integer dot products.
- 512 GB/s memory bandwidth per card.
- 32 GB VRAM per card.

Those are substantial computational resources. \[[AMD Radeon™ PRO V620](https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html?utm_source=chatgpt.com)\]

An optimized V620 GEMM implementation could improve the performance of your existing heterogeneous configuration without changing the hardware.

I would specifically profile the prefill GEMMs, inspect the generated `gfx1030` assembly, and determine whether the kernels issue packed FP16/dot-product instructions or spend excessive time performing conversions and scalar arithmetic.

If the V620s currently spend most of prefill in inefficient GEMM kernels, optimized RDNA 2 code might provide a larger improvement than another R9700.

**The relevant question is no longer simply how much faster RDNA 4 is. It is how close Maya gets to the V620's actual hardware capabilities.** Until we measure that, treating the V620s as inherently poor prefill accelerators would be premature.

## Prompt:
11/5/2026, 8:51:10 AM

## 10. W4A8 MoE prefill kernel (2026-09-24)

`vllm/model_executor/layers/fused_moe/rdna_w4a8_moe.py`, opt-in with `VLLM_RDNA_MOE_W4A8=1`, hooked at the top of `invoke_fused_moe_wna16_triton_kernel` (the modular `TritonExperts` path calls it directly). A prefill profile put the int4 MoE Triton kernel at 28 % of prefill and \~4 TFLOPS, about 11 % of the V620's measured fp16 dot rate: the kernel converts every int4 weight to fp16 in the inner loop (shift, mask, int→fp32, subtract, scale, fp32→fp16). The W4A8 kernel quantises activations per token to int8 once per GEMM, unpacks weights to int8 with integer ops only, accumulates `tl.dot(int8, int8) → int32` (lowered to `v_dot4_i32_i8`) over each 128-wide quantisation group, applies the group scale once per output element and the token scale at the store. Symmetric int4 only; decode-shaped configs (BLOCK_SIZE_M < 32) keep the stock path, so decode is unchanged.

Tiles matter more than usual on RDNA2: any K block ≥ 64 hits the 256-VGPR cap and spills 190–390 registers (up to 5× slower). Swept best at Flash-Next per-rank shapes: 64×128×16, 8 warps, 2 stages — 2.0–2.45× over W4A16 from M = 256 to 2048, exact against a torch reference of the same maths (2e-4). BLOCK_SIZE_M is taken from the caller because the token sort was aligned to it.

In-server (4× V620, TP=4/EP=4, 120 W caps): prefill 908 → 1,160 t/s @3.3k, 957 → 1,217 @14k, 962 → 1,232 @26k (+27–28 %, more than the MoE share alone predicts — halving MoE time also shrinks the all-reduce wait behind the slowest rank); decode 62.8–62.9 t/s unchanged; validate PASS; teacher-forced NLL/token vs W4A16 −0.1 % general / −0.2 % identifier-heavy (within noise), deterministic run to run.

Side finding: the MoE kernel's time is nearly flat in chunk size (per-rank \~40 tokens per expert per 2048 chunk, so every expert fills one 64-row block regardless) — the 2048-token prefill chunk was tuned before this kernel and is worth re-sweeping.

So how does hat improve prfill?

## Response:
11/5/2026, 8:51:13 AM · gpt-6-thinking

The W4A8 kernel improves prefill by replacing expensive per-weight floating-point conversions with integer dot-product operations that RDNA 2 can execute efficiently.

The reported result is a **27-28% increase in end-to-end prefill throughput**, without changing decoding performance.

## What the kernel changes

The original W4A16 implementation processes INT4 weights and FP16 activations. Inside the matrix multiplication loop, it repeatedly converts quantized weights into FP16 values.

For each weight, it performs operations equivalent to\:

1. Extract the 4-bit integer from its packed representation.
2. Convert the integer to floating point.
3. Subtract the quantization zero point.
4. Apply the quantization scale.
5. Convert the result to FP16.
6. Multiply it by the activation and accumulate.

That is a substantial amount of work for each multiply-accumulate operation.

The new kernel changes the arithmetic.

Old: W4A16

INT4 weights

Unpack + convert + scale

FP16 dot products

New: W4A8

INT4 weights + INT8 activations

Integer unpacking

INT8 dot products + scaling

The W4A8 kernel moves scaling outside the inner dot-product loop, reducing repeated conversion work.

RDNA 2 supports `v_dot4_i32_i8`, which multiplies four pairs of INT8 values and accumulates into an INT32 result.

The kernel therefore uses native packed integer arithmetic instead of reconstructing FP16 weights before multiplication.

## Why this produces a 28% prefill improvement

Your quoted benchmarks show\:

The 26K-token test improved from 962 to 1,232 tokens/s, a **28.1% increase**\.

The old MoE kernel accounted for approximately 28% of prefill execution time. If that component alone became 2.45 times faster, a simple Amdahl's-law estimate gives\:

$$
S=\frac{1}{0.72+0.28/2.45}\approx1.20
$$

That predicts approximately 20% higher throughput.

The measured improvement is greater, at 28%. The explanation provided in the benchmark is that accelerating the MoE computation also reduces synchronization delays across the four GPUs.

In tensor/expert-parallel execution, a faster rank may have to wait for the slowest rank to complete its work before collective operations can proceed. Reducing the slowest rank's execution time can improve overall efficiency beyond the time saved in the MoE kernel itself.

## Why the tile dimensions are important

The reported optimal tile is\:

`BLOCK_M=64, BLOCK_N=128, BLOCK_K=16`

Each tile computes a 64×128 portion of the output matrix while iterating over the reduction dimension in groups of 16.

The small K dimension matters because RDNA 2 has a limited vector-register budget. With K blocks of 64 or greater, the compiler reportedly spills 190-390 registers.

Spilling forces temporary values out of registers into slower storage, potentially destroying the performance gains from faster dot products.

The optimized tile avoids that problem and retains enough parallelism to feed the GPU.

## What this means for your R9700 decision

This is strong evidence that **V620 prefill performance can improve substantially through kernel optimization alone**\.

However, the benchmark concerns a particular quantized MoE implementation and model. It does not demonstrate that every model or Maya's GEMM implementation would improve by 28%.

It does suggest a sensible order of operations for your system\:

- Optimize RDNA 2 MoE kernels using native INT8/FP16 dot products.
- Profile dense-layer and attention kernels separately.
- Measure whether prefill becomes limited by computation, memory transfers, or cross-GPU synchronization.
- Only then evaluate whether adding a second R9700 addresses the remaining bottleneck.

One numerical point stands out: the optimized W4A8 kernel reportedly achieves 2.0-2.45× the performance of W4A16, yet overall prefill improves only 28%. That indicates substantial prefill time still occurs outside the accelerated MoE kernel.

**The next opportunity is identifying those remaining stages.** In particular, if dense projections or attention are still executing inefficiently on the V620s, additional optimized kernels could improve performance without buying another GPU.

## Prompt:
11/5/2026, 8:52:00 AM

Does Proect Maya and Llama have kernels like this

## Response:
11/5/2026, 8:52:04 AM · gpt-6-thinking

I’ll check the current Project Maya and llama.cpp implementations, specifically whether they use RDNA 2 dot-product instructions for quantized MoE prefill, rather than merely supporting generic GPU matrix multiplication.

> **Searched 5 websites**
>
> 
>
> Worked for 36s

**Yes. Both Project Maya and llama.cpp have quantized matrix-multiplication kernels that exploit packed integer dot products on AMD GPUs.** They are conceptually similar to the W4A8 kernel you found, although they differ in quantization format, fusion, scheduling, and implementation.

The more interesting finding is that **Project Maya already has an experimental INT8 MMQ path for DeepSeek prefill on AMD**, but disables it by default.

## Project Maya

Maya uses ggml-derived quantized matrix-multiplication (MMQ) kernels. Its current settings documentation exposes\:

`MAYA_DS4_EXPERT_MMQ=1`

This enables an existing Q8\_1/INT8 MMQ path for supported DeepSeek expert operations during prefill.

According to Maya's documentation\:

- It applies to supported quantized expert operations with at least 32 input rows.
- It quantizes prompt activations and the intermediate SwiGLU output to Q8\_1.
- Unsupported shapes retain the original execution path.
- It runs on HIP, making it potentially relevant to your V620s.
- It changes numerical behavior and can affect model outputs, which is why Maya leaves it disabled.

This is much closer to the W4A8 approach than the ordinary FP16 GEMM path we discussed earlier. \[[project-maya/docs/SETTINGS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md?utm_source=chatgpt.com)\]

I have not found published V620 benchmarks demonstrating its effect on DeepSeek V4.1 prefill. Nor have I verified that its particular quantization formats and kernels achieve the same efficiency as the Triton implementation you quoted.

## llama.cpp

llama.cpp also implements quantized matrix multiplication with packed dot products.

Its HIP backend shares much of the CUDA MMQ implementation, including code in\:

- [github.com](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh?utm_source=chatgpt.com)
- [github.com](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu?utm_source=chatgpt.com)

The vector-dot code contains HIP-specific integer unpacking and uses `ggml_cuda_dp4a` for quantized dot products. This is the same general approach as the W4A8 implementation: retain integer representations during arithmetic and apply quantization scales to accumulated results. \[[llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh?utm_source=chatgpt.com); [llama.cpp/ggml/src/ggml-cuda/mmq.cu at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu?utm_source=chatgpt.com)\]

However, llama.cpp does not necessarily select MMQ for every prefill workload. Its dispatch logic considers architecture, batch dimensions, expert count, and available matrix hardware.

You can force its quantized MMQ path at build time\:

```bash
cmake -B build -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1030 \
  -DGGML_CUDA_FORCE_MMQ=ON
```

This requests custom MMQ kernels rather than the BLAS alternative. It does **not** guarantee better performance; llama.cpp's documentation explicitly warns that forced MMQ can reduce large-batch throughput. \[[llama.cpp/docs/build.md at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md?plain=1&utm_source=chatgpt.com); [llama.cpp/ggml/src/ggml-hip/CMakeLists.txt at master · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hip/CMakeLists.txt?utm_source=chatgpt.com)\]

## Comparison for your V620s

| Capability | Project Maya | llama.cpp |
|--------|--------|--------|
| Quantized GPU GEMM | Yes | Yes |
| Packed INT8 dot-product implementation | Yes, through MMQ | Yes, through MMQ |
| AMD HIP implementation | Yes | Yes |
| RDNA 2 `gfx1030` support | Experimental | Supported HIP compilation target |
| Quantized MoE prefill | Yes | Yes |
| DeepSeek-specific Q8\_1 expert prefill | Experimental opt-in | Not the same Maya-specific path |
| Equivalent to the cited Triton W4A8 kernel | Not established | Not established |
| Verified 27-28% V620 speedup | No | No |

Maya also has a separate fused INT8 WMMA expert prefill implementation, enabled by default on certain RDNA 3 and Strix Halo devices. Its documentation explicitly says other AMD architectures retain the conventional MMQ implementation, so this particular fused kernel does not currently provide the same optimization on your V620s. \[[project-maya/docs/SETTINGS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md?utm_source=chatgpt.com)\]

## What I would do

For your DeepSeek configuration, the most useful experiment is to benchmark `MAYA_DS4_EXPERT_MMQ=1` against its default.

That directly tests whether Maya's existing quantized INT8 expert path provides an advantage on the V620s. Because it changes intermediate quantization, I would also compare output quality using teacher-forced log-likelihood or another fixed evaluation set.

I would then compare the two engines' kernels, particularly\:

- Whether their `gfx1030` binaries actually issue `v_dot4_i32_i8`\.
- Whether quantization and dequantization happen outside the inner multiplication loop.
- Whether their tile configurations spill VGPRs.
- Whether they fuse activation functions and intermediate operations.
- How much time the complete prefill spends in expert GEMMs versus attention, dense layers, and transfers.

**Conclusion\:** Maya and llama.cpp already implement much of the underlying technique. The question is whether their implementations exploit RDNA 2 as effectively as the specialized Triton W4A8 kernel. I found no controlled benchmark establishing that either does. Given the 27-28% improvement you quoted, optimizing the existing V620 kernels deserves serious consideration before purchasing your second R9700.

## Prompt:
11/5/2026, 8:53:07 AM

Well let's enable it!  How do I do that

## Response:
11/5/2026, 8:53:12 AM · gpt-6-thinking

> **Searched 1 website**
>
> 
>
> Worked for 16s

To enable Maya's experimental INT8 expert prefill kernel, set\:

```bash
MAYA_DS4_EXPERT_MMQ=1
```

Project Maya documents this setting for AMD HIP. It defaults to disabled and applies to supported DeepSeek IQ expert operations with at least 32 input rows. \[[project-maya/docs/SETTINGS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md?utm_source=chatgpt.com)\]

## Enable it in your Maya configuration

From your Project Maya directory, run\:

```bash
./maya.sh --setup --env MAYA_DS4_EXPERT_MMQ=1
```

This stores the setting in Maya's configuration so it persists across restarts. Restart your DeepSeek server afterward.

Alternatively, for a one-time test without modifying the saved configuration, launch Maya with the environment variable\:

```bash
MAYA_DS4_EXPERT_MMQ=1 ./maya.sh --serve
```

Use the same remaining launch arguments or configuration you normally use.

## Important: your Q8\_K\_XL model

You're running DeepSeek V4.1 Flash UD-Q8\_K\_XL. Maya's documentation specifically says the new MMQ path supports **IQ expert formats**, not every quantization format. \[[project-maya/docs/MODELS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/MODELS.md?utm_source=chatgpt.com)\]

I cannot confirm that it will activate for your Q8\_K\_XL pack. Setting the variable may therefore have no effect on that model.

You can still enable it safely as an experiment. It should fall back to the original path for unsupported formats.

To verify whether it actually activates, inspect startup logs and compare prefill throughput with the setting enabled and disabled using identical prompts. Do not interpret an unchanged result as proof that the kernel itself offers no benefit.

**Also, this is an approximate computation path.** It introduces Q8\_1 activation rounding, which can change outputs. Since you've preferred to avoid unvalidated quality optimizations, I would use it for an A/B benchmark before making it permanent.

To disable it again\:

```bash
./maya.sh --setup --env MAYA_DS4_EXPERT_MMQ=0
```

Source: [Project Maya settings documentation](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md?utm_source=chatgpt.com)\.

## Prompt:
11/5/2026, 8:55:51 AM

What has changed in maya since 1.1.49 that may improve my performance

## Response:
11/5/2026, 8:55:53 AM · gpt-6-thinking

I’ll review the releases and commits after v1.1.49, then separate changes that could improve your AMD HIP prefill or decode performance from unrelated features.

> **Searched 1 website**
>
> 
>
> Worked for 33s

As of **November 5, 2026, Project Maya v1.1.54 is the latest release**\. Since v1.1.49, several changes could improve your performance, particularly on DeepSeek V4.1 Flash.

For your configuration-large Q8\_K\_XL model, AMD GPUs, tensor parallelism, long context, and substantial system RAM-**v1.1.51 is especially relevant**\. The latest release also introduces significant AMD prefill optimizations, but some target different GPU architectures or model storage formats.

## Changes most relevant to your hardware

| Release | Improvement | Reported benefit | Relevance |
|--------|--------|--------|--------|
| **1.1.50** | Optional peer-to-peer all-reduce | Up to 29% faster prefill on tested systems | High, if supported by your AMD topology |
| **1.1.51** | Improved VRAM allocation across TP GPUs | Up to 10% faster decode | High |
| **1.1.51** | Larger prefill chunks when experts reside in RAM | Up to 12% faster long prefill | High |
| **1.1.52** | More parallel CPU readers for DeepSeek n-gram tables | Faster cold prompts | Moderate |
| **1.1.53** | Experimental flow prefill optimizations | 29-72% faster on 4× RTX 3090 | Potentially high; AMD unverified |
| **1.1.54** | Skip unused streamed experts during prefill | Up to 3.5× on tested AMD APU | Conditional; different hardware and quantization |

These percentages come from the release authors' measurements, not tests on your system. They are neither additive nor directly transferable. \[[Releases · mw00/project-maya](https://github.com/mw00/project-maya/releases?utm_source=chatgpt.com); [Release v1.1.53 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.53?utm_source=chatgpt.com); [Release v1.1.51 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.51?utm_source=chatgpt.com); [Release v1.1.54 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.54?utm_source=chatgpt.com)\]

### 1\. Better tensor-parallel VRAM utilization - v1.1.51

Previously, Maya divided expert placement more evenly across tensor-parallel GPUs.

Now it accounts for the VRAM remaining after allocating other resources, including the speculative drafter, head, and buffers.

The developer reports approximately 10% more resident experts and up to 10% faster decoding.

This should be particularly relevant to your DeepSeek configuration if your tensor-parallel GPUs have unequal memory commitments. It is enabled by default, and `DS4_TP_SHARES=off` restores the old behavior. \[[Release v1.1.51 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.51?utm_source=chatgpt.com)\]

### 2\. Faster long-context prefill - v1.1.51

Maya can temporarily borrow VRAM normally assigned to resident experts to process longer prompt chunks, then restore those experts from the page cache.

The reported effect is up to 12% faster prefill for prompts above approximately 10K tokens when most experts live in RAM. Shorter prompts can become slightly slower.

This directly targets the sort of long-context workload you have been benchmarking. However, your manually configured `--prefill-chunk 32768` could change the benefit, so it needs testing rather than assuming the default improvement applies. \[[Release v1.1.51 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.51?utm_source=chatgpt.com)\]

### 3\. Peer-to-peer all-reduce - v1.1.50

This is one of the first options I would benchmark.

```bash
DS4_AR_P2P=1
```

It replaces part of the usual synchronization and reduction mechanism with direct peer-to-peer transfers through ReBAR.

The release reports up to 29% faster prefill on tested systems, although some configurations regressed. v1.1.51 also corrected BAR-size detection, improving the chances that the feature activates on compatible hardware. \[[Release v1.1.50 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.50?utm_source=chatgpt.com); [Release v1.1.51 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.51?utm_source=chatgpt.com)\]

Given your previously measured GPU-to-GPU bandwidth of approximately 67 GB/s, this deserves investigation. However, that measurement alone does not establish which pairs have usable peer-to-peer access or whether Maya's implementation supports your HIP configuration.

### 4\. Experimental layer-split prefill - v1.1.53

This release introduces several independent opt-in optimizations\:

| Setting | Function | Output consistency |
|--------|--------|--------|
| `DS4_FLOW_COPY_DMA` | Transfers weights through pinned RAM using DMA | Same answers reported |
| `DS4_AI_RUN` | Adjusts indexer processing to prompt size | Same answers reported |
| `DS4_CHUNK_AUTO` | Adjusts prefill chunk size to prompt length | Slight changes reported |
| `DS4_YOCO_FLOW` | Changes the YOCO skip path | Slight changes reported |
| `DS4_LENT_RAM_GB` | Keeps borrowed experts in pinned RAM for long conversations | Intended to reduce rereads |

The developer reports 29-72% faster prefill on four RTX 3090s for the combined experimental changes. **Those measurements are from a layer-split NVIDIA configuration, not an AMD tensor-parallel configuration.** I would not assume the optimizations activate or help on your machine. \[[Release v1.1.53 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.53?utm_source=chatgpt.com)\]

### 5\. New AMD prefill optimization - v1.1.54

This release is particularly interesting, but its applicability needs care.

For models whose experts are streamed from model files, Maya now fetches only the experts needed by the current prompt chunk rather than fetching all of them.

On a Gorgon Halo APU running a DeepSeek EXL3 2.9-bpw model, the developer reports\:

\~2K-token prompt

# 3.5×

Prefill throughput

\~9K-token prompt

# 2.78×

Prefill throughput

The change reportedly preserves the answers. However, your Q8\_K\_XL deployment uses a different quantization and memory arrangement, so I cannot establish that it will benefit similarly.

The same release enables RDNA 3/3.5 matrix-core indexer acceleration by default. **That particular optimization does not target your RDNA 2 V620s or RDNA 4 R9700.** \[[Release v1.1.54 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.54?utm_source=chatgpt.com)\]

## What I would test on your DeepSeek server

Your earlier four-V620 benchmarks provide a baseline of approximately 460-480 tokens/s prefill for long prompts.

I would proceed in this order\:

1. **Update to v1.1.54**, preserving your current model configuration and inference settings.
2. Benchmark the new default behavior at 8K, 32K, and 64K tokens.
3. Compare `DS4_TP_SHARES=off` with the new default.
4. Test `DS4_AR_P2P=1` independently and verify that the peer path actually engages.
5. Test the v1.1.53 flow optimizations individually, but only if your deployment uses their supported layer-split path.
6. Separately test `MAYA_DS4_EXPERT_MMQ=1`, as discussed earlier, if its supported formats match your quantization.

For each experiment, retain the same sampling parameters, context size, quantization, expert distribution, and cache conditions. Record both prefill and decode. For changes that alter numerical behavior, run a quality comparison before retaining them.

One important warning: **do not enable Turbo for these comparisons.** The v1.1.50 Turbo feature can enable greedy decoding and expert skipping, which conflicts with your requirement to preserve inference quality. \[[Release v1.1.50 · mw00/project-maya](https://github.com/mw00/project-maya/releases/tag/v1.1.50?utm_source=chatgpt.com)\]

The most promising developments for your particular machine are the v1.1.51 expert-placement improvements and the potential reduction in collective communication overhead. I would evaluate those before spending money on another GPU.

Sources: [v1.1.54 release](https://github.com/mw00/project-maya/releases/tag/v1.1.54?utm_source=chatgpt.com) · [v1.1.53 release](https://github.com/mw00/project-maya/releases/tag/v1.1.53?utm_source=chatgpt.com) · [v1.1.51 release](https://github.com/mw00/project-maya/releases/tag/v1.1.51?utm_source=chatgpt.com) · [v1.1.50 release](https://github.com/mw00/project-maya/releases/tag/v1.1.50?utm_source=chatgpt.com)

## Prompt:
11/5/2026, 8:57:13 AM

So to be clear the Q8 and Q4 versionsof deepseek are almost identical

## Response:
11/5/2026, 8:57:22 AM · gpt-6-thinking

> **Searched 3 websites**
>
> 
>
> Worked for 9s

**Yes, substantially. For DeepSeek V4 Flash, the UD-Q8\_K\_XL and UD-Q4\_K\_XL versions are much closer than their names suggest.** There is an important distinction between the quantizations, though.

Unsloth explains that both versions preserve the model's native MXFP4 expert weights. The primary difference lies in the non-MoE layers\:

| Property | UD-Q8\_K\_XL | UD-Q4\_K\_XL |
|--------|--------|--------|
| MoE expert weights | Native MXFP4 | Native MXFP4 |
| Attention and other dense weights | BF16 | Q8\_0 |
| Approximate model size (V4 Flash 0731) | 162 GB | 155 GB |
| Fidelity | Closest to original | Slight additional quantization error |
| Memory savings with Q4 | - | \~7 GB (4.3%) |

Unsloth explicitly describes the two as virtually equivalent, while reporting approximately **96% top-1 agreement** for Q4 against its reference. That means differences do occur; it does not establish a 4% reduction in task accuracy. \[[unsloth/DeepSeek-V4-Flash-0731-GGUF · KLD Benchmarks \+ why Q8\_K\_XL vs MXFP4 naming](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/discussions/11?utm_source=chatgpt.com); [README.md](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/main/README.md?utm_source=chatgpt.com)\]

The practical implication is that Q4 may save some memory without significantly reducing model quality, but the saving is relatively small because the experts already use four-bit floating-point weights.

**One qualification for your setup\:** You are running *DeepSeek V4.1 Flash 1017*, not the earlier V4 Flash 0731 used for those figures. I have not verified that your particular V4.1 Q8\_K\_XL and Q4\_K\_XL packs use exactly the same quantization recipe.

If they do, I would expect the quality difference to be small. I would not claim that they produce almost identical outputs without measuring agreement and log-likelihood on the actual V4.1 packs.

For your machine, the more consequential question may be whether Q4 saves enough VRAM to hold additional experts and improve throughput. If it saves only a few gigabytes, the performance improvement could be modest.

## Prompt:
11/5/2026, 8:58:01 AM

Right.  So that's why I run v8

## Response:
11/5/2026, 8:58:04 AM · gpt-6-thinking

Exactly. You run **UD-Q8\_K\_XL** because it preserves the higher-precision non-expert weights while retaining the same native MXFP4 expert weights as UD-Q4\_K\_XL.

If the size difference is only a few gigabytes, Q8 is a reasonable choice for maximizing fidelity. You sacrifice relatively little memory capacity, although Q8 could still incur additional bandwidth or computational costs during prefill.

The remaining question is whether Q4's smaller dense weights offer a measurable performance advantage. Quality alone provides little reason to switch, assuming the quantization recipes are as described.

## Prompt:
11/5/2026, 8:58:42 AM

1. Update to v1.1.54, preserving your current model configuration and inference settings.
2. Benchmark the new default behavior at 8K, 32K, and 64K tokens.
3. Compare `DS4_TP_SHARES=off` with the new default.
4. Test `DS4_AR_P2P=1` independently and verify that the peer path actually engages.
5. Test the v1.1.53 flow optimizations individually, but only if your deployment uses their supported layer-split path.
6. Separately test `MAYA_DS4_EXPERT_MMQ=1`, as discussed earlier, if its supported formats match your quantization.

Fine.  Write me a script to do all of this while I go out today

## Response:
11/5/2026, 8:58:48 AM · gpt-6-thinking

Maya’s built-in `--bench` only tests 2K and 8K prefill, so it will not cover the 32K/64K measurements you want. I found its actual engine benchmark protocol and will use that for the longer prompts, with separate configurations and logs for each experiment.

> **Preparing the benchmark script**
>
> I’m drafting a self-contained shell script that avoids touching the running service, uses an isolated checkout, validates available benchmark options, captures logs, and preserves existing configuration. I’m keeping assumptions explicit where the project’s exact command-line interface isn’t confirmed.
>
> **Prepared the benchmark script**
>
> 
>
> **Searched 2 websites**
>
> 
>
> **Drafting the benchmark runner**
>
> I’m shaping an isolated script that updates a separate checkout, preserves production settings, reuses a cached corpus, runs the requested variants and flows, records timing and quality metrics, and clearly marks unsupported tests instead of guessing.
>
> **Drafted the benchmark runner**
>
> 
>
> **Searched 1 website**
>
> 
>
> **Creating Maya benchmark runner script**
>
>
> **Created Maya benchmark runner script**
>
> 
>
> Worked for 2m 15s

I created an unattended benchmark script based on Project Maya v1.1.54's actual benchmark implementation, including its engine protocol for measuring prefill beyond the built-in 8K limit. \[[GitHub](https://raw.githubusercontent.com/mw00/project-maya/v1.1.54/maya.py?utm_source=chatgpt.com)\]

[Download the Maya v1.1.54 benchmark script](sandbox:/mnt/data/maya_benchmark_1154.py)

Run it on your Maya machine with the server stopped\:

```bash
python3 maya_benchmark_1154.py \
  /path/to/project-maya \
  /path/to/project-maya/maya-ds4-hip.json
```

Replace the paths with your installation directory and actual DeepSeek configuration file.

The script will fetch and check out v1.1.54, invoke Maya's engine rebuild mechanism, and run the following matrix\:

| Test | 8K | 32K | 64K | Decode |
|--------|--------|--------|--------|--------|
| v1.1.54 defaults | 2 runs | 2 runs | 2 runs | Yes |
| `DS4_TP_SHARES=off` | 2 runs | 2 runs | 2 runs | Yes |
| `DS4_AR_P2P=1` | 2 runs | 2 runs | 2 runs | Yes |

It preserves your original JSON configuration, GPU selection, quantization, context settings, and inference parameters. It writes separate engine logs, a CSV file, and a performance summary. It refuses to run if the Git checkout has uncommitted changes or detects an active Maya engine.

**Two tests remain deliberately excluded\:** the v1.1.53 flow optimizations, because their applicability to your tensor-parallel arrangement requires verification, and `MAYA_DS4_EXPERT_MMQ=1`, because its documentation specifies supported IQ expert formats rather than your Q8\_K\_XL pack. The script records those omissions.

The P2P test enables the setting and retains the engine log, but does not yet automatically certify that the P2P path engaged.

I verified the relevant Maya source code, but **I have not executed this script on your hardware**\. An engine build failure or an unsupported configuration will stop it and preserve the available logs rather than silently continue with invalid measurements.

The results will appear in a timestamped `maya-bench-1154-*` directory beside the Maya repository.

## Prompt:
11/5/2026, 11:29:21 AM

[Attachment omitted: Screenshot 2026-11-05 at 11.29.10.png; signed download URL removed.]

## Response:
11/5/2026, 11:29:24 AM · gpt-6-thinking

This looks like a Project Maya inference timing report for a **48,794-token input**, with extensive prompt-cache reuse.

## Performance

| Metric | Value |
|--------|--------|
| Total prompt tokens | 48,794 |
| Cached prompt tokens | 47,087 |
| Newly processed tokens | 1,707 |
| Prompt processing speed | **170.4 tokens/s** |
| Generated tokens | 4,602 |
| Generation speed | **31.2 tokens/s** |
| Draft acceptance rate | 71.9% |
| Total reported tokens | 27,517 |

### Prefill

The important figures are\:

- `cache_n: 47087`
- `prompt_n: 1707`
- `prompt_ms: 10020.1`
- `prompt_per_second: 170.4`

Only 1,707 tokens required actual prefill computation. The engine reused 47,087 tokens from the cache.

Your measured prefill speed is therefore\:

$$
\frac{1707}{10.0201} \approx 170.4\text{ tokens/s}
$$

**That is considerably slower than your earlier 460–480 tokens/s long-context prefill measurements.**

However, this is not an equivalent workload. The model processed a small uncached suffix with an existing context of roughly 47K tokens, rather than performing a fresh 48K-token prefill. Prefill throughput depends on batch size, attention context, caching behavior, and execution overhead.

### Decode

Your decode speed is **31.2 tokens/s**, with speculative decoding enabled.

The engine reports\:

- 3,552 drafted tokens.
- 2,556 accepted draft tokens.
- Approximately 71.9% draft acceptance.

The reported `predicted_per_second` measures the end-to-end generation rate, including speculative decoding overhead.

## What concerns me

The main observation is that **your 170.4 tokens/s measurement does not indicate a regression by itself**\.

At only 1,707 uncached input tokens, the engine may not achieve the same GPU utilization as a 32K or 64K fresh prefill. A short suffix appended to a long context is a very different benchmark.

There is also an accounting discrepancy: `prompt_tokens` plus `completion_tokens` equals 53,396, not the reported `total_tokens` of 27,517. The fields may represent different accounting scopes, but I cannot determine their exact meanings from this screenshot.

**For your v1.1.54 benchmark, I would rely on fresh 8K/32K/64K prefill measurements with caching disabled or explicitly controlled.** This screenshot is useful for measuring interactive performance and speculative decoding, but it should not serve as your baseline for comparing prefill optimizations.

## Prompt:
11/5/2026, 11:31:45 AM

[Attachment omitted: Screenshot 2026-11-05 at 11.31.35.png; signed download URL removed.]

## Response:
11/5/2026, 11:31:49 AM · gpt-6-thinking

This screenshot is useful. It confirms that **Project Maya v1.1.54 is running your DeepSeek V4.1 UD-Q8\_K\_XL model across all four V620s**, with substantial expert residency in VRAM.

## Current configuration

| Metric | Observed |
|--------|--------|
| Model | DeepSeek V4.1 UD-Q8\_K\_XL |
| Context capacity | 1,024K tokens |
| GPUs | 4× Radeon Pro V620 |
| GPU VRAM allocated | 116.6 / 120 GB |
| Experts in VRAM | 3,102 |
| Experts in pinned RAM | 12,258 |
| VRAM hit rate | 82.3% |
| System RAM | 250.4 / 2,004 GB |
| Decode throughput | 31.2 tokens/s |
| Last-request prefill | 170 tokens/s |

A few things stand out.

## 1\. You are already getting a high VRAM hit rate

The dashboard shows 82.3% VRAM hits despite only 3,102 of 15,360 experts residing in GPU memory.

That suggests your expert caching and placement are effective for this workload. The hit rate is likely influenced by repeated conversation context and the model's expert-routing patterns.

Your reported CPU expert processing cost is 45.16 ms/token, and promotion overhead is only 0.18 ms/token. This suggests CPU-side expert execution remains a significant component of your workload.

However, I would not interpret the 45.16 ms/token as an additive contribution to the 32.1 ms/token decode latency; these counters may aggregate overlapping work.

## 2\. Your recent prefill results vary considerably

The dashboard shows\:

| Prompt tokens | Prefill throughput |
|--------|--------|
| 10,205 | 174 tok/s |
| 15,437 | 239 tok/s |
| 29,194 | 278 tok/s |
| 32,645 | 388 tok/s |
| 33,589 | 423 tok/s |
| 35,472 | 176 tok/s |
| 47,087 | 370 tok/s |
| 48,794 | 170 tok/s |

The wide spread does not necessarily indicate inconsistent GPU performance.

These are conversational requests with different amounts of cache reuse. The prefill metric reflects the uncached portion, while the dashboard's prompt column reports the total prompt length.

Your latest request processed just 1,707 new tokens against approximately 47K cached tokens, explaining why its measured throughput was lower.

For comparing v1.1.49 with v1.1.54, controlled fresh-prefill benchmarks remain necessary.

## 3\. P2P is enabled, but engagement is unverified

Your tmux status line shows\:

```bash
DS4_AR_P2P=1 ./maya.sh
```

That establishes that you requested the peer-to-peer all-reduce optimization.

It does **not** establish that Maya activated the P2P path. We need the engine initialization logs to check its actual selection.

You can inspect them with\:

```bash
cd /opt/project-maya

grep -Ei 'p2p|all.?reduce|peer|rebar|bar' \
  /path/to/your/maya/logfile
```

If you use the dashboard, press `Tab` to switch to its log view and look for messages about P2P or all-reduce initialization.

## 4\. A potential performance opportunity: VRAM allocation

Each V620 reports approximately 29.0-29.2 GB used out of 30 GB available to Maya.

That leaves relatively little free VRAM, which is expected with an expert residency cache, but also means that any additional prefill workspace may require evicting experts.

The v1.1.51 changes to expert placement and prefill chunk borrowing could affect this tradeoff.

## My assessment

Your current decode speed of approximately 31 tokens/s is reasonable relative to your earlier results, but the screenshot alone does not establish an improvement from v1.1.49.

More importantly, **you appear to have substantial CPU-side expert activity despite the 82% VRAM hit rate**\. That is where I would focus the next investigation.

I would first confirm that `DS4_AR_P2P=1` actually engages, then compare the default v1.1.54 configuration against `DS4_TP_SHARES=off` using fresh, uncached prompts.

One detail I would also investigate is why the dashboard reports 15,360 total experts. That count may include the model's full expert inventory rather than only the experts used for each token, but confirming how Maya counts residency would help interpret the cache statistics.

## Prompt:
11/5/2026, 11:32:42 AM

3,102 of 15,360

How can that be?  I'm usin 120GB of VRAM as a cache.  The model isn't that large

## Response:
11/5/2026, 11:32:46 AM · gpt-6-thinking

> **Searched 2 websites**
>
> 
>
> Worked for 12s

The numbers actually make sense. The key is that **15,360 counts individual experts across all MoE layers, not the number of experts within a single layer.**

DeepSeek V4.1 Flash has\:

- **40 MoE layers**
- **384 routed experts per layer**
- **15,360 routed experts total**
- **6 routed experts activated per token per layer**

Therefore\:

$$
40 \times 384 = 15{,}360
$$

A token activates only $40 \times 6 = 240$ routed experts, but the model needs all 15,360 available because routing varies by token. \[[vllm/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml at main · vllm-project/vllm](https://github.com/vllm-project/vllm/blob/main/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml?utm_source=chatgpt.com); [We tripled DeepSeek's speed by giving it FEWER CPU threads: 15 → 41 tok/s on one GPU (and --calibrate skipped the test) · Issue \#433 · mw00/project-maya](https://github.com/mw00/project-maya/issues/433?utm_source=chatgpt.com)\]

## Where your model lives

Your screenshot shows\:

Expert placement across 15,360 routed experts

VRAM

3,102 (20.2%)

Pinned system RAM

12,258 (79.8%)

The crucial distinction is that your 116.6 GB VRAM allocation does **not** consist entirely of cached experts. Maya also stores dense model weights, attention weights, KV cache, activation buffers, draft-model data, and other working memory on the GPUs.

The Q8\_K\_XL version of DeepSeek V4.1 needs approximately **306-310 GB of VRAM plus system RAM** for its model weights and associated runtime requirements, excluding the large n-gram tables that Maya can access separately. \[[project-maya/docs/MODELS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/MODELS.md?utm_source=chatgpt.com); [project-maya/maya.py at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/maya.py?utm_source=chatgpt.com)\]

Your system has enough memory to keep all the routed experts in VRAM or pinned RAM, with none restricted to SSD. That's consistent with the dashboard.

## What about your 82.3% VRAM hit rate?

This is actually encouraging. Although only 20.2% of routed experts are resident in VRAM, 82.3% of expert accesses hit VRAM.

Maya preferentially caches frequently activated experts, rather than distributing VRAM equally among all experts. The expert profile and runtime activity guide that selection. \[[project-maya/docs/SETTINGS.md at main · mw00/project-maya](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md?utm_source=chatgpt.com)\]

So the dashboard is not saying you have only 20% of your model loaded. **Your entire model is accessible from memory, while the 20% most valuable routed experts, by Maya's placement policy, occupy GPU cache slots.**

The next useful question is whether your current 82% hit rate could improve by reallocating GPU memory away from less frequently used buffers or by changing expert placement. But 3,102 of 15,360 is not itself evidence of a problem.

## Prompt:
11/5/2026, 11:33:58 AM

Right, so I have enough VRAM for 1/3 of experts in RAM yet only 20% are in RAM

## Response:
11/5/2026, 11:34:02 AM · gpt-6-thinking

> **Searched 1 website**
>
> 
>
> Worked for 23s

Yes. Your intuition is correct that **3,102 experts in 120 GB of VRAM seems unexpectedly low**, particularly given how much memory your system has available.

But we need to distinguish total VRAM allocation from memory actually available for experts.

Your dashboard reports\:

- 15,360 routed experts total.
- 3,102 experts in VRAM (20.2%).
- 12,258 experts in system RAM (79.8%).
- 116.6 GB of VRAM allocated.

The 116.6 GB includes expert weights, dense model weights, KV cache, and other runtime buffers. It is not a 116.6 GB expert-only cache.

I found the relevant model sizes, and they make your observation more significant.

For DeepSeek V4.1 Flash UD-Q8\_K\_XL, the routed experts occupy approximately **288.8 GB**, or 18.8 MB per expert. The dense weights occupy another 16.8 GB, and the roughly 203 GB Engram tables reside separately. \[[Strata-T8/docs/DEEPSEEK41.md at main · T8mars/Strata-T8](https://github.com/T8mars/Strata-T8/blob/main/docs/DEEPSEEK41.md?utm_source=chatgpt.com)\]

That gives us\:

| Allocation | Approximate size |
|--------|--------|
| All 15,360 experts | 288.8 GB |
| Your 3,102 VRAM-resident experts | 58.3 GB |
| Dense model weights | 16.8 GB |
| Reported GPU memory usage | 116.6 GB |

**There is an apparent discrepancy of roughly 41.5 GB**, before accounting for KV cache, speculative decoding buffers, tensor-parallel replication, and other allocations.

Your 120 GB GPU allocation should theoretically accommodate around 5,000 experts after the dense weights and reasonable overhead, although the precise count depends on the execution layout. I cannot yet determine why Maya reports only 3,102.

Tensor-parallel replication or substantial reserved workspace could explain it. I would not assume either without checking the allocation log.

Run\:

```bash
cd /opt/project-maya

grep -Ei 'layers.*weights|experts.*VRAM|tensor parallel|cache|reserve|free' \
  maya*.log
```

Or inspect Maya's startup log through the dashboard. Look for lines like\:

```text
ds4: HIP0 layers ... weights ... experts ... caches ... free
```

Those entries should tell us how much of each GPU's memory actually holds expert weights.

**Your concern is justified.** The dashboard's 20.2% figure is not enough to explain where the other GPU memory went. The per-GPU startup allocations are necessary to distinguish legitimate overhead from inefficient expert placement.

## Prompt:
11/5/2026, 11:44:35 AM

[Attachment: Pasted markdown(5).md]

## Response:
11/5/2026, 11:44:37 AM · gpt-6-thinking

> **Inspecting Log Runs and Startup Details**
>
>
> **Inspecting Markdown Sections 900-945**
>
>
> **Inspected log runs, startup details, and Markdown sections**
>
> 
>
> Worked for 21s

The logs reveal the problem. **Maya was leaving substantial VRAM unused on GPU 1 in an earlier configuration, and later startups show that its allocation changed.**

Your grep output contains several separate Maya initializations, so we must distinguish the old allocations from the newer ones.

## Earlier allocation

| GPU | Expert weights | Reported free VRAM | Resident expert entries |
|--------|--------|--------|--------|
| V620 0 | 13.64 GB | 3.0 GB | 1,047 |
| V620 1 | 6.42 GB | **10.4 GB** | 663 |
| V620 2 | 14.93 GB | 3.0 GB | 1,568 |
| V620 3 | 14.93 GB | 3.1 GB | 1,568 |

GPU 1 had approximately 7.4 GB more free VRAM than the other cards. That was a genuine imbalance, not merely a misleading dashboard percentage.

The allocation uses tensor parallelism across two-device groups and splits layers between those groups. The expert-entry counts therefore cannot simply be added to obtain the number of distinct model experts in VRAM.

## Newer allocation: the imbalance is corrected

Later in the same log, Maya reports\:

| GPU | Expert weights | Free VRAM | Expert entries |
|--------|--------|--------|--------|
| V620 0 | 16.17 GB | 1.0 GB | 1,590 |
| V620 1 | 16.02 GB | 1.0 GB | 1,581 |
| V620 2 | 17.08 GB | 1.0 GB | 1,797 |
| V620 3 | 17.08 GB | 1.1 GB | 1,797 |
| **Total** | **66.35 GB** | **\~4.1 GB** | **6,765 entries** |

This shows a much better-balanced allocation. GPU 1 no longer has 10 GB sitting unused.

The logs also explain why your expert-residency estimate looked unexpectedly low: **Maya allocates only about 66 GB of the 120 GB VRAM budget to expert weights in this configuration.**

The remaining VRAM includes dense weights, KV-cache capacity, intermediate buffers, runtime allocations, and other memory commitments. The log's explicit weight/cache/free categories do not account for every allocation, so I cannot break down the full remainder precisely.

The 6,765 expert entries are per-device counts under tensor parallelism. They are not 6,765 distinct experts. This also means we should not assume the dashboard's 3,102 count measures the same thing.

## What I would change

Your later configuration reserves only about 1 GB per GPU, compared with the earlier 3 GB target. I would **keep the balanced allocation rather than reduce the reserve further** until we establish peak memory requirements for long-context prefill and decoding.

The important result is that the logs demonstrate an improvement in VRAM allocation. They do not indicate an obvious unused 10–20 GB remaining in the latest configuration.

One caveat: this grep output contains historical restarts, not timestamps that definitively establish the currently running configuration. The later entries are consistent with an improved allocation, but we should confirm that the active v1.1.54 process uses those settings.

**Bottom line\:** Your original concern exposed a real allocation imbalance, and the newer initialization appears to have corrected it. The next useful measurement is whether that redistribution improves your expert-cache hit rate and decode throughput compared with the previous setup.

## Prompt:
11/5/2026, 11:45:49 AM

Put everything including environment vars to test in here so I can go take a nap

python3 maya_benchmark_1154.py \
  /path/to/project-maya \
  /path/to/project-maya/maya-ds4-hip.json

and the paths are as shown -> /opt/ and whatever the ds4 default config was

## Response:
11/5/2026, 11:45:57 AM · gpt-6-thinking

> **Inspecting Benchmark Script Contents**
>
>
> **Inspected benchmark script contents**
>
> 
>
> **Clarifying the command**
>
> I’m treating this as a request for one runnable command using `/opt`, the DS4 default configuration, and environment variables. The current script still tests only three variants, so other cases remain skipped.
>
> **Clarified the command**
>
> 
>
> **Searched 1 website**
>
> 
>
> Worked for 21s

Use this command on the Maya server. The correct default HIP configuration path is `/opt/project-maya/maya-ds4-hip.json`\.

```bash
cd /opt/project-maya

env \
  -u DS4_AR_P2P \
  -u DS4_TP_SHARES \
  -u MAYA_DS4_EXPERT_MMQ \
  -u DS4_FLOW_COPY_DMA \
  -u DS4_AI_RUN \
  -u DS4_CHUNK_AUTO \
  -u DS4_YOCO_FLOW \
  python3 maya_benchmark_1154.py \
    /opt/project-maya \
    /opt/project-maya/maya-ds4-hip.json \
  2>&1 | tee /opt/maya-benchmark-run.log
```

The `-u` options clear inherited overrides so the baseline measurements remain valid.

**Important\:** The current script automatically tests v1.1.54 defaults, `DS4_TP_SHARES=off`, and `DS4_AR_P2P=1`, at 8K, 32K, and 64K tokens, with two repetitions each.

It does **not yet test** the flow flags or `MAYA_DS4_EXPERT_MMQ=1`\. Those require additional script changes, and I would not claim that this command tests them. Your Q8\_K\_XL expert format may not support the experimental MMQ path.

The script also requires Maya to be stopped before execution. It refuses to kill your running server automatically.

If the script is on your Mac rather than in `/opt/project-maya`, copy it there first. Once installed, the command above runs the supported benchmark matrix and writes its results under `/opt/maya-bench-1154-*`\.

## Prompt:
11/5/2026, 11:47:09 AM

It does not yet test the flow flags or `MAYA_DS4_EXPERT_MMQ=1`. Those require additional script changes, and I would not claim that this command tests them. Your Q8_K_XL expert format may not support the experimental MMQ path.

THEN do that for me please

## Response:
11/5/2026, 11:47:13 AM · gpt-6-thinking

I’ll revise the existing script to include the flow-option experiments and the MMQ test, with capability checks, isolated logs, and a consolidated report. Unsupported options will be marked as skipped rather than counted as successful benchmarks.

The release documentation confirms that your TP=2 layout has two layer-split GPU groups, so the flow options are applicable. It also documents a separate HIP-specific peer all-reduce setting, `MAYA_DS4_HIP_PEER_AR`, which is more directly relevant to your V620s than the older generic P2P flag. I’m including both as independent tests.

> **Inspecting benchmark script and verifying flags**
>
>
> **Inspected benchmark script and verified flags**
>
> 
>
> **Reviewing implementation details**
>
> I’m checking configuration formatting, benchmark compatibility, environment isolation, and eligibility for layer-splitting. I’m also verifying GPU-group parsing, opt-in combinations, repository state, and the requested memory test.
>
> **Reviewed implementation details**
>
> 
>
> **Searched 2 websites**
>
> 
>
> **Patching Maya benchmark test matrix**
>
>
> **Patched the Maya benchmark test matrix**
>
> 
>
> **Validating the script**
>
> I’m checking the base script’s assumptions, especially configuration paths, GPU argument parsing, inherited tuning, and eligibility detection. I’ll preserve existing behavior while tightening syntax and avoiding false claims about unsupported cases.
>
> **Validated the script**
>
> 
>
> **Searched 1 website**
>
> 
>
> **Fixing GPU Eligibility Detection and Git Safety**
>
>
> **Fixed GPU detection, Git safety, and validated Maya benchmark compatibility**
>
> 
>
> Worked for 1m 44s

I updated the benchmark script to include the previously omitted experiments.

**[Download the revised maya_benchmark_1154.py](sandbox:/mnt/data/maya_benchmark_1154.py)**

The script now runs these configurations independently\:

| Experiment | Environment variable |
|--------|--------|
| v1.1.54 baseline | Defaults |
| Original TP expert partition | `DS4_TP_SHARES=off` |
| Generic P2P all-reduce | `DS4_AR_P2P=1` |
| HIP peer all-reduce | `MAYA_DS4_HIP_PEER_AR=auto` |
| Pinned-memory DMA | `DS4_FLOW_COPY_DMA=1` |
| Indexer optimization | `DS4_AI_RUN=1` |
| Automatic prefill chunking | `DS4_CHUNK_AUTO=1` |
| YOCO flow optimization | `DS4_YOCO_FLOW=1` |
| Combined flow optimizations | All four flow options |
| Quantized expert MMQ | `MAYA_DS4_EXPERT_MMQ=1` |
| Borrowed-expert RAM cache | `DS4_LENT_RAM_GB=8`, conditional |

Each experiment measures 8K, 32K, and 64K prefill with two repetitions, plus decode throughput.

The script also saves logs, records failed experiments, and generates `results.csv` and `summary.txt`\. It attempts the MMQ experiment even though your Q8\_K\_XL format may not activate the kernel. The flow tests run only when the configuration identifies multiple GPU groups. \[[GitHub](https://raw.githubusercontent.com/mw00/project-maya/v1.1.54/docs/SETTINGS.md?utm_source=chatgpt.com); [GitHub](https://raw.githubusercontent.com/mw00/project-maya/v1.1.54/docs/SETTINGS.md?utm_source=chatgpt.com); [GitHub](https://raw.githubusercontent.com/mw00/project-maya/v1.1.54/docs/SETTINGS.md?utm_source=chatgpt.com)\]

## Run it

Copy the updated script to `/opt/project-maya/`, stop the currently running Maya server, and execute\:

```bash
cd /opt/project-maya

python3 maya_benchmark_1154.py \
  /opt/project-maya \
  /opt/project-maya/maya-ds4-hip.json \
  2>&1 | tee /opt/maya-benchmark-run.log
```

The script preserves the original JSON configuration and inference parameters. It checks out v1.1.54 if necessary and builds the HIP engine.

**Important limitations\:** I syntax-checked the revised script, but I could not execute it against your Maya installation. The P2P report uses initialization logs rather than hardware counters, and the approximate-computation experiments do not establish numerical equivalence. Do not adopt those settings permanently until their quality has been checked.

The script leaves the production Maya server stopped after benchmarking; it does not restart it automatically.

## Prompt:
11/5/2026, 11:51:11 AM

The file is messed up and not loading

## Response:
11/5/2026, 11:51:15 AM · gpt-6-thinking

I’ll inspect the generated Python file for truncation or syntax errors, repair it, and verify that the replacement compiles before sending it.

> **Inspecting and validating benchmark file**
>
>
> **Inspected and validated benchmark files and script structure**
>
> 
>
> **Searched 1 website**
>
> 
>
> **Downloading and validating GitHub API script**
>
>
> **Downloaded and validated GitHub API script**
>
> 
>
> **Fixing the package**
>
> I’m spotting a likely configuration mismatch: the script checks `gpus` while the configuration may use `gpu`, so execution can be skipped incorrectly. I’m also checking imports and packaging, then I’ll provide a clean, re-exported version.
>
> **Searched 1 website**
>
> 
>
> **Fixing Maya Benchmark Configuration Issues**
>
> 
>
> Worked for 47s

I inspected the previous

I inspected the previous script. It compiles, but I found several problems, including incorrect GPU-count detection that could cause the flow experiments to be skipped.

I corrected those issues and repackaged the script.

**[Download the corrected script (ZIP)](sandbox:/mnt/data/maya_benchmark_1154_fixed.zip)**

Alternatively: [Python file](sandbox:/mnt/data/maya_benchmark_1154_fixed.py)

Extract the ZIP, copy the Python file to `/opt/project-maya/`, and run\:

```bash
cd /opt/project-maya

python3 maya_benchmark_1154_fixed.py \
  /opt/project-maya \
  /opt/project-maya/maya-ds4-hip.json \
  2>&1 | tee /opt/maya-benchmark-run.log
```

The script includes the baseline, TP allocation, P2P, HIP peer all-reduce, flow optimizations, and experimental MMQ configurations.

**Validation\:** I verified Python syntax and checked the benchmark protocol against Maya v1.1.54's source. I could not execute the complete GPU benchmarks on your hardware.

The script requires the production Maya server to be stopped before running.

## Prompt:
11/5/2026, 2:21:47 PM

❯ cd /opt/project-maya

python3 maya_benchmark_1154_fixed.py \

  /opt/project-maya \

  /opt/project-maya/maya-ds4-production.json \

  2>&1 | tee /opt/maya-benchmark-run.log

## Response:
11/5/2026, 2:21:50 PM · gpt-6-thinking

Use this command. The configuration path is now `/opt/project-maya/maya-ds4-production.json`\.

```bash
cd /opt/project-maya

python3 maya_benchmark_1154_fixed.py \
  /opt/project-maya \
  /opt/project-maya/maya-ds4-production.json \
  2>&1 | tee /opt/maya-benchmark-run.log
```

**Important\:** Each backslash must be the final character on its line, with no trailing spaces. Your pasted command contains blank lines that could cause the shell to execute it incorrectly.

The script should run the benchmark matrix and save its output to `/opt/maya-benchmark-run.log`\.

If it fails, send me the error output.

## Prompt:
11/5/2026, 2:22:07 PM

FileExistsError: [Errno 17] File exists: '/opt/maya-bench-1154-20261105-142159'

❯ cd /opt/project-maya

python3 maya_benchmark_1154_fixed.py \

  /opt/project-maya \

  /opt/project-maya/maya-ds4-production.json \

  2>&1 | tee /opt/maya-benchmark-run.log

Results: /opt/maya-bench-1154-20261105-142201

Traceback (most recent call last):

  File "/opt/project-maya/maya_benchmark_1154_fixed.py", line 222, in <module>

    try: main()

         ~~~~^^

  File "/opt/project-maya/maya_benchmark_1154_fixed.py", line 52, in main

    import strata_tokenizer as ST

  File "/opt/project-maya/tools/strata_tokenizer.py", line 22, in <module>

    import regex

ModuleNotFoundError: No module named 'regex'

## Response:
11/5/2026, 2:22:12 PM · gpt-6-thinking

> **Inspecting Benchmark Script Errors**
>
>
> **Inspected benchmark script errors**
>
> 
>
> Worked for 17s

The script failed before running any benchmarks. The immediate cause is a missing Python dependency\:

```text
ModuleNotFoundError: No module named 'regex'
```

The earlier `FileExistsError` is a separate issue: the script names result directories with timestamps precise only to the second.

## Fix the dependency

Run this on your Maya server\:

```bash
cd /opt/project-maya

python3 -m venv --system-site-packages .benchmark-venv
.benchmark-venv/bin/python -m pip install regex
```

Then run the benchmark using that Python environment\:

```bash
cd /opt/project-maya

.benchmark-venv/bin/python maya_benchmark_1154_fixed.py \
  /opt/project-maya \
  /opt/project-maya/maya-ds4-production.json \
  2>&1 | tee /opt/maya-benchmark-run.log
```

This avoids modifying your system Python installation. The `regex` package supplies the module imported by Maya's `strata_tokenizer.py`\.

## Fix the result-directory collision

Run this once to make the script generate unique result directories\:

```bash
python3 - <<'PY'
from pathlib import Path

p = Path("/opt/project-maya/maya_benchmark_1154_fixed.py")
s = p.read_text()

s = s.replace(
    "import copy, csv, datetime, json, os, pathlib, re, shutil, subprocess, sys, time, traceback, statistics, signal",
    "import copy, csv, datetime, json, os, pathlib, re, shutil, subprocess, sys, time, traceback, statistics, signal, tempfile"
)

s = s.replace(
    "out=root.parent/f'maya-bench-1154-{stamp}'\n    out.mkdir(mode=0o700)",
    "out=pathlib.Path(tempfile.mkdtemp(prefix=f'maya-bench-1154-{stamp}-', dir=root.parent))"
)

p.write_text(s)
print("Patched:", p)
PY
```

That prevents collisions even when the script starts twice within one second.

**One further concern\:** The script had already checked out v1.1.54 before the missing dependency caused it to fail. It had not reached the GPU benchmarks. The commands above should get past these errors, but I cannot guarantee there are no additional runtime problems. If another error occurs, the next traceback will identify it.

## Prompt:
11/5/2026, 3:10:36 PM

❯ cd /opt/project-maya

.benchmark-venv/bin/python maya_benchmark_1154_fixed.py \

  /opt/project-maya \

  /opt/project-maya/maya-ds4-production.json \

  2>&1 | tee /opt/maya-benchmark-run.log

Results: /opt/maya-bench-1154-20261105-142331

Starting 10 variants, 8K/32K/64K, two repeats each

[default] {}

{'variant': 'default', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 324.65, 'status': 'ok'}

{'variant': 'default', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 324.86, 'status': 'ok'}

{'variant': 'default', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 458.86, 'status': 'ok'}

{'variant': 'default', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 459.86, 'status': 'ok'}

{'variant': 'default', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 480.6, 'status': 'ok'}

{'variant': 'default', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 478.2, 'status': 'ok'}

[tp_even] {'DS4_TP_SHARES': 'off'}

{'variant': 'tp_even', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 323.74, 'status': 'ok'}

{'variant': 'tp_even', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 324.49, 'status': 'ok'}

{'variant': 'tp_even', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 458.93, 'status': 'ok'}

{'variant': 'tp_even', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 458.67, 'status': 'ok'}

{'variant': 'tp_even', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 479.2, 'status': 'ok'}

{'variant': 'tp_even', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 478.43, 'status': 'ok'}

[p2p_generic] {'DS4_AR_P2P': '1', 'DS4_AR_P2P_LOG': '1'}

{'variant': 'p2p_generic', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 324.07, 'status': 'ok'}

{'variant': 'p2p_generic', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 324.23, 'status': 'ok'}

{'variant': 'p2p_generic', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 459.12, 'status': 'ok'}

{'variant': 'p2p_generic', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 458.12, 'status': 'ok'}

{'variant': 'p2p_generic', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 480.11, 'status': 'ok'}

{'variant': 'p2p_generic', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 478.7, 'status': 'ok'}

[p2p_hip_auto] {'MAYA_DS4_HIP_PEER_AR': 'auto'}

## Response:
11/5/2026, 3:10:39 PM · gpt-6-thinking

The benchmark is running successfully. The first three configurations have completed their prefill tests, and the results are extremely consistent.

## Results so far

Mean throughput across two runs, tokens/second

| Configuration | 8K | 32K | 64K |
|--------|--------|--------|--------|
| v1.1.54 default | 324.76 | 459.36 | 479.40 |
| `DS4_TP_SHARES=off` | 324.12 | 458.80 | 478.82 |
| `DS4_AR_P2P=1` | 324.15 | 458.62 | 479.41 |

## Interpretation

**1\. The new TP allocation algorithm does not improve prefill in this test.**

Disabling it changes throughput by approximately -0.1%. However, the new allocation algorithm may still improve *decoding* through increased expert residency. These prefill measurements do not evaluate that benefit.

**2\. The generic P2P setting makes no measurable difference.**

At 64K, the difference is effectively zero\:

- Default: 479.40 tokens/s
- P2P: 479.41 tokens/s

This suggests either that the optimization is not active, that its impact is negligible in this configuration, or that another bottleneck dominates.

**3\. The baseline reproduces your earlier performance.**

Your previous long-prompt results were approximately 460-480 tokens/s. The v1.1.54 measurements are nearly identical, so the new default configuration has not produced a major prefill improvement.

## What comes next

The script has reached\:

```text
[p2p_hip_auto] {'MAYA_DS4_HIP_PEER_AR': 'auto'}
```

This experiment is particularly relevant because it targets Maya's AMD HIP peer all-reduce implementation.

The remaining flow and MMQ experiments may reveal larger differences than the first three configurations.

One concern: the output so far shows only `prefill` rows. I do not see decode measurements. If the script does not produce decode results, we will need a separate decode benchmark to evaluate the v1.1.51 expert-placement improvement.

**Do not change the configuration while the benchmark runs.** The current results provide a stable baseline for the remaining experiments.

## Prompt:
11/5/2026, 3:29:25 PM

'repeat': 1, 'tokens_per_s': 316.12, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 323.38, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 459.94, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 458.55, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 479.22, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 478.42, 'status': 'ok'}

[flow_dma] {'DS4_FLOW_COPY_DMA': '1'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 323.56, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 325.21, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 458.99, 'status': 'ok'}

## Response:
11/5/2026, 3:29:28 PM · gpt-6-thinking

The HIP-specific P2P experiment also shows **no measurable prefill improvement**\. The DMA experiment is still running, but its first measurements look similar to baseline.

## Updated results

Mean prefill throughput, tokens/second

| Configuration | 8K | 32K | 64K |
|--------|--------|--------|--------|
| Default | 324.76 | 459.36 | 479.40 |
| TP shares off | 324.12 | 458.80 | 478.82 |
| Generic P2P | 324.15 | 458.62 | 479.41 |
| HIP peer AR | 319.75 | 459.25 | 478.82 |
| DMA flow | 324.39 | 458.99\* | — |

\*One completed run; second run pending.

The HIP peer all-reduce setting produces a 0.12% decrease at 64K, which is within ordinary measurement variation. The 8K result is 1.5% lower, driven mainly by the first run at 316.12 tokens/s.

### What this suggests

The nearly identical 32K and 64K results across configurations suggest these environment variables are either not activating different execution paths or are changing operations that contribute very little to total prefill time.

The logs will help distinguish those possibilities. **An `ok` status establishes that the benchmark completed, not that the requested optimization activated.**

Your results are also remarkably stable at longer context lengths, which makes future performance changes easier to detect.

I would pay particular attention to the upcoming `DS4_AI_RUN`, `DS4_CHUNK_AUTO`, `DS4_YOCO_FLOW`, and `MAYA_DS4_EXPERT_MMQ` experiments.

One caveat: if the script launches a new engine process with each setting but bypasses the relevant model-loading or runtime initialization path, some environment variables may have no effect. We should inspect the per-variant logs before concluding that an optimization provides no benefit.

So far, **there is no evidence that changing TP shares, generic P2P, or HIP peer all-reduce improves your four-V620 prefill performance**\.

## Prompt:
11/5/2026, 3:31:50 PM

]\

## Response:
11/5/2026, 3:31:53 PM · gpt-6-thinking

I received only `]`\. Did you intend to paste additional benchmark output?

## Prompt:
11/5/2026, 3:57:33 PM

[p2p_hip_auto] {'MAYA_DS4_HIP_PEER_AR': 'auto'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 316.12, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 323.38, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 459.94, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 458.55, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 479.22, 'status': 'ok'}

{'variant': 'p2p_hip_auto', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 478.42, 'status': 'ok'}

[flow_dma] {'DS4_FLOW_COPY_DMA': '1'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 323.56, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 325.21, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 458.99, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 458.41, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 479.14, 'status': 'ok'}

{'variant': 'flow_dma', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 478.71, 'status': 'ok'}

[flow_ai] {'DS4_AI_RUN': '1'}

{'variant': 'flow_ai', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 324.74, 'status': 'ok'}

{'variant': 'flow_ai', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 307.15, 'status': 'ok'}

{'variant': 'flow_ai', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 460.64, 'status': 'ok'}

{'variant': 'flow_ai', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 461.24, 'status': 'ok'}

{'variant': 'flow_ai', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 481.43, 'status': 'ok'}

{'variant': 'flow_ai', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 480.34, 'status': 'ok'}

[flow_chunk] {'DS4_CHUNK_AUTO': '1'}

{'variant': 'flow_chunk', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 287.27, 'status': 'ok'}

{'variant': 'flow_chunk', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 285.42, 'status': 'ok'}

## Response:
11/5/2026, 3:57:36 PM · gpt-6-thinking

The new results show the first potentially useful improvement, along with a clear regression.

**`DS4_AI_RUN=1` improves long-context prefill by approximately 0.3%, while `DS4_CHUNK_AUTO=1` reduces 8K prefill throughput by nearly 12%.**

## Updated benchmark results

Average of two runs, tokens/second. Percentage change relative to v1.1.54 defaults.

| Configuration | 8K | 32K | 64K |
|--------|--------|--------|--------|
| Default | 324.76 | 459.36 | 479.40 |
| TP shares off | 324.12 | 458.80 | 478.82 |
| Generic P2P | 324.15 | 458.62 | 479.41 |
| HIP peer AR | 319.75 | 459.25 | 478.82 |
| DMA flow | 324.39 | 458.70 | 478.93 |
| AI run | 315.95 | **460.94** | **480.89** |
| Auto chunk | **286.35** | Pending | Pending |

Change in 8K prefill throughput

## Interpretation

### `DS4_AI_RUN=1`

The long-context improvement is small but consistent across both measurements.

- 32K: +0.34%
- 64K: +0.31%

However, at 8K, one repetition dropped to 307.15 tokens/s while the other remained near baseline. That suggests an intermittent slowdown rather than establishing a consistent 8K regression.

I would not enable this permanently based on these measurements alone.

### `DS4_CHUNK_AUTO=1`

The 8K results are much more conclusive\:

- Baseline: 324.76 tokens/s
- Auto chunk: 286.35 tokens/s
- Difference: **-11.8%**

Both repetitions agree closely, making ordinary measurement noise an unlikely explanation.

Your existing fixed `--prefill-chunk 32768` configuration appears substantially better for this 8K workload.

The 32K and 64K results will tell us whether automatic chunking trades short-prompt performance for longer-prompt performance.

## One important concern with the benchmark

The stability across nearly every configuration suggests that several flags may not be changing the actual execution path.

This is particularly relevant to the P2P and DMA experiments. Before concluding that those optimizations are ineffective, we need to verify their activation in the engine logs.

For now, **retain the default configuration**\. The only improvement observed so far is approximately 0.3%, which is too small to justify changing production settings without further measurements.

## Prompt:
11/5/2026, 8:12:18 PM

{'variant': 'flow_combined', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 424.77, 'status': 'ok'}

{'variant': 'flow_combined', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 460.44, 'status': 'ok'}

{'variant': 'flow_combined', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 459.85, 'status': 'ok'}

[expert_mmq] {'MAYA_DS4_EXPERT_MMQ': '1'}

{'variant': 'expert_mmq', 'kind': 'prefill', 'tokens': 8192, 'repeat': 1, 'tokens_per_s': 324.17, 'status': 'ok'}

{'variant': 'expert_mmq', 'kind': 'prefill', 'tokens': 8192, 'repeat': 2, 'tokens_per_s': 325.03, 'status': 'ok'}

{'variant': 'expert_mmq', 'kind': 'prefill', 'tokens': 32768, 'repeat': 1, 'tokens_per_s': 458.98, 'status': 'ok'}

{'variant': 'expert_mmq', 'kind': 'prefill', 'tokens': 32768, 'repeat': 2, 'tokens_per_s': 459.66, 'status': 'ok'}

{'variant': 'expert_mmq', 'kind': 'prefill', 'tokens': 65536, 'repeat': 1, 'tokens_per_s': 480.23, 'status': 'ok'}

{'variant': 'expert_mmq', 'kind': 'prefill', 'tokens': 65536, 'repeat': 2, 'tokens_per_s': 479.63, 'status': 'ok'}

Project Maya v1.1.54 DeepSeek HIP benchmarks

Results:

default      8192 tokens:    324.8 t/s (n=2)

default     32768 tokens:    459.4 t/s (n=2)

default     65536 tokens:    479.4 t/s (n=2)

tp_even      8192 tokens:    324.1 t/s (n=2)

tp_even     32768 tokens:    458.8 t/s (n=2)

tp_even     65536 tokens:    478.8 t/s (n=2)

p2p_generic   8192 tokens:    324.1 t/s (n=2)

p2p_generic  32768 tokens:    458.6 t/s (n=2)

p2p_generic  65536 tokens:    479.4 t/s (n=2)

p2p_hip_auto   8192 tokens:    319.8 t/s (n=2)

p2p_hip_auto  32768 tokens:    459.2 t/s (n=2)

p2p_hip_auto  65536 tokens:    478.8 t/s (n=2)

flow_dma     8192 tokens:    324.4 t/s (n=2)

flow_dma    32768 tokens:    458.7 t/s (n=2)

flow_dma    65536 tokens:    478.9 t/s (n=2)

flow_ai      8192 tokens:    315.9 t/s (n=2)

flow_ai     32768 tokens:    460.9 t/s (n=2)

flow_ai     65536 tokens:    480.9 t/s (n=2)

flow_chunk   8192 tokens:    286.3 t/s (n=2)

flow_chunk  32768 tokens:    420.8 t/s (n=2)

flow_chunk  65536 tokens:    458.1 t/s (n=2)

flow_yoco    8192 tokens:    324.4 t/s (n=2)

flow_yoco   32768 tokens:    458.9 t/s (n=2)

flow_yoco   65536 tokens:    478.7 t/s (n=2)

flow_combined   8192 tokens:    286.5 t/s (n=2)

flow_combined  32768 tokens:    425.0 t/s (n=2)

flow_combined  65536 tokens:    460.1 t/s (n=2)

expert_mmq   8192 tokens:    324.6 t/s (n=2)

expert_mmq  32768 tokens:    459.3 t/s (n=2)

expert_mmq  65536 tokens:    479.9 t/s (n=2)

Transport verification (log-based, not hardware-counter proof):

  p2p_generic: ds4: DS4_AR_P2P: the group CUDA0 BAR1 0 MiB CUDA1 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA0 BAR1 0 MiB CUDA1 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA0 BAR1 0 MiB CUDA1 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA0 BAR1 0 MiB CUDA1 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA2 BAR1 0 MiB CUDA3 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA2 BAR1 0 MiB CUDA3 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA2 BAR1 0 MiB CUDA3 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access) | ds4: DS4_AR_P2P: the group CUDA2 BAR1 0 MiB CUDA3 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access)

  p2p_hip_auto: ds4: peer prefill all-reduce unavailable on GPU0/GPU1 (GPU0 has 1026 MiB free, needs 1984 (auto: + the probe's input row)); retaining host transport | ds4: peer prefill all-reduce unavailable on GPU2/GPU3 (GPU2 has 1006 MiB free, needs 1984 (auto: + the probe's input row)); retaining host transport

Approximate variants (flow_chunk, flow_yoco, flow_combined, expert_mmq) require numerical quality validation before deployment.

The MMQ flag on Q8_K_XL may be a no-op; compare logs/kernels before attributing a speed difference to MMQ.

No original configuration was modified. Production engine remains stopped after benchmarking.

❯

## Response:
11/5/2026, 8:12:20 PM · gpt-6-thinking

The benchmark is complete, and the results are informative. **None of the tested optimizations meaningfully improves prefill on your four V620s.** More importantly, the logs explain why P2P did not help.

## 1\. Final performance comparison

Change relative to v1.1.54 defaults

| Configuration | 8K | 32K | 64K |
|--------|--------|--------|--------|
| Default | 324.8 | 459.4 | 479.4 |
| TP shares off | -0.2% | -0.1% | -0.1% |
| Generic P2P | -0.2% | -0.2% | 0.0% |
| HIP peer AR | -1.5% | 0.0% | -0.1% |
| DMA flow | -0.1% | -0.1% | -0.1% |
| AI run | -2.7% | \+0.3% | \+0.3% |
| Auto chunk | **-11.8%** | **-8.4%** | **-4.4%** |
| YOCO flow | -0.1% | -0.1% | -0.1% |
| Combined flow | **-11.8%** | **-7.5%** | **-4.0%** |
| Expert MMQ | -0.1% | 0.0% | \+0.1% |

The results favor retaining your original 32,768-token prefill chunk rather than enabling automatic chunk selection.

The 0.3% improvement from `DS4_AI_RUN=1` is too small to establish a meaningful advantage with only two repetitions.

## 2\. Your P2P optimization never activated

This is the most useful diagnostic result.

### Generic P2P

The logs report\:

```text
BAR1 0 MiB
stays host-mapped
```

Maya detected insufficient BAR1 aperture or unavailable peer access and retained its host-mapped transport.

Therefore, `DS4_AR_P2P=1` did not actually test peer-to-peer all-reduce performance.

### HIP peer all-reduce

The logs report\:

```text
GPU0 has 1026 MiB free, needs 1984 MiB
retaining host transport
```

The same condition occurred for GPUs 2 and 3.

**This is actionable.** The HIP peer implementation appears available, but Maya cannot allocate the approximately 2 GB of additional workspace it requires.

Your recent effort to maximize expert residency by reducing free VRAM to approximately 1 GB per card has prevented this optimization from activating.

There is therefore a potential tradeoff between expert residency and communication performance.

## 3\. A targeted experiment I would run next

Increase available GPU memory during initialization enough to let HIP peer all-reduce activate.

Your earlier configuration reserved approximately 3 GB per GPU. That would likely satisfy the reported 1,984 MiB requirement, assuming the reserve translates into free memory at the point Maya performs the allocation.

Test\:

```bash
MAYA_DS4_HIP_PEER_AR=auto
```

with a 3 GB reserve per GPU.

Then verify that the logs indicate actual peer all-reduce activation rather than the host fallback.

However, this would reduce expert residency, potentially lowering your cache-hit rate and decode throughput. A prefill improvement alone would not establish that it is a better production configuration.

## 4\. The MMQ experiment tells us very little

Your results are virtually identical\:

| Prompt length | Default | MMQ enabled |
|--------|--------|--------|
| 8K | 324.8 | 324.6 |
| 32K | 459.4 | 459.3 |
| 64K | 479.4 | 479.9 |

This strongly suggests that the experimental MMQ path either does not activate for your Q8\_K\_XL expert weights or provides negligible improvement under this workload.

Given the documented IQ-format restriction, I suspect it is not activating. We need a kernel trace or explicit dispatch log to establish that.

The result does not demonstrate that INT8 dot-product acceleration would be ineffective on the V620.

## 5\. What I would retain

| Setting | Recommendation |
|--------|--------|
| `DS4_TP_SHARES` | Leave at default |
| `DS4_AR_P2P` | Disabled until BAR1 access is resolved |
| `MAYA_DS4_HIP_PEER_AR` | Test with more free VRAM |
| `DS4_FLOW_COPY_DMA` | Disabled |
| `DS4_AI_RUN` | Default |
| `DS4_CHUNK_AUTO` | Disabled |
| `DS4_YOCO_FLOW` | Disabled |
| `MAYA_DS4_EXPERT_MMQ` | Disabled until activation and quality verified |
| Prefill chunk | **32,768 tokens** |

## 6\. Implications for another R9700

These results suggest that your current bottleneck is not addressed by any of the tested high-level switches.

The existing GPU kernels, expert execution, and memory architecture deserve further profiling.

A second R9700 would add substantial compute throughput, but nothing in this benchmark establishes how efficiently Maya could use it for prefill. I would still prioritize investigating RDNA 2 GEMM performance.

**The next concrete opportunity is the HIP peer all-reduce path.** You now have an explicit reason it failed: insufficient free VRAM. Testing with approximately 3 GB free per GPU will establish whether enabling it provides enough benefit to compensate for reduced expert residency.

## Prompt:
11/5/2026, 8:13:18 PM

Give me the test command for that one thing

## Response:
11/5/2026, 8:13:24 PM · gpt-6-thinking

> **Inspecting Maya Benchmark Script Options**
>
>
> **Preparing a focused test**
>
> I’m narrowing this to one transport variant, using a temporary configuration with a 3072 MiB reservation and automatic HIP peer selection. I’ll avoid altering the original script or configuration while running the benchmark.
>
> **Prepared a focused test**
>
> 
>
> **Inspected Maya benchmark script options**
>
> 
>
> Worked for 27s

This runs **only the HIP peer all-reduce test**, with 3,072 MiB reserved per GPU instead of approximately 1,024 MiB. It preserves your production configuration.

Run with Maya stopped\:

```bash
cd /opt/project-maya

python3 - <<'PY'
import json
from pathlib import Path

root = Path("/opt/project-maya")

# Create a temporary configuration with a 3 GiB reserve.
config = json.loads((root / "maya-ds4-production.json").read_text())
args = config["args"]

if "--reserve-mib" in args:
    args[args.index("--reserve-mib") + 1] = "3072"
else:
    args.extend(["--reserve-mib", "3072"])

(root / "maya-ds4-p2p-test.json").write_text(json.dumps(config, indent=2))

# Restrict the benchmark to the HIP peer all-reduce variant.
source = (root / "maya_benchmark_1154_fixed.py").read_text()
source = source.replace(
    "    print('Starting',len(environments)",
    "    environments = [v for v in environments if v[0] == 'p2p_hip_auto']\n"
    "    print('Starting',len(environments)"
)

(root / "maya_benchmark_p2p_test.py").write_text(source)
PY

.benchmark-venv/bin/python maya_benchmark_p2p_test.py \
  /opt/project-maya \
  /opt/project-maya/maya-ds4-p2p-test.json \
  2>&1 | tee /opt/maya-p2p-test.log
```

This will benchmark 8K, 32K, and 64K prefill twice each, using `MAYA_DS4_HIP_PEER_AR=auto`\.

**Check the final transport verification output.** We want to see that Maya actually enabled peer all-reduce rather than reporting `retaining host transport`\.

If it activates, compare the throughput against your existing baseline of 324.8, 459.4, and 479.4 tokens/s. If it still fails, the logs should indicate whether insufficient VRAM or unavailable peer access prevented activation.

## Prompt:
11/5/2026, 8:15:25 PM

Why twice

## Response:
11/5/2026, 8:15:27 PM · gpt-6-thinking

To check **repeatability** and distinguish genuine performance differences from measurement noise.

However, for this particular test, the first priority is determining whether HIP peer all-reduce actually activates with 3 GB reserved per GPU. One run at each prompt length is sufficient for that initial check.

We can repeat the benchmarks later if P2P activates and shows a meaningful improvement.

## Prompt:
11/5/2026, 8:35:16 PM

(root / "maya_benchmark_p2p_test.py").write_text(source)

PY

.benchmark-venv/bin/python maya_benchmark_p2p_test.py \

  /opt/project-maya \

  /opt/project-maya/maya-ds4-p2p-test.json \

  2>&1 | tee /opt/maya-p2p-test.log

Results: /opt/maya-bench-1154-20261105-201458

Starting 1 variants, 8K/32K/64K, two repeats each

[p2p_hip_auto] {'MAYA_DS4_HIP_PEER_AR': 'auto'}

[p2p_hip_auto] FAILED: p2p_hip_auto: engine exited before READY (continuing to next variant)

Project Maya v1.1.54 DeepSeek HIP benchmarks

Results:

Transport verification (log-based, not hardware-counter proof):

  p2p_hip_auto: PEER_AR_AUTO gpu=0,1 floats=167772160 samples=0 host_ms=0.000000 ar2_ms=0.000000 peer_ms=0.000000 selected=host elapsed_ms=10295.2 complete=0

Approximate variants (flow_chunk, flow_yoco, flow_combined, expert_mmq) require numerical quality validation before deployment.

The MMQ flag on Q8_K_XL may be a no-op; compare logs/kernels before attributing a speed difference to MMQ.

No original configuration was modified. Production engine remains stopped after benchmarking.

Exception ignored in: <_io.TextIOWrapper name=5 encoding='UTF-8'>

Traceback (most recent call last):

  File "/opt/project-maya/maya_benchmark_p2p_test.py", line 223, in <module>

    try: main()

BrokenPipeError: [Errno 32] Broken pipe

❯ 

❯ tmux a

sessions should be nested with care, unset $TMUX to force

## Response:
11/5/2026, 8:35:30 PM · gpt-6-thinking

> **Inspecting Benchmark Script and Logs**
>
>
> **Inspecting Script Lines 105–205**
>
>
> **Inspected benchmark script and relevant log sections**
>
> 
>
> Worked for 19s

The P2P test failed during engine initialization, before any benchmarks ran.

The important diagnostic is\:

```text
PEER_AR_AUTO gpu=0,1 floats=167772160
selected=host elapsed_ms=10295.2 complete=0
```

This tells us\:

- Maya entered the HIP peer all-reduce probe.
- The probe involved 167,772,160 floats, approximately 640 MiB of FP32 data.
- It had not completed after 10.3 seconds.
- Maya had not selected the peer transport; `selected=host` is not proof of a completed fallback.
- The benchmark harness reported that the engine exited before becoming ready.

**My leading suspicion is that the benchmark harness terminated the engine because initialization took too long.** The reported 10.3-second probe duration makes that plausible, but we need to inspect the engine log to distinguish a timeout from an allocation failure, crash, or hang.

The final `BrokenPipeError` likely occurred during the benchmark script's cleanup. It does not identify why the engine failed.

## Check the actual failure

Run\:

```bash
LOG=/opt/maya-bench-1154-20261105-201458/p2p_hip_auto.engine.log

tail -n 100 "$LOG"
```

And check for GPU allocation failures\:

```bash
grep -Ein \
  'error|fail|alloc|out of memory|hipError|peer|killed|signal|traceback' \
  "$LOG" | tail -n 60
```

I inspected the benchmark script. **It does not impose a 10-second startup timeout.** It waits for `READY` or EOF, so the failure indicates that the engine process exited or closed its output pipe. The 10.3 seconds came from Maya's own probe.

The log should establish whether the 3 GB reserve allowed allocation but exposed another failure.

Also, the `tmux` message is unrelated. It simply warns that you attempted to attach to tmux from inside an existing tmux session.

Do not reduce the reserve or change the P2P setting again yet. We need the engine's actual error first.

## Prompt:
11/5/2026, 8:37:50 PM

❯ tmux a

sessions should be nested with care, unset $TMUX to force

❯ LOG=/opt/maya-bench-1154-20261105-201458/p2p_hip_auto.engine.log

tail -n 100 "$LOG"

ds4: CUDA0 (gfx1030): layer streams at the default priority (RDNA 2; DS4_STREAM_PRIO=1 to raise them)

ds4: CUDA1 (gfx1030): layer streams at the default priority (RDNA 2; DS4_STREAM_PRIO=1 to raise them)

ds4: CUDA2 (gfx1030): layer streams at the default priority (RDNA 2; DS4_STREAM_PRIO=1 to raise them)

ds4: CUDA3 (gfx1030): layer streams at the default priority (RDNA 2; DS4_STREAM_PRIO=1 to raise them)

ds4: the one-GPU decode path (in-place pulls, windows streamed, landings): off (v1.1.27's path: not NVIDIA + one GPU)

ds4: DSpark blocks spread over the drafter's group (the experts do not all fit the GPUs): block 0 on CUDA0, 1 on CUDA1, 2 on CUDA1 (MAYA_DS4_DRAFT_SPREAD=0: all on the lead)

ds4: Engram RAM file 7 locked 0.27 / 101.38 GB

ds4: Engram RAM file 7 locked 8.86 / 101.38 GB

ds4: Engram RAM file 7 locked 17.45 / 101.38 GB

ds4: Engram RAM file 7 locked 26.04 / 101.38 GB

ds4: Engram RAM file 7 locked 34.63 / 101.38 GB

ds4: Engram RAM file 7 locked 43.22 / 101.38 GB

ds4: Engram RAM file 7 locked 51.81 / 101.38 GB

ds4: Engram RAM file 7 locked 60.40 / 101.38 GB

ds4: Engram RAM file 7 locked 68.99 / 101.38 GB

ds4: Engram RAM file 7 locked 77.58 / 101.38 GB

ds4: Engram RAM file 7 locked 86.17 / 101.38 GB

ds4: Engram RAM file 7 locked 94.76 / 101.38 GB

ds4: Engram RAM file 7 locked 101.38 / 101.38 GB

ds4: Engram RAM file 7 locked 0.27 / 101.38 GB

ds4: Engram RAM file 7 locked 8.86 / 101.38 GB

ds4: Engram RAM file 7 locked 17.45 / 101.38 GB

ds4: Engram RAM file 7 locked 26.04 / 101.38 GB

ds4: Engram RAM file 7 locked 34.63 / 101.38 GB

ds4: Engram RAM file 7 locked 43.22 / 101.38 GB

ds4: Engram RAM file 7 locked 51.81 / 101.38 GB

ds4: Engram RAM file 7 locked 60.40 / 101.38 GB

ds4: Engram RAM file 7 locked 68.99 / 101.38 GB

ds4: Engram RAM file 7 locked 77.58 / 101.38 GB

ds4: Engram RAM file 7 locked 86.17 / 101.38 GB

ds4: Engram RAM file 7 locked 94.76 / 101.38 GB

ds4: Engram RAM file 7 locked 101.38 / 101.38 GB

ds4: Engram RAM ONLY: 2 complete tables (Q6_K/Q5_K/IQ4_NL/FP8), 202.758 GB locked; reads use memcpy

ds4: the experts VRAM leaves over (~238.1 GB) are read in place from the model's files (copied into RAM after the GPUs' share when the RAM holds them all; --mmap-experts on keeps them in place) (2131.1 GB of RAM free to keep them: the most-routed 238.1 GB read ahead; prompts in chunks of 32768)

ds4: strict RAM: all 12715 CPU experts selected, no disk fallback

ds4: copied expert cache: DONTNEED hints accepted for 238.894 GB of whole source pages (eviction is advisory)

ds4: the copied experts pinned (239.9 GB in 19.8 s): the prompt path DMAs them in place

ds4: 12715 of the CPU lane's 12715 experts copied into RAM (239.1 GB, 1 NUMA part) in 221.5 s

PEER_AR_AUTO gpu=0,1 floats=167772160 samples=0 host_ms=0.000000 ar2_ms=0.000000 peer_ms=0.000000 selected=host elapsed_ms=10295.2 complete=0

maya-ds4: startup transport probe did not drain; reload required

## Response:
11/5/2026, 8:38:00 PM · gpt-6-thinking

The log establishes the cause. **Maya's HIP peer all-reduce startup probe did not complete.** This was not a benchmark timeout or a normal fallback due to insufficient VRAM.

The decisive lines are\:

```text
PEER_AR_AUTO gpu=0,1 floats=167772160
samples=0
selected=host
elapsed_ms=10295.2
complete=0

maya-ds4: startup transport probe did not drain; reload required
```

This tells us\:

- Maya reached the transport-selection benchmark.
- It attempted to test a 640 MiB FP32 workload between GPUs 0 and 1.
- It collected **zero completed timing samples**\.
- After approximately 10.3 seconds, the probe had not drained.
- Maya terminated initialization and requested a reload.

The earlier failure with only 1 GB free was a workspace-capacity rejection. This failure is different: the probe got further but could not complete.

The log does not identify whether the underlying cause was a HIP synchronization problem, a stalled transfer, a GPU kernel, or another implementation issue.

## What I recommend

**Stop testing `MAYA_DS4_HIP_PEER_AR=auto` for now.** The problem is no longer simply insufficient reserved VRAM. The peer transport initialization itself needs investigation.

Restore your standard configuration and explicitly disable the experimental peer path\:

```bash
cd /opt/project-maya

MAYA_DS4_HIP_PEER_AR=off \
DS4_AR_P2P=0 \
./maya.sh --serve --config maya-ds4-production.json
```

This assumes your installed `maya.sh` accepts the `--serve --config` syntax. If your usual production launch differs, retain that launch command and prepend the two environment assignments instead.

Your production JSON was not modified by the benchmark, so there is no reason to retain the temporary 3 GB reserve configuration.

There is one additional useful observation: Maya successfully copied all 12,715 CPU-lane experts into RAM before the transport probe failed. The failure occurred afterward, during peer all-reduce initialization, rather than during model loading.

I would now treat the peer transport as a separate debugging problem. To diagnose it properly, we need either a smaller standalone HIP peer-transfer test or instrumentation around the probe's completion and synchronization operations. Increasing the reserve further has no demonstrated justification.

## Prompt:
11/5/2026, 8:38:36 PM

❯ cd /opt/project-maya

MAYA_DS4_HIP_PEER_AR=off \

DS4_AR_P2P=0 \

./maya.sh --serve --config maya-ds4-production.json

usage: maya.py [-h] [--backend {cuda,hip,metal}] [--setup] [--check] [--no-start] [--yes] [--plain] [--import-model FOLDER] [--copy] [--download-model] [--model MODEL] [--model-dir MODEL_DIR]

               [--ds4-quant {gguf-q8,gguf-q4,maya-s,maya-m,maya-xs,exl3,exl3-3.5,abliterated-128g,abliterated-256g,gguf-iq1_s,gguf-iq1_m,gguf-iq2_xxs,gguf-q2_k_xl,gguf-iq3_xxs,gguf-iq3_s,gguf-q3_k_xl,abliterated}]

               [--mm3-quant {exl3,gguf-xs,gguf-s,maya-xs,maya-s}] [--qwen-quant {q5_k_m,ud-iq2_xs}] [--draft {mtp,dflash2}] [--no-vision] [--gguf-dir GGUF_DIR] [--models-dir MODELS_DIR] [--gpu GPU] [--gpus GPUS]

               [--context CONTEXT] [--parallel N] [--vram-reserve-mib N] [--mmap-experts {auto,on,off}] [--port PORT] [--host HOST] [--api-key API_KEY] [--env KEY=VALUE] [--nvcc NVCC] [--host-compiler HOST_COMPILER]

               [--llama-dir LLAMA_DIR] [--rebuild] [--repack] [--report] [--bench] [--calibrate] [--speed-calibration] [--drafting {measure,on,off}] [--no-speed-calibration] [--config CONFIG]

maya.py: error: unrecognized arguments: --serve

## Response:
11/5/2026, 8:38:39 PM · gpt-6-thinking

My mistake. Maya does not accept `--serve`\. It starts the server by default.

Use\:

```bash
cd /opt/project-maya

MAYA_DS4_HIP_PEER_AR=off \
DS4_AR_P2P=0 \
./maya.sh --config maya-ds4-production.json
```

This uses your production configuration and disables both experimental P2P paths without modifying the configuration file.

## Prompt:
11/5/2026, 8:51:13 PM

Why do I need those flags?  Production doesn't set these t 1

## Response:
11/5/2026, 8:51:16 PM · gpt-6-thinking

You do not need those flags. I added them unnecessarily.

The benchmark script passed the experimental environment variables only to its child processes. They do not persist in your shell or modify your production configuration.

Just start Maya normally\:

```bash
cd /opt/project-maya
./maya.sh --config maya-ds4-production.json
```

This uses your production configuration and Maya's default environment settings.

## Prompt:
11/6/2026, 3:44:15 PM

Switch to work mode.

## Response:
11/6/2026, 3:44:19 PM · gpt-6-thinking

Work mode handoff was not accepted. You can select **Work** from the mode selector to continue the Project Maya investigation with the existing conversation context.

## Prompt:
11/6/2026, 3:45:00 PM

Switch to work mode.
Working for 1s

## Response:
11/6/2026, 3:45:04 PM · gpt-6-thinking

The Work mode handoff was accepted. A new chat has been created with the Project Maya investigation context.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)