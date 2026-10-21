# Entry 17 — Why STREAM bandwidth is not a decode prediction

**Recorded:** October 21, 2026. I compared the [July scheduler experiment](09-session-9-building-and-benchmarking-the-patch.md), [2 TB STREAM record](11-stream-rebaseline-2tb.md), and historical [GLM](15-glm53-flash-placement.md) and [DeepSeek](16-deepseek41-dspark.md) captures to understand why decode throughput remained below my bandwidth-based estimate.

My initial expectation was that a machine sustaining roughly 150 GB/s in STREAM should generate tokens at a rate close to that bandwidth divided by the active expert weights. That ratio is a useful bound for an assumed memory-traffic model. It is not a measured inference rate. STREAM exercises long, regular memory operations; quantized MoE inference must select experts, read and decode their weights, perform arithmetic, and complete the rest of the model before the next dependent token can finish.

If the relevant weights require W bytes per token and sustainable bandwidth is B bytes/s, then W/B is an optimistic time for that traffic alone. A claim that the measured token time should equal W/B assumes the traffic estimate is correct, the inference kernels sustain B, and other work is either negligible or overlaps completely. I have not measured those quantities for GLM-5.3-Flash or DeepSeek-V4.1. I also need a separate bytes-per-token estimate for each model; the older GLM-5.2 estimate describes a different set of weights and operations.

## Where the CPU and GPUs compute the experts

Storing routed expert weights in RAM does not establish which device computes the projections. CPU kernels can read those weights directly, while the scheduler can transfer eligible prefill work to a GPU. My shared scheduler patch deliberately acts on sufficiently large prefill batches and leaves decode selection unchanged. I cannot blame CPU-to-GPU copying for slow decode simply because I placed the weights in RAM. I need to trace the operations and transfers to determine whether copying contributes materially to token time.

Moving computation to the GPU also requires supplying its weights. For a model larger than aggregate VRAM, that means some combination of resident experts, caching, streaming, and CPU work. A cache helps only when reuse avoids enough work to repay its management and transfer costs, while leaving space for attention state and compute buffers. The GLM capture contains both a slower cache setting and a faster one, plus an allocation failure. Increasing the requested budget did not consistently improve throughput.

## Prefill and decode are different experiments

I increased pp8192 throughput in July only after combining distribution with the routing readback bypass. It did not improve decode. This is consistent with reducing synchronization in a phase that can amortize weight movement over many tokens, without solving the costs of a largely serial token-generation path. I still need to test the current port on Galactus before determining whether the Flash models benefit from the same changes.

Likewise, the GLM record's 19.1–19.4 t/s under automatic placement is a useful operating point, but it does not prove that a custom kernel caused the gain. I first need to establish where the weights and operations ran, then use kernel timings, transfer timings, and cache statistics to explain the difference.

## Measurements needed to identify the bottleneck

For each model I need the four combinations of speculation off/on and automatic/forced-RAM placement, holding the prompt, generation length, split settings, and explicit cache budget constant. I need the source revision and model hashes, actual allocations on each GPU, repeated warm timings, and draft acceptance statistics. For the bandwidth explanation I also need phase timings or a profile that separates CPU expert work, GPU work, transfers, and synchronization.

With those measurements, I could distinguish a compute or memory-access limit from transfer overhead or poor placement. So far, I have measured slower GLM generation with forced-RAM placement and inconsistent gains from increasing the cache budget. The V4.1 DSpark test failed before generation, leaving that comparison open.
