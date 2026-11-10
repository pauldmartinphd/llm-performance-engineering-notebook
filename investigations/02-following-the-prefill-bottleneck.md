# 2. Following the prefill bottleneck

[Previous: the baseline](01-establishing-a-baseline.md) · [Chapter index](README.md) · [Next: speculation](03-making-speculation-useful.md)

## The question

Galactus had four GPUs, yet GLM-5.2 read prompts much more slowly than I expected. The weights were available, the memory channels worked, and the model generated answers. I needed to find where prompt processing was spending its time.

Prompt processing is often called **prefill**. It reads the input and builds the state used to generate an answer. Unlike generation, it can process many input tokens together. That should give the GPU enough work to make large matrix operations worthwhile.

The July investigation moved through three questions: was I measuring the intended workload, was work reaching all four GPUs, and was the engine waiting when it could have continued?

## First, the requested batch was not the effective batch

Early tests appeared to show that sending expert operations to the GPU could hurt performance. The diagnostic work found an important interaction: in the tested llama-bench build, the prompt length limited the effective microbatch size. Asking for a microbatch of 8192 while testing a 512-token prompt still exercised the smaller case.

A **microbatch** is the group of tokens the engine processes together in one part of the calculation. The distinction matters because transferring expert weights has a cost. A larger group can reuse those weights for more work before moving on.

I repeated the comparison with an 8192-token prompt and verified the effective sizes. The [completed ladder](../results/lab-notebook/2026-07-14-diagnostic-reconstruction.md) showed a clear change:

| Microbatch | Expert operations offloaded to GPU | CPU-expert control |
|---|---:|---:|
| 512 | 25.90 | 31.88 |
| 1024 | 41.68 | 32.89 |
| 2048 | 62.64 | 33.31 |
| 4096 | 84.62 | 33.54 |
| 8192 | 104.97 | 33.77 |

Rates are tokens/s. At 512, offload was slower. At 8192, it was about three times as fast as the control. The small-batch result was real, but it answered a narrower question than I had intended.

## Next, I followed where the work went

The scheduler decides which device runs each part of the computation. Its trace showed **731 of 1186 execution splits on ROCm0**. That concentration suggested that the offloaded expert work was not being spread across the cards as I expected.

A split count is a count of scheduled work segments, not a direct measure of time. Equal counts would show a change in placement, but would not prove that the cards were equally busy or that the model ran faster. I therefore measured both placement and throughput.

The [placement tests](../results/lab-notebook/2026-07-14-1813-multi-card-placement.md) and [scheduler traces](../results/lab-notebook/2026-07-14-glm52-scheduler-traces.md) led to a round-robin change. It distributed the eligible work across the GPUs. The measured prefill rate moved from **104.97 to 105.71 tokens/s**—too little to establish a useful gain.

That null result mattered. The uneven distribution was visible, but fixing it alone did not remove the main delay.

## The remaining wait was in the execution path

The engine also read the expert-routing indices back before continuing. That read introduced a synchronization point: work had to wait for the information even though the weight transfers themselves could be asynchronous.

For large prefill batches, the combined patch treated all experts as used and copied their weights without waiting for that index read. This does more transfer work than copying only the selected experts, but it removes a dependency from the path. On this workload, avoiding the wait was worth more than avoiding those copies.

The [July 21 test](../results/lab-notebook/2026-07-21-prefill-patch.md) compared the combined change with the baseline:

| Configuration | Prefill, tokens/s | What it established |
|---|---:|---|
| Original scheduler | 104.97 | Baseline |
| Distributed work only | 105.71 | Placement changed; throughput barely changed |
| Distribution plus removal of the routing-index wait | 119.36 | A measured 13.7% prefill gain |

The final split counts were 285, 300, 294, and 292. Decode did not show a measured gain from this prefill change. The [patch documentation](../patches/prefill/README.md) preserves the exact edits and checks.

## Other explanations were tested along the way

I checked container overhead, memory topology, CPU threads, affinity, and transfer behavior. The container's memory bandwidth was within about 1% of the host's. The sampled diagnostic intervals showed no swap traffic or reported CPU steal. A four-stream host-to-device test reached 65.7 GB/s.

Those observations narrowed the search. They did not mean that every transfer path could sustain that rate or that utilization was balanced at every moment. The complete tests and failures remain in the [July records](../results/lab-notebook/README.md#july-2026) and the [unsuccessful-approach summary](../takeaways/refuted-hypotheses.md).

## What this changed

The useful result was a specific prefill improvement on the tested GLM-5.2 build. It was not a general claim that distributing work always helps, or that the patch improved every model.

The investigation also changed my method. I needed to verify the settings the engine actually used, inspect where time could be serialized, and measure the final effect. A better-looking GPU histogram was only an intermediate result.

Later Flash-model comparisons did not isolate this patch on their own builds. The broader matched comparison remained unfinished when the work moved to Maya. The successful July result remains valid within its original conditions.
