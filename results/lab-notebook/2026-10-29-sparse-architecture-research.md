# Research questions: learned memory, sparse attention sharing, and active experts

**Recorded:** October 29, 2026 (America/New_York). **Status:** research ideas and proposed experiments; no experiments run. **Source:** [Merging Sparse Attention Architectures](https://chatgpt.com/c/6add5a03-c34c-83ea-b385-07288fc12cd3) and the October 29 notebook continuation.

These notes preserve four research questions from the architectural discussion, separately from the memo, *Toward a More Efficient Trillion-Parameter Language Model: Predicting the Architecture of the Next Wave of Large Open Models*. They describe experiments that could test the ideas, rather than predictions about a particular future model.

Galactus has 2 TiB of RAM and is not serving model components from SSD in this discussion. There are no SSD benchmarks behind this entry. Earlier assistant claims about SSD measurements and subsequent external searches are excluded; they do not establish an observation from this work. The SSD ideas concern a possible future architecture whose learned tables exceed the available RAM budget, not a diagnosis or proposed reconfiguration of the current server.

The memo's parameter sizes and ratios were illustrative. Neither the proposed learned-memory allocation nor the active parameter budgets came from an optimization experiment. Model-specific descriptions of CSA2 and QSA also need confirmation against the chosen implementation before an architectural ablation can be specified precisely.

## SSD-resident Engram as a future capacity experiment

The question is whether a model could retain a large learned n-gram table on SSD while keeping its computational backbone and a bounded lookup cache in RAM or accelerator memory. The hypothesis is that sparse, predictable table accesses might permit learned capacity to grow beyond available RAM without an unacceptable increase in generation latency. This is untested here.

The distinction worth preserving is between calculating an address and retrieving its contents. Deterministic addressing may avoid searching the table, but it does not establish the physical cost of a read. Table size, lookup fanout, row width, locality, storage-page size, and the time available before an embedding is consumed would all affect the result.

A first experiment would capture lookup addresses and consumption times from a compatible implementation, then replay that trace against a table with the same row layout. Compare RAM-resident access, SSD demand reads, and asynchronous SSD reads with a fixed RAM-cache budget. Separate warm-cache and cold-cache conditions and measure physical I/O; a memory-mapped file alone does not demonstrate that reads reached the SSD. A synthetic capacity sweep can test storage behavior, but cannot establish that a larger learned table improves model quality.

Record logical embedding bytes and physical bytes read per token, cache hits, read amplification, request latency, and time spent waiting for embeddings on the execution critical path. An end-to-end follow-up must also measure prompt processing and generation separately, peak RAM/VRAM use, and output correctness against the RAM-resident control. Hold table precision and model weights fixed so that a placement comparison does not also become a quantization comparison.

The useful result would be a measured tradeoff between RAM saved and added latency under a specified workload. If demand reads remain on the critical path even after overlap and caching, that would limit the proposed architecture on the tested hardware. No throughput estimate or acceptable slowdown is assumed in advance.

## Speculative decoding as a possible prefetch mechanism

Draft tokens might reveal future n-gram addresses before target-model verification needs the corresponding embeddings. This suggests a second use for speculation: providing storage lookahead. It is a hypothesis contingent on the SSD-resident design above, not an explanation of the notebook's existing Maya or DSpark speedups.

The first implementation question is when each address becomes available relative to its target-layer consumption deadline. Confirm the tokenization, n-gram history, and address calculation used by the target. Derive addresses for candidate continuations without committing rejected tokens to the target's logical state, and retain a correct demand-read path when predictions fail.

Use three paired configurations: target-only SSD access; speculative decoding with demand reads; and the same speculative decoder with candidate-address prefetch. Keep the draft model, verification method, cache budget, and workload fixed between the last two. Their difference isolates the incremental prefetch effect; comparing only the first and third would mix ordinary speculative acceleration with storage overlap.

Measure emitted tokens per second, draft acceptance, accepted tokens per verification, useful prefetches completed before consumption, unused prefetched bytes, cache eviction effects, and target wait time. Reads for rejected candidates may still be required during verification, so rejection alone is not a sufficient definition of wasted I/O. Track whether each read was consumed and whether it arrived early enough to help. Test low-acceptance workloads as well as favorable ones.

Proceed only if prefetch reduces end-to-end time beyond speculation alone without changing decoding correctness. Higher IOPS or more outstanding requests would not by themselves show a benefit. This experiment is relevant only after storage waits have been demonstrated in the first experiment.

## Shared KV representations and independent sparse indexing

The architectural distinction is between the representations retained in memory and the decisions that select which representations each layer reads. Sharing stored KV representations does not logically require every consuming layer to select the same positions or blocks. Whether independently selected reads remain useful depends on the shared representation and the queries trained to use it.

The proposed comparison begins with native compressed sparse attention, then adds a Gated DeltaNet (GDN) hybrid while retaining the native selection mechanism. A third configuration keeps that hybrid and its shared representations but uses independent per-layer sparse selection. The first comparison tests the hybrid; the second tests independent selection within it. These should be trained or adapted under an explicit protocol, not treated as interchangeable inference settings in an unchanged checkpoint.

Before training, map which layers produce stored representations, which consume them, and which own or reuse index lists. Removing an attention consumer does not necessarily remove a stored KV allocation. Estimate memory from actual unique buffers, then measure persistent KV, recurrent state, index data, and temporary workspace separately.

Hold context length, retrieval budget, compression precision, and training data constant where possible. Record any additional indexer parameters and computation rather than treating independent selection as free. Measure held-out loss, long-context retrieval across positions and distractor counts, generation latency, and peak memory. Small models and controlled retrieval tasks are the initial feasibility test; they would not establish trillion-parameter behavior.

Independent indexing is worth pursuing if its quality improvement justifies its additional selection cost. A memory benefit from the GDN hybrid must be demonstrated through reduced allocations, not inferred from the fraction of attention layers replaced.

## Active expert count as a scaling dimension

The longer-term hypothesis is that models with the same total expert capacity and learned-memory capacity could offer different capability levels primarily by changing the number of experts activated per token. This would separate stored capacity from active computation. It does not imply that increasing top-k in a checkpoint trained for a smaller budget will improve it.

Start with a small compatible MoE family whose expert count, expert width, backbone, and learned-memory allocation remain fixed. Train several activation budgets. Use both a fixed-token comparison, which exposes the additional training compute required by higher activation, and a fixed-training-compute comparison, which generally gives different token budgets. These answer different questions and cannot be presented as a single comparison controlling both resources.

A separate experiment would train one model to support a range of activation budgets and evaluate each supported budget against the corresponding fixed-budget model. Record routing normalization, load balancing, expert utilization, token dropping, and the activation-budget sampling schedule. Distinguish shared experts from routed experts when reporting active parameters.

Measure held-out loss and task quality alongside actual training compute, inference latency, memory traffic, and resident capacity. More active parameters are not a sufficient measure of useful computation. If varying activation produces a useful quality/cost curve, compare it with alternatives that increase total expert capacity or learned-memory capacity. Without those alternatives, the experiment cannot establish that active expert count is the preferred scaling direction.

## Next work and evidence boundary

The immediate notebook contribution is the separation of these questions and their controls. No server change, training run, storage benchmark, or implementation validation was performed. The SSD and speculative-prefetch questions form one conditional research sequence; the attention-sharing and expert-scaling questions can be studied independently in smaller trained models.

Before execution, choose the implementation and resource budget, retain its revision and configuration, and define the quality and latency tolerances for the comparison. Add results only when there are attributable captures. Existing inference measurements elsewhere in the notebook remain evidence for their recorded configurations, not evidence for these proposed architectural combinations.
