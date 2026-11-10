# Important findings

The experiments established useful gains on specific local configurations. This page gives the result, the reason it matters, and the boundary of the evidence. The [investigation chapters](../investigations/README.md) explain the full sequence. The [lab notebook](../results/lab-notebook/README.md) retains every test.

## 1. Large models can be useful without fitting in GPU memory

Galactus has about 120 GB of usable GPU memory across four V620s, far less than the weight size of several tested models. With system RAM holding the remaining data, Maya's October 23 tests nevertheless reached **31.6 generated tokens/s for GLM-5.3-Flash** and **29.3 for DeepSeek-V4.1-Flash** in separate runs.

This matters because model capacity is not set by VRAM alone. The engine's placement, caching, CPU work, and speculative path determine whether the larger model is practical to use. The tests do not show those rates under simultaneous load, and the two model workloads were different.

[Experiment and controls](../results/lab-notebook/2026-10-23-maya-glm53-speculation.md) · [Why the work moved to Maya](../investigations/04-from-the-fork-to-maya.md)

## 2. Verifying the effective batch revealed a large prefill opportunity

In the July GLM-5.2 ladder, GPU operation offload processed an 8192-token prompt at **25.90 tokens/s with microbatch 512** and **104.97 with microbatch 8192**. At the smaller batch, offload was slower than the CPU-expert control. At the larger batch, it was about three times as fast as that control.

Larger batches let the engine reuse transferred weights for more work. The important finding was not simply “use the largest batch.” It was that the original tests had exercised a smaller effective batch than intended. Correcting that measurement changed which optimization was worth pursuing.

[Complete ladder](../results/lab-notebook/2026-07-14-1108-v3-diagnostic-run.md#follow-up-tests-and-hardware-monitoring) · [Scheduler investigation](../investigations/02-following-the-prefill-bottleneck.md)

## 3. Removing a wait helped where distribution alone did not

The July scheduler experiment improved GLM-5.2 prefill from **104.97 to 119.36 tokens/s**, a **13.7% gain**, by combining distributed offload with removal of a routing-index readback wait. Distributing the work alone reached 105.71, with little practical gain.

The result separated a visible placement change from a useful speed change. It supports examining dependencies across the whole execution path. It establishes this patch's benefit on the tested July build, not on every later model or engine. Decode did not show a measured gain from the prefill patch.

[Patch experiment](../results/lab-notebook/2026-07-21-prefill-patch.md) · [Exact edits](../patches/prefill/README.md)

## 4. Speculative decoding gave useful generation gains

Maya's GLM-Flash comparison measured **19.9 tokens/s without speculation and 31.6 with it**, a **58.8% gain**. DeepSeek's separate DSpark calibration measured **21.2 and 29.3**, a **38% gain**. Earlier common-build tests also recorded useful MTP and DSpark generation results.

Drafting can reduce the number of expensive target steps per output token. It also adds work. The draft-length tests showed why acceptance percentage alone does not identify the fastest setting. The gains above are within each test; they are not a ranking between models.

[Speculation chapter](../investigations/03-making-speculation-useful.md) · [October A/B and calibration](../results/lab-notebook/2026-10-23-maya-glm53-speculation.md)

## 5. DeepSeek long-prefill tuning reduced a real waiting cost

The October 28 sweep selected **32K chunks and an 800 MiB transfer ring**. At a 64K prompt target, it reached about **480 tokens/s**, compared with **434** for the opening baseline. The roughly 11% improvement corresponds to about 14 seconds less prefill time for that input length, using the rounded rates.

The repeated closing baseline stayed close to the opening one. Later fresh production requests remained near the selected range. The benefit depended on the combination of chunk and ring size; it was not a general result for every larger buffer.

[Full sweep](../results/lab-notebook/2026-10-28-maya-prefill-and-regression.md) · [Long-prompt chapter](../investigations/05-tuning-long-prompts.md)

## Later operating choices with more limited evidence

These results informed the current service, but need more testing before they support strong claims about repeatability or answer quality.

| Choice | What the tests showed | What remains unresolved |
|---|---|---|
| DeepSeek FP16-output prefill on two V620s | Median 389.55 / 501.95 / 589.70 tokens/s at 8K / 16K / 32K; about 23–30% above the same-version TP2 fallback observations | No interleaved requested-TP2 control; local answer-quality equivalence unmeasured |
| GLM-Flash reserve reduced from 3072 to 1024 MB per GPU | Built-in mean generation rose from 18.7 to 21.5 tokens/s in successive runs | One run per setting; placement and warm-up may contribute |
| GLM-Flash INT8 KV | About 11–13% higher average generation than its paired FP16 control; long prefill near 292 tokens/s | Short samples varied; local quality and full-context behavior unmeasured |
| Two resident Maya services | Reported operating setup uses two V620s for each model; individual-model tests support the allocation | No controlled overlapping-request benchmark of the final pair |

The [two-model chapter](../investigations/06-sharing-galactus.md) connects these choices to the experiments. The [other-machines chapter](../investigations/07-other-local-machines.md) records the separate M2 Max and Ryzen AI Max+ results.

## Findings about the platform and the method

The August upgrade doubled Galactus's RAM capacity while measured bandwidth remained near 150 GB/s. That expanded the set of models that could stay in memory without doubling the rate at which the CPU could read their weights. [Memory remeasurement](../results/lab-notebook/2026-08-15-stream-rebaseline-2tb.md).

Across the investigations, requested settings and actual execution often differed. Examples include clamped microbatches, TP4 requests that ran as TP2, GLM chunks reduced to fit workspace, and an ANE toggle that did not produce ANE work. These observations make runtime verification part of the result, rather than a setup detail to assume.

The [measurement guide](../results/methodology.md) explains the checks. The [earlier detailed method lessons](measurement-lessons.md) retain the original benchmark pitfalls and cost models. [Unsuccessful approaches](refuted-hypotheses.md) remain a separate reference.
