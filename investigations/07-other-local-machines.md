# 7. Testing other local machines

[Previous: sharing Galactus](06-sharing-galactus.md) · [Chapter index](README.md) · [Important findings](../takeaways/general-principles.md)

## The question

Galactus separates system RAM from GPU memory. I also wanted to understand machines where the CPU and GPU share a larger memory pool. Would that make large local models easier to host, and which limits would remain?

The notebook contains two separate investigations: oMLX on an M2 Max MacBook Pro called Magneto, and Gufo on a Ryzen AI Max+ laptop called SilverSurfer. They use different models and software. Their results describe each machine; they do not form a controlled hardware ranking.

## Magneto: the requested acceleration was not always the path that ran

Magneto is a 14-inch MacBook Pro with an M2 Max, 38 GPU cores, and 64 GB of unified memory. I tested Qwen3.8-27B with oMLX, including Lightning MTP and Apple's Neural Engine, or ANE.

The initial single-request baseline had MTP enabled and ANE prefill disabled. At 4K context it measured **156.1 tokens/s prompt processing and 16.4 tokens/s generation**. At 16K it measured 152.9 and 19.1. These are different prompt depths, not repetitions of one test.

After ANE tuning, the measured prompt rates rose to **170.1 at 4K and 165.6 at 16K**. That was about 8–9% above the initial rows. Generation was slower in those runs, although the reported full-request time still fell. The tuner had predicted a larger improvement, so its estimate could not substitute for the benchmark.

The [model record](../results/qwen-3.8-27b.md) preserves the full context sweep, a later screenshot, draft statistics, and timing details. Some early rows survive as conversation transcriptions rather than original benchmark exports. Thermal state was not matched across all runs, so I cannot assign the whole difference to ANE.

## The DFlash trial exposed a runtime mismatch

I then requested DFlash with tuned ANE and Lightning MTP off. The measured generation rate was lower: 15.5 tokens/s at 4K and 11.2 at 16K. The corresponding initial MTP rows were 16.4 and 19.1.

The logs showed ANE enabled in the settings but doing no MLP or GDN operations. They also reported a slower attention path. This means the intended combination was not the combination actually tested. The trial was a useful failure, but it could not establish the performance of a correctly running ANE-plus-DFlash configuration.

I also investigated the difference from a higher M2 Max leaderboard result. A chip name and RAM size do not identify the laptop chassis. The record contains thermal observations, but no matched comparison proving that chassis or cooling caused the gap. The [dated discussion](../results/lab-notebook/undated-magneto-omlx-qwen38.md) preserves that uncertainty.

## SilverSurfer: enough total memory did not guarantee a successful allocation

SilverSurfer is an HP ZBook Ultra G1a with a Ryzen AI Max+ PRO 395 and 128 GB of shared memory. I used openSUSE Aeon and a Podman container to run Gufo 0.28.1 with Qwen3.8-Flash-Next UD-Q4_K_XL and an MTP predictor.

The first attempts failed for different reasons. One could not open the model file because of access restrictions. After that was addressed, the model failed with an allocation error. Memory diagnostics and later configuration changes were therefore separate parts of the investigation.

The initial system reported about 94 GiB of host memory with a 32 GiB firmware reservation for the GPU. Later, a successful load had about 119,129 MiB available beforehand. Kernel memory limits had changed, and a smaller firmware reservation had been recommended. The final BIOS value was not captured, so I cannot credit one setting alone with the fix.

The successful runtime reported about 97.65 GiB of GPU-device memory in use. On a shared-memory machine, that is not a separate physical pool to add to the host-memory counters. It does show that the GPU could use more than the initial firmware-reserved slice.

## A retained on-battery result

The direct API timing event records **318 generated tokens at 29.85 tokens/s on battery**, after a 63-token prompt. The model loaded in about 25 seconds. The server requested a 262,144-token context ceiling, but the short completion did not test that capacity.

A later Jan discussion reports 539–740 tokens/s for newly processed prompt tokens as the conversation grew to 54,781 total tokens. The original attachment behind those figures is missing, so those values have weaker support than the retained timing JSON. They also include conversation history and cache reuse, which differ from a fresh long-prompt sweep.

The [full SilverSurfer entry](../results/lab-notebook/2026-11-02-silversurfer-gufo.md) keeps the container launch, memory readbacks, failures, model-ID correction, and timing event.

## What these tests add

Shared memory changes the capacity problem, but it does not remove allocation limits, execution choices, thermal effects, or measurement pitfalls. On Magneto, a saved toggle did not prove that ANE ran. On SilverSurfer, physical memory capacity did not prove that the first allocation path would work.

Both investigations reinforced the same method used on Galactus: check the effective runtime, preserve the actual counters, and compare the complete request. Neither provides a matched Gufo-versus-Maya or Apple-versus-AMD benchmark. Those comparisons remain open.
