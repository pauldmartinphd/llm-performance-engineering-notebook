# Maya releases, memory budgets, and alternative deployment plans

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

**Discussion: November 2–8, 2026; no new benchmark in this entry.**

The matrix changed the deployment question from maximizing one model's rate to deciding how much capacity each service should retain. You considered putting DeepSeek on a separate eight-channel DDR4 server with 512 GB RAM, one V620 and an R9700, while leaving GLM on Galactus. You also considered Qwen and speech, OCR, translation and TTS services. These were deployment proposals; the source files contains no matched R9700 run or simultaneous multi-service benchmark.

## Release changes as test candidates

The November 2 discussion quotes v1.1.50 (`10e69a0eb11541004abb24f9fa47147e9002326f`): automatic prompt chunks, a GLM CPU prefill gate, model load/unload APIs, an optional peer all-reduce path, and a Turbo mode that trades quality for speed. These were upstream descriptions, not local speedups. In particular, the proposed approximately 575 tokens/s prefill from applying a release percentage to the prior 483.4 was a forecast, not a measurement.

The November 5 discussion quotes v1.1.55 (`c4effee67cb95fb4303720919b075fddf04b4e69`), including API compatibility, prompt memoization, GLM INT8 KV by default, and several engine defaults. You kept the established DeepSeek settings and treated INT8 KV as a separate GLM experiment. The [November 5 sweep](2026-11-05-maya-v1154-prefill-and-peer-ar.md) already tested several relevant switches; it did not show a useful automatic-chunk or peer-transport improvement.

The GPU peer-bandwidth discussion recalled about 67 GB/s. The earlier notebook contains different transfer methods and directions; that recollection is not a measured bandwidth for Maya's all-reduce. The failed HIP peer probe remains the relevant local result.

## Configuration and operational distinctions

GLM calibration on November 2 explicitly declined to tune HIP: it measures the CUDA CPU lane and retained the HIP engine settings. Copying DeepSeek's calibration fields into GLM would not establish a tuned GLM profile, since the engines have different execution paths and controls. The November 9 attempt produced the same limitation.

On November 8 you asked how to bind the service to `0.0.0.0`. The conversation supplied a host override and a persistent config edit; no readback establishes that either was applied. The November 10 discussion about missing tags shows a non-shallow checkout at v1.1.57, not an absent release history. `git fetch --unshallow` failed because the repository was already complete. The later build and tmux instructions are operational guidance without captured execution results. You preserve these states separately from measured engine changes.

## Sources

**Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md).
