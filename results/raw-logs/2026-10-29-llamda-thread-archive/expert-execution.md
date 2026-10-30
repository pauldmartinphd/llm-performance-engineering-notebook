# Experimental expert execution

All Maya-derived changes live on `experimental/moe-execution`, based on `amd-main`. They are not enabled or present on `amd-main`, `rdna-dev`, `halo-dev`, or `vega-dev`. This branch is the shared development point; promotion requires correctness and performance validation on the affected hardware. The clean upstream routing submission remains separate.

## Implemented: MXFP4 CPU token batching

The AVX2 CPU MUL_MAT_ID path groups up to four tokens routed to the same expert. It loads and decodes each pair of MXFP4 weight blocks once, then applies them to the group's Q8_0 activation rows. This reduces repeated weight loads and nibble decoding during speculative verification and other multi-token CPU expert work. Single-token expert groups retain the existing path. Non-AVX2 builds retain their existing execution path.

The implementation preserves each output's existing accumulation order, FP16 activation scales, and tail handling. It accepts nonadjacent activation rows through explicit pointers, including broadcast inputs. It does not change quantization, expert selection, model weights, or speculative acceptance.

This follows Project Maya's multi-token expert-weight reuse approach, adapted to llama.cpp's existing arithmetic. Maya's source is [ds4_mx_kern.inc](https://github.com/mw00/project-maya/blob/75798d4280dcc282b2f2224417fe8fa11e26ea89/src/ds4/ds4_mx_kern.inc). Gufo applies related grouping to GPU expert down projections in [ds4_rocm_moe.hip.hpp](https://github.com/gufo-org/gufo/blob/d4eaaad438a498bf9ba541afe3d655f70dd4196d/src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_moe.hip.hpp).

## Validation

The x86_64 AVX2 build runs locally under macOS x86 emulation. The targeted MUL_MAT_ID suite covers 36 combinations of token counts, broadcast inputs, and odd block counts. Direct batched-versus-single dot tests cover 24 combinations and require bitwise equality. These checks do not establish EPYC throughput or HIP performance.

On Galactus, build the same revision with normal AVX2 CPU support and run:

```sh
cmake --build build --target test-quantize-fns test-backend-ops
build/bin/test-quantize-fns
build/bin/test-backend-ops test -b CPU -o MUL_MAT_ID -p 'type_a=mxfp4.*m=17'
build/bin/test-backend-ops perf -b CPU -o MUL_MAT_ID -p 'type_a=mxfp4'
```

Use the same CPU affinity and thread settings for comparisons. End-to-end DSpark benefits depend on whether its expert verification actually runs on the CPU. Enabling the existing VRAM expert cache can move those operations to the GPU, where this CPU change does not apply.

## Implemented: bounded pinned upload staging

Cache misses and uncached portions of large prefill expert transfers on discrete GPUs use the same two 8 MiB pinned host buffers when the MoE cache is enabled. A CPU copy fills one buffer while an earlier upload can consume the other. Each slot has a backend event; the host waits for that event before overwriting the slot. Destruction also drains outstanding uploads. The requested VRAM cache budget is unchanged; staging adds up to 16 MiB of host memory per active cache device, included in the memory breakdown.

Backends without asynchronous copies, pinned buffers, or events retain direct uploads. Failed allocations, including a backend's silent fallback to pageable memory, also retain the original path. Integrated GPUs skip staging. Weights already in the backend's pinned buffer type use direct uploads. Large-batch cache hits still use device-to-device copies; the staging path handles only the remaining host ranges, including the padding required by MMQ. With the MoE cache disabled, transfers retain the original behavior.

`LLAMA_MOE_CACHE_STAGING=0` disables staging for an A/B comparison with the same model, cache budget, and placement. A startup log reports the staging allocation when enabled. The allocation test uses delayed asynchronous copies to check buffer reuse, partial chunks, nonzero destination offsets, destruction with pending work, and fallback after allocation/event failure. This host-side protocol has been tested locally; HIP transfer throughput has not.

This overlaps host staging with uploads. It does not yet overlap CPU expert computation with GPU expert computation, or execute prefill experts as streaming groups arrive. The current graph still orders expert execution after its required uploads.

## Imported: host-weight pipeline parallelism

Roman Marchenko's [PR 29963](https://github.com/ggml-org/llama.cpp/pull/29963), pinned to `3ca7eed13274e406822fa1ba3e69a883560566bc`, supplies the implementation, imported as a seven-file diff against its upstream merge base. It shares a single device copy of host weights across pipeline slots, uploads user inputs before the splits, preserves worst-case graph reservations, assigns host-weight operations to their layer's device, and balances layer slices during automatic fitting.

This is inter-GPU, multi-ubatch prompt processing. It is not concurrent CPU/GPU expert decoding or Maya's within-layer streamed expert execution. Eligibility requires multiple devices, layer splitting, sufficient GPU-layer offload, asynchronous execution/events, and sufficiently distributed host weights. Short prompts offer little overlap; asymmetric GPU/PCIe systems can regress. The source discussion reports such regressions, so this remains experimental.

The integration resolves the overlapping routing-readback logic by using the imported pipeline policy during pipeline execution and retaining Paul Martin's existing density policy otherwise. The existing scheduler routing patch remains present. The imported code replaces the cache-disabled-only pipeline restriction; GPU testing must cover cache both enabled and disabled, successive prompts, and prompt/decode transitions.

Local validation: the ARM CPU library and common library compile, and all 33 allocator tests pass. These tests do not validate multiple HIP streams or establish a speedup. No AMD hardware benchmark has been run for this import.

## Candidate selection

| Source revision | Decision | Reason |
| --- | --- | --- |
| drrros PR 29963, `3ca7eed13274e406822fa1ba3e69a883560566bc` | Imported | Uses shared backend APIs and directly addresses multi-GPU host-weight prompt execution. |
| leloch/moe-cache-v2-pr, `e3096b046bb809f7f80bc47801f6579aed1cbc60` | Not imported | `moe-cache.cu` compiles only stubs for HIP/MUSA. Its CPU-miss/GPU-hit design is relevant, but requires an actual AMD port. |
| GenerelSchwerz/moe-cache, `f3f31a788f05d99f9404afe90f9de4baaa2f6e6f` | Not imported | The newer source executor still rejects HIP storage/host-alias paths. It is a substantially larger CUDA runtime replacement, not a drop-in AMD cache. |
| ap03906101/moe-hotcache, `beda4496714f03e74bded372402cba5224e29baf` | Rejected for this integration | Its README retracts corrupted-output measurements and documents unresolved large-prompt corruption. Its GPU reads from host RAM also do not provide the requested CPU/GPU compute overlap. |
| Maya v1.1.24 and Gufo revisions above | Design references | Their model-specific runtimes cannot be substituted for llama.cpp's generic scheduler as standalone kernels. Existing Halo fusions overlap Gufo's work. |

The earlier CPU batching and upload staging remain provisional, separately tested changes. The pipeline import does not replace either mechanism. No hybrid CPU-miss/GPU-hit implementation has been enabled by this selection.

## Work still required

The larger Maya mechanisms remain unported: concurrent CPU and GPU expert execution, asynchronous hot-expert promotion, and pipelined prefill expert staging. These need explicit buffer-lifetime and synchronization handling in the shared scheduler. The CPU batching change alone does not reproduce Maya's measured end-to-end speedup.

The Gufo review found that halo-dev already has normalization/RoPE/KV-write fusion and fused activation quantization. Importing equivalent kernels would duplicate existing functionality. Further Gufo changes require a demonstrated missing path and HIP correctness checks before becoming defaults.
