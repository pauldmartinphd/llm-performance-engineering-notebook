# Entry 14 — llAMDa.cpp: an AMD integration fork

**Recorded:** October 21, 2026. **Project:** [pauldmartinphd/llAMDa.cpp](https://github.com/pauldmartinphd/llAMDa.cpp). I have not benchmarked this revision on Galactus. The results below come from the earlier V620 build.

I started with a V620-specific llama.cpp fork to run newer large MoE models on Galactus. I expanded that work into llAMDa.cpp, an AMD-focused llama.cpp fork for RDNA, Strix Halo, and Vega. I wanted the changes that can run by default in a common branch, without requiring a collection of helper scripts to enable them. I kept architecture-specific development work in separate branches and preserved credits for the forks and authors whose changes I imported.

I made `amd-main` the default branch and added the Unsloth DeepSeek-V4.1/DSpark port and my MoE prefill changes to it. The `rdna-dev`, `halo-dev`, and `vega-dev` branches build on that common base. I still need to test their architecture-specific changes on the corresponding hardware.

## AMD patches and source credits

I imported normalization launch handling from [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), small UMA readback handling from [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp), and qualified RDNA3 Q8_0/MXFP4 paths from [Edwin Brown's RDNA fork](https://github.com/edwinbrowwn/llama.cpp-rdna2). The [source credits](https://github.com/pauldmartinphd/llAMDa.cpp/blob/9fcd346d4/CREDITS-AMD.md) identify the original revisions and adaptations. Upstream llama.cpp already supplies gfx900, RDNA3.5, and BF16 fallback work; I did not add those features.

For DeepSeek, I replaced the older port with Daniel Han and Unsloth's [PR 30817](https://github.com/ggml-org/llama.cpp/pull/30817), source revision `b72050d03ae14bb3cb5138ecefab3f3f65157d31`. I based the integration on upstream `9b56ccf00d755b0bf780140f6352a67db1ac0378`. Their implementation supports V4.1-Flash-1017 and its DSpark graph, sparse attention, associated tensor operations, and conversion support. I documented compatibility and validation in the [integration notes](https://github.com/pauldmartinphd/llAMDa.cpp/blob/9fcd346d4/docs/deepseek41.md). Since I changed the implementation, I need to rerun the model tests before comparing its performance with the earlier port.

## Adding my round-robin patch to all four branches

I found my prefill patch on `rdna-dev`, but not on the other three branches. In commit [`9fcd346d4`](https://github.com/pauldmartinphd/llAMDa.cpp/commit/9fcd346d4), I added both parts to `amd-main`: layer-based distribution across eligible GPU backends, with round-robin fallback for unnamed tensors, and the routing-index readback bypass for sufficiently large prefill batches. I then merged that commit into all three development branches.

| Branch | Published revision after promotion |
|---|---|
| amd-main | `9fcd346d4` |
| rdna-dev | `45ea2f058` |
| halo-dev | `42d49821d` |
| vega-dev | `f7394fdac` |

The current policy applies to eligible host-weight MoE operations with at least 32 tokens. Layer-based selection keeps gate, up, and down projections together when their eligible backend lists match. The copy bypass also requires at least eight routing selections per model expert; it copies the full expert tensor instead of synchronously reading the selected IDs. Under skewed routing this can transfer unused weights. `GGML_MOE_PREFILL=0` disables both changes for a controlled comparison. Neither change makes experts persist in VRAM, enables tensor parallelism, or promises a decode speedup.

I kept both edits because the July experiment showed that distribution alone did not solve the problem. In the [July GLM-5.2 comparison](09-session-9-building-and-benchmarking-the-patch.md), distribution alone changed pp8192 from 104.97 to 105.71 t/s, while distribution plus the readback bypass reached 119.36 t/s, a 13.7% gain. Decode stayed at about 5.51 t/s. Distributing the work more evenly did not improve throughput appreciably until I also removed the routing readback synchronization.

I built all four branches for the CPU backend and checked the actual selection and copy code with a test harness covering four-device selection, eligible subsets, projection affinity, fallback behavior, bypass boundaries, numerical equivalence under skewed routing, and the disable switch. Those checks passed, but they do not exercise HIP transfers. I still need to measure transfer overlap and throughput on Galactus.

## Kernel performance depends on where the work runs

The GPU patches can help only when the model uses their supported tensor types and shapes and actually executes those operations on the GPU. If the CPU computes a routed expert projection, tuning the GPU kernel for that projection will not make it faster. I need to establish where each configuration runs the work before attributing a throughput difference to the kernels.

The earlier V620 build also printed `b0-unknown`, which prevents me from tying those measurements to an exact source revision. Future comparisons need the source revision, model export, and full launch settings together.

I discuss the GLM measurements in [Entry 15](15-glm53-flash-placement.md), the DeepSeek tests in [Entry 16](16-deepseek41-dspark.md), and the remaining bandwidth questions in [Entry 17](17-bandwidth-and-placement.md).
