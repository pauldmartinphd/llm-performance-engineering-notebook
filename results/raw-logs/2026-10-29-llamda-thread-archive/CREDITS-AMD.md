# AMD source credits

This fork builds on ggml-org/llama.cpp and its contributors. Source authorship is preserved in the cherry-picked commits, along with the original revision references. Upstream history and licenses remain intact.

| Author and source | Original commit | Included change |
| --- | --- | --- |
| mxxm-t, [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) | [f6da2e696](https://github.com/mxxm-t/mx-llama.cpp/commit/f6da2e69660fe787e61ce5021162e58fd538b2ab) | Compile normalization channel/sample loops only for grids that exceed the launch limits. |
| Laurent Zuijdwijk, [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) | [db73f798f](https://github.com/halo-box/strix-llama.cpp/commit/db73f798f2e5b3ee2e1eba26e2eac79a246e8551) | Keep small UMA readbacks direct while retaining the staged path for large uncached reads. |
| Edwin Brown, [llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2) | [d5295d36d](https://github.com/edwinbrowwn/llama.cpp-rdna2/commit/d5295d36d9b750d8b11a735c68b6c68a43c2c399) | Widen RDNA3.0 MXFP4 matrix-vector dot processing. |
| Edwin Brown, [llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2) | [f35976c38](https://github.com/edwinbrowwn/llama.cpp-rdna2/commit/f35976c38cb25e580b02572455dd875db1bde16f) | Widen RDNA3.0 Q8_0 matrix-vector dot processing. |

The normalization cherry-pick was adapted to retain upstream's newer vectorized RMSNorm dispatcher. The Q8_0 kernel was retained without importing its source fork's launcher documentation; the relevant measurements are linked in the audit.

The default branch also inherits upstream work for gfx900, RDNA3.5 MMQ, and BF16 fallback on devices without native BF16 acceleration. These are not represented as new work by this fork.

The previous combined AMD import was reverted in `9b229d705`. Its source references remain in Git history; the milpster Vega table and the Strix-Halo transpose/concat import are no longer in the current tree. Credits for the separate V620 model integration remain on `rdna-dev`.

## DeepSeek-V4.1 and DSpark

Daniel Han and Unsloth supply the shared DeepSeek-V4.1-Flash-1017 implementation through [PR 30817](https://github.com/ggml-org/llama.cpp/pull/30817), revision `b72050d03ae14bb3cb5138ecefab3f3f65157d31`, retained as an attributed cherry-pick. This replaces the older V4.1 implementation. Upstream llama.cpp supplies the general speculative runtime. See [source review and validation notes](docs/deepseek41.md).

## Multi-GPU MoE prefill distribution

Paul D. Martin, Ph.D., supplies the layer-keyed distribution with round-robin fallback and the routing-readback bypass, adapted from the [performance notebook patch](https://github.com/pauldmartinphd/llm-performance-engineering-notebook/blob/main/patches/prefill/README.md). See [port details and validation](docs/moe-prefill-offload.md). Both are enabled by default for eligible prefill operations; `GGML_MOE_PREFILL=0` disables them.

## Expert execution research

Project Maya (Peasant Smith, Niko1221, and contributors), version 1.1.24 at `75798d4280dcc282b2f2224417fe8fa11e26ea89`, supplies the design reference for reusing decoded MXFP4 expert weights across multiple routed tokens and for bounded, event-protected expert-transfer staging. The initial CPU implementation adapts this approach to llama.cpp's existing Q8_0 layout and dot-product arithmetic; it does not import Maya's activation layout or runtime. See [implementation and validation](docs/expert-execution.md).

Gufo contributors, revision `d4eaaad438a498bf9ba541afe3d655f70dd4196d`, supply an additional reference for grouped expert computation and Strix Halo fusion. This review is not a claim that Gufo's runtime or kernels have been imported.

Roman Marchenko (drrros) supplies the host-weight pipeline implementation from [upstream PR 29963](https://github.com/ggml-org/llama.cpp/pull/29963), revision `3ca7eed13274e406822fa1ba3e69a883560566bc`. The imported seven-file diff adds single-copy host weights and user inputs, pipeline graph reservation, layer-local offload, and balanced automatic fitting. The integration retains Paul Martin's existing non-pipeline routing/readback policy. This import is confined to `experimental/moe-execution`; it retains the source project's MIT license.
