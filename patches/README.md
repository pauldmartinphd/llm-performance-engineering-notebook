# Patches

Patches to llama.cpp that I developed as a result of the experiments in this repo. Each patch lives in its own subdirectory with the exact edits, how to apply them, and the before/after measurements that justify it.

| Patch | What it does | Measured effect | Status |
|---|---|---|---|
| [prefill/](prefill/) | Distributes offloaded MoE expert matmuls across all GPUs (layer-keyed) and skips the expert-ids readback at prefill batch sizes, removing a per-split serializing synchronize | +13.7% prefill on GLM-5.2 (104.97 → 119.36 t/s pp8192); decode unchanged | Being submitted upstream to llama.cpp |

The path from experiment to patch is documented end to end: the split-histogram measurement that found the bottleneck ([Session 7](../results/lab-notebook/07-session-7-multi-card-placement-and-split-histogram.md)), the pre-registered null result for distribution alone and the identification of the true serializer ([Session 9](../results/lab-notebook/09-session-9-building-and-benchmarking-the-patch.md)), and the general principle it yielded ([general-principles](../takeaways/general-principles.md)). Code here is MIT — see [LICENSING](../LICENSING.md).
