# Galactus build record — March 23, 2026

**Source period:** March 2026. **Added to this repository:** September 30, 2026.

This entry recovers the early Galactus configuration from my [build notebook](../reference-notes/2026-03-23-galactus-build.md). The retained source is the September 30 edited version of that dated note, rather than a pristine March export or a raw benchmark capture. Its commands and trial tables are useful reproducibility material, but its displayed rates have no repetition counts or statistical spreads. I therefore keep them separate from the later [August common baseline](12-common-baseline-2tb.md).

## The initial installation

The March system used an EPYC 7713 and four V620 GPUs under Proxmox VE 9.1.1, kernel `6.17.2-1-pve`, with BTRFS on NVMe. Inference ran in an unprivileged Debian 13 LXC container, initially configured for 128 logical CPUs, a 770 GB memory ceiling, and 2 TB of disk. llama.cpp was build `b8477-ec2b787eb`, with ROCm 7.2.0 and the HIP backend targeting `gfx1030`. Open WebUI used port 8080 and the local llama-server endpoint on port 8081.

These conditions differ from the July investigation's ZFS/SATA storage, later binary, and unrestricted container. The shared machine name does not make the two software environments one continuous benchmark recipe.

The setup script exposed the render devices but omitted `/dev/kfd`. I added the bind mount recorded in the source and used a world-accessible device permission as a workaround. That solved the installation's access problem while granting every host user access to the device; it is a historical setup choice, rather than a general permission recommendation.

Debian's system clang could not read the installed ROCm device bitcode. Building with ROCm's bundled compiler resolved the mismatch. The source retains the CMake command, install command, and hostname-migration incident. Nothing in this import reruns those commands or changes a machine configuration.

## Memory population and measured rates

The early mixed population was 4 × 128 GB plus 4 × 64 GB. Its large STREAM test reported about 30 GB/s. Replacing it with 8 × 128 GB matched DDR4-2933 DIMMs produced 141 GB/s Copy and 107 GB/s Triad in the March note. These are recorded STREAM rates; they are not the later July RFO-adjusted 152 GB/s reference.

| March Qwen 3.5 397B configuration | Recorded generation rate |
|---|---:|
| Q6_K_L, CPU experts, mixed DIMMs | 4.4 tokens/s |
| Q6_K_L, 16 expert layers placed on GPUs, mixed DIMMs | 5.5 tokens/s |
| UD-Q4_K_XL, 24 expert layers placed on GPUs, mixed DIMMs | 8.5 tokens/s |
| UD-Q4_K_XL, same recorded 24-layer placement, matched DIMMs | 12.4–12.7 tokens/s |

The placement comparison raised Q6 generation from 4.4 to 5.5 tokens/s. The next comparison changed both quantization and the amount of expert residency, so it does not isolate a quantization gain. The DIMM replacement then improved generation at the same recorded Q4 placement, supporting a material memory-system constraint. It does not establish the original explanation that particular address ranges were operating in single-channel mode; the note contains no controller mapping evidence that would distinguish that mechanism from other causes.

The production command allocated a 16K context and reported 17.4 tokens/s prompt processing. Context capacity is not measured prompt length, so this figure is not a `pp16384` benchmark. Its workload and warm/cold cache conditions are not sufficiently specified for normalization against the later prompt tests.

## Placement and unsuccessful experiments

The March production command assigned six layers' expert tensors to each GPU, with the specific patterns preceding the `exps=CPU` fallback. The [complete source command](../reference-notes/2026-03-23-galactus-build.md#production-configuration-recorded-in-march) preserves the patterns, model path, sampling settings, and server flags. An eight-layer allocation on ROCm0 failed with OOM; aggregate capacity did not establish headroom on that card.

The note also records the Kimi K2.5 thread and placement trials, including a best displayed rate of 4.0 tokens/s with CPU experts on the mixed DIMMs. This result cannot be compared directly with the later Kimi releases or treated as the machine's limit after the memory change.

No gain was visible at the reported precision after `--no-repack`. Polling, the tested flash-attention settings, and the 9B speculative draft did not produce a recorded improvement. The latter lacks acceptance and verification timing, leaving its failure mechanism unresolved. Other attempts failed to load or were unavailable in the tested build, including a KV-cache quantization segfault and the attempted ik_llama.cpp build. These are March observations about particular implementations, not permanent backend limitations.

The unresolved items at the end of the source remain part of that historical record. They do not become a new task list or supersede later completed experiments. This entry adds the build's provenance and early measurements without changing the current common baseline.
