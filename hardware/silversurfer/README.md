# SilverSurfer — platform

SilverSurfer is the integrated-GPU workstation laptop in the fleet, alongside [Galactus](../galactus/README.md), [Borg](../borg/README.md), and [Magneto](../magneto/README.md). Like Magneto, it is a unified-memory machine, with one LPDDR5X pool for the CPU and a large integrated GPU, but on the x86 and ROCm side, which makes it the closest laptop analogue to the Galactus software stack. The [methodology](../../results/methodology.md) applies here too, starting with the memory-bandwidth limit (§1). The [November 2 Gufo record](../../results/lab-notebook/2026-11-02-silversurfer-gufo.md) now supplies the first retained model timing; STREAM bandwidth remains unmeasured.

## Compute

The machine is an HP ZBook Ultra G1a, 14-inch, with an AMD Ryzen AI Max+ PRO 395 ("Strix Halo"): 16 Zen 5 cores and 32 threads.

## Memory

The memory is 128 GB of LPDDR5X-8000, unified and shared by the CPU and the integrated GPU.

The theoretical bandwidth is 256 GB/s, computed from the 256-bit LPDDR5X-8000 memory bus (32 bytes × 8000 MT/s), which gives the Strix Halo specification. This is a calculation; I have not measured STREAM bandwidth. Since the memory type and controller differ from Galactus, its efficiency ratio does not establish the achievable bandwidth here. I have not measured either the bandwidth or the ratio on SilverSurfer.

The firmware assigns a fixed slice of the 128 GB to the GPU, selectable in the BIOS at 512 MB, 4 GB, 8 GB, and up to 96 GB (75% of RAM); the remainder stays visible to the CPU. That reservation reduces the CPU-visible pool, but it is not a hard ceiling on GPU-accessible memory: the November Gufo run used the shared-memory allocation path and reported 99,994 MiB of GPU usage. The initial 32 GiB reservation and approximately 47 GiB GTT limit were captured; raising the kernel limits alone still left loading unsuccessful. A subsequent load succeeded after advice to reduce the reservation, but the final BIOS value was not read back. See the experiment record for the exact diagnostics and limits on the causal conclusion.

## GPU

The integrated GPU is an AMD Radeon 8060S (RDNA 3.5, 40 compute units), and it is ROCm-capable. It uses the same vendor stack as Galactus's V620s, so I expect most of the Galactus tooling (llama.cpp with ROCm, and the `GGML_SCHED_DEBUG` split-histogram check) to apply. Gufo 0.28.1 identified this unit as **gfx1151**. VMM and llama.cpp split-buffer behavior remain unconfirmed. Galactus's gfx1030 could not do `-sm row`; this part may differ.

## Storage

The NVMe SSD is 8 TB.

## Status and LLM duty

On November 2, Gufo 0.28.1 on openSUSE Aeon loaded Qwen3.8-Flash-Next UD-Q4_K_XL with the shared Q8_0 MTP predictor, one session, and a configured 262,144-token context. A retained API timing event reports **318 generated tokens at 29.85 tokens/s on battery**, with 63 prompt tokens processed at 112.32 tokens/s and zero cache hits. The short prompt does not measure sustained prefill.

The later Jan conversation summary records **539–740 tokens/s prefill** at total contexts from 5,563 to 54,781 tokens, with prefix reuse. Its original log attachment is missing from the export, so those rows remain summary-only. The [full record](../../results/lab-notebook/2026-11-02-silversurfer-gufo.md) retains the working command, loading failures, memory observations, and evidence distinctions. There is no controlled Maya comparison, AC-power repetition, full-262K test, or STREAM efficiency measurement in this session.

## Open items

- Measure the STREAM baseline with the RFO correction, per methodology §1.
- Read back the final BIOS reservation and effective memory limits; the successful run did not capture those values.
- Test whether the llama.cpp `-sm` and VMM limits match Galactus's gfx1030; the GPU target is now confirmed as gfx1151.
- Add controlled long-prompt and MTP-off measurements, recover the Jan attachment, and measure `C` and bytes per token for the two-term model.
