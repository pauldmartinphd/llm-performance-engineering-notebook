# Borg — platform

Second machine in the fleet, alongside [Galactus](../galactus/README.md). Dual-role: LLM inference for agentic coding, and multi-era Windows support (native 3D audio/video from Win98 through 11) via period-correct GPU and sound hardware. The [methodology](../../docs/methodology.md) transfers as written; figures marked unmeasured have no benchmark on record yet.

## Compute

AMD Threadripper Pro 3995WX (Zen 2, 64C/128T, WRX80, 8-channel).

## Memory

**512 GB DDR4 ECC RDIMM, 8 channels, 2400 MT/s, Samsung.**

Bandwidth: **unmeasured.** Theoretical 8-ch DDR4-2400 = 153.6 GB/s; at Galactus-like efficiency (81% of theoretical) expect on the order of ~120–125 GB/s real — roughly 20% below Galactus's 152 GB/s, which sets proportionally lower CPU-MoE decode expectations for shared models. The STREAM sweep (spread binding, RFO-corrected, per methodology §1) is the prerequisite before any decode budget here is trusted.

## GPUs

**Modern (inference):** AMD Radeon AI PRO R9700 (32 GB, RDNA4 gfx1201) + AMD Radeon Pro W6800 (32 GB, RDNA2 gfx1030, same family as Galactus's V620s; VMM limitations presumed shared). 64 GB modern VRAM total, mixed-architecture ROCm pair.

**Vintage (era support):** NVIDIA Quadro K4200 (Kepler, 2014 era); NVIDIA Quadro FX 1300 (2004 era).

## Audio (era support)

Sound Blaster X-Fi Titanium (PCIe); Sound Blaster Audigy 2 NX (USB). Selected for native hardware 3D audio (EAX-class) across the Win98–11 span.

## Storage

6 × 8 TB HDD; 6 × 3.84 TB Micron 5100 (SATA SSD); 2 × 3.84 TB Crucial NVMe.

## LLM duty (throughput unmeasured)

- **Qwen3.8 27B** — agentic coding.
- **DeepSeek-V4-Flash-0731** — agentic coding. Shared with Galactus — see the [Galactus results](../../docs/results-and-takeaways/deepseek-v4-flash.md), which do **not** transfer: Borg has ~20% less memory bandwidth (estimated), 64 GB VRAM across two mixed-arch cards vs 120 GB across four.

## Open items

- STREAM baseline (prerequisite for everything).
- V4-Flash baseline, then whether DSpark (see [speculative decoding](../../docs/speculative-decoding.md)) reproduces the Galactus gain — the drafter (10.9 GB) fits the R9700; needs llama.cpp ≥ PR #25784.
- Record quants in use and placement flags.
