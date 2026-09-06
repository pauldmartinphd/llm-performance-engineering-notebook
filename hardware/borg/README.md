# Borg — platform

Borg is the second machine in the fleet, alongside [Galactus](../galactus/README.md). It has two roles: LLM inference for agentic coding, and multi-era Windows support, with native 3D audio and video from Windows 98 through 11 through period-correct GPU and sound hardware. The [methodology](../../results/methodology.md) transfers as written. Figures marked unmeasured have no benchmark on record yet.

## Compute

The CPU is an AMD Threadripper Pro 3995WX (Zen 2, 64C/128T, WRX80, 8-channel).

## Memory

The memory is 512 GB of DDR4 ECC RDIMM (8 channels, 2400 MT/s, Samsung).

The bandwidth is unmeasured. The theoretical 8-channel DDR4-2400 peak is 153.6 GB/s; at Galactus-like efficiency (81% of theoretical), expect roughly 120 to 125 GB/s in practice, about 20% below Galactus's 152 GB/s. That sets proportionally lower CPU-MoE decode expectations for the shared models. The STREAM sweep (spread binding, RFO-corrected, per methodology §1) is the prerequisite before any decode budget here can be trusted.

## GPUs

For inference, Borg has an AMD Radeon AI PRO R9700 (32 GB, RDNA4, gfx1201) and an AMD Radeon Pro W6800 (32 GB, RDNA2, gfx1030, the same family as Galactus's V620s, so the VMM limitations are presumed shared): 64 GB of modern VRAM in total, as a mixed-architecture ROCm pair.

For era support, it has an NVIDIA Quadro K4200 (Kepler, 2014) and an NVIDIA Quadro FX 1300 (2004).

## Audio (era support)

The audio hardware is a Sound Blaster X-Fi Titanium (PCIe) and a Sound Blaster Audigy 2 NX (USB), chosen for native hardware 3D audio (EAX-class) across the Windows 98 to 11 span.

## Storage

The storage is 6 × 8 TB HDD, 6 × 3.84 TB Micron 5100 (SATA SSD), and 2 × 3.84 TB Crucial NVMe.

## LLM duty (throughput unmeasured)

- Qwen3.8 27B, for agentic coding.
- DeepSeek-V4-Flash-0731, for agentic coding, shared with Galactus. The [Galactus results](../../results/deepseek-v4-flash.md) do not transfer: Borg has about 20% less memory bandwidth (estimated) and 64 GB of VRAM across two mixed-architecture cards, against Galactus's 120 GB across four.

## Open items

- Run the STREAM baseline; it is the prerequisite for everything.
- Run the V4-Flash baseline, then test whether DSpark (see [speculative decoding](../../takeaways/speculative-decoding.md)) reproduces the Galactus gain; the drafter (10.9 GB) fits the R9700, and it needs llama.cpp ≥ PR #25784.
- Record the quants in use and the placement flags.
