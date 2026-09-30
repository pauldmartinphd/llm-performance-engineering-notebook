# Lab hardware

The llama.cpp performance measurements in this repository come from [Galactus](galactus/README.md), an AMD EPYC 7713 server with four Radeon Pro V620 GPUs, running llama.cpp with ROCm in an LXC container on Proxmox. The July investigation used 1 TB of DDR4-2933 memory (8 × 128 GB). The August measurements used 2 TB (8 × 256 GB), after replacement of a failing module. Corrected STREAM bandwidth was about 152 GB/s on the earlier population and 148–151 GB/s on the replacement population, a difference of about 2%.

[Magneto](magneto/README.md) now has [oMLX/Qwen3.8-27B measurements](../results/qwen-3.8-27b.md), including thermal observations on its 14-inch chassis. The other machines have no published inference benchmarks here yet.

| Machine | Configuration and records |
|---|---|
| [Galactus](galactus/README.md) | EPYC 7713, eight-channel DDR4, four Radeon Pro V620 GPUs; specifications, costs, and raw system captures |
| [Borg](borg/README.md) | Threadripper Pro 3995WX, eight-channel DDR4, Radeon AI PRO R9700 and Radeon Pro W6800 |
| Vision | Lenovo ThinkStation P620 with a Threadripper Pro processor and eight-channel DDR4; bought broken and under repair |
| [SilverSurfer](silversurfer/README.md) | HP ZBook Ultra G1a, Ryzen AI Max+ PRO 395, 128 GB unified LPDDR5X |
| [Magneto](magneto/README.md) | 14-inch MacBook Pro, Apple M2 Max, 38 GPU cores, 64 GB unified LPDDR5; oMLX results and thermal observations |

See the [results](../results/README.md) for measured performance and the [reproduction guide](../REPRODUCE.md) for applying the method to another machine.
