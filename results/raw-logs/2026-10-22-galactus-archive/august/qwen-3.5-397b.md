ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 337.43 GiB |   396.35 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |       249.61 ± 19.78 |
| qwen35moe 397B.A17B Q6_K       | 337.43 GiB |   396.35 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |          9.37 ± 0.16 |

build: 3653e6d6d (10326)
