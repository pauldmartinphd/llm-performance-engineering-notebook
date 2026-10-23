ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| glm-dsa 744B.A40B Q4_K - Medium | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |         96.17 ± 3.49 |
| glm-dsa 744B.A40B Q4_K - Medium | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |          5.21 ± 0.03 |

build: 3653e6d6d (10326)
