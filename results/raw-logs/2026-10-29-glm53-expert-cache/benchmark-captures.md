# GLM-5.3-Flash expert-cache benchmark captures

Extracted October 29, 2026 from conversation `6adace95-5744-83e9-8348-9e84b6b03966`. Run dates are not printed. These are the two supplied terminal captures; discussion, hypotheses, diagnostic proposals, and the untested patch are consolidated in the [dated notebook](../../lab-notebook/2026-10-29-glm53-expert-cache.md). No new benchmark was run.

## Incomplete tg128 attempt

```text
❯ llamda-bench -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf -ngl 99 -ot "exps=CPU" --moe-cache-mib 81920 -fa on -t 32 -b 8192 -ub 8192 -p 8192 -n 128 -r 2 -o md
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | moe_cache_mib | threads | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ------: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
```

## Completed pp8192/tg256 benchmark

```text
❯ llamda-bench -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf -ngl 99 -ot "exps=CPU" --moe-cache-mib 81920 -fa on -t 32 -b 8192 -ub 8192 -p 8192 -n 256 -r 2 -o md
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | moe_cache_mib | threads | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ------: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| glm5-next 320B.A18B Q4_K - Medium | 185.98 GiB |   320.76 B | ROCm       |  99 |         81920 |      32 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |       279.51 ± 15.37 |
| glm5-next 320B.A18B Q4_K - Medium | 185.98 GiB |   320.76 B | ROCm       |  99 |         81920 |      32 |    8192 |     8192 |   1 | exps=CPU              |           tg256 |          8.63 ± 2.84 |

build: 754aa4151 (11811)
```
