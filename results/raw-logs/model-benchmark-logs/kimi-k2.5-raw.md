# Baseline
 llama-bench       -m /root/models/Kimi-K2.5-GGUF/UD-Q4_K_XL/Kimi-K2.5-UD-Q4_K_XL-00001-of-00013.gguf       -ngl 99       -nopo 1       -mmp 0       -t "$t"       -ctk q8_0       -ctv q8_0       -fa 1       -b 4096       -ub 4096       -ot "exps=CPU"       -p 512       -n 128       -r 3

## Thread Sweep

LOG="llama_bench_cpu_experts_thread_sweep_$(date +%Y%m%d_%H%M%S).log"

for t in 16 32 48 64 96 128; do
  {
    echo "============================================================"
    echo "THREADS=$t  TIME=$(date -Is)"
    echo "============================================================"
	
	llama-bench -m /root/models/Kimi-K2.5-GGUF/UD-Q4_K_XL/Kimi-K2.5-UD-Q4_K_XL-00001-of-00013.gguf   -ngl 99   -nopo 1   -mmp 0   -t "$t"   -ctk q8_0   -ctv q8_0   -fa 1   -b 4096   -ub 4096   -ot "exps=CPU"   -p 512   -n 128   -r 3
	
	echo
  } 2>&1 | tee -a "$LOG"
done

echo "Saved log to: $LOG"

============================================================
THREADS=16  TIME=2026-04-07T10:26:36-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         17.74 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          6.58 ± 0.02 |

build: 58190cc84 (8671)

============================================================
THREADS=32  TIME=2026-04-07T10:54:14-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         31.36 ± 0.06 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          6.76 ± 0.02 |

build: 58190cc84 (8671)

============================================================
THREADS=48  TIME=2026-04-07T11:18:36-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         38.52 ± 0.04 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          6.72 ± 0.02 |

build: 58190cc84 (8671)

============================================================
THREADS=64  TIME=2026-04-07T11:36:16-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         43.37 ± 0.01 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          6.68 ± 0.01 |

build: 58190cc84 (8671)

============================================================
THREADS=96  TIME=2026-04-07T11:54:58-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         44.41 ± 0.03 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          6.56 ± 0.02 |

build: 58190cc84 (8671)

============================================================
THREADS=128  TIME=2026-04-07T12:12:09-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         42.82 ± 2.44 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          3.25 ± 0.57 |

build: 58190cc84 (8671)

# ROCm

root@openwebui:~/llama.cpp# ./build/bin/llama-bench \
  -m /root/models/Kimi-K2.5-GGUF/UD-Q4_K_XL/Kimi-K2.5-UD-Q4_K_XL-00001-of-00013.gguf \
  -ngl 99 -nopo 1 -mmp 0 \
  -t 32 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 \
  -ot "exps=CPU" \
  -p 512 -n 128 -r 3
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         31.31 ± 0.08 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          6.74 ± 0.03 |

build: 58190cc84 (8671)

# Vulkan
root@openwebui:~/llama.cpp# GGML_VK_FORCE_MAX_ALLOCATION_SIZE=6442450944 \
GGML_VK_FORCE_MAX_BUFFER_SIZE=6442450944 \
./build/bin/llama-bench \
  --device Vulkan0/Vulkan1/Vulkan2/Vulkan3 \
  -m /root/models/Kimi-K2.5-GGUF/UD-Q4_K_XL/Kimi-K2.5-UD-Q4_K_XL-00001-of-00013.gguf \
  -ngl 99 -nopo 1 -mmp 0 \
  -t 32 -ctk q8_0 -ctv q8_0 -fa 1 -b 4096 -ub 4096 \
  -ot "exps=CPU" \
  -p 512 -n 128 -r 3
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 1 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 2 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
ggml_vulkan: 3 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | dev          | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | ------------ | --------------------- | ---: | ---: | --------------: | -------------------: |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | Vulkan0/Vulkan1/Vulkan2/Vulkan3 | exps=CPU              |    0 |    1 |           pp512 |         29.81 ± 0.02 |
| deepseek2 671B Q4_K - Medium   | 579.28 GiB |  1026.41 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | Vulkan0/Vulkan1/Vulkan2/Vulkan3 | exps=CPU              |    0 |    1 |           tg128 |          5.44 ± 0.02 |

build: 58190cc84 (8671)