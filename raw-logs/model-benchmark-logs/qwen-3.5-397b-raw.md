# Baseline
llama-bench       -m /root/models/Qwen3.5-397B-A17B-GGUF/Qwen_Qwen3.5-397B-A17B-Q6_K_L/Qwen_Qwen3.5-397B-A17B-Q6_K_L-00001-of-00009.gguf       -ngl 99       -nopo 1       -mmp 0       -t "$t"       -ctk q8_0       -ctv q8_0       -fa 1       -b 4096       -ub 4096       -ot "exps=CPU"       -p 512       -n 128       -r 3

## Thread Sweep

LOG="llama_bench_cpu_experts_thread_sweep_$(date +%Y%m%d_%H%M%S).log"

for t in 16 32 48 64 96 128; do
  {
    echo "============================================================"
    echo "THREADS=$t  TIME=$(date -Is)"
    echo "============================================================"
	
	llama-bench       -m /root/models/Qwen3.5-397B-A17B-GGUF/Qwen_Qwen3.5-397B-A17B-Q6_K_L/Qwen_Qwen3.5-397B-A17B-Q6_K_L-00001-of-00009.gguf       -ngl 99       -nopo 1       -mmp 0       -t "$t"       -ctk q8_0       -ctv q8_0       -fa 1       -b 4096       -ub 4096       -ot "exps=CPU"       -p 512       -n 128       -r 3
	
	echo
  } 2>&1 | tee -a "$LOG"
done

echo "Saved log to: $LOG"

============================================================
THREADS=16  TIME=2026-04-07T19:47:50-04:00
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
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         36.28 ± 0.01 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          9.46 ± 0.04 |

build: 58190cc84 (8671)

============================================================
THREADS=32  TIME=2026-04-07T19:57:25-04:00
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
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         62.81 ± 0.05 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          9.69 ± 0.02 |

build: 58190cc84 (8671)

============================================================
THREADS=48  TIME=2026-04-07T20:07:46-04:00
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
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         76.05 ± 0.11 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          9.63 ± 0.01 |

build: 58190cc84 (8671)

============================================================
THREADS=64  TIME=2026-04-07T20:17:06-04:00
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
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         84.64 ± 0.21 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          9.56 ± 0.02 |

build: 58190cc84 (8671)

============================================================
THREADS=96  TIME=2026-04-07T20:26:24-04:00
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
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         87.54 ± 0.41 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          9.42 ± 0.00 |

build: 58190cc84 (8671)

============================================================
THREADS=128  TIME=2026-04-07T20:36:23-04:00
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
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         85.61 ± 1.65 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          4.57 ± 0.64 |

build: 58190cc84 (8671)

##### **Result**: THREADS=64 best pp and tg balance

# ROCm

## Layer Offloading

### 4 Layers Per Card

======================================= ROCm System Management Interface =======================================

================================================= Concise Info =================================================

Device  Node  IDs              Temp    Power  Partitions          SCLK  MCLK   Fan  Perf  PwrCap  VRAM%  GPU%   

              _(DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)_                                               

================================================================================================================

0       4     0x73a1,   8042   28.0°C  8.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  78%    0%     

1       2     0x73a1,   48907  28.0°C  8.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  76%    0%     

2       3     0x73a1,   60616  28.0°C  6.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  73%    0%     

3       1     0x73a1,   20282  27.0°C  8.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  78%    0%       

================================================================================================================

============================================= End of ROCm SMI Log ==============================================

root@openwebui:~/llama.cpp# llama-bench \
  -m /root/models/Qwen3.5-397B-A17B-GGUF/Qwen_Qwen3.5-397B-A17B-Q6_K_L/Qwen_Qwen3.5-397B-A17B-Q6_K_L-00001-of-00009.gguf \
  -ngl 99 -nopo 1 -mmp 0 -t 64 \
  -ctk q8_0 -ctv q8_0 -fa 1 \
  -b 4096 -ub 4096 \
  -ot "blk\.([0-3])\.ffn_.*_exps\.=ROCm0;blk\.([4-7])\.ffn_.*_exps\.=ROCm1;blk\.(8|9|10|11)\.ffn_.*_exps\.=ROCm2;blk\.(12|13|14|15)\.ffn_.*_exps\.=ROCm3;exps=CPU" \
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
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-3])\.ffn_.*_exps\.=ROCm0;blk\.([4-7])\.ffn_.*_exps\.=ROCm1;blk\.(8|9|10|11)\.ffn_.*_exps\.=ROCm2;blk\.(12|13|14|15)\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        102.68 ± 0.11 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-3])\.ffn_.*_exps\.=ROCm0;blk\.([4-7])\.ffn_.*_exps\.=ROCm1;blk\.(8|9|10|11)\.ffn_.*_exps\.=ROCm2;blk\.(12|13|14|15)\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         10.81 ± 0.04 |

build: 58190cc84 (8671)

##### **Result**: 1 t/s speedup over no offload

### 5 Layers Per Card
======================================= ROCm System Management Interface =======================================

================================================= Concise Info =================================================

Device  Node  IDs              Temp    Power  Partitions          SCLK  MCLK   Fan  Perf  PwrCap  VRAM%  GPU%  

              _(DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)_                                               

================================================================================================================

0       4     0x73a1,   8042   28.0°C  8.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  94%    0%      

1       2     0x73a1,   48907  28.0°C  8.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  93%    0%       

2       3     0x73a1,   60616  27.0°C  6.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  89%    0%       

3       1     0x73a1,   20282  27.0°C  9.0W   N/A, N/A, 0         0Mhz  96Mhz  0%   auto  250.0W  96%    0%       

================================================================================================================

============================================= End of ROCm SMI Log ==============================================

root@openwebui:~/llama.cpp# llama-bench \
  -m /root/models/Qwen3.5-397B-A17B-GGUF/Qwen_Qwen3.5-397B-A17B-Q6_K_L/Qwen_Qwen3.5-397B-A17B-Q6_K_L-00001-of-00009.gguf \
  -ngl 99 -nopo 1 -mmp 0 -t 64 \
  -ctk q8_0 -ctv q8_0 -fa 1 \
  -b 4096 -ub 4096 \
  -ot "blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU" \
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
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        107.95 ± 0.49 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.10 ± 0.05 |

build: 58190cc84 (8671)

##### **Result**: 1 t/s speedup over 4 Layers per card

### Transparent hugepages
```
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag
cat /sys/kernel/mm/transparent_hugepage/enabled
echo 3 > /proc/sys/vm/drop_caches
echo always > /sys/kernel/mm/transparent_hugepage/defrag
echo 1 > /proc/sys/vm/compact_memory
```

root@openwebui:~# cat /sys/kernel/mm/transparent_hugepage/enabled 
[always] madvise never
root@openwebui:~# llama-bench \
  -m /root/models/Qwen3.5-397B-A17B-GGUF/Qwen_Qwen3.5-397B-A17B-Q6_K_L/Qwen_Qwen3.5-397B-A17B-Q6_K_L-00001-of-00009.gguf \
  -ngl 99 -nopo 1 -mmp 0 -t 64 \
  -ctk q8_0 -ctv q8_0 -fa 1 \
  -b 4096 -ub 4096 \
  -ot "blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU" \
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
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        107.61 ± 0.43 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm,Vulkan |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.14 ± 0.03 |

build: 58190cc84 (8671)

##### **Result**: No difference vs no THP

# 2933

============================================================
THREADS=16  TIME=2026-04-10T18:31:03-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |         51.14 ± 0.04 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.52 ± 0.01 |

build: 0893f50f2 (8746)

============================================================
THREADS=32  TIME=2026-04-10T18:42:12-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |         82.87 ± 0.15 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.67 ± 0.07 |

build: 0893f50f2 (8746)

============================================================
THREADS=48  TIME=2026-04-10T18:46:41-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |         97.35 ± 0.22 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.63 ± 0.03 |

build: 0893f50f2 (8746)

============================================================
THREADS=64  TIME=2026-04-10T18:50:28-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        107.21 ± 0.50 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.51 ± 0.04 |

build: 0893f50f2 (8746)

============================================================
THREADS=96  TIME=2026-04-10T18:54:13-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        110.23 ± 0.25 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         11.34 ± 0.02 |

build: 0893f50f2 (8746)

============================================================
THREADS=128  TIME=2026-04-10T18:57:59-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        104.75 ± 1.73 |
| qwen35moe 397B.A17B Q6_K       | 319.21 GiB |   396.35 B | ROCm       |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-4])\.ffn_.*_exps\.=ROCm0;blk\.([5-9])\.ffn_.*_exps\.=ROCm1;blk\.(1[0-4])\.ffn_.*_exps\.=ROCm2;blk\.(1[5-9])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |          5.78 ± 0.69 |

build: 0893f50f2 (8746)