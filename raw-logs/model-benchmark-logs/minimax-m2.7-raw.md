# Baseline
llama-bench   -m /root/models/MiniMax-M2.7-UD-Q5_K_M/UD-Q5_K_M/MiniMax-M2.7-UD-Q5_K_M-00001-of-00005.gguf  -ngl 99   -nopo 1   -mmp 0   -t 64   -ctk q8_0   -ctv q8_0   -fa 1   -b 4096   -ub 4096   -ot "exps=CPU"   -p 512   -n 128   -r 3

## Thread Sweep

LOG="llama_bench_cpu_experts_thread_sweep_$(date +%Y%m%d_%H%M%S).log"

for t in 16 32 48 64 96 128; do
  {
    echo "============================================================"
    echo "THREADS=$t  TIME=$(date -Is)"
    echo "============================================================"
	
	llama-bench       -m /root/models/MiniMax-M2.7-UD-Q5_K_M/UD-Q5_K_M/MiniMax-M2.7-UD-Q5_K_M-00001-of-00005.gguf       -ngl 99       -nopo 1       -mmp 0       -t "$t"       -ctk q8_0       -ctv q8_0       -fa 1       -b 4096       -ub 4096       -ot "exps=CPU"       -p 512       -n 128       -r 3
	
	echo
  } 2>&1 | tee -a "$LOG"
done

echo "Saved log to: $LOG"

============================================================
THREADS=16  TIME=2026-04-17T09:13:56-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         39.48 ± 0.01 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      16 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |         14.39 ± 0.01 |

build: 0893f50f2 (8746)

============================================================
THREADS=32  TIME=2026-04-17T09:18:02-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         71.82 ± 0.06 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      32 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |         14.76 ± 0.01 |

build: 0893f50f2 (8746)

============================================================
THREADS=48  TIME=2026-04-17T09:22:02-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         88.67 ± 0.02 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      48 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |         14.60 ± 0.01 |

build: 0893f50f2 (8746)

============================================================
THREADS=64  TIME=2026-04-17T09:25:21-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |        100.82 ± 0.33 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |         14.42 ± 0.03 |

build: 0893f50f2 (8746)

============================================================
THREADS=96  TIME=2026-04-17T09:28:47-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |        101.71 ± 0.13 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |      96 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |         14.05 ± 0.03 |

build: 0893f50f2 (8746)

============================================================
THREADS=128  TIME=2026-04-17T09:32:18-04:00
============================================================
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           pp512 |         99.96 ± 1.16 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |     128 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | exps=CPU              |    0 |    1 |           tg128 |          8.96 ± 2.84 |

build: 0893f50f2 (8746)

# ROCm

## Layer Offloading

root@openwebui:~# llama-bench \
  -m /root/models/MiniMax-M2.7-UD-Q5_K_M/UD-Q5_K_M/MiniMax-M2.7-UD-Q5_K_M-00001-of-00005.gguf \
  -ngl 42 -nopo 1 -mmp 0 -t 64 \
  -ctk q8_0 -ctv q8_0 -fa 1 \
  -b 4096 -ub 4096 \
  -ot "blk\.([0-9]|10)\.ffn_.*_exps\.=ROCm0;blk\.(1[1-9]|2[0-1])\.ffn_.*_exps\.=ROCm1;blk\.(2[2-9]|3[0-1])\.ffn_.*_exps\.=ROCm2;blk\.(3[2-9]|4[0-1])\.ffn_.*_exps\.=ROCm3;exps=CPU" \
  -p 512 -n 128 -r 3
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa | ot                    | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------------- | ---: | ---: | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  42 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-9]|10)\.ffn_.*_exps\.=ROCm0;blk\.(1[1-9]|2[0-1])\.ffn_.*_exps\.=ROCm1;blk\.(2[2-9]|3[0-1])\.ffn_.*_exps\.=ROCm2;blk\.(3[2-9]|4[0-1])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           pp512 |        154.93 ± 4.76 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  42 |    4096 |     4096 |   q8_0 |   q8_0 |  1 | blk\.([0-9]|10)\.ffn_.*_exps\.=ROCm0;blk\.(1[1-9]|2[0-1])\.ffn_.*_exps\.=ROCm1;blk\.(2[2-9]|3[0-1])\.ffn_.*_exps\.=ROCm2;blk\.(3[2-9]|4[0-1])\.ffn_.*_exps\.=ROCm3;exps=CPU |    0 |    1 |           tg128 |         17.37 ± 0.04 |

build: 0893f50f2 (8746)