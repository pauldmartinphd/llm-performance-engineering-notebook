## Session 4 — Monday, July 14, 2026, 11:08–14:25 — The v3 diagnostic run (machine log)

This session is the machine's log, not the dialogue: the 16-phase battery `galactus-diag.sh` ran unattended on Galactus from 11:08:13 to 14:25:10 ET (total runtime 3h16m) while the conversation of Sessions 3 and 5 continued in parallel. Run directory `/root/diag-20260714-110813`; MAIN.log (delivered to the dialogue as results.txt, 175,463 lines) is the source for everything below. Build under test: `657e01125 (10001)`. Hang guard: 10800 s per invocation; deadline: none. Wall-clock times below are mapped from the log's elapsed markers against the 11:08 start.

**Artifacts:** the script `galactus-diag.sh` (756 lines) was saved at 11:05:36 ET; `results.txt` (the copy of MAIN.log) was saved at 14:33:44 ET.

### Session preamble — the BRIEFING block (log lines 1–184, condensed)

The script opens with a self-contained briefing ("If you are a language model reading this log with no prior context, this section is the entire background. Everything below it is evidence.").

- **The machine ("Galactus").** CPU: AMD EPYC 7713, Zen3/Milan, 64 cores / 128 threads, single socket, 8 CCDs × 8 cores, 32 MB L3 per CCD, AVX2 + FMA, no AVX-512, no VNNI. RAM: 1 TB DDR4, 8 channels, NPS1 (single NUMA node), verified. GPU: 4× Radeon Pro V620 (gfx1030/RDNA2), 30704 MiB each = 122816 MiB, PCIe 4.0 x16 each, GTT pool 497.8 GiB (half of RAM, the amdgpu default) — "REMEMBER THIS." Storage: local ZFS on SATA SSD array, dd direct 1.5 GB/s, buffered 2.1 GB/s, not a bottleneck. Host: Proxmox 7.0.6-2-pve; llama.cpp in LXC "openwebui" with full capabilities and no cgroup limits (cpu.max=max, cpuset=0-127, memory.max=max, nr_throttled=0). Topology verified from `/sys/.../cache/index3/shared_cpu_list`: CCD0=CPUs 0–7, CCD1=8–15, CCD2=16–23, CCD3=24–31, CCD4=32–39, CCD5=40–47, CCD6=48–55, CCD7=56–63; CPUs 64–127 are SMT siblings (cpu0 siblings = "0,64") — the `-C` masks are correct.
- **Memory bandwidth — settled.** STREAM, corrected for the RFO artifact (GCC turns Copy into memcpy with non-temporal stores, which pays no read-for-ownership; Scale/Add/Triad use ordinary vector stores and do pay it — multiply Scale ×1.5, Add/Triad ×1.333; proof: if Copy paid RFO its real traffic would be 227.7 GB/s, above the 204.8 GB/s theoretical ceiling):

```
threads:   8     16     32     48     64     96    128
GB/s:    ~150   152    150    143    145    140    141     <- all four kernels agree
```

  The platform delivers ~152 GB/s (74% of theoretical for 8-channel DDR4-3200) and saturates at 16 threads. Packed placement (OMP_PROC_BIND=close) at 16 threads collapses to ~90 GB/s Copy / 68 GB/s Triad — CCD spread is worth ~2×; each Milan CCD reaches the IOD over one GMI2 link, and 3–4+ CCDs must be active to saturate DRAM. Container STREAM equals host STREAM within 1%.
- **The model — measured, not estimated.** GLM-5.2, Unsloth UD-Q4_K_XL, 11 shards, 435.19 GiB, 753.86 B params; llama.cpp arch `glm-dsa`. 78 layers (blk.0–blk.77); first_k_dense_replace=3, so blk.3–blk.77 are the 75 MoE layers; 256 routed experts, 8 active, 1 shared expert; MLA attention (kv_lora_rank=512, q_lora_rank=2048); 1M context. blk.78 = MTP/NextN, flagged TENSOR_SKIP, never allocated — `--spec-type draft-mtp` cannot work for glm-dsa. `-sm tensor` is not supported for glm-dsa (throws); `-sm row` is allowed. Expert tensor types verified by gguf_dump and load log: gate 1728 MiB Q4_K, up 1728 MiB Q4_K (both CPU_REPACK-capable), down 2112 MiB Q5_K (not repackable); exceptions blk.8 (Q5_K gate/up + Q6_K down) and blk.75/76/77 (Q6_K down). CPU_REPACK (AVX2 q4_K_8x8_q8_K) covers 3456 of 5568 MiB per layer = 62% of expert bytes. Measured buffers at `-ngl 99 -ot exps=CPU`: ROCm0 4987.08, ROCm1 4431.34, ROCm2 4431.34, ROCm3 4952.45 MiB (18.36 GiB on GPU); ROCm_Host 420964.22 MiB (411 GiB of routed experts). Decode reads per token: 13,125 MiB = 12.82 GiB = 13.77 GB from DDR4.
- **The budget — and the hole.** Measured hybrid (`-ngl 99 -ot exps=CPU -mmp 0 -t 64 -nopo 1`, build 9942): pp512 = 37.6 t/s, tg128 = 5.15 t/s (= 201 ms/token). CPU 13.77 GB @ 152 GB/s → 91 ms; GPU ~18.7 GB dense path read serially one card at a time under `-sm layer` at ~400 GB/s effective → ~47 ms; ideal ~138 ms (7.2 t/s); actual 201 ms (5.0 t/s); unaccounted ~63 ms (31%). Graph splits = 155 per token (verified via GGML_SCHED_DEBUG=1), 6063 nodes; boundary payload at tg 24 KB — pure latency (~155 stream syncs + ~76 threadpool fork/joins per token); 63 ms / 155 splits = 0.4 ms per split, "exactly what a sleeping 64-thread pool costs to wake." **"FINDING THAT 63 ms IS THE POINT OF THIS SCRIPT."**
- **Why every v2 phase crashed.** Cause 1 — ZenDNN: registers as an ACCEL backend; rejects the routed experts (256 > its 32-expert cap; Q4_K unsupported) but accepts Q8_0 MUL_MAT, and all 872 Q8_0 attention tensors are CPU-resident at `-ngl 0` → splits 1088 (bs=512, shredded) versus 155 (`-ngl 99`); every crashed run printed the LIBXSMM banner; B0 survived only because `-n 1 --no-warmup` never built a 512-token graph. Rebuilt with `-DGGML_ZENDNN=OFF`; Phase 0.9 verifies. Cause 2 — the 411 GiB pinned host buffer: with `-mmp 0`, `make_cpu_buft_list()` puts the GPU's pinned host buffer ahead of CPU_REPACK, so llama.cpp attempts `hipHostMalloc(420964 MiB)` — 14-minute loads (only ~5 min of it I/O), AnonHugePages stuck at 0 kB (the huge-page hypothesis was never actually tested), and after one run took and released 411 GiB the next failed and fell back to a plain CPU buffer. With `-nopo 1` the pin buys nothing. v3 therefore uses `-mmp 1` almost everywhere ("this is why v3 can afford ~15 invocations where v2 could only afford 14").
- **The three model configurations** (a reload happens only when `-m -ngl -ncmoe -sm -mg -ts -mmp -dio -dev --no-host -ot` change; everything else — `-t -C --cpu-strict --poll -b -ub -p -n -d -fa -nopo -ctk -ctv` — sweeps free inside one load): M1 = `-mmp 1` (CPU_Mapped, no pin, no THP, no repack, loads in seconds — the iteration workhorse); M2 = `-mmp 0 --no-host 1` (CPU_REPACK 62% + plain CPU, anonymous memory so THP-eligible, no pin — the production candidate); M3 = `-mmp 0` default (ROCm_Host, 411 GiB pinned, slow load — the only config with fast H2D for op_offload). op_offload fires from any host buffer (the scheduler checks `ggml_backend_buffer_is_host`), so it works under M1 too, just with slower bounce-buffered H2D; M3 exists to measure that cost.
- **`-C`/`--cpu-strict` semantics** (verified in `ggml_thread_cpumask_next`): strict 0 (default) gives every thread the full mask and lets the kernel migrate freely; strict 1 pins thread i to the i-th set bit in ascending CPU order. One mask therefore means different things at different `-t`:

```
mask                    -t 16                 -t 32
0303030303030303   ->   8 CCDs, 2 thr/CCD     (WRAPS: IGNORE THAT ROW)
0f0f0f0f0f0f0f0f   ->   4 CCDs, 4 thr/CCD     8 CCDs, 4 thr/CCD
00000000ffffffff   ->   2 CCDs, 8 thr/CCD     4 CCDs, 8 thr/CCD
```

- **Settled facts — do not re-investigate:** PCIe 16.0 GT/s x16 all four; storage 1.5–2.1 GB/s; container unconstrained; NUMA NPS1; `GGML_CUDA_REGISTER_HOST` exists but `ggml_backend_cuda_register_host_buffer()` is exported-but-never-called (does nothing); GPU P2P is opt-in via `GGML_CUDA_P2P` and irrelevant here (only three 24 KB tensors cross GPUs per token under `-sm layer`) — deliberately not tested; HIP graphs are compiled in (GGML_HIP_GRAPHS defaults ON) and the "cc < AMPERE" gate does not fire on AMD, so kernel-launch overhead is not the explanation.
- **How to read the results:** 1. Phase 0.9 — is ZenDNN gone? 2. Phase 0.7 — IOMMU mode, want "identity". 3. Phase A — does anything run at all (A1 plain, A2 with masks). 4. Phase B — the main event, `-t` × `--poll`; if `--poll 100` produces a step change, the 63 ms is threadpool sleep/wake and the fix is one flag. 5. Phase C — CCD spread at fixed thread count (STREAM says worth 2×). 6. Phase D — `-ngl 0` denominator, ~27+ GB/token, zero splits; D versus B at matched `-t` is the split overhead. 7. Phases F/G — op_offload; prefill is CPU-compute-bound at 1.69 TFLOP/s; on the GPUs it becomes PCIe-bound, ~410 GiB per graph eval over 4× PCIe4 x16, ~4 s per ubatch regardless of size, so pp scales ~linearly with `-ub`; ceilings ub 512 → ~125 t/s, ub 2048 → ~500 t/s, now 37 t/s. 8. The M2 check — does `CPU_REPACK model buffer size` appear, does AnonHugePages climb. 9. The VRAM fill — ~104 GiB idle, expected only ~1.25×, lowest priority. (The briefing's "PHASE I" and "PHASE M" labels for these last two lag the final script — in the run they are Phase L and Phase J respectively.) Reference numbers: decode reads 13.77 GB/token (hybrid) or ~27 GB/token (`-ngl 0`); 152 GB/s → 91 ms hybrid CPU part; current 201 ms/token = 4.97 t/s, 155 splits, ~63 ms unaccounted; current prefill 37.6 t/s = 1.69 TFLOP/s.

```
RUN STARTED : 2026-07-14T11:08:13-04:00
HOSTNAME    : openwebui
OUTDIR      : /root/diag-20260714-110813
MODEL       : /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf
DEADLINE    : NONE.
HANG GUARD  : 10800s per invocation (0 = off).
```

### ~11:08 — Phase 0: system inventory (pre-flight)

Purpose: verify the just-rebooted, just-rebuilt system before spending any benchmark time. Every command and its result:

- Kernel/OS/container: `uname -a` → `Linux openwebui 7.0.14-4-pve #1 SMP PREEMPT_DYNAMIC PMX 7.0.14-4 (2026-07-07T07:27Z) x86_64`; `/etc/os-release` → Debian GNU/Linux 13 (trixie) userland; `cat /proc/cmdline` → `initrd=\EFI\proxmox\7.0.14-4-pve\initrd.img-7.0.14-4-pve root=UUID=f23a25fb-9be5-4b2e-9a3e-e5a5787c9754 amd_iommu=on iommu=pt` — the just-added IOMMU flags are present; `uptime` → up 5 min (load 0.12/0.16/0.09, fresh boot); `systemd-detect-virt` → lxc; cgroups: `cpu.max` = `max 100000`, `cpuset.cpus.effective` = `0-127`, `memory.max` = `max`, `nr_throttled 0` / `throttled_usec 0`; `ulimit -l` → 8192 (irrelevant to hipHostMalloc, which pins via GTT).
- CPU topology: lscpu → AMD EPYC 7713 64-Core, 2 threads/core, 1 socket, scaling 83%, max 3720.7029 MHz, min 1500.0000 MHz, L3 256 MiB (8 instances), 1 NUMA node (CPUs 0–127); `/proc/cpuinfo` flags include `fma avx bmi1 avx2 bmi2 sha_ni`, no avx512f; `thread_siblings_list` for cpu0 → `0,64`; the index3 shared_cpu_list walk confirms the eight-CCD map (0-7,64-71 / 8-15,72-79 / 16-23,80-87 / 24-31,88-95 / 32-39,96-103 / 40-47,104-111 / 48-55,112-119 / 56-63,120-127) — the `-C` masks are correct.
- Clocks/idle: 128× `performance` governor; boost = 1; cpuidle POLL/C1/C2 all `disable=1`; average 3099 MHz across 128 CPUs.
- Memory/THP: `free -h` → 995Gi total, 860Mi used, 994Gi free, 391Mi buff/cache, swap 0B; THP enabled = `[always] madvise never`, defrag = `always defer [defer+madvise] madvise never`; `AnonHugePages: 0 kB` baseline, `Hugepagesize: 2048 kB`; swap totals 0 kB; dmidecode unavailable in the container ("STREAM already confirmed 8ch @ full speed").
- NUMA: `numactl --hardware` → 1 node, cpus 0–127, node 0 size 1019408 MB, free 1010025 MB.
- GPUs: 4× `AMD RADEON PRO V620 Azure`, GFX Version gfx1030; per GPU `VRAM Total Memory (B): 32195477504`, VIS_VRAM the same, `GTT Total Memory (B): 534463733760` (497.8 GiB pool confirmed); `/sys/module/amdgpu/parameters/gttsize` unreadable in the container (`[exit=1]`); `pcie clock level: 1 (16.0GT/s x16)` on all four; topology link type all PCIE (no XGMI).
- IOMMU: `cat /sys/kernel/iommu_groups/*/type | sort | uniq -c` → **`96 identity`** — passthrough took; 96 groups; dmesg not readable in the container.
- Storage: `findmnt` → `/ galactus-datastore/subvol-100-disk-0 zfs rw,relatime,xattr,posixacl,casesensitive`; the model directory holds 11 shards — 00001 = 9,423,744 B; 00002 = 49,433,942,336 B; 00003–00010 = 48,566,415,136 B each; 00011 = 29,314,424,736 B (456,344,681 KB of blocks total).
- llama.cpp build — the ZenDNN check: `git log -1` → `657e01125aa49577a62a5531fde24cbcc007006d Tue Jul 14 13:15:41 2026 +0300 tests: export-graph-ops: exit gracefully when called w/o arguments (#25619)`. CMakeCache grep:

```
CMAKE_BUILD_TYPE:STRING=Release
GGML_CUDA_NO_PEER_COPY:BOOL=OFF
GGML_HIP:BOOL=ON
GGML_HIP_GRAPHS:BOOL=ON
GGML_NATIVE:BOOL=ON
GGML_ZENDNN:BOOL=ON        <- NOTE: the CMakeCache STILL SAYS ON
```

  `llama-bench --list-devices` lists only the four V620s (`ROCm0..ROCm3: AMD Radeon Pro V620 (30704 MiB, 30618 MiB free)`) — no ZenDNN device; `ldd $(which llama-bench)` links `libggml.so.0, libggml-base.so.0, libggml-cpu.so.0, libggml-hip.so.0` — libggml-zendnn not linked; `/usr/local/lib/libggml-cpu.so*` symlinks to `libggml-cpu.so.0.16.0` built Jul 14 08:28 (older 0.9.8/0.9.11/0.13.0/0.15.3 versions also present). The stale `GGML_ZENDNN:BOOL=ON` line is CMake cache text from an old build dir; the runtime evidence (`--list-devices`, `ldd`, and later Phase D's `graph splits = 1`) is unambiguous — ZenDNN is gone.
- Model layout: gguf_dump on shard 2 confirms `blk.3.ffn_down_exps.weight Q5_K (2048, 6144, 256)`, `blk.3.ffn_gate_exps.weight Q4_K (6144, 2048, 256)`, `blk.3.ffn_up_exps.weight Q4_K`; blk.4 the same pattern; blk.8 down Q6_K, gate/up Q5_K (the exception layer); each exps tensor 3,221,225,472 bytes.
- STREAM re-baseline (MB/s; the log reminds the reader to multiply Scale ×1.5 and Add/Triad ×1.333 for real DRAM traffic): spread — 8T Copy 147458.6 / Triad 117643.1; 16T 153472.7 / 114581.9; 32T 151818.7 / 111994.1; 64T 145415.0 / 109747.9; 128T 141346.7 / 107244.8. Packed — 16T 90185.1 / 68462.1 (~half, as expected); 32T 140080.4 / 100401.6.

### ~11:08 — Phase 1: background monitors (pre-flight)

turbostat is not installed (needs CAP_SYS_RAWIO + MSRs; would have to run on the host); a fallback MHz sampler writes to `monitors/cpumhz.txt`, and rocm-smi / vmstat / meminfo samplers write to `monitors/`. The log prints its own reminder verbatim:

```
*** AnonHugePages was 0 kB in EVERY v2 memsnap because of the 411 GiB
    pinned buffer. Watch it during PHASE I (M2, -mmp 0 --no-host 1).
    If it finally climbs, the THP hypothesis becomes testable at last. ***
```

(The monitors write outside MAIN.log; MAIN.log itself carries only before/after-phase memsnaps.)

### ~11:08 — Phase 2: prewarm — reading all 11 shards

Purpose: pull the 435 GiB model through the storage stack once so subsequent mmap loads are warm.

```bash
time cat /models/GLM-5.2/UD-Q4_K_XL/*.gguf > /dev/null
```

```
real	5m14.163s
user	0m1.815s
sys	5m10.983s
[exit=0  314s]
```

Memsnap before: MemFree 1,042,595,372 kB, Cached 377,772 kB, AnonHugePages 0 kB. After: MemFree 1,042,602,748 kB, Cached 377,892 kB — Cached did not grow. The log's own caveat: "If 'Cached' did not grow by ~435 GiB, ZFS is serving from ARC rather than the page cache. -mmp 1 loads will still be fast; just note it."

**Observations**

- **Refuted (in effect):** the prewarm did not populate the page cache — the ZFS ARC absorbed the read. Consequence realized in Phase A1: the first mmap load paid the full 435 GiB read (~25 min); after that, Cached sat at ~456.75 GB (456,752,756 kB) for the rest of the run and every later `-mmp 1` load took ~45–90 s.

### ~11:14 (elapsed 0h06m) — Phase A1: SMOKE TEST (plain)

Purpose: M1 (`-mmp 1`), plain — no `-C`, no `--cpu-strict`. If this crashes, ZenDNN is still linked; watch the load time; the `load_tensors` lines should read `CPU_Mapped`, not `ROCm_Host`.

```bash
timeout --kill-after=180 10800 llama-bench -m '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 32 -r 1
```

Load: 1809 tensors from 11 GGUFs; tensor-type census `f32: 709, q8_0: 872, q4_K: 150, q5_K: 74, q6_K: 4`; `file size = 435.19 GiB (4.96 BPW)`. Buffers, verbatim:

```
load_tensors:   CPU_Mapped model buffer size = 46166.88 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB   (x8 shard buffers)
load_tensors:   CPU_Mapped model buffer size = 22067.47 MiB
load_tensors:        ROCm0 model buffer size =  4987.08 MiB
load_tensors:        ROCm1 model buffer size =  4431.34 MiB
load_tensors:        ROCm2 model buffer size =  4431.34 MiB
load_tensors:        ROCm3 model buffer size =  4952.45 MiB
```

CPU_Mapped as demanded — no ROCm_Host, no pin; GPU dense = 18.36 GiB, matching the briefing. pp512 context: KV buffers ROCm0–2 11.25 MiB each, ROCm3 10.69 MiB; compute buffers ROCm0–2 205.05 MiB, ROCm3 435.05 MiB, ROCm_Host 217.01 MiB. `sched_reserve: graph splits = 155`; `reserve took 247.42 ms, sched copies = 1`.

| test | t/s |
|---|---|
| pp512 | 34.65 ± 0.00 |
| tg32 | 5.35 ± 0.00 |

`llama_perf_context_print: load time = 1468266.44 ms` (~24.5 min — the one-time ZFS→page-cache pull; see Phase 2). `build: 657e01125 (10001)`. `[A1 ... exit=0 took 25 min]`. Memsnap after: MemFree 585,176,260 kB, Cached 456,752,756 kB, AnonHugePages 0 kB.

**Observations**

- Log verdict, verbatim: **"IT RUNS. ZenDNN crashes are gone."**
- **Confirmed:** the M1 configuration behaves exactly as designed (mapped buffers, 155 splits, no pinned allocation).

### ~11:39 (elapsed 0h31m) — Phase A2: SMOKE TEST with -C / --cpu-strict

Purpose: A1 plus the pinning flags every crashed v2 run had — "prove it rather than assume." If this crashes and A1 did not, drop `-C` and skip Phase C.

```bash
timeout --kill-after=180 10800 llama-bench -m '...00001-of-00011.gguf' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 64 --poll 100 --cpu-strict 1 -C ffffffffffffffffffffffffffffffff -b 2048 -ub 2048 -p 512 -n 32 -r 1
```

Same buffers and splits as A1 (graph splits = 155).

| test | cpu_mask | cpu_strict | t/s |
|---|---|---|---|
| pp512 | ffffffffffffffffffffffffffffffff | 1 | 34.29 ± 0.00 |
| tg32 | ffffffffffffffffffffffffffffffff | 1 | 5.39 ± 0.00 |

`load time = 65827.20 ms` (~66 s — the model now in page cache; "seconds" as promised). `[A2 ... exit=0 took 1 min]`.

**Observations**

- **Confirmed:** `-C`/`--cpu-strict` is not the crasher; the script sets `CPU_MASK_OK=1` and Phase C proceeds.

### ~11:41 (elapsed 0h33m) — Phase B: *** THE MAIN EVENT *** thread × poll sweep

Purpose (from the script): "M1. THE experiment. Decode only (-p 0)... STREAM saturates at 16 threads and DECAYS beyond. llama.cpp is being run with 64... --poll 100 keeps the threadpool SPINNING instead of futex-sleeping between the ~76 CPU splits per token. 63 ms unaccounted / 155 splits = 0.4 ms per split, which is exactly what waking a sleeping 64-thread pool costs. *** IF --poll 100 PRODUCES A STEP CHANGE, THE 63 ms IS FOUND AND THE FIX IS ONE FLAG. *** 16 combos, ONE model load."

```bash
timeout --kill-after=180 10800 llama-bench -m '...00001-of-00011.gguf' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 8,16,24,32,48,64,96,128 --poll 0,100 -b 2048 -ub 2048 -p 0 -n 64 -r 3
```

First load: `load time = 44187.45 ms`. Every combo reported `sched_reserve: graph splits = 155`. All 16 rows (tg64):

| `-t` | `--poll 0` | `--poll 100` |
|---|---|---|
| 8 | 3.89 ± 0.01 | 3.90 ± 0.01 |
| 16 | 5.27 ± 0.02 | 5.28 ± 0.01 |
| 24 | 5.53 ± 0.01 | **5.54 ± 0.00** |
| 32 | 5.53 ± 0.03 | 5.46 ± 0.09 |
| 48 | 5.49 ± 0.01 | 5.48 ± 0.01 |
| 64 | 5.40 ± 0.06 | 4.84 ± 1.01 |
| 96 | 2.76 ± 0.01 | 2.76 ± 0.01 |
| 128 | 1.29 ± 0.06 | 1.31 ± 0.10 |

`[B ... exit=0 took 16 min]`

**Observations**

- The answer to the main question: decode peaks at t=24–32 (5.53–5.54 t/s), is flat from 16 to 64, then collapses at 96 (2.76) and 128 (1.29) once SMT siblings are engaged.
- **Refuted:** `--poll 100` produces no step change anywhere (identical within noise; at t=64 it is actually noisier and worse, 4.84 ± 1.01). The 63 ms is not threadpool sleep/wake.
- Best hybrid decode ≈ 5.54 t/s = 181 ms/token — still ~43 ms above the 138 ms ideal.

### ~11:58 (elapsed 0h50m) — Phase B2: prefill at the best thread counts

Purpose: "Prefill is CPU-COMPUTE-bound (1.69 TFLOP/s...), NOT bandwidth-bound... it should keep scaling with threads long after decode has flattened."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 16,32,48,64,96,128 --poll 100 -b 2048 -ub 2048 -p 512 -n 0 -r 2
```

| `-t` | pp512 (t/s) |
|---|---|
| 16 | 12.81 ± 0.00 |
| 32 | 21.17 ± 0.02 |
| 48 | 26.91 ± 0.03 |
| 64 | 30.73 ± 0.06 |
| 96 | **32.92 ± 0.04** |
| 128 | 32.48 ± 0.79 |

`[B2 ... exit=0 took 7 min]`

**Observations**

- **Confirmed:** prefill is compute-bound — pp keeps climbing to t=96 (32.92) and only then flattens, the opposite shape to decode.
- Under M1/mmap, pp at t=64 is 30.73 versus the 34.6 seen in A1 and later in Phases L/N (a mmap-versus-`-mmp 0` prefill difference revisited at Phase L/N).

### ~12:06 (elapsed 0h58m) — Phase C: CCD placement at fixed thread count

Purpose: "Isolates Infinity-Fabric bandwidth from thread count... STREAM's answer is unambiguous: at 16 threads, SPREAD = 152 GB/s Copy, PACKED = 90 GB/s Copy / 68 GB/s Triad. CCD spread is worth ~2x... If llama.cpp shows the same, the right production config may be '-t 16 -C 0303030303030303 --cpu-strict 1'." Row map: t=16 — 0303=8 CCDs 2/CCD, 0f0f=4 CCDs 4/CCD, ffffffff=2 CCDs 8/CCD; t=32 — 0303 wraps (ignore), 0f0f=8 CCDs 4/CCD, ffffffff=4 CCDs 8/CCD.

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 16,32 --cpu-strict 1 --poll 100 -C 0303030303030303,0f0f0f0f0f0f0f0f,00000000ffffffff -b 2048 -ub 2048 -p 0 -n 64 -r 3
```

| `-t` | mask | placement | tg64 (t/s) |
|---|---|---|---|
| 16 | 0303030303030303 | 8 CCDs, 2/CCD | 3.09 ± 0.01 |
| 16 | 0f0f0f0f0f0f0f0f | 4 CCDs, 4/CCD | 3.14 ± 0.01 |
| 16 | 00000000ffffffff | 2 CCDs, 8/CCD | 3.13 ± 0.01 |
| 32 | 0303030303030303 | WRAPS — log says IGNORE | 0.44 ± 0.00 |
| 32 | 0f0f0f0f0f0f0f0f | 8 CCDs, 4/CCD | 5.53 ± 0.01 |
| 32 | 00000000ffffffff | 4 CCDs, 8/CCD | 5.07 ± 0.01 |

`[C ... exit=0 took 12 min]`

**Observations**

- **Refuted:** llama.cpp does not reproduce STREAM's 2× CCD-spread advantage. At t=32, 8-CCD spread beats 4-CCD packing by only 9% (5.53 versus 5.07), and the spread value merely ties the unpinned t=32 result (5.53).
- At t=16, all strict-pinned placements (3.09–3.14) are far worse than unpinned t=16 (5.27–5.28) regardless of spread — strict pinning itself costs ~40% at 16 threads.
- **Dead end:** the "-t 16 spread" production idea; `-t 24–32` unpinned (or 0f0f @ t=32) is the decode sweet spot.

### ~12:19 (elapsed 1h11m) — Phase D: -ngl 0 CPU-only denominator

Purpose: "M1, everything on CPU. Reads ~27 GB/token with... ZERO graph splits... In v2 this produced 'graph splits = 1088'... It should now say 1. *** D vs B AT MATCHED -t AND --poll IS THE SPLIT OVERHEAD... At the measured 152 GB/s, 27 GB/token = 178 ms = 5.6 t/s. If -ngl 0 matches or beats the hybrid's 4.97 t/s, the four V620s are contributing NOTHING NET."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 0 -mmp 1 -nopo 1 -t 16,32,48,64 --poll 0,100 -b 2048 -ub 2048 -p 512 -n 64 -r 3
```

Every combo reported **`sched_reserve: graph splits = 1`** — the v2 1088-split shredding is gone. All rows:

| `-t` | `--poll` | pp512 | tg64 |
|---|---|---|---|
| 16 | 0 | 8.32 ± 0.00 | 3.62 ± 0.00 |
| 16 | 100 | 8.32 ± 0.00 | 3.62 ± 0.00 |
| 32 | 0 | 14.20 ± 0.01 | **3.87 ± 0.01** |
| 32 | 100 | 14.18 ± 0.01 | 3.87 ± 0.02 |
| 48 | 0 | 18.40 ± 0.02 | 3.72 ± 0.00 |
| 48 | 100 | 17.27 ± 0.02 | 1.53 ± 0.00 |
| 64 | 0 | 20.04 ± 0.09 | 1.44 ± 0.02 |
| 64 | 100 | 19.73 ± 0.08 | 1.42 ± 0.00 |

`[D ... exit=0 took 32 min]`

**Observations**

- **Refuted** (the net-negative-GPUs hypothesis): the GPUs do contribute — `-ngl 0` peak decode is 3.87 t/s (t=32) versus the hybrid's 5.53; the V620s are worth +43%.
- Even with graph splits = 1 and zero device syncs, CPU-only decode reaches only 3.87 t/s = 258 ms/token where the 27 GB/token bandwidth model predicts 178 ms (5.6 t/s): ~80 ms of overhead exists without any splits — so the hybrid's 63 ms hole is largely not split overhead either, consistent with Phase B's null `--poll` result.
- The `-ngl 0` t=48/poll 100 and t=64 rows collapse to 1.4–1.5 t/s — heavy-thread CPU-only decode is pathological (matching Phase B's collapse at t ≥ 96).

### ~12:51 (elapsed 1h43m) — Phase E: graph-split count re-verification

Purpose: "-n 1 -r 1 --no-warmup so the dump is one graph... Expect: 'sched_reserve: graph splits = 155'."

```bash
env GGML_SCHED_DEBUG=1 timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 64 --poll 100 -b 2048 -ub 2048 -p 0 -n 1 -r 1 --no-warmup
```

`sched_reserve: graph splits = 155` re-confirmed, with the full split dump printed. Structure, verbatim (token-generation graph):

```
## SPLIT #0: CPU # 0 inputs
## SPLIT #1: ROCm0 # 4 inputs: [embd (  24K)] [leaf_8 (   0K)] [leaf_11 (   0K)] [attn_inp_kq_mask (   0K)]
## SPLIT #2: CPU # 2 inputs: [ffn_norm-3 (reshaped) (  24K)] [ffn_moe_topk-3 (   0K)]
## SPLIT #3: ROCm0 # 1 inputs: [ffn_moe_down-3 ( 192K)]
## SPLIT #4: CPU # 2 inputs: [ffn_norm-4 (reshaped) (  24K)] [ffn_moe_topk-4 (   0K)]
   ... (CPU/ROCm alternation for every MoE layer) ...
## SPLIT #36: ROCm1 # 4 inputs: [l_out-19 (  24K)] [leaf_8 (   0K)] [leaf_11 (   0K)] [attn_inp_kq_mask (   0K)]   <- device hop GPU0->GPU1 at layer 20
   ...
## SPLIT #153: CPU # 2 inputs: [ffn_norm-77 (reshaped) (  24K)] [ffn_moe_topk-77 (   0K)]
## SPLIT #154: ROCm3 # 1 inputs: [ffn_moe_down-77 ( 192K)]
```

Per-boundary payloads at tg: 24K (ffn_norm), 192K (ffn_moe_down back to GPU), ~0K (topk ids) — pure latency, as the briefing stated. (In the prefill-graph dump the same tensors are 384K / 3M / 15K.) Result row: `tg1 = 1.30 ± 0.00` (a single cold token, no warmup). `[E ... exit=0 took 1 min]`

### ~12:53 (elapsed 1h45m) — Phase F: op_offload ON, ubatch ladder (M1 = pageable H2D)

Purpose: "*** THE 3-10x ON PREFILL. *** op_offload streams CPU-resident weights to the GPU for any op whose batch dim >= 32... ~410 GiB of expert weights per graph eval spread over 4x PCIe4 x16, i.e. roughly 4 s per ubatch REGARDLESS of ubatch size. So pp scales ~LINEARLY with -ub. ub 512 -> ~125 t/s, ub 2048 -> ~500 t/s, current: 37 t/s. Under M1 the weights are in a PAGEABLE mmap, so H2D is bounce-buffered and SLOWER than it could be." Config: M1, `-nopo 0` (op_offload ON), `-b 4096`.

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 0 -t 64 --poll 100 -b 4096 -ub 128,256,512,1024,2048,4096 -p 512 -n 0 -r 2
```

| `-ub` | pp512 (t/s) |
|---|---|
| 128 | 9.35 ± 0.14 |
| 256 | 15.76 ± 1.58 |
| 512 | 25.77 ± 5.58 |
| 1024 | 25.94 ± 5.60 |
| 2048 | 25.95 ± 5.58 |
| 4096 | 25.87 ± 5.63 |

`[F ... exit=0 took 9 min]` — no crash, no OOM.

**Observations**

- **Refuted (as measured here):** the predicted 3–10× did not happen. op_offload over pageable mmap plateaus at ~25.9 t/s with very large run-to-run variance (± 5.6) — slower than plain CPU prefill (30.7 at t=64 in B2).
- ub > 512 cannot help a 512-token prompt, so the plateau above ub=512 is expected in this test; the salient point is the absolute level — ~26, not ~125. (Session 5, 14:37, identifies the deeper problem: `-p 512` silently clamped n_ubatch to 512 in every row, so the large-ubatch regime was never actually exercised.)

### ~13:02 (elapsed 1h54m) — Phase G: op_offload + HIP managed memory

Purpose: "GGML_CUDA_ENABLE_UNIFIED_MEMORY makes ggml_cuda_device_malloc() call cudaMallocManaged()... permitting VRAM OVERSUBSCRIPTION. If F aborts on VRAM exhaustion, this should make it SURVIVE (slowly)... Diagnostic only."

```bash
env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 0 -t 64 --poll 100 -b 2048 -ub 512,2048 -p 512 -n 0 -r 2
```

| `-ub` | pp512 (t/s) |
|---|---|
| 512 | 7.24 ± 0.05 |
| 2048 | 7.18 ± 0.30 |

(`sched_reserve: reserve took ~1100 ms` versus ~100–250 ms elsewhere — managed memory is slow even to reserve.) `[G ... exit=0 took 8 min]`

**Observations**

- **Dead end:** managed memory runs but at 7.2 t/s — 3.6× slower than Phase F. Since F never OOM'd, G's only lesson is the cost of hipMallocManaged itself.

### ~13:11 (elapsed 2h03m) — Phase H: -sm row

Purpose: "Under -sm layer only ONE card is live at a time... -sm row parallelises each layer across all four cards... may fail on MLA. A failure is itself a result." (P2P caveat noted in the script.)

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -sm row -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 64 -r 3
```

Failed during load, verbatim:

```
load_tensors: loading model tensors, this can take a while... (mmap = true, direct_io = false)
llama_model_load: error loading model: device ROCm0 does not support split buffers
llama_model_load_from_file_impl: failed to load model
llama_bench: error: failed to load model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
[H  -sm row  exit=1  took 0 min]
!!! H  -sm row FAILED (exit=1). Continuing.
!!!   132 = SIGILL   139 = SIGSEGV   124 = timeout
!!!   If ZenDNN is still linked (see PHASE 0.9), that is the cause.
```

**Observations**

- **Dead end (and the clean failure is itself the result):** ROCm on these V620s reports `VMM: no`, and llama.cpp's row-split buffer type requires it — `-sm row` is impossible on this hardware/stack. The ~47 ms serial-GPU dense cost under `-sm layer` cannot be attacked this way.

### ~13:11 (elapsed 2h03m) — Phase I: GGML_CUDA_GRAPH_OPT=1

Purpose: "Enables ggml-cuda's graph optimisation / concurrent stream-event launching (off by default)... second-order. Compare against the -t 64 --poll 100 row of PHASE B."

```bash
env GGML_CUDA_GRAPH_OPT=1 timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 64 --poll 100 -b 2048 -ub 2048 -p 0 -n 64 -r 3
```

Result: `tg64 = 5.40 ± 0.02` (graph splits = 155). `[I ... exit=0 took 1 min]`

**Observations**

- **Refuted:** 5.40 versus Phase B's t=64 rows (5.40 ± 0.06 / 4.84 ± 1.01) — no effect.

### ~13:13 (elapsed 2h05m) — Phase J: FILL THE IDLE VRAM (fitter)

Purpose: "18.36 GiB of 120 GiB VRAM is used; ~104 GiB is IDLE. Each MoE layer's routed experts are ~5.44 GiB, so ~17 of the 75 fit. Predicted: CPU reads drop from 13.77 to ~10.6 GB/token => decode ~1.3x. Real, but not where the 63 ms is. DO NOT USE -ncmoe... The fitter computes -ngl, --tensor-split AND the per-layer overrides together. If the measured gain is MUCH larger than 1.3x, that is itself evidence about the split overhead."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -fitt 2048 -fitc 8192 -mmp 1 -nopo 1 -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 128 -r 3
```

The phase's bulk (~123k log lines) is the fitter itself: 52 loader passes — 50 `no_alloc` dry-run probes plus 2 real loads (one per benchmark) — probing layouts whose trial reserves reported graph splits ranging over 1, 3, 5, 8, 13, 17, 133, 135, 137–147, 151, 152, 155. No allocation failures or OOMs anywhere in the phase. Final fitted layout (real load, verbatim buffers):

```
load_tensors: offloading output layer to GPU
load_tensors: offloading 78 repeating layers to GPU
load_tensors: offloaded 79/80 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1371.03 MiB
load_tensors:   CPU_Mapped model buffer size = 42638.96 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB   (x6)
load_tensors:   CPU_Mapped model buffer size = 22067.47 MiB
load_tensors:        ROCm0 model buffer size = 27636.72 MiB
load_tensors:        ROCm1 model buffer size = 26636.33 MiB
load_tensors:        ROCm2 model buffer size = 26645.77 MiB
load_tensors:        ROCm3 model buffer size = 27696.32 MiB
```

108,615 MiB ≈ 106.1 GiB on the four GPUs (versus 18.36 GiB in the exps=CPU config). Placement mechanism: 177 of the 225 exps tensors overridden to ROCm_Host (mmap → CPU-resident), 5 overridden to specific GPUs (e.g. `tensor blk.6.ffn_down_exps.weight (2112 MiB q5_K) buffer type overridden to ROCm1`), and the remaining ~14–15 expert layers left resident on their assigned GPU — roughly 15–16 of 75 expert layers now live in VRAM, matching the "~17 fit" prediction. KV cache: layer 0 on CPU, the rest spread ROCm0 (7), ROCm1 (5), ROCm2 (4), with ROCm3 carrying the balance of the 63-layer assignment (the experts of most ROCm3 layers overridden to host). Final config: `sched_reserve: graph splits = 137` (down from 155).

| test | ngl | fitt | fitc | t/s |
|---|---|---|---|---|
| pp512 | -1 (fitter-chosen) | 2048 | 8192 | **40.11 ± 0.14** |
| tg128 | -1 (fitter-chosen) | 2048 | 8192 | **6.01 ± 0.05** |

`[J  FILL THE IDLE VRAM (fitter)  exit=0  took 5 min]`

**Observations**

- **Confirmed (and best of run):** pp512 40.11 t/s, tg128 6.01 t/s — the best numbers in the entire battery; prefill 40.11 beats every other configuration including the op_offload phases.
- Decode gain over the 5.40–5.54 hybrid is 1.09–1.11× — less than the predicted ~1.3× bandwidth gain (removing ~16/75 layers of CPU reads should cut 13.77 → ~10.8 GB/token). Splits only dropped 155 → 137. Consistent with Phases B and D: the non-bandwidth overhead does not shrink proportionally.

### ~13:18 (elapsed 2h10m) — Phase K: decode at context depth — CRASHED

Purpose: "MLA keeps the KV cache small (44 MiB at 512 ctx), so degradation should be mild — confirm it."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 1 -nopo 1 -t 64 --poll 100 -b 2048 -ub 2048 -d 0,4096,16384,65536 -p 0 -n 64 -r 2
```

Rows obtained before the crash:

| test | t/s |
|---|---|
| tg64 | 5.41 ± 0.00 |
| tg64 @ d4096 | 5.29 ± 0.01 |

KV buffers: d0 = 5.62 ×3 + 5.34 MiB; d4096 = 95.62 ×3 + 90.84 MiB; d16384 reserved successfully (365.62 ×3 + 347.34 MiB) — then, during the depth-state restore, verbatim:

```
state_read_meta: cell_count = 16384, dest_seq_id = 0
ROCm error: an illegal memory access was encountered
  current device: -1, in function ggml_backend_cuda_buffer_set_tensor at /root/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:718
  hipMemcpyAsync((char *) tensor->data + offset, data, size, hipMemcpyHostToDevice, ((hipStream_t)2))
/root/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:104: ROCm error
[backtrace: ggml_abort <- libggml-hip <- llama_io_read_host dtor <- llama_context::state_seq_set_data <- llama_bench]
./galactus-diag.sh: line 457: 14483 Aborted                 timeout --kill-after=180 10800 llama-bench ...
[K  decode at CONTEXT DEPTH  exit=134  took 13 min]
!!! K  decode at CONTEXT DEPTH FAILED (exit=134). Continuing.
!!!   132 = SIGILL   139 = SIGSEGV   124 = timeout
```

**Observations**

- **Confirmed:** depth degradation is mild while it works — 5.41 → 5.29 at d4096, −2.2%.
- **Dead end (new bug):** restoring a 16384-cell saved KV state into GPU KV buffers dies in hipMemcpyAsync H2D — llama-bench's `-d` state save/restore path, unrelated to ZenDNN. The d16384 and d65536 rows were never produced.

### ~13:32 (elapsed 2h24m) — Phase L: *** THE PRODUCTION CANDIDATE *** M2 = -mmp 0 --no-host 1

The log's banner precedes this block: "## SLOW PHASES — these use -mmp 0 and each load costs 5-15 minutes." Purpose (from the script, abridged): "THIS IS THE CONFIG THAT SHOULD WIN. --no-host 1 removes the pinned host buffer... 1. NO 411 GiB hipHostMalloc. 2. The experts land in ANONYMOUS memory => THP-ELIGIBLE for the first time. 3. CPU_REPACK (AVX2 q4_K_8x8_q8_K) claims the Q4_K gate+up tensors = 62% of expert bytes... GREP THE load_tensors LINES: 'CPU_REPACK model buffer size' should be ~253 GiB; 'CPU model buffer size' ~158 GiB; 'ROCm_Host model buffer size' MUST NOT APPEAR... NOTE: CPU_REPACK's buffer reports is_host = nullptr, so op_offload CANNOT fire for repacked weights. This config is therefore CPU-prefill only."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 0 --no-host 1 -nopo 1 -t 16,32,48,64 --poll 0,100 -b 2048 -ub 2048 -p 512 -n 64 -r 3
```

Load: per-tensor messages `tensor blk.N.ffn_gate_exps.weight (1728 MiB q4_K) buffer type overridden to CPU_REPACK` (and ffn_up likewise) for every Q4_K expert layer. Buffers, verbatim:

```
load_tensors:          CPU model buffer size = 165220.22 MiB
load_tensors:   CPU_REPACK model buffer size = 255744.00 MiB
load_tensors:        ROCm0 model buffer size =  4987.08 MiB
load_tensors:        ROCm1 model buffer size =  4431.34 MiB
load_tensors:        ROCm2 model buffer size =  4431.34 MiB
load_tensors:        ROCm3 model buffer size =  4952.45 MiB
```

CPU_REPACK appeared: 255,744 MiB = 249.75 GiB repacked; plain CPU 165,220 MiB = 161.3 GiB; no ROCm_Host line. First-bench `load time = 255453.97 ms` (~4.3 min — no pin, includes the repack work; versus 14 min pinned in v2). Graph splits = 155. All rows:

| `-t` | `--poll` | pp512 | tg64 |
|---|---|---|---|
| 16 | 0 | 13.73 ± 0.00 | 5.39 ± 0.01 |
| 16 | 100 | 13.74 ± 0.00 | 5.40 ± 0.01 |
| 32 | 0 | 23.13 ± 0.00 | **5.52 ± 0.01** |
| 32 | 100 | 23.13 ± 0.01 | 5.52 ± 0.01 |
| 48 | 0 | 28.84 ± 0.00 | 5.42 ± 0.02 |
| 48 | 100 | 28.85 ± 0.01 | 5.41 ± 0.01 |
| 64 | 0 | **34.64 ± 0.14** | 5.39 ± 0.01 |
| 64 | 100 | 34.28 ± 0.26 | 5.38 ± 0.02 |

`[L ... exit=0 took 20 min]`

Post-phase check, verbatim:

```
$ grep AnonHugePages /proc/meminfo
  # AnonHugePages after M2
    AnonHugePages:         0 kB
```

**Observations**

- **Refuted:** the production candidate did not win on decode — tg peaks at 5.52 @ t=32, identical to M1 mmap (5.53) and M3 pinned (5.52–5.54). Repack plus anonymous memory bought zero decode.
- Prefill: 34.64 @ t=64 versus M1's 30.73 (+13%) but identical to M3's 34.60 — the pp gain comes from leaving mmap, not from repack. The 62% repack coverage moved nothing measurable at the token level.
- The AnonHugePages readings in MAIN.log (post-phase and memsnap) are both taken after the process exited and freed its memory; the per-run trajectory lives in `monitors/meminfo.txt`, which is not part of MAIN.log — MAIN.log alone does not settle the THP question.

### ~13:53 (elapsed 2h45m) — Phase M: M3 = -mmp 0 default (411 GiB pinned) + op_offload

Purpose: "The ONLY config with fast, DMA-direct H2D for op_offload... Now that iommu=pt is set, this is the config that should give the real prefill number. EXPECT: a 5-15 minute load and 'ROCm_Host model buffer size = 420964.22 MiB'... Compare pp against PHASE F (same thing over a pageable mmap). The delta IS the value of the pin."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 0 -nopo 0 -t 64 --poll 100 -b 4096 -ub 512,1024,2048,4096 -p 512 -n 0 -r 2
```

Load succeeded: `load_tensors: ROCm_Host model buffer size = 420964.22 MiB` (the 411 GiB pin; no GTT OOM — fresh boot plus released allocations). First-bench `load time = 342653.46 ms` (~5.7 min, in the predicted range).

| `-ub` | pp512 (t/s) |
|---|---|
| 512 | 29.72 ± 0.21 |
| 1024 | 29.54 ± 0.76 |
| 2048 | 29.60 ± 0.17 |
| 4096 | 29.80 ± 0.11 |

`[M ... exit=0 took 10 min]`

**Observations**

- The "real prefill number" with pinned DMA H2D is ~29.7 t/s, flat across `-ub` — versus F's pageable ~25.9 (the pin is worth ~15%) and versus plain CPU prefill at 34.6.
- **Refuted (as measured here):** op_offload is a net loss on this machine; the predicted 125–500 t/s ceilings were off by ~4–17×; PCIe streaming of 410 GiB per eval never approached line rate. (Session 5's clamp discovery reopens the question: every row here also ran at an effective n_ubatch of 512.)

### ~14:04 (elapsed 2h56m) — Phase N: M3 control, op_offload OFF

Purpose: "The original production config, for a clean apples-to-apples... This is the 37.6 t/s / 5.15 t/s baseline that started this whole investigation."

```bash
timeout --kill-after=180 10800 llama-bench -m '...' -fa 1 --progress -o md -v -ngl 99 -ot exps=CPU -mmp 0 -nopo 1 -t 16,32,64 --poll 0,100 -b 2048 -ub 2048 -p 512 -n 64 -r 3
```

Load: `ROCm_Host model buffer size = 420964.22 MiB` again (the 411 GiB pin); `load time = 345268.21 ms`. All rows:

| `-t` | `--poll` | pp512 | tg64 |
|---|---|---|---|
| 16 | 0 | 13.10 ± 0.00 | 5.27 ± 0.02 |
| 16 | 100 | 13.06 ± 0.00 | 5.28 ± 0.01 |
| 32 | 0 | 22.34 ± 0.02 | 5.52 ± 0.04 |
| 32 | 100 | 22.34 ± 0.01 | **5.54 ± 0.02** |
| 64 | 0 | **34.60 ± 0.06** | 5.40 ± 0.07 |
| 64 | 100 | 34.46 ± 0.04 | 5.38 ± 0.05 |

`[N ... exit=0 took 20 min]`

**Observations**

- The re-measured baseline on build 10001: 34.60 pp / 5.40 tg at the old settings (t=64), versus the briefing's 37.6 / 5.15 on build 9942.
- **Confirmed:** cross-config decode at matched threads is a three-way tie — M1 5.53, M2 5.52, M3 5.54 at t=32. Pinning, mmap-versus-anonymous, and repack all wash out; decode is configuration-insensitive. Only thread count (and the fitter's VRAM fill) move it.

### ~14:25 (elapsed 3h16m) — SUMMARY and TEARDOWN

The log's closing SUMMARY, verbatim:

```
## SUMMARY — READ IN THIS ORDER
================================================================================
   1. PHASE 0.9    Is ZenDNN gone? If --list-devices still lists it, STOP — every
                   crash will repeat and nothing else in this log is meaningful.
   2. PHASE 0.7    IOMMU mode: 'identity' = passthrough took; DMA/DMA-FQ = it did not.
   3. PHASE A1     Did it run? Load time under -mmp 1 should be SECONDS.
   4. PHASE B      *** THE MAIN EVENT. *** The -t x --poll grid.
                   - tg flattens by t=16-32          -> bandwidth-bound; drop -t.
                   - --poll 100 gives a step change  -> the 63 ms was threadpool
                                                        sleep/wake. Fix = one flag.
                   - tg climbs past t=64             -> kernel-bound, not bandwidth.
   5. PHASE D vs B at matched -t/--poll  *** THE DELTA IS THE 63 ms. ***
                   Also: does D now report graph splits = 1 (not 1088)?
   6. PHASE C      CCD spread. STREAM says this is worth 2x. Does llama.cpp agree?
   7. PHASE F/M    Largest surviving ubatch with op_offload, and the pp there.
                   *** This is the 3-10x on prefill. ***
   8. PHASE L      Does 'CPU_REPACK model buffer size' appear? Does AnonHugePages
                   finally climb off zero?
   9. PHASE H/J    -sm row and the VRAM fill. Lowest priority.
  Reference numbers:
      decode reads 13.77 GB/token (hybrid) / ~27 GB/token (-ngl 0)
      platform bandwidth 152 GB/s => 91 ms for the hybrid CPU part
      GPU dense path ~18.7 GB/token, serialised under -sm layer => ~47 ms
      ideal 138 ms (7.2 t/s) vs actual 201 ms (4.97 t/s)
      155 graph splits per token; 63 ms unaccounted; 0.4 ms per split
      prefill 37.6 t/s = 1.69 TFLOP/s on 64 Zen3 cores (AVX2, no VNNI)
All phases attempted. Total runtime: 3h16m
```

TEARDOWN, verbatim content:

```
## TEARDOWN
$ free -h
  # final memory
                   total        used        free      shared  buff/cache   available
    Mem:           995Gi       859Mi       558Gi       140Ki       436Gi       994Gi
    Swap:             0B          0B          0B
$ rocm-smi --showmeminfo vram
  # final VRAM
    GPU[0..3]: VRAM Total Memory (B): 32195477504 ; VRAM Total Used Memory (B): 17182720   (all four)
PHASES ATTEMPTED : 16
TOTAL RUNTIME    : 3h16m
ARTIFACTS        : /root/diag-20260714-110813/MAIN.log
                   /root/diag-20260714-110813/monitors/
=== END OF RUN 2026-07-14T14:25:10-04:00 ===
```

Clean teardown: VRAM back to ~16.4 MiB used per GPU; 436 GiB of model still resident in buff/cache. MAIN.log was delivered into the dialogue as results.txt at 14:34 (artifact saved 14:33:44 ET).

### State of knowledge at end of session

- All 16 phases attempted; 14 succeeded; H failed by design-relevant error (`device ROCm0 does not support split buffers`, exit=1); K crashed in llama-bench's `-d` KV-state restore (hipMemcpyAsync illegal access at 16384 cells, exit=134) — a new, non-ZenDNN bug.
- ZenDNN is confirmed gone at runtime (`--list-devices`, `ldd`, and `-ngl 0` splits = 1 versus v2's 1088), the stale `GGML_ZENDNN:BOOL=ON` CMakeCache line notwithstanding; IOMMU passthrough took (96 identity groups).
- Decode peaks at 5.53–5.54 t/s at t=24–32 and is configuration-insensitive (M1 mmap 5.53 / M2 repack 5.52 / M3 pinned 5.54 at t=32); SMT thread counts are catastrophic (t=96 → 2.76, t=128 → 1.29); `--poll`, `-C`/`--cpu-strict` (−40% at t=16), and `GGML_CUDA_GRAPH_OPT` (5.40) all null or harmful.
- `-ngl 0` peaks at 3.87 t/s (t=32) with graph splits = 1 — the GPUs are net +43%, and ~80 ms of non-bandwidth overhead exists even with zero splits.
- Prefill is compute-bound (scales to t=96: 32.92 t/s); `-mmp 0` (pinned or repacked) adds ~13% prefill over mmap at t=64 (34.6 versus 30.7).
- op_offload prefill lost everywhere as run: 25.9 t/s pageable (F), 29.7 pinned (M), 7.2 managed (G), against 34.6 CPU-only — the pin worth only ~15% on H2D.
- The fitter (Phase J) is the only real win: 106.1 GiB of VRAM filled (~15–16 expert layers resident), splits 155 → 137, pp512 40.11 / tg128 6.01 — decode 1.09–1.11×, below the predicted ~1.3×.
- CPU_REPACK engaged exactly as designed (255,744 MiB repacked + 165,220 MiB plain) and bought nothing measurable; the THP question remains unsettled by MAIN.log (all AnonHugePages readings post-exit 0 kB).
- Context-depth cost is mild while measurable: 5.41 → 5.29 t/s at d4096 (−2.2%).
- Artifacts: `/root/diag-20260714-110813/MAIN.log` (= results.txt, saved 14:33:44 ET) and `monitors/`; script `galactus-diag.sh` saved 11:05:36 ET.

---

