## Session 2 — Monday, July 14, 2026 (morning) — Baselines, memory bandwidth, and the first diagnostic script

*Executed 07:00–08:46. Work begins at the `root@openwebui` container prompt (where llama.cpp and the model live) and moves to the bare-metal host `root@galactus` at 08:04. llama.cpp build f84a51940 (9942), backends ROCm,ZenDNN; model at `/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf` (11 shards).*

### 07:00 — First llama-bench baseline from the openwebui container

Paul ran the first GLM-5.2 benchmark and asked three things: (1) explain what each flag does, (2) propose flag changes to improve performance, (3) list new flags in llama.cpp master that could further improve performance. Command as run (including the failed `nice`):

```bash
root@openwebui:~# nice -n -15 numactl --physcpubind=0-63 llama-bench   -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf   -ngl 99   -nopo 1   -mmp 0   -t 64   -b 4096   -ub 4096   -fa 1   -ot "exps=CPU"   -p 512   -n 128   -r 5
```

Error in output: `nice: cannot set niceness: Permission denied` (the benchmark ran anyway at normal priority). Device initialization:

```
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
```

Results:

| model | size | params | backend | threads | n_batch | n_ubatch | fa | ot | mmap | nopo | test | t/s |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm,ZenDNN | 64 | 4096 | 4096 | 1 | exps=CPU | 0 | 1 | pp512 | 37.63 ± 3.69 |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm,ZenDNN | 64 | 4096 | 4096 | 1 | exps=CPU | 0 | 1 | tg128 | 5.15 ± 0.40 |

`build: f84a51940 (9942)`

**Observations**

- Baseline numbers established: pp512 = 37.63 ± 3.69 t/s; tg128 = 5.15 ± 0.40 t/s.
- llama.cpp reports the model as 435.19 GiB / 753.86 B params (matching HF's decimal 467 GB ≈ 434.9 GiB).
- The container evidently lacks CAP_SYS_NICE — `nice -n -15` fails even as root inside it.

### 07:09 — Flag-by-flag audit: the "worst of both worlds" diagnosis

Every flag in the baseline command was audited and llama.cpp master surveyed for new levers.

**Flag breakdown.**

- `nice -n -15` — failed; root inside a container lacking CAP_SYS_NICE; currently a no-op.
- `numactl --physcpubind=0-63` — pins to logical CPUs 0–63 (on an EPYC 7713 with SMT, normally thread 0 of each of 64 physical cores; 64–127 are SMT siblings). Sets CPU affinity only, no memory policy.
- `-m …-00001-of-00011.gguf` — first shard; llama.cpp discovers the other ten.
- `-ngl 99` — offloads all 78 layers to GPU; with `-ot`, means "everything on GPU except what the override pulls back."
- `-nopo 1` — `--no-op-offload 1`; disables op_offload. Called "the single most damaging flag in the command."
- `-mmp 0` — disables mmap; the correct choice with CPU tensor overrides (llama.cpp recommends it).
- `-t 64` — 64 compute threads matching 64 physical cores.
- `-b 4096` — logical batch size; `-ub 4096` — physical batch size, determines compute-buffer allocation. Note: with `-p 512` only 512 tokens are submitted, so the actual ubatch is 512 — reserving VRAM for 4096 while measuring at 512.
- `-fa 1` — flash attention on (`-fa` now takes on|off|auto; 1 parses as truthy).
- `-ot "exps=CPU"` — unanchored regex_search for `exps`; matches `blk.N.ffn_{gate,up,down}_exps` on all 75 MoE layers → CPU. Does not match `ffn_*_shexp` or `ffn_gate_inp` (shared expert and router stay on GPU). Functionally identical to `--cpu-moe`.
- `-p 512 / -n 128 / -r 5` — prefill 512, generate 128, 5 repetitions.

**Central diagnosis.** Mechanism 1: `make_cpu_buft_list()` (llama-model.cpp) priority order for CPU-resident weights is (1) ACCEL device buffers, (2) pinned host buffer from the first GPU (unless `--no-host 1`), (3) the `CPU_REPACK` extra buffer (AVX2 `q4_K_8x8_q8_K` kernels — verified available on Zen3, supports MUL_MAT_ID), (4) plain CPU buffer. Pinned host sits ahead of CPU_REPACK (source comment: the host buffer is useful when large batches are offloaded to GPU), so by default ~410 GiB of expert weights land in `ROCm_Host` pinned memory and the repacked AVX2 kernels are silently disabled. Mechanism 2: `op_offload` (default true) makes ggml_backend_sched stream CPU-resident weights to GPU for any op with batch dimension ≥ 32 (`GGML_OP_MUL_MAT_ID` uses `ne[2]` = token count); prefill would run expert GEMMs on the four V620s; generation (batch 1) never triggers it. With `-nopo 1` and default `--no-host 0`: expert weights are pinned but not repacked, and there is no GPU offload — "You get neither optimization." Corroboration: pp512 37.63 t/s → 512 tokens in 13.6 s over ~23 TFLOP of expert GEMM ≈ ~1.7 TFLOP/s — exactly what 64 Zen3 cores do on non-repacked Q4_K.

| Config | Expert weights live in | Prefill | Decode |
|---|---|---|---|
| A `-nopo 0 --no-host 0` (both defaults) | ROCm_Host (pinned) | GPU, streamed over PCIe | CPU, generic Q4_K |
| B `-nopo 1 --no-host 1` | CPU_REPACK | CPU, AVX2 8×8 | CPU, AVX2 8×8 |
| Current `-nopo 1 --no-host 0` | ROCm_Host (pinned) | CPU, generic Q4_K | CPU, generic Q4_K |

Config B cannot use op_offload at all: the repack buffer reports `is_host = nullptr`, so the scheduler's offload path (`ggml_backend_buffer_is_host`) never fires. The actual regime is verifiable from the `load_tensors:` lines — `ROCm_Host model buffer size` versus `CPU_REPACK model buffer size`.

**Decode bandwidth.** tg128 = 5.15 t/s = 194 ms/token. Per token: 8 experts × 3 matrices × 6144 × 2048 ≈ 302 M params per MoE layer; × 75 CPU-resident layers at ~4.5–5.0 bpw = ~13–14 GB/token → effective ~65–72 GB/s (byte figures ±15% due to the mixed-precision UD layout). Against 187.7 GB/s theoretical for 8-channel DDR4-2933 and "typically 150–170 GB/s achievable" (a claim corrected at 07:56): "You are at roughly a third of peak." Causes proposed, in order: (1) missing repack; (2) NUMA — `--physcpubind` sets no memory policy, and with `-mmp 0` weights fault in under the default local-node policy; if the BIOS were NPS2/NPS4, a large fraction of 435 GiB could sit on the wrong node. "Run `numactl --hardware` first." If >1 node: prepend `numactl --interleave=all` and add `--numa numactl`. Also: confirm 0–63 are distinct physical cores via `lscpu -e | head -70` or `cat /sys/devices/system/cpu/cpu1/topology/thread_siblings_list`.

**Idle VRAM.** 4 × 30,704 MiB = 119.9 GiB total; `-ot "exps=CPU"` leaves ~25 GiB on GPU (attention, norms, shared experts, 3 dense layers, embeddings) → ~95 GiB idle. Each MoE layer's routed experts ~5.4 GiB → 14–16 layers of experts fit in VRAM. (Both numbers corrected at 08:00 to ~107 GiB / 17–18 layers.) Warning: do not just add `-ncmoe N` — it keeps experts of layers 0…N−1 on CPU; with `--tensor-split` unset, layers ~59–77 all go to ROCm3; `-ncmoe 62` would put 16 heavy layers (~86 GiB) on one 30 GiB card → OOM. Instead use the fitter:

```bash
llama-bench -m … -fitt 2048 -fitc 8192 -t 64 -mmp 0 -fa 1 -p 4096 -n 128 -r 3
```

`-fitt` computes `-ngl`, `--tensor-split`, and per-layer expert overrides together; it aborts if `-ngl`/`-ot`/`-ncmoe`/`-ts` are user-set.

**Benchmark shape.** `-p 512` with `-ub 4096` measures at ubatch 512 while paying VRAM for 4096. Match them: `-p 2048,4096` with `-ub 4096`, or sweep `-ub 1024,2048,4096`. Add `-d 4096,32768` for realistic context depth.

**Smaller items.** Drop `nice` or add `--cap-add=SYS_NICE`; try `--prio 2`; `--poll 100` (busy-wait at the thread barrier); sweep `-t 48,56,60,64`; use `-r 3` while sweeping and `-r 5` for finals (each run reads 435 GiB from disk with `-mmp 0`).

**ZenDNN is inert for this model.** Its `supports_op` for GGML_OP_MUL_MAT_ID rejects >32 experts (GLM-5.2 has 256) and only accepts weight types F32, BF16, Q8_0 (excludes Q4_K); its buffer type is plain `ggml_backend_cpu_buffer_type()` so it is skipped during buffer selection. "It costs nothing, but do not attribute any performance to it."

**Master flags worth adopting** (build f84a51940 (9942); check `--help`): `--no-host <0|1>` (the CPU_REPACK lever, rated highest value); `-fitt/--fit-target <MiB>` + `-fitc/--fit-ctx <n>`; `-d/--n-depth <n>`; `--numa <distribute|isolate|numactl>`; `--prio <-1..3>`, `--poll <0..100>`, `-C/--cpu-mask`, `--cpu-strict`; `-dio/--direct-io <0|1>` (load time only); `--progress`; `-ncmoe/--n-cpu-moe <n>` (with the multi-GPU caveat). Environment: `GGML_OP_OFFLOAD_MIN_BATCH` (default 32) sets the offload batch threshold; leave alone.

**Suggested next run:**

```bash
# 0. Check this first — it may dominate everything else
numactl --hardware

# 1. Config A: GPU-offloaded prefill (both flags at default)
numactl --physcpubind=0-63 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -t 64 -b 4096 -ub 4096 -fa 1 \
  --numa numactl --prio 2 -p 4096 -n 128 -r 3

# 2. Config B: CPU repack
numactl --physcpubind=0-63 llama-bench \
  -m … -ngl 99 -ot "exps=CPU" -mmp 0 -t 64 -b 4096 -ub 4096 -fa 1 \
  -nopo 1 --no-host 1 \
  --numa numactl --prio 2 -p 4096 -n 128 -r 3

# 3. Fill the VRAM (fitter handles the 4-way split; no -ngl/-ot/-ts)
numactl --physcpubind=0-63 llama-bench \
  -m … -fitt 2048 -fitc 16384 -mmp 0 -t 64 -b 4096 -ub 4096 -fa 1 \
  --numa numactl --prio 2 -p 4096 -n 128 -r 3
```

Run each with `-v` once; confirm from `load_tensors:` whether experts landed in `ROCm_Host` or `CPU_REPACK`.

**Observations**

- **Hypothesis:** the current configuration is the worst of both worlds — pinned-host placement disables CPU_REPACK while `-nopo 1` forgoes GPU prefill; prefill is CPU-compute-bound at ~1.7 TFLOP/s.
- **Prediction:** Config A wins prefill by a wide margin (expert GEMMs on ~4× RDNA2, streaming ~1.8 GiB per expert tensor over PCIe per graph eval, amortizing with larger ubatch). **Prediction:** Config B likely wins decode (8×8 repack path materially faster than the generic kernel).
- **Hypothesis:** decode at ~65–72 GB/s effective may be NUMA-limited; `numactl --hardware` decides.
- **Dead end (flagged in advance):** `-sm tensor` — `llm_arch_supports_sm_tensor()` explicitly returns false for `LLM_ARCH_GLM_DSA` and throws at load (`-sm row` is available but little is expected). **Dead end:** MTP self-speculation (`--spec-type draft-mtp`) — the glm-dsa loader marks NextN block blk.78 TENSOR_SKIP, tensors never allocated; whatever worked for DeepSeek-V4-Flash will not work here in mainline. Upside: blk.78 costs zero RAM, so resident weights sit ~7 GiB below the reported 435.19 GiB.
- Supporting estimates from the analysis: V620 = Navi21, 72 CUs ≈ RX 6800 XT, ~40 TFLOP/s fp16 per card (~160 combined); PCIe 4.0 x16 ≈ 25–27 GB/s practical per card (link width/generation unverified); per-batch expert streaming ~405 GiB for 75 layers → ~4 s/batch at ~100 GB/s aggregate → prefill ceilings ~125 t/s at ub512, ~1000 t/s at ub4096; the prior DeepSeek-V4-Flash ~80 t/s pp2048 was noted as "also quite low."

### 07:44 — Config B is flat; enabling op_offload crashes with a ROCm abort

Paul ran two of the proposed configurations. Run 1 (Config B — repack attempt):

```bash
root@openwebui:~# numactl --physcpubind=0-63 llama-bench   -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf   -ngl 99   -mmp 0   -t 64   -b 4096   -ub 4096   -fa 1   -ot "exps=CPU" -nopo 1 --no-host 1
```

Same 4-device `ggml_cuda_init` header (Total VRAM: 122816 MiB; 4x V620, 30704 MiB each). Results (note the added `noh` column = 1):

| model | size | params | backend | threads | n_batch | n_ubatch | fa | ot | mmap | nopo | noh | test | t/s |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm,ZenDNN | 64 | 4096 | 4096 | 1 | exps=CPU | 0 | 1 | 1 | pp512 | 37.29 ± 3.24 |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm,ZenDNN | 64 | 4096 | 4096 | 1 | exps=CPU | 0 | 1 | 1 | tg128 | 4.97 ± 0.45 |

Run 2 (op_offload enabled — `-nopo` and `--no-host` dropped):

```bash
root@openwebui:~# numactl --physcpubind=0-63 llama-bench   -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf   -ngl 99   -mmp 0   -t 64   -b 4096   -ub 4096   -fa 1   -ot "exps=CPU"
```

After printing the table header, the run aborted:

```
/root/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:104: ROCm error
/usr/local/lib/libggml-base.so.0(+0x18665) [0x73de4d212665]
/usr/local/lib/libggml-base.so.0(ggml_print_backtrace+0x1df) [0x73de4d212a3f]
/usr/local/lib/libggml-base.so.0(ggml_abort+0x11e) [0x73de4d212bce]
/usr/local/lib/libggml-hip.so.0(+0x2e67d12) [0x73de4bf2ad12]
/usr/local/lib/libggml-hip.so.0(+0x2e6c519) [0x73de4bf2f519]
/usr/local/lib/libggml-base.so.0(ggml_backend_sched_graph_compute_async+0x369) [0x73de4d22ee89]
/usr/local/lib/libllama.so.0(_ZN13llama_context13graph_computeEP11ggml_cgraphb+0xa1) [0x73de4d3a5311]
/usr/local/lib/libllama.so.0(_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_status+0xea) [0x73de4d3a8fca]
/usr/local/lib/libllama.so.0(_ZN13llama_context6decodeERK11llama_batch+0x368) [0x73de4d3ae758]
/usr/local/lib/libllama.so.0(llama_decode+0xb) [0x73de4d3b033b]
/usr/local/lib/libllama-bench-impl.so(+0x1746e) [0x73de4defd46e]
/usr/local/lib/libllama-bench-impl.so(_Z11llama_benchiPPc+0xedc) [0x73de4df0e13c]
/lib/x86_64-linux-gnu/libc.so.6(+0x29ca8) [0x73de4da88ca8]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x85) [0x73de4da88d65]
llama-bench(+0x11b1) [0x582e35ac21b1]
LIBXSMM_VERSION: feature_print_bw-1.17-3780 (25693892)
LIBXSMM_TARGET: hsw [AMD EPYC 7713 64-Core Processor]
Registry and code: 13 MB
Command: llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf -ngl 99 -mmp 0 -t 64 -b 4096 -ub 4096 -fa 1 -ot exps CPU
Uptime: 1041.934304 s
Aborted
```

Paul also asked: "Can we track progress too by the way".

**Observations**

- Config B: pp512 37.29 ± 3.24 / tg128 4.97 ± 0.45 — statistically indistinguishable from the baseline in both directions.
- The op_offload run crashed after 1041.9 s of uptime — essentially all of it the model load — before producing a single result row.
- The LIBXSMM_TARGET line (`hsw [AMD EPYC 7713 64-Core Processor]`) confirms an AVX2-class code path, no AVX-512.

### 07:50 — Reading the two runs: crash triage and the number that does not add up

Analysis of both runs pointed to the next diagnostic commands.

**Run 1 — no signal.** 37.29 ± 3.24 / 4.97 ± 0.45 versus 37.63 ± 3.69 / 5.15 ± 0.40 — inside the error bars both directions. "The repack hypothesis is not supported" — and it should have been visible, because prefill is compute-bound and repack is a compute optimization; had CPU_REPACK engaged, pp would have moved. **Hypothesis (two candidate explanations):** (1) the expert tensors are not a type the repacker handles — `ggml_repack_get_optimal_repack_type()` covers exactly Q4_0, Q4_K, Q2_K, IQ4_NL, and "Q4_K - Medium" comes from `general.file_type` in the GGUF header (a single scalar Unsloth sets by hand) that says nothing about per-tensor types; UD quants deliberately mix (possibly Q5_K/Q6_K/IQ-something); (2) buffer selection fell through to plain CPU for some other reason.

**Run 2 — crash triage.** ggml-cuda.cu:104 is the generic `ggml_cuda_error()` → GGML_ABORT; the actual HIP error string is emitted by GGML_LOG_ERROR on the line before, and llama-bench suppresses llama.cpp's log unless `-v`. **Hypothesis (crash cause):** VRAM exhaustion — with op_offload the scheduler must materialize the entire 256-expert weight tensor on GPU per MUL_MAT_ID; `blk.N.ffn_up_exps` is [6144, 2048, 256], roughly 1.8 GiB at Q4_K, three per layer; plus `-ub 4096` MoE activation buffers; ~24 GiB free VRAM per card goes fast. Test by collapsing `-ub`: 512, 1024, 2048.

**"The number that does not add up" (decode math):**

```
8 experts × 3 matrices × 6144 × 2048  = 302 M params per MoE layer
                          × 75 layers = 22.65 B params/token
       at 4.5–5.0 bpw                 = 12.7–14.2 GB/token
       / 0.201 s                      = 63–70 GB/s effective
```

4.97 t/s = 201 ms/token. Against the (later retracted) "140–160 GB/s achievable in practice": "You are running at 40–50% of what the platform will deliver." Prefill: 512 tokens in 13.7 s = 23.2 TFLOP of expert GEMM at 1.69 TFLOP/s (a normal AVX2 Q4_K rate for 64 Zen3 cores); prefill reads the same 12.7 GB once per batch = 0.93 GB/s — nowhere near bandwidth-limited. Two diseases: prefill is CPU-compute-bound (fix = op_offload on the GPUs — fix the crash); decode is CPU-memory-bound at roughly half the believed-available bandwidth. **Hypothesis:** prime suspect for decode is NUMA (`--physcpubind` sets no memory policy; an NPS4 quadrant on Milan ≈ 50 GB/s, "uncomfortably close to what you are measuring").

Commands proposed ("Run these three things before another benchmark"):

```bash
# 1. NUMA topology and actual page placement
numactl --hardware
lscpu | grep -iE 'numa|thread|core|socket'

# 2. Where the weights actually went, and which CPU backend variant loaded
numactl --physcpubind=0-63 llama-bench -m … -ngl 99 -mmp 0 -t 64 -fa 1 \
  -ot "exps=CPU" -nopo 1 -p 128 -n 8 -r 1 -v --progress 2>&1 \
  | grep -E 'load_backend|system_info|load_tensors'
```

Wanted in that output: `load_backend: loaded CPU backend from …libggml-cpu-haswell.so` (not the x64 baseline), `AVX2 = 1` in system_info, and one of `ROCm_Host model buffer size` / `CPU_REPACK model buffer size` / `CPU model buffer size`.

```bash
# 3. Actual expert tensor types (shard 1 is metadata-only at 9 MB; use shard 2)
python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
  /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00002-of-00011.gguf \
  | grep -E 'ffn_(up|gate|down)_exps' | head -12
```

If those come back Q5_K/Q6_K/IQ4_XS rather than Q4_K, `--no-host` is dead as a lever. Then, if more than one NUMA node exists:

```bash
numactl --interleave=all --physcpubind=0-63 llama-bench … --numa numactl
```

with `numastat -p <pid>` to confirm pages spread.

**Progress tracking and the iteration loop.** `--progress` prints per-test progress; `-v` un-suppresses the llama.cpp log (model-load percentage plus the buffer breakdown); `-o jsonl` / `-oe jsonl` give machine-readable output. Live monitoring:

```bash
watch -n1 'rocm-smi --showmemuse --showuse'   # VRAM + GPU utilisation
numastat -m                                    # per-node memory placement
htop                                           # core utilisation, thread count
```

Load time is the real bottleneck: Uptime 1041 s ≈ entirely the 435 GiB read with `-mmp 0`. From `cmd_params_instance::equal_mparams` (bench.cpp): a model reload is forced only by `-m`, `-ngl`, `-ncmoe`, `-sm`, `-mg`, `-ts`, `-mmp`, `-dio`, `-dev`, `--no-host`, `-ot`. Everything else sweeps free from one load — including `-b`, `-ub`, `-t`, `--poll`, `-ctk/-ctv`, `-fa`, `-p`, `-n`, `-d`, and `-nopo`. Combined sweep proposed (24 measurements from one load; answers the op_offload, ubatch/OOM, and thread-barrier questions; if it OOMs at `-ub 2048`, only that combination fails):

```bash
numactl --interleave=all --physcpubind=0-63 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -fa 1 \
  --numa numactl --prio 2 --progress \
  -t 64 -b 4096 -ub 512,1024,2048 -nopo 0,1 --poll 0,100 \
  -p 2048 -n 128 -r 3
```

Storage: 435 GiB in ~1000 s ≈ 430 MB/s. "If that is babel over the network, copy the model to local NVMe — it is worth more than any flag in this thread." If local: page-cache thrash — with `-mmp 0` the kernel holds 435 GiB page cache plus 425 GiB anonymous = 860 GiB of 1 TB. Leave `-dio 0` (default) so repeat loads hit the page cache. Fast crash-repro recipe: `-p 128 -n 8 -r 1 -v`.

**Observations**

- **Refuted:** the repack hypothesis is not supported by Run 1 (flat within error bars).
- **Dead end (pending):** the op_offload crash cannot be diagnosed without `-v`; the real HIP error string was suppressed.
- **Correction (self):** "The `--no-host` idea is provisionally dead pending the tensor-type dump — I proposed it on the basis that the experts were plain Q4_K... That label is not authoritative and I should have said so."
- **Decision:** remaining levers in order — (1) decode bandwidth (NUMA first, then `--poll`; the CPU↔GPU boundary is crossed twice per MoE layer ≈ ~150 thread-pool barriers per token); (2) prefill on GPUs (shrink `-ub` until op_offload survives, then walk back up). Both are independent of the ~95 GiB idle VRAM.

### 07:53 — NUMA topology check: a single node

Paul ran the first proposed check:

```bash
root@openwebui:~# numactl --hardware
```

Output:

```
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 0 size: 1019408 MB
node 0 free: 990002 MB
node distances:
node     0 
   0:   10
```

**Observations**

- Single NUMA node (NPS1); node 0 size 1019408 MB, 990002 MB free at check time.
- **Refuted:** the NUMA-misplacement hypothesis for the decode deficit.
- The 990 GB free also implies the 435 GiB just read is no longer in the page cache — every benchmark run pays the full ~17-minute load.

### 07:56 — NUMA ruled out; bandwidth claim retracted; the decisive thread sweep proposed

The leading hypothesis fell: NUMA is clean, so that hypothesis was wrong. The NUMA line of investigation was discarded.

**Correction (explicit):** "I said 140–160 GB/s was 'achievable' on this platform. That figure is a STREAM extrapolation, not a measurement of llama.cpp's quantized MoE kernel, and I presented it with more confidence than it deserved." Cross-check: Paul's own DeepSeek-V4-Flash baseline (13B active, MXFP4 experts, 7.16 t/s → roughly 38–42 GB/s expert-weight traffic) versus GLM-5.2's 63–70 GB/s — both far below STREAM on the same box. "A gather-heavy quantized GEMV with 75 CPU fork/join cycles per token does not behave like a triad benchmark." Open question: is decode limited by DRAM bandwidth or by the kernel? The two answers demand completely different next steps.

"The decisive experiment, and it is free" (`-t` does not force a reload):

```bash
numactl --physcpubind=0-127 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -fa 1 -nopo 1 \
  -t 16,32,48,64,96,128 --poll 0,100 \
  -b 2048 -ub 2048 -p 0 -n 128 -r 3 --progress
```

Interpretation: tg128 flattens by ~32–48 threads → DRAM-bandwidth-bound, no CPU-side win, the only lever is VRAM; tg128 keeps climbing to 64+ → kernel/compute-bound (thread count, `--poll`, repack matter; SMT `-t 128` may help). `-p 0` skips prefill. Affinity check:

```bash
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list   # expect "0,64"
```

If it prints `0,1`, `--physcpubind=0-63` has been binding to 32 physical cores with both SMT threads and "every number so far is suspect."

**P0 — fix the op_offload crash** ("unambiguously the biggest lever"). Prefill runs at 1.69 TFLOP/s on CPU; streamed to four V620s it becomes PCIe-bound: ~410 GiB of expert weights per graph eval over 4 cards ≈ 4 s per ubatch regardless of ubatch size → prefill scales roughly linearly with `-ub`. **Prediction** (ceilings at PCIe 4.0 x16, ~25 GB/s/card):

| `-ub` | ceiling |
|---:|---:|
| 512 | ~125 t/s |
| 2048 | ~500 t/s |
| 4096 | ~1000 t/s |

"Real numbers will be well below those ceilings, but even the pessimistic case is a 3× on your 37 t/s."

```bash
# actual link width and generation — halve/quarter the table above if these are x8 or Gen3
rocm-smi --showbus
lspci -vv -s <bdf> | grep -E 'LnkCap|LnkSta'

# the real HIP error (ggml-cuda.cu:104 is the generic abort; the message is on the prior line)
llama-bench -m … -ngl 99 -ot "exps=CPU" -mmp 0 -t 64 -fa 1 -ub 512 -p 128 -n 8 -r 1 -v
```

Then walk `-ub` up 512 → 1024 → 2048 → 4096; `-ub 512,1024,2048 -nopo 0,1` runs from a single load.

**P1 — reclaim ~95 GiB idle VRAM:** ~5.4 GiB/MoE layer, 15–16 layers fit; moves ~20% of decode bytes from 65 GB/s DDR4 to GDDR6. **Prediction:** decode ~6.2 t/s. Use `-fitt 2048`, not `-ncmoe`. **P2 — `--no-host`/repack now low priority;** still run the gguf_dump command from 07:50; if the types are not Q4_0/Q4_K/Q2_K/IQ4_NL, the repacker declines and `--no-host` is a dead flag.

**Iteration loop.** 990 GB free ⇒ the 435 GiB just read is not in page cache; `use_direct_io` defaults false so the cache was populated, but 435 (cache) + 425 (anon) + pinned pages against 1019 GB forces eviction — every run pays the full ~17-minute read. **Decision/advice:** "Sweep with `-mmp 1`" — with mmap the weights are the page cache (~435 GiB resident instead of ~860 GiB); subsequent loads near-instant. Cost: mmap'd tensors are demoted out of the pinned host buffer (and CPU_REPACK); op_offload H2D goes through a bounce buffer. Measure the cost once:

```bash
# two loads, one comparison — is mmap actually costing you anything with op_offload on?
llama-bench -m … -ngl 99 -ot "exps=CPU" -t 64 -fa 1 -ub 1024 -mmp 0,1 -p 2048 -n 128 -r 3
```

Storage advice repeated: 435 GiB in ~1000 s ≈ 430 MB/s — "If /models is on babel over the network, move the model to local NVMe. That is worth more to you right now than any flag in this thread."

**Observations**

- **Refuted/Dead end:** NUMA as the decode limiter (NPS1, single node).
- **Correction:** the 140–160 GB/s "achievable" figure retracted as an unverified STREAM extrapolation.
- **Hypothesis (context):** community MoE decode reports put 8-channel DDR4 Milan at ~3–4 t/s ≈ 73 GB/s on DeepSeek-R1-class Q4 — matching the observed band; 12-channel DDR5 Genoa reports ~8 t/s (~166 GB/s of 460 peak).
- **Decision:** the thread/poll sweep is the decisive, free experiment; op_offload crash triage (P0) is the biggest lever; VRAM fill (P1) is a ~1.25× that can wait.

### 07:57 — lscpu: Zen3 confirmed — AVX2 and FMA, no AVX-512

Paul ran the topology check:

```bash
root@openwebui:~# lscpu | grep -iE 'numa|thread|core|socket'
```

Output:

```
Model name:                              AMD EPYC 7713 64-Core Processor
Thread(s) per core:                      2
Core(s) per socket:                      64
Socket(s):                               1
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-127
```

The full CPU flags line was also pasted; condensed, it includes fpu, sse–sse4_2, ht, avx, avx2, fma, f16c, aes, sha_ni, vaes, vpclmulqdq, bmi1/bmi2, adx, rdrand/rdseed, clflushopt, clwb, fsrm, topoext, cpb, ibrs/ibpb/stibp/ssbd, svm/npt (virtualization), rapl, etc. No avx512* flags are present.

**Observations**

- **Confirmed:** single socket, 64 cores, 2 threads/core, one NUMA node spanning CPUs 0–127.
- **Confirmed:** AVX2 + FMA, no AVX-512, no VNNI — the instruction-set ceiling for the CPU path.

### 08:00 — ISA settled; corrected sizing arithmetic; the decode budget gap; a three-load plan

What lscpu settles was consolidated and the earlier arithmetic corrected against the llama-bench table figures.

**What lscpu settles.** Zen3, 64C/128T, single socket, NPS1. AVX2 + FMA, no AVX-512, no VNNI → llama.cpp's Q4_K×Q8_K dot product runs the `maddubs`/`madd` path — the ceiling for prefill on this box, which explains the measured 1.69 TFLOP/s. "Nothing pathological." Still unverified — the CPU enumeration:

```bash
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list   # want "0,64"
```

If it prints `0,1`: every number so far was produced on 32 physical cores.

**Corrected arithmetic** (supersedes the ~25 GiB non-expert / ~95 GiB idle estimates). From the table figures — 753.86 B params, 435.19 GiB = 4.96 bpw average:

```
routed experts, 75 MoE layers   3 × 256 × 6144 × 2048 × 75  = 724.8 B params
in-file experts incl. blk.78                                = 734.5 B params
everything else                                             =  19.4 B params
```

Solving at plausible precisions: experts ≈ 4.92 bpw → 5.53 GiB per MoE layer, ~415 GiB resident; only ~13 GiB of non-expert weights (not 25 GiB). Consequences: ~107 GiB VRAM idle (not 95); 17–18 MoE layers fit (after KV and compute buffers), not 15. Decode reads 13.9 GB/token from DDR4 (8 × 3 × 6144 × 2048 × 0.6146 B × 75); at 4.97 t/s that is 69 GB/s.

**"The decode budget does not close."** STREAM on a 7713 "lands around 145–165 GB/s" (still assumed at this point); even discounting to 120 GB/s, 13.9 GB should take 116 ms; actual is 201 ms → ~85 ms/token unaccounted. Exactly two places it can be: (1) **Hypothesis:** the CPU really is at ~69 GB/s — the kernel cannot pull more than half the platform bandwidth; (2) **Hypothesis:** the ~85 ms/token is overhead — with `-ot exps=CPU` the graph splits at every MoE layer: 75 CPU splits, ~150 device boundaries per token; each CPU split is a full 64-thread fork/join; `--poll` defaults to 50 (spins briefly then sleeps); 1.1 ms per split × 75 = exactly 85 ms. `-t` and `--poll` are context params — the test costs nothing.

**One consolidated sequence** — `-mmp 1` throughout (the weights become the page cache, ~435 GiB resident, second and third loads near-instant; this demotes pinned-host and repack buffers, "that is fine, and in one case below it is exactly what you want").

Load 1 — decode diagnosis (no reload across the sweep):

```bash
numactl --physcpubind=0-127 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -fa 1 -nopo 1 \
  -t 16,32,48,64,96,128 --poll 0,100 --cpu-strict 0,1 \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

Interpretation: tg flattens by t=32–48 and `--poll` makes no difference → DRAM-bandwidth-bound at ~69 GB/s → go straight to VRAM. `--poll 100` produces a step change → threadpool sleep/wake was eating the ~85 ms/token → pin `--poll 100 --cpu-strict 1` and re-baseline. tg climbs to t=96/128 → kernel-bound, SMT hides unpack latency → 128 threads with `--physcpubind=0-127`.

Load 2 — the op_offload / repack question in one shot:

```bash
numactl --physcpubind=0-127 llama-bench \
  -m … -ngl 99 -ot "exps=CPU" -mmp 1 -fa 1 --no-host 1 \
  -t <best from load 1> --poll <best> \
  -nopo 0,1 -b 512 -ub 512 -p 512 -n 0 -r 2 -v --progress
```

**Hypothesis:** `--no-host 1` keeps the experts out of the 415 GiB pinned-host mapping, "which is itself a plausible contributor to the ROCm abort" — pinning 415 GiB via hipHostMalloc consumes GART resources, and RDNA2 without large-BAR could exceed limits; notably, Run 1 with `--no-host 1` did not crash while Run 2 with default pinning did. Diagnostic logic: pp jumps with `-nopo 0` → op_offload fired → weights sit in a plain host CPU buffer → the repacker declined them (explaining the flat `--no-host` result while also handing prefill to the GPUs). pp does not move → weights are in CPU_REPACK (is_host null, op_offload cannot fire) — but repack should then have raised pp and did not → a contradiction pointing at the build; check `system_info: AVX2 = 1` and the `load_backend:` line in the `-v` output. Start at `-ub 512`; walk up 1024, 2048, 4096.

Load 3 — reclaim the VRAM:

```bash
numactl --physcpubind=0-127 llama-bench \
  -m … -mmp 1 -fa 1 -fitt 2048 -fitc 16384 \
  -t <best> --poll <best> -nopo <best> \
  -b 2048 -ub 2048 -p 2048 -n 128 -r 3 --progress
```

**Prediction:** 17–18 of 75 expert layers to GDDR6 → CPU reads drop to ~10.8 GB/token → decode ~6.0–6.4 t/s. "Real but unexciting on its own."

**Where the real money is (ranked):** (1) prefill on GPUs (Load 2): 37 t/s → low hundreds, "the only 3–10× available"; (2) the 85 ms/token gap (Load 1): if it is threadpool overhead, decode goes 5 → 8+ t/s for a flag; (3) VRAM fill (Load 3): ~1.25×, compounds with both. While waiting on a load, grab link topology:

```bash
rocm-smi --showbus
lspci -vv -s <bdf> | grep -E 'LnkCap|LnkSta'
```

**Observations**

- **Correction:** non-expert weights ≈ 13 GiB, not ~25 GiB; idle VRAM ≈ 107 GiB, not 95; 17–18 expert layers would fit, not 15.
- Decode traffic re-derived: 13.9 GB/token → 69 GB/s effective at 4.97 t/s.
- **Hypothesis:** ~85 ms/token of the budget is either a kernel bandwidth ceiling or threadpool fork/join overhead — 1.1 ms × 75 splits matches exactly.
- **Hypothesis:** the 415 GiB pinned-host mapping is a plausible contributor to the ROCm abort.
- **Decision:** three-load plan (decode sweep, op_offload/repack probe at `-ub 512`, fitter run), with `-mmp 1` to amortize loads.

### 08:00 — Decision: VRAM filling deferred; hugepages question raised

Paul redirected the priorities: "Check other things like hugepages or something? And let's not optimize for the idle VRAM until we figure out the throughput issue."

**Observations**

- **Decision (Paul):** VRAM filling is postponed until decode throughput is understood.
- New question on the table: transparent hugepages.

### 08:03 — THP finding, STREAM recipe, and the firmware/power checklist

The deferral held — VRAM is a 1.25× and it will still be there later; the 69 GB/s number is the thing that either has a 2× hiding in it or does not, and right now nobody knows which — and hugepages were investigated.

**Hugepages finding.** ggml allocates CPU tensor buffers through `ggml_aligned_malloc()` = `posix_memalign()` (ggml.c:367). "There is no `madvise(MADV_HUGEPAGE)` and no `MAP_HUGETLB` anywhere in ggml or llama.cpp." glibc services a 415 GiB request with anonymous mmap — THP-eligible only if the system policy is `always`; on `madvise` (the Ubuntu/Debian default) llama.cpp never asks → "your entire 415 GiB working set is running on 4 KiB pages."

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled   # likely [madvise]
cat /sys/kernel/mm/transparent_hugepage/defrag
```

Zen3's L2 TLB has 2048 entries: 8 MiB coverage at 4 KiB pages, 4 GiB at 2 MiB. "But the TLB is not the main story — the prefetcher is. AMD's L2 stream prefetcher does not cross 4 KiB page boundaries." On 2 MiB pages it streams uninterrupted for 32,768 cache lines instead of restarting every 64.

```bash
echo always      > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
# during the next load, confirm it actually took:
watch -n5 "grep -E 'AnonHugePages|Hugepagesize' /proc/meminfo"
```

`AnonHugePages` should climb toward ~415 GiB. **Prediction (calibrated):** 5–25% gain, not 2×. Works only with `-mmp 0` (file-backed mappings are not THP-eligible without large-folio page cache) → "THP and fast reloads are mutually exclusive." Explicit 1 GiB hugetlbfs pages (64 GiB TLB coverage on Zen3): no llama.cpp hooks; hugectl/libhugetlbfs will not reliably cover this; not worth the effort. Side estimates from the analysis: the 415 GiB working set is ~109 million 4 KiB pages (~830–872 MB of page tables); TLB page-walk overhead itself is small (~3% for streaming, ~0.2% of bandwidth for PTE reads) — prefetcher continuity is the real THP mechanism.

**Method correction.** "Establish the ceiling first — everything else is guesswork without it. I have twice now reasoned from an assumed memory bandwidth. Stop assuming."

1. DIMM slots:

```bash
dmidecode -t memory | grep -E 'Locator|^\s+Size|Configured Memory Speed|^\s+Speed|Rank' | grep -v 'No Module'
```

Want 8 or 16 populated DIMMs at `Configured Memory Speed: 2933 MT/s`. If 4 DIMMs, or 2666/2400 MT/s: peak is 102 or 170 GB/s rather than 187.7, and 69 GB/s effective is an ordinary 65–70% — "no software bug exists and this whole line of inquiry closes."

2. STREAM build and run:

```bash
apt-get install -y build-essential wget
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c
gcc -O3 -march=znver3 -fopenmp -mcmodel=medium \
    -DSTREAM_ARRAY_SIZE=1000000000 -DNTIMES=10 stream.c -o stream
OMP_NUM_THREADS=64 OMP_PROC_BIND=spread OMP_PLACES=cores ./stream
```

(1e9 doubles × 3 arrays = 24 GB, far past the 256 MB of L3. Triad is the number.) Interpretation:

| STREAM Triad | llama.cpp at 69 GB/s | Conclusion |
|---|---|---|
| 150–165 GB/s | 42% | Real software gap. Chase THP, --poll, C-states, thread count. |
| 100–115 GB/s | ~62% | Normal for llama.cpp's quantized GEMV. Decode essentially done; limit is DIMM config. |
| 75–90 GB/s | 80%+ | At the wall. Only a hardware change moves it. |

3. Firmware and power:

```bash
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor    # want: performance
cpupower idle-info
turbostat --quiet --show Core,Bzy_MHz,PkgWatt --interval 2      # run during llama-bench
```

The 7713 is base 2.0 / boost 3.675 GHz. All-core AVX2 load should sit ~2.6–3.0 GHz. If Bzy_MHz reads ~2.0 or lower → BIOS Determinism Slider on Power or a cTDP cap → a 30–40% clock deficit affecting everything. C-states are "disproportionately important for this specific workload" (75 full 64-thread fork/joins per token):

```bash
cpupower frequency-set -g performance
cpupower idle-set -D 0        # disable C-states deeper than C0
```

BIOS if reachable: disable DF C-states (Data Fabric); confirm NPS1 and memory interleaving across all channels. `--poll 100` attacks the same problem from userspace — "Test both; they are not redundant."

4. Sanity checks that cost nothing:

```bash
free -h                          # is anything swapping?
vmstat 1 5                       # si/so must be 0
cat /sys/fs/cgroup/memory.max    # "max" = unlimited; a container limit would explain the cache eviction
cat /proc/cmdline | tr ' ' '\n' | grep -i iommu   # want iommu=pt for the GPU streaming later
```

Build confirmation lines to look for in any `-v` run:

```
load_backend: loaded CPU backend from .../libggml-cpu-haswell.so   ← not libggml-cpu-x64.so
system_info: ... AVX2 = 1 | FMA = 1 | AVX512 = 0 ...
```

Order of operations: (1) dmidecode + STREAM ("Do not skip this" — it decides whether 2–4 are worth doing); (2) turbostat during a tg run; (3) THP `always`, re-run tg with `-mmp 0`, watch AnonHugePages; (4) `-t 16,32,48,64,96,128 --poll 0,100` in one llama-bench invocation (no reload; `-p 0 -n 64 -r 3`). "If STREAM comes back at 150+ and tg is stuck at 69, we have a real target. If STREAM comes back at 100, the decode path is finished and the only remaining lever in this entire thread is getting prefill onto the GPUs."

**Observations**

- **Confirmed (source):** ggml never requests hugepages; under the default `madvise` THP policy the whole working set runs on 4 KiB pages.
- **Prediction:** THP `always` is worth 5–25% (prefetcher continuity across page boundaries), not 2×; incompatible with `-mmp 1` fast reloads.
- **Correction (method):** stop reasoning from an assumed bandwidth; measure the ceiling with dmidecode + STREAM before any further tuning.
- **Hypothesis:** BIOS power determinism or C-states could be depressing all-core clocks — turbostat during a run decides.

### 08:04 — STREAM thread sweep on the bare-metal host: first results

The prompt changed from `root@openwebui:~#` to `root@galactus:~/STREAM#` — Paul was now on the host, not the container, with the binary built as `./stream_c`. He launched a full thread sweep:

```bash
for t in 16 24 32 40 48 56 64 80 96 112 128; do   echo "===== THREADS=$t =====";   OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores ./stream_c |     awk -v t="$t" '/Copy:|Scale:|Add:|Triad:/{print "Threads=" t, $0}';   echo; done | tee stream_sweep.log
```

Results pasted so far (t=16 and t=24 only):

| Threads | Kernel | Best rate (MB/s) | Avg time (s) | Min time (s) | Max time (s) |
|---:|---|---:|---:|---:|---:|
| 16 | Copy | 151831.4 | 0.105755 | 0.105380 | 0.109707 |
| 16 | Scale | 103367.2 | 0.155244 | 0.154788 | 0.156321 |
| 16 | Add | 111945.6 | 0.216259 | 0.214390 | 0.226713 |
| 16 | Triad | 112513.9 | 0.214323 | 0.213307 | 0.217742 |
| 24 | Copy | 142439.9 | 0.113202 | 0.112328 | 0.125969 |
| 24 | Scale | 97448.0 | 0.164636 | 0.164190 | 0.169676 |
| 24 | Add | 105331.5 | 0.228328 | 0.227852 | 0.231805 |
| 24 | Triad | 105418.5 | 0.228264 | 0.227664 | 0.231854 |

*File artifact:* `galactus_triad.txt` (saved 08:02:59 ET) — the STREAM thread sweep; the sweep also tees to `stream_sweep.log`.

**Observations**

- Key figures at t=16: Copy 151.8 GB/s, Scale 103.4, Add 111.9, Triad 112.5. At t=24: Copy 142.4, Scale 97.4, Add 105.3, Triad 105.4.
- Throughput *drops* going 16 → 24 threads — every kernel loses ~6%.
- The Copy-versus-Scale gap (151.8 versus 103.4) is conspicuous and unexplained at this point.

### 08:05 — DIMM population captured with dmidecode

While the STREAM discussion continued, the DIMM inventory prescribed in the 08:03 checklist was captured to a file on the host:

```bash
dmidecode -t memory | grep -E 'Locator|^\s+Size|Configured Memory Speed|^\s+Speed|Rank' | grep -v 'No Module'
```

*File artifact:* `galactus_dmidecode_memory.txt` (saved 08:05:58 ET) — DIMM population.

**Observations**

- The output was not pasted into the dialogue; two minutes later the STREAM analysis rendered the question moot ("Skip the dmidecode; the memory subsystem is fine").

### 08:07 — RFO accounting: all four STREAM kernels converge on ~150 GB/s

Analysis of the partial sweep: the STREAM numbers are better than they look. The Copy-versus-Scale discrepancy (151.8 versus 103.4) is an artifact of how STREAM counts bytes: at -O3, GCC's loop-distribute turns the Copy kernel into memcpy, and glibc memcpy uses non-temporal stores above the L3 threshold (NT skips read-for-ownership). Scale/Add/Triad get ordinary vectorized stores and pay RFO — every cache line written is first read, and STREAM does not count that traffic:

| Kernel | Reported (GB/s) | Counted B/iter | Actual B/iter | Real DRAM traffic |
|---|---:|---:|---|---:|
| Copy | 151.8 | 16 | 16 (NT) | 152 GB/s |
| Scale | 103.4 | 16 | 24 (RFO) | 155 GB/s |
| Add | 111.9 | 24 | 32 (RFO) | 149 GB/s |
| Triad | 112.5 | 24 | 32 (RFO) | 150 GB/s |

"All four converge on ~150 GB/s. That is 80% of the 187.7 GB/s theoretical for eight channels of DDR4-2933 — a healthy, normal figure, and it confirms all eight channels are populated and running at speed. Skip the dmidecode; the memory subsystem is fine." Caveat (marked): the NT/RFO reading is inference from the shape of the numbers, not verified in the binary — but Copy at 151.8 GB/s is a hard floor either way, and llama.cpp weight reads are pure-read, which on Milan typically runs above Triad: "call the ceiling ~150–170 GB/s."

"Bandwidth peaks at ≤16 threads and is already declining at 24." Going 16 → 24 threads costs 6% on every kernel — the memory subsystem saturates and then degrades under queueing pressure. "You are running llama.cpp with 64." llama.cpp's Q4_K kernel has real compute per byte (nibble unpack, 6-bit scale unpack, maddubs/madd chains), so its optimum will be higher than STREAM's — but thread count is "a first-class tuning axis, not a rounding error." Let the sweep finish to 128.

**Correction ("Correcting my own arithmetic"):** the earlier "69 GB/s" was total_bytes / total_time, which wrongly charges GPU time to the CPU. Proper per-token budget:

```
per token @ 201 ms (4.97 t/s):

GPU  ~13.3 GB   dense weights (MLA projections, shared experts,
                3 dense layers, output head) — read serially, one
                card at a time under layer split, ~400 GB/s
             → ~33 ms

CPU   13.9 GB   routed experts (8 × 3 × 6144 × 2048 × 75 @ 4.92 bpw)
                at the measured 150 GB/s ceiling
             → ~93 ms

ideal          ~126 ms  →  7.9 t/s
actual          201 ms  →  5.0 t/s
unaccounted    ~75 ms/token
```

"There is roughly a 1.6× sitting in overhead, before touching VRAM at all." The GPU figure is an estimate; pin it down with:

```bash
llama-bench -m … -ngl 99 -ot "exps=CPU" -mmp 0 -t 64 -fa 1 -nopo 1 \
  -p 0 -n 8 -r 1 -v 2>&1 | grep -E 'load_tensors|system_info|load_backend'
```

(prints per-device model buffer sizes).

**Where 75 ms/token can go.** With `-ot exps=CPU` the graph splits at every MoE layer — per token: 75 CPU fork/joins, ~150 device boundaries, ~375 intra-split thread barriers (gate → up → silu → mul → down). If the threadpool sleeps between splits or cores drop into C1/C2, all of it lands on the critical path. 75 ms / 75 splits = 1 ms per split — "entirely plausible for a sleeping 64-thread pool on a 64-core box." Three non-redundant knobs:

```bash
# userspace: keep the threadpool spinning
--poll 100 --cpu-strict 1

# kernel: keep the cores out of idle states
cpupower frequency-set -g performance
cpupower idle-set -D 0

# BIOS, if you can reach it: disable DF C-states, Determinism = Performance
```

Note: "You are now on galactus rather than the container, so cpupower and nice will actually work."

**The battery, in order:**

1. Decode sweep — one model load, no reload (32 combinations; `-t`, `--poll`, `--cpu-strict` are all context params):

```bash
numactl --physcpubind=0-127 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -fa 1 -nopo 1 \
  -t 16,24,32,40,48,64,96,128 --poll 0,100 --cpu-strict 0,1 \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

2. Clocks during that run:

```bash
turbostat --quiet --show Core,Bzy_MHz,PkgWatt --interval 2
```

(base 2.0 / boost 3.675; want 2.6–3.0 GHz all-core AVX2; Bzy_MHz near 2.0 → Determinism Slider on Power or a cTDP cap = "flat 30% tax on everything").

3. Transparent huge pages:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled     # almost certainly [madvise]
echo always      > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
# during load:
watch -n5 "grep -E 'AnonHugePages' /proc/meminfo"   # should climb toward ~415 GiB
```

(ggml uses posix_memalign(), never madvise(MADV_HUGEPAGE); the gain is mostly prefetcher continuity — AMD's L2 stream prefetcher stops at 4 KiB boundaries, restarting every 64 cache lines versus running 32,768. Expect 5–25%, not 2×. Requires `-mmp 0`.)

4. Finish the STREAM sweep to 128 threads — "If it keeps declining, the llama.cpp -t optimum is likely well below 64 and that alone may be a large part of the 75 ms."

Closing: once the `-t`/`--poll` grid and the turbostat readings exist, "we will know whether the 75 ms is scheduling overhead, clock throttling, or thread oversubscription — and those have completely different fixes."

**Observations**

- **Confirmed:** the memory subsystem is healthy — ~150 GB/s real DRAM traffic, 73% of theoretical; all eight channels populated and at speed; the dmidecode check is unnecessary.
- **Correction:** the per-token budget now separates GPU (~13.3 GB dense, ~33 ms) from CPU (13.9 GB experts, ~93 ms): ideal ~126 ms (7.9 t/s) versus actual 201 ms → ~75 ms/token unaccounted.
- **Hypothesis:** the 75 ms is split/synchronization overhead — 75 fork/joins, ~150 device boundaries, ~375 barriers per token; 1 ms per split is plausible for a sleeping threadpool.
- Supporting estimates: V620 GDDR6 peak 512 GB/s/card, ~350–400 GB/s effective GEMV; under layer split only one GPU is active at a time at batch 1; the 7713's 8 CCDs each reach the IOD at ~51.2 GB/s read (GMI2 at FCLK 1600) — aggregate CCD→IOD exceeds DRAM, so not the bottleneck; active non-expert weights ~18.45 B params ≈ 13.3 GB/token; KV cache negligible at short context.

### 08:07 — Full STREAM thread sweep, 16–128 threads

Paul pasted the completed sweep from `root@galactus:~/STREAM#` (same command as at 08:04):

```bash
for t in 16 24 32 40 48 56 64 80 96 112 128; do   echo "===== THREADS=$t =====";   OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores ./stream_c |     awk -v t="$t" '/Copy:|Scale:|Add:|Triad:/{print "Threads=" t, $0}';   echo; done | tee stream_sweep.log
```

| Threads | Kernel | Best rate (MB/s) | Avg time (s) | Min time (s) | Max time (s) |
|---:|---|---:|---:|---:|---:|
| 16 | Copy | 151831.4 | 0.105755 | 0.105380 | 0.109707 |
| 16 | Scale | 103367.2 | 0.155244 | 0.154788 | 0.156321 |
| 16 | Add | 111945.6 | 0.216259 | 0.214390 | 0.226713 |
| 16 | Triad | 112513.9 | 0.214323 | 0.213307 | 0.217742 |
| 24 | Copy | 142439.9 | 0.113202 | 0.112328 | 0.125969 |
| 24 | Scale | 97448.0 | 0.164636 | 0.164190 | 0.169676 |
| 24 | Add | 105331.5 | 0.228328 | 0.227852 | 0.231805 |
| 24 | Triad | 105418.5 | 0.228264 | 0.227664 | 0.231854 |
| 32 | Copy | 149915.0 | 0.106820 | 0.106727 | 0.106895 |
| 32 | Scale | 100844.0 | 0.159070 | 0.158661 | 0.162049 |
| 32 | Add | 110649.5 | 0.217652 | 0.216901 | 0.224602 |
| 32 | Triad | 110620.6 | 0.217168 | 0.216958 | 0.217430 |
| 40 | Copy | 131550.9 | 0.121885 | 0.121626 | 0.124288 |
| 40 | Scale | 90391.4 | 0.177199 | 0.177008 | 0.178226 |
| 40 | Add | 97973.6 | 0.246132 | 0.244964 | 0.263584 |
| 40 | Triad | 97975.6 | 0.245351 | 0.244959 | 0.248582 |
| 48 | Copy | 143287.2 | 0.111816 | 0.111664 | 0.112936 |
| 48 | Scale | 96297.4 | 0.166480 | 0.166152 | 0.169055 |
| 48 | Add | 105824.3 | 0.227362 | 0.226791 | 0.232588 |
| 48 | Triad | 105867.3 | 0.227469 | 0.226699 | 0.233245 |
| 56 | Copy | 143317.8 | 0.112044 | 0.111640 | 0.114641 |
| 56 | Scale | 98433.6 | 0.162873 | 0.162546 | 0.164845 |
| 56 | Add | 108064.8 | 0.222399 | 0.222089 | 0.224996 |
| 56 | Triad | 108131.0 | 0.222771 | 0.221953 | 0.234898 |
| 64 | Copy | 144635.2 | 0.110785 | 0.110623 | 0.111242 |
| 64 | Scale | 98470.7 | 0.162965 | 0.162485 | 0.167849 |
| 64 | Add | 108750.8 | 0.221193 | 0.220688 | 0.223312 |
| 64 | Triad | 108813.0 | 0.220953 | 0.220562 | 0.223116 |
| 80 | Copy | 129120.8 | 0.124188 | 0.123915 | 0.124362 |
| 80 | Scale | 88254.0 | 0.181481 | 0.181295 | 0.181940 |
| 80 | Add | 96860.2 | 0.248104 | 0.247780 | 0.248866 |
| 80 | Triad | 96941.4 | 0.247976 | 0.247572 | 0.248696 |
| 96 | Copy | 140384.2 | 0.114090 | 0.113973 | 0.114273 |
| 96 | Scale | 94622.9 | 0.169583 | 0.169092 | 0.175544 |
| 96 | Add | 104775.6 | 0.229277 | 0.229061 | 0.229888 |
| 96 | Triad | 104869.4 | 0.229013 | 0.228856 | 0.229596 |
| 112 | Copy | 139904.0 | 0.114847 | 0.114364 | 0.117856 |
| 112 | Scale | 96152.1 | 0.166661 | 0.166403 | 0.168776 |
| 112 | Add | 106291.5 | 0.225995 | 0.225794 | 0.226519 |
| 112 | Triad | 106299.6 | 0.226158 | 0.225777 | 0.229384 |
| 128 | Copy | 140847.4 | 0.114825 | 0.113598 | 0.119273 |
| 128 | Scale | 96315.3 | 0.167290 | 0.166121 | 0.172207 |
| 128 | Add | 107100.2 | 0.225652 | 0.224089 | 0.231465 |
| 128 | Triad | 107157.2 | 0.225205 | 0.223970 | 0.229781 |

**Observations**

- Bandwidth saturates at 16 threads and stays essentially flat (mildly declining) through 128.
- Reproducible dips at 40 and 80 threads on every kernel.

### 08:09 — Sweep decoded: saturation at 16 threads; the -ngl 0 experiment proposed

The full sweep was decoded. The RFO artifact is provable from the numbers: if Copy paid RFO, its real traffic would be 227.7 GB/s — above the 187.7 GB/s theoretical ceiling, which is impossible. So GCC turned the Copy loop into `memcpy`, glibc used non-temporal stores (threshold ~192 MB), and Copy's counted bytes equal its actual bytes; Scale/Add/Triad keep ordinary vector stores and pay uncounted RFO. Applying the corrections (Copy ×1.0, Scale ×1.5, Add/Triad ×1.333):

| threads | Copy | Scale | Add | Triad |
|---:|---:|---:|---:|---:|
| 16 | 152 | 155 | 149 | 150 |
| 32 | 150 | 151 | 148 | 148 |
| 48 | 143 | 144 | 141 | 141 |
| 64 | 145 | 148 | 145 | 145 |
| 96 | 140 | 142 | 140 | 140 |
| 128 | 141 | 144 | 143 | 143 |

Conclusion 1: the platform delivers ~150 GB/s real DRAM traffic — 73% of theoretical, normal; all eight channels populated and clocked correctly; the memory subsystem is healthy; "Skip `dmidecode`." Conclusion 2: bandwidth saturates at 16 threads and then decays slightly — 16→64 costs ~4%, 16→128 costs ~6%; the dips at 24, 40, 80 are jitter, not structure (judged Infinity Fabric clock / power-frequency effects, not CCD imbalance — not worth investigating). Decode ceiling called at ~150 GB/s (pure-read on Milan runs at or slightly above Triad).

Updated budget:

```
per token, current hybrid config (201 ms, 4.97 t/s):

CPU   13.9 GB  routed experts @ 150 GB/s  →   93 ms
GPU   13.3 GB  dense path (MLA, shared experts, dense layers, output head)
               read serially, one card at a time under layer split,
               ~3.3 GB/card @ ~380 GB/s effective  →   35 ms
                                                    ────────
ideal                                                 128 ms  →  7.8 t/s
actual                                                201 ms  →  5.0 t/s
unaccounted                                            73 ms     (36%)
```

Supporting arithmetic behind the model: 22.65 B active params → 45.3 GFLOP MAC work/token; at the measured prefill rate of 1.69 TFLOP/s that is ~26.8 ms, but at batch 1 the kernel is bandwidth-bound — 2 flops per 0.56 bytes = 3.5 flop/byte means sustaining 150 GB/s needs only 525 GFLOP/s against the 1690 GFLOP/s measured at large batch (~3.2× headroom): compute is not the limiter. GPU-side dense weights: non-expert at ~6.5 bpw ≈ 14.06 GB, minus ~0.77 GB (the token embedding is a single-row lookup) → ~13.3 GB/token; ~3.5 GB/card sequential ≈ 27 ms at full GDDR6 or ~10 ms/card ≈ 40 ms at ~350 GB/s effective; GPU attention reads ~122 MB/layer at 512 GB/s ≈ 0.24 ms/layer → ~19 ms over 78 layers; plus 20–30 kernels/layer ≈ ~1,950 launches/token at 5–10 µs ≈ 13–14 ms if ungraphed; GPU-side total ~30 ms. Successive budget passes gave gaps of 68, 73, and 78 ms.

**Decision: "The experiment I should have proposed three turns ago" — run `-ngl 0`.** Pure CPU eliminates all ~150 device boundaries (KV cache on CPU). **Prediction:** the CPU reads 27.2 GB/token (13.9 experts + 13.3 dense) at 150 GB/s → 181 ms → ~5.5 t/s, with zero device boundaries, PCIe syncs, or graph splits. The hybrid with four V620s currently gives 4.97 t/s. If `-ngl 0` matches or beats it, the GPUs are net-zero or net-negative and the 73 ms gap is the hybrid-split cost (75 CPU fork/joins, ~150 device boundaries, ~2,000 un-graphable kernel launches per token). If it instead measures ~3.5 t/s, the CPU is really running ~100 GB/s.

```bash
# ~3 minutes of compute, one model load
numactl --physcpubind=0-127 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 0 -mmp 0 -fa 1 -t 32,64 --poll 0,100 \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

Then the thread sweep:

```bash
numactl --physcpubind=0-127 llama-bench \
  -m … -ngl 99 -ot "exps=CPU" -mmp 0 -fa 1 -nopo 1 \
  -t 16,24,32,48,64,96,128 --poll 0,100 --cpu-strict 0,1 \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

Interpretation keys: tg peaks at 16–32 threads → the Q4_K GEMV is bandwidth-bound like STREAM; 48 surplus threads buy nothing while paying a 64-wide barrier ~375 times per token → drop `-t`. tg keeps climbing past 48 → nibble/scale unpack is compute-limited, the kernel is not reaching 150 GB/s, the thread count was correctly set, and the 73 ms is elsewhere.

Row split noted: under `-sm layer` only one card is live at a time → the dense path costs 35 ms rather than ~9; `-sm row` parallelizes across four cards and could return ~20 ms/token if it works for glm-dsa (no ROW architecture exclusion in source, unlike TENSOR); requires a model reload, so run as its own invocation. THP and turbostat items reiterated. **Decision:** "Run the `-ngl 0` baseline first. It is the cheapest and it will tell us more than the other three combined."

**Observations**

- **Confirmed:** ~150 GB/s is the platform's real DRAM ceiling; saturation at 16 threads; 16→64 −4%, 16→128 −6%.
- **Correction:** the ideal per-token budget is 128 ms (7.8 t/s); the unaccounted gap is 73 ms (36%).
- **Prediction:** `-ngl 0` lands near 5.5 t/s if the CPU truly runs at ceiling; near 3.5 t/s if the kernel only manages ~100 GB/s.
- **Decision:** `-ngl 0` is the decisive next experiment.

### 08:10 — SMT topology check: sibling pair 0,64

Paul ran the outstanding enumeration check on the host:

```bash
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list   # want "0,64"
```

Result: `0,64`

**Observations**

- **Confirmed:** CPUs 0–63 are one thread per physical core; 64–127 are the SMT siblings. `--physcpubind=0-63` was correct and `-t 64` really was 64 physical cores — no prior number is invalidated.

### 08:13 — CCD placement analysis, -C affinity masks, and the HIP-graphs hypothesis ruled out

The `0,64` confirmation triggered a placement insight. Milan CCD facts: the EPYC 7713 has 8 CCDs × 8 cores × 32 MB L3; each CCD connects to the IOD via one GMI2 link (~32 bytes/clock read) ≈ 47–51 GB/s per CCD (~50 GB/s used in the analysis). Two CCDs cap at ~102 GB/s; at least 3–4 CCDs must be active to reach 150 GB/s, regardless of thread count. CPU enumeration follows CCD order (cores 0–7 = CCD0, 8–15 = CCD1, …).

**Hypothesis:** 16 llama.cpp threads packed on CPUs 0–15 occupy only 2 CCDs and cap near ~100 GB/s; spread 2-per-CCD they reach the full 152 GB/s. STREAM hit 152 GB/s at 16 threads only because `OMP_PROC_BIND=spread OMP_PLACES=cores` spread the threads. llama.cpp does not do this by default — verified in `ggml_thread_cpumask_next()`: `--cpu-strict 0` (default) gives every thread the entire mask and the kernel scheduler migrates at will; `--cpu-strict 1` pins thread i to the i-th set bit. `parse_cpu_mask()` reads `-C` as ordinary big-endian hex, bit 0 = CPU 0, up to 128 hex digits (16 hex digits cover CPUs 63…0).

| `-t` | per CCD | `-C` mask |
|---:|---|---|
| 8 | 1 | `0101010101010101` |
| 16 | 2 | `0303030303030303` |
| 24 | 3 | `0707070707070707` |
| 32 | 4 | `0f0f0f0f0f0f0f0f` |
| 48 | 6 | `3f3f3f3f3f3f3f3f` |
| 64 | 8 | `ffffffffffffffff` |
| 16 packed | 8 on 2 CCDs | `000000000000ffff` ← control |

The last row is the experiment: 16 spread versus 16 packed at the same thread count. If spread wins big, the expert GEMV is fabric-limited and thread placement is a first-class knob never touched before.

Outstanding measurement A — the `-ngl 0` baseline (27.2 GB/token, zero device boundaries):

```bash
numactl --physcpubind=0-127 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 0 -mmp 0 -fa 1 -nopo 1 \
  -t 32,64 --poll 0,100 -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

If pure-CPU lands near 5.5 t/s while hybrid gives 4.97, the four V620s contribute nothing net and the whole 73 ms is split cost. Outstanding measurement B — the placement sweep (one load):

```bash
numactl --physcpubind=0-127 llama-bench \
  -m … -ngl 99 -ot "exps=CPU" -mmp 0 -fa 1 -nopo 1 \
  --cpu-strict 1 --poll 100 \
  -t 16 -C 0303030303030303,000000000000ffff \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

then widen to `-t 16,32,48,64` with matching spread masks.

**Hypothesis ruled out: HIP graphs disabled.** `GGML_HIP_GRAPHS` was verified to default ON, and the `cc < GGML_CUDA_CC_AMPERE` disable gate does not fire on AMD (AMD compute-capability values carry a `0x1000000` offset, so gfx1030 comfortably clears the threshold). With `-ot exps=CPU`, each split gets its own graph key and cached graph — 75–150 small graphs replayed is workable (75 × 10 µs ≈ 0.75 ms). Kernel-launch overhead is not the explanation for the ~2,000-launches-per-token concern.

Supporting overhead arithmetic from the analysis: a spin barrier costs 1–3 µs at 64 threads, a futex-based one 20–100 µs; 375 barriers/token at 64 threads with futex ≈ 18.75 ms/token; fork/join wakeups at 75 CPU–GPU split boundaries add 0.75–3.75 ms; total threading overhead plausibly 5–25 ms — not the full 73 ms. Scheduler-sync accounting: 75 MoE layers × 2 syncs × ~20 µs ≈ 3 ms, threadpool wake-ups ~60 µs/layer, total ~10 ms — also nowhere near 73 ms. **Correction (self):** the 93 ms CPU estimate assumes STREAM-level bandwidth; the Q4_K kernel (144-byte block = 256 weights; 1.78 weights/byte; 3.56 flop/byte; needs ~534 GFLOP/s int8 to sustain 150 GB/s) likely achieves only 60–80% of STREAM → 13.9 GB at ~110 GB/s ≈ 126 ms CPU + 35 ms GPU → ~40 ms of overhead is more realistic than 73. If `-ngl 0` measures 4.5 t/s on 27.2 GB → 222 ms/token → ~122 GB/s effective = 81% of STREAM ("quite reasonable"). A recalculation also put non-expert weights at 19.4 B params × ~6.5 bpw ≈ 15.8 GB total (~3.9 GB/GPU, ~39.5 ms serial), and MLA kernel launches at 25–40/layer × 78 layers = 2,000–3,000/token (2,500 × 6 µs ≈ 15 ms if ungraphed).

**Decision:** "Run A first."

**Observations**

- **Confirmed:** SMT enumeration (0,64) — no prior measurement invalidated.
- **Hypothesis:** CCD/fabric placement may cap llama.cpp's effective bandwidth well below the 152 GB/s that spread-placement STREAM achieves; the 16-spread versus 16-packed mask pair is the designed test.
- **Refuted:** the HIP-graphs-off explanation for kernel-launch overhead (`GGML_HIP_GRAPHS` defaults ON; the Ampere gate never fires on AMD).
- **Correction:** barrier/sync arithmetic suggests threading overhead of 5–25 ms, not 73 ms; a kernel running at 60–80% of STREAM would account for much of the rest.

### 08:13 — Kernel tuning applied on the host; request for an all-day diagnostic script

Paul reported having just set, on the host:

```bash
echo always      > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
cpupower idle-set -D 0        # disable C-states deeper than C0
```

*File artifact:* `galactus_kernel_tuning.txt` (saved 08:11:58 ET) — the THP and C-state commands.

He was leaving the house for the day and requested one large script that logs everything possible to a file for full diagnostics later. The output might be fed to a different LLM ("like a newer version of you"), so the script must log all system context (NUMA info and the like) and contain all context itself; it can run many llama-bench commands testing different flags; "This will run all day."

**Observations**

- **Decision (Paul):** move from interactive iteration to a single unattended, self-documenting diagnostic battery.
- Kernel state now: THP `always`, defrag `defer+madvise`, C-states deeper than C0 disabled (host-side).

### 08:22 — galactus-diag.sh v1: Phase 0 inventory plus twelve benchmark loads, 11-hour deadline

The script was designed and delivered. Design constraints worked through: a model load takes ~17 minutes with mmap disabled (435 GiB in 1041 s ≈ 428 MB/s — slow for NVMe at 3 GB/s; **Hypothesis:** the models may sit on network storage — a Phase 0 raw-read throughput test was added). `-mmp 1` would drop the footprint to 435 GiB and make repeat loads nearly free, but disables THP for the weights and the pinned-host/repack buffers; **Decision:** run everything with `-mmp 0` (the production configuration, THP-eligible) and design for 8–14 total invocations. With `-mmp 0`: 415 GiB anonymous + 435 GiB page cache ≈ 850 GiB on 1 TB — tight; log `free` and `/proc/meminfo` before and after each load. Timing model: tg128 at ~5 t/s → ~26 s/rep, ~50 s/combo at 3 reps plus warmup; ~20 combos ≈ 20 minutes of bench per invocation; 10–14 invocations ≈ 3 h loading + 4 h benchmarking ≈ 7 h. Crash risk handled: `-nopo 0` crashes with a ROCm error at certain `-ub` thresholds and llama-bench aborts the whole invocation, losing all data → risky combos isolated in their own invocation, and the ubatch ladder runs ascending (llama-bench prints rows incrementally, so results before a crash survive). The mask/thread cartesian-product problem: masks need at least as many set bits as the largest `-t`; with `--cpu-strict 1` only the first N set bits matter → the main `-t` sweep runs `--cpu-strict 0` (what users actually run) with `--poll 0,100`, and a separate invocation at t=16 varies only the `-C` masks (8/4/2-CCD variants: `0x0303030303030303` for 2/CCD, `0x000000000f0f0f0f` for 4/CCD on 4 CCDs, `0x000000000000ffff` for 8 on 2 CCDs). `--prio` mapping checked: `--prio 2` = HIGH, `--prio 3` = SCHED_FIFO realtime — 64 spinning threads at realtime could make the box unresponsive → the script never touches `--prio`. Phase structure derives from `equal_mparams` in the source (reload forced only by `-m`, `-ngl`, `-ncmoe`, `-sm`, `-ot`, `-mmp`, …); noted that `-fitt` (fit_params_target) is missing from the `equal_mparams` list — a possible upstream bug/oversight. The v1 phase plan: Phase 0 inventory (no model load), then B-phases of exactly one load each — CPU-only baseline sweep, CCD masks CPU-only, hybrid sweep, hybrid CCD masks, op_offload ubatch ladder (risky, isolated), `--no-host` variant, mmap comparison, `-sm row`, THP A/B (two loads, madvise versus always, run last, restore setting), VRAM fitter, depth test — ~11–13 loads ≈ 3 h of loads + ~2.5 h of bench ≈ 5.5–6 h. Prewarming the page cache via `cat`/`dd` was rejected (takes as long as a load); instead Phase 0 logs memory stats and measures raw read speed with direct I/O. `-v` runs only on the first invocation of each distinct model configuration; background monitors sample GPU memory, clocks, and resources every few seconds to CSVs. A syntax error in the `cap()` helper (bad parameter expansion) was found and fixed; the script was syntax-checked and made executable.

Run instructions:

```bash
cd /root/STREAM          # so it finds ./stream_c
chmod +x galactus-diag.sh
nohup ./galactus-diag.sh > /root/galactus-console.txt 2>&1 &
```

Output lands in `/root/diag-<timestamp>/MAIN.log` plus a `monitors/` directory. MAIN.log opens with a self-contained briefing (machine, model architecture, every measurement so far, the six hypotheses, and how to read the results) so a fresh model can pick it up cold. Phase 0 is the inventory "we have been guessing at": DIMM population and configured speed (settles the 150 GB/s story), PCIe LnkSta for all four V620s (hard ceiling on GPU prefill), the CCD/L3 domain map (verifies the `-C` masks), GGUF expert tensor types (settles whether CPU_REPACK can ever engage), raw versus buffered read throughput on a model shard (explains the 17-minute loads), the build's CPU backend variant, THP state, C-state disable flags, IOMMU mode, and cgroup limits. Phases B1–B12: twelve llama-bench invocations, each exactly one model load, all context params swept inside, structured from `equal_mparams`. The two that matter most: B1 (`-ngl 0`) versus B3 (hybrid) at matched `-t`/`--poll` — "The delta *is* the 73 ms. If `-ngl 0` matches or beats 4.97 t/s, the four V620s are net-zero and the split is the whole problem." And B6 — the op_offload ubatch ladder, ordered small→large with `-v` on; a crash at `-ub 4096` still leaves the 128/256/512/1024 rows in the log and captures the real HIP error string. B2/B4 hold the thread count at 16 and vary only CCD spread (8/4/2 CCDs) — separating the fabric limit from the thread-count limit. Guardrails: `set -uo pipefail` but deliberately not `-e`; every phase wrapped so a ROCm abort logs and moves on; each invocation under a 2-hour `timeout --kill-after`; an 11-hour wall-clock deadline (`DEADLINE_HOURS=11`) skips remaining phases; an exit trap restores THP to `always`; `--prio` never touched. Budget: ~30 min inventory + 12 loads + ~2.5 h benching; at 17-minute loads, expect 6–7 hours total.

*File artifact:* `galactus-diag.sh` (v1, delivered via the conversation at `/mnt/user-data/outputs/galactus-diag.sh`).

**Observations**

- **Decision:** production configuration (`-mmp 0`) throughout, one model load per phase, crash-risky combinations isolated, results crash-survivable via incremental rows.
- **Hypothesis (carried into Phase 0):** model storage may be network-backed (428 MB/s observed load rate).
- Possible upstream llama.cpp bug noted: `-fitt` absent from `equal_mparams`.

### 08:27 — Deadline removal requested

Paul: "Can you remove the deadline entirely. I am going to let it run overnight if need be"

**Observations**

- **Decision (Paul):** the battery runs to completion regardless of wall-clock time.

### 08:37 — galactus-diag.sh v2: deadline removed; B0, B7, B11 added; a dead-code discovery

v2 was delivered. The deadline logic is gone — no `DEADLINE_HOURS`, no `deadline_ok()`, no skipping; every phase runs to completion. The per-invocation `timeout` is kept as "a hang guard, not a budget", raised to `BENCH_TIMEOUT=21600` (six hours; `BENCH_TIMEOUT=0` disables — GNU timeout treats 0 as none), so one wedged ROCm call cannot eat the night. Runtime estimates: B1 = 24 combos × ~170 s worst case + 17-minute load ≈ 85 minutes. The B12 depth test was trimmed: `-d 65536` needs a 65,536-token prefill at ~37 t/s ≈ 30 min/rep (1 h at 2 reps) → trimmed to `-d 0,4096,16384`, keeping rows under 15 minutes at `-r 2`; the full run ≈ 59 minutes → ~80 with overhead.

Additions: **B0** — a `GGML_SCHED_DEBUG=1` split dump, run with `-n 1 -r 1 --no-warmup` and piped through `head -300` (the scheduler prints splits on every graph compute; 64 tokens would flood the log). It "measures H1 instead of inferring it — we finally get the real number of graph splits per token rather than my estimate of ~150"; costs one ~17-minute load for ~2 minutes of work; "the most direct test in the whole battery." **B7** — `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (switches `ggml_cuda_device_malloc()` to `cudaMallocManaged()`, permitting VRAM oversubscription): if the B6 abort is really an OOM this makes it survive; if it still aborts, it is not an OOM — decisive either way. **B11** — `GGML_CUDA_GRAPH_OPT=1` (graph optimization + concurrent stream-event launching; off by default; second-order but free). Dual-purpose CCD masks: with ascending-bit semantics one 32-bit mask means different things at t=16 versus t=32, so the masks were picked to be meaningful at both — B2/B4 now cover five CCD configurations instead of three at no extra reload; the candidate mask `000000000000ffff000000000000ffff` (32 threads on 2 CCDs with SMT) was rejected as conflating two variables; Phase 0.2 dumps `/sys/.../cache/index3/shared_cpu_list` to verify the core→CCD mapping rather than assume it.

Environment variables verified by reading the `getenv()` calls in `ggml-cuda.cu`: `GGML_CUDA_GRAPH_OPT`, `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_NO_PINNED`, `GGML_CUDA_P2P` (skipped — direction unverifiable), `GGML_CUDA_ALLREDUCE`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY`, `GGML_SCHED_DEBUG` (=1 prints split counts, =2 per-node assignments), `GGML_OP_OFFLOAD_MIN_BATCH` (default 32; sweep skipped). `GGML_CUDA_DISABLE_GRAPHS` is not in current master. **Dead end:** `GGML_CUDA_REGISTER_HOST` looks ideal (calls `cudaHostRegister` on the CPU weight buffer, page-locking for fast H2D — promising with `-mmp 1`), but `ggml_backend_cuda_register_host_buffer()` is exported and never called from llama.cpp `src/` — dead code; "Setting it does nothing." (Also noted a 415 GiB `cudaHostRegister` could fail or be slow anyway.) Recorded in the briefing so nobody wastes time on it later.

The log's briefing is self-contained: the machine, the GLM-5.2 architecture with every derived number, the corrected STREAM table with the RFO reasoning, the 201 ms budget and the 73 ms hole, all six hypotheses mapped to phases, five settled questions (including ZenDNN being inert and MTP being unavailable), and the `equal_mparams` mechanics dictating the phase structure. "Fourteen phases, each exactly one model load. Ordered by information value" — inventory (no load) → B0 sched-debug split dump → `-ngl 0` baseline sweep (the denominator) → CCD masks at `-ngl 0` → hybrid sweep (the main event) → hybrid CCD masks → hybrid ubatch ladder with/without op_offload → unified memory → `--no-host` → `-mmp` → row split → graph-opt A/B → THP A/B → VRAM fitter → depth test (the working estimate behind this ordering was ~15 loads ≈ ~8 h). Per-phase timing instrumentation added (phase counter plus running elapsed total), and env-var prefix support in the bench helper.

*File artifact:* `galactus-diag.sh` (v2, same delivery path `/mnt/user-data/outputs/galactus-diag.sh`).

**Observations**

- **Decision:** no wall-clock deadline; six-hour per-invocation hang guard only.
- **Dead end:** `GGML_CUDA_REGISTER_HOST` is dead code in llama.cpp — eliminated before it could waste a phase.
- B0 (split dump) converts the central overhead hypothesis from an estimate (~150 splits/token) into a measurement; B7 makes the B6 crash diagnosis decisive either way.

### 08:40 — STREAM missing inside the container

Paul: "Oh, I don't have STREAM inside the container. Give me instructions to download and build it"

**Observations**

- This surfaced a configuration split that had gone unnoticed: the morning's tooling had been installed in two different places.

### 08:43 — Host/container split identified; STREAM build for the container; capability audit

A key fact surfaced directly: `llama-bench` lives in the `openwebui` container, but STREAM, `cpupower`, and the THP sysfs writes were all done on the `galactus` host — half of the script's inventory would not work in the container (the earlier `nice: cannot set niceness: Permission denied` was already evidence of missing CAP_SYS_NICE). The container has ~24 GB of headroom — enough for STREAM's 22.4 GiB. Running STREAM inside the container is itself a diagnostic: if it comes back materially slower than the host's 152 GB/s, the container is CPU-throttled (cgroup quota/cpuset), which could explain part of the 73 ms gap. THP is kernel-wide, so the host's `always` applies inside the container; only the toggle needs host access. Running llama-bench on the host instead was considered and rejected (ROCm userspace and the model are containerized).

Build instructions:

```bash
apt-get update && apt-get install -y build-essential wget

mkdir -p /root/STREAM && cd /root/STREAM
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c

gcc -O3 -march=native -fopenmp -mcmodel=medium \
    -DSTREAM_ARRAY_SIZE=1000000000 \
    -DNTIMES=10 \
    stream.c -o stream_c
```

Non-optional flags: `-mcmodel=medium` — the STREAM arrays are `static`; at 1e9 doubles that is 8 GB each, 24 GB total, past the 2 GB small-code-model limit; without it the link fails with `relocation truncated to fit: R_X86_64_PC32`. `-DSTREAM_ARRAY_SIZE=1000000000` — derived from Paul's own output: Copy reported 151831.4 MB/s at 0.105380 s best time = 16,000 MB moved; Copy touches 2 arrays × 8 B × N → N = 1e9; the same value keeps the container run comparable to the host. `-O3` — makes GCC turn the Copy loop into memcpy with NT stores (the source of the RFO artifact); keep applying the ×1.5 / ×1.333 corrections. Verify:

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=cores ./stream_c
```

It should print `Array size = 1000000000`, `Total memory required = 22888.2 MiB`, and Copy near 152000 MB/s. "If it does not — that is a finding, not noise" (container CPU-throttled). Immediate container checks (any one failing invalidates the whole battery):

```bash
cat /sys/fs/cgroup/cpu.max              # "max 100000" = unlimited. A number = CFS quota throttling.
cat /sys/fs/cgroup/cpuset.cpus.effective # should be 0-127
cat /sys/fs/cgroup/memory.max            # "max" = unlimited
```

**Hypothesis:** a CFS quota throttling the 64-thread pool during the ~75 fork/joins per token "would look precisely like the 'unaccounted 73 ms.'"

What breaks inside the container: `turbostat` needs `/dev/cpu/*/msr` plus CAP_SYS_RAWIO — will fail → H6 (clock throttling) goes untested; `dmidecode` needs `/sys/firmware/dmi/entries` — usually absent → DIMM population/speed unknown; `dmesg` needs CAP_SYSLOG → no IOMMU/PCIe error visibility; writes to `/sys/kernel/mm/transparent_hugepage/enabled` — sysfs read-only → the THP A/B phase silently fails; `lspci -vv` usually works (PCIe LnkSta fine); `rocm-smi`, `numactl`, `lscpu`, `/proc/meminfo` all work. Fix Option A — restart the container privileged (also fixes `nice`):

```bash
docker run ... --privileged ...
# or, minimally:
#   --cap-add=SYS_NICE --cap-add=SYS_RAWIO --cap-add=SYSLOG
```

"For an overnight run on a home-lab box, `--privileged` is the right call." Fix Option B — the script was patched with `MODE=all|host|bench`: `MODE=host` runs the inventory then parks turbostat in the foreground for 8 hours (the only way to test H6 if the container cannot reach MSRs); `MODE=bench` skips the inventory and runs the battery. Container detection added to Phase 0.1: `cpu.max`, `cpuset.cpus.effective`, `cpu.stat nr_throttled` (>0 = actively throttled right now), and `CapEff` decoded for CAP_SYS_NICE/CAP_SYS_RAWIO. The THP A/B phase auto-detects read-only sysfs and, instead of failing silently, prints the current kernel-wide THP state, points to `monitors/meminfo.txt` for `AnonHugePages`, and gives exact host-side A/B commands. Run options:

```bash
# restart openwebui with --privileged (or at minimum:
#   --cap-add=SYS_NICE --cap-add=SYS_RAWIO --cap-add=SYSLOG)
cd /root/STREAM && nohup ./galactus-diag.sh > /root/console.txt 2>&1 &
```

or split:

```bash
# terminal 1, on galactus (host):
cd /root/STREAM && MODE=host ./galactus-diag.sh

# terminal 2, in the openwebui container:
cd /root/STREAM && MODE=bench nohup ./galactus-diag.sh > /root/console.txt 2>&1 &
```

then concatenate the two MAIN.log files (host run: dmidecode, dmesg, turbostat, writable-sysfs checks; container run: the fourteen benchmark phases). "One thing to check before you leave": run STREAM once in the container and compare Copy at 16 threads against the host's 151831 MB/s; if slower, stop and look at `cpu.max` and `cpu.stat`.

*File artifact:* `galactus-diag.sh` (patched with MODE support, same delivery path).

**Observations**

- **Confirmed:** the benchmark environment (container) and the tuning/measurement environment (host) had silently diverged all morning.
- **Hypothesis (new, testable in minutes):** container CFS-quota throttling could masquerade as the unaccounted 73 ms.
- **Decision:** in-container STREAM plus three cgroup reads before launch; MODE split as the non-privileged fallback.

### 08:44 — Container STREAM run and cgroup checks

Paul built and ran STREAM inside the container:

```bash
gcc -O3 -march=native -fopenmp -mcmodel=medium \
    -DSTREAM_ARRAY_SIZE=1000000000 \
    -DNTIMES=10 \
    stream.c -o stream_c
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=cores ./stream_c
```

STREAM header output: version $Revision: 5.10 $; 8 bytes per array element; `Array size = 1000000000 (elements), Offset = 0 (elements)`; `Memory per array = 7629.4 MiB (= 7.5 GiB)`; `Total memory required = 22888.2 MiB (= 22.4 GiB)`; each kernel executed 10 times; Number of Threads requested = 16, counted = 16; clock granularity/precision ~1 microseconds; each test on the order of 103736 microseconds (= 103736 clock ticks). Results:

| Kernel | Best rate (MB/s) | Avg time (s) | Min time (s) | Max time (s) |
|---|---:|---:|---:|---:|
| Copy | 153234.8 | 0.104556 | 0.104415 | 0.104733 |
| Scale | 104661.4 | 0.153106 | 0.152874 | 0.153387 |
| Add | 114007.8 | 0.211182 | 0.210512 | 0.213627 |
| Triad | 114308.0 | 0.211748 | 0.209959 | 0.220083 |

`Solution Validates: avg error less than 1.000000e-13 on all three arrays`

Cgroup checks:

```bash
cat /sys/fs/cgroup/cpu.max              # "max 100000" = unlimited. A number = CFS quota throttling.
cat /sys/fs/cgroup/cpuset.cpus.effective # should be 0-127
cat /sys/fs/cgroup/memory.max            # "max" = unlimited
```

Results: `max 100000`; `0-127`; `max`.

**Observations**

- The container's STREAM numbers slightly *exceed* the host's.
- No CFS quota, no cpuset restriction, no memory cap.

### 08:45 — Container exonerated; ceiling pinned at 152 GB/s; launch instructions

"The container is clean. That closes the hypothesis." Comparison at 16 threads (MB/s):

| 16 threads | Host (galactus) | Container (openwebui) |
|---|---:|---:|
| Copy | 151,831 | 153,235 |
| Scale | 103,367 | 104,661 |
| Add | 111,946 | 114,008 |
| Triad | 112,514 | 114,308 |

The container is 1–2% faster — pure run-to-run noise. RFO corrections (Copy ×1.0, Scale ×1.5, Add/Triad ×1.333) give 153 / 157 / 152 / 152 GB/s; all four converge, matching galactus exactly. The cgroups are wide open: `cpu.max = max`, `cpuset = 0-127`, `memory.max = max` — no CFS quota, no cpuset restriction, no memory cap. **Dead end closed:** container throttling ruled out. "The container is not costing you anything on the memory path, and every measurement in this thread stands. The ~75 ms is real and it is inside llama.cpp, not the sandbox." **Decision:** pin the ceiling at 152 GB/s → CPU expert reads 13.9 GB ÷ 152 GB/s = 91 ms; ideal total ~126 ms (7.9 t/s) versus actual 201 ms (4.97 t/s). The container still lacks CAP_SYS_NICE (why `nice -n -15` failed — Docker's default capability set excludes it) and /sys is read-only; neither affects the benchmarks themselves; only turbostat and the THP toggle need the host.

Launch, on galactus (host):

```bash
# install turbostat if you don't have it — this is the ONLY test of H6 (clock throttle)
apt install -y linux-tools-common linux-tools-$(uname -r)

cd /root/STREAM
MODE=host SKIP_STREAM=1 nohup ./galactus-diag.sh > /root/host-console.txt 2>&1 &
```

(grabs dmidecode, dmesg, and writable-sysfs state; parks turbostat sampling for 8 hours to capture the container's load from outside). In the openwebui container:

```bash
cd /root/STREAM        # stream_c is here now, so Phase 0.11 will find it
nohup ./galactus-diag.sh > /root/container-console.txt 2>&1 &
```

`MODE` defaults to `all`; the THP A/B phase auto-detects read-only sysfs, skips, and prints what to run later. Tomorrow's reading order: (1) `monitors/turbostat.txt` on the host — `Bzy_MHz` under load: 2.6–3.0 GHz = H6 dead; ~2.0 GHz = BIOS throttling 30% "and everything else is a footnote"; (2) B0 — the actual graph split count per token; (3) B1 versus B3 at matched `-t`/`--poll` — the 75 ms; (4) B6 — whether op_offload survives at any ubatch, and what prefill does when it does.

**Observations**

- **Confirmed:** in-container corrected bandwidths 153/157/152/152 GB/s — within noise of the host (host corrected range 151.8–155.1).
- **Refuted/Dead end:** container CPU throttling as a contributor to the decode gap.
- **Decision:** platform ceiling pinned at 152 GB/s; the working budget is 91 ms (experts) + ~35 ms (GPU dense) ≈ 126 ms ideal against 201 ms actual.

### 08:46 — Output locations and live monitoring

Paul launched the script and asked: "I'm running the script. Where will the output go?" Answer — two locations, one per run, both timestamped:

```
/root/diag-YYYYMMDD-HHMMSS/
├── MAIN.log          ← the whole thing: briefing, inventory, every bench
└── monitors/
    ├── turbostat.txt   (host run)  or  cpumhz.txt (fallback)
    ├── rocm-smi.txt
    ├── vmstat.txt
    └── meminfo.txt
```

Plus the nohup redirect files `/root/container-console.txt` and `/root/host-console.txt` — near-duplicates of MAIN.log (everything is tee'd to both), free redundancy. The container's `/root/diag-*` lives inside the container's filesystem and the host's on galactus; different timestamps, no collision. Find and watch:

```bash
D=$(ls -dt /root/diag-*/ | head -1); echo "$D"
tail -f "$D/MAIN.log"
```

Phase progress:

```bash
grep '^## ' "$D/MAIN.log" | tail -5
```

Monitors worth peeking at:

```bash
tail -3 "$D/monitors/meminfo.txt"   # AnonHugePages — should climb toward ~415 GiB
tail -5 "$D/monitors/vmstat.txt"    # si/so columns MUST stay 0
tail -8 "$D/monitors/rocm-smi.txt"  # VRAM use — B6/B7 will spike this
```

Getting the container's log onto the host before losing it (if openwebui is Docker and /root is not a volume, the log dies with the container; files are written incrementally):

```bash
# from galactus
docker cp openwebui:/root/$(docker exec openwebui bash -c 'ls -dt /root/diag-*/ | head -1 | xargs basename') /root/
```

Bundling both for handing to a model:

```bash
cd /root && tar czf galactus-diag.tar.gz diag-*/
```

Either MAIN.log stands alone (self-contained briefing), but both are wanted: the host run carries turbostat/dmidecode/dmesg; the container run carries the fourteen benchmark phases.

**Observations**

- The v2 battery is now running: `MODE=all` (default) in the container, with the `MODE=host SKIP_STREAM=1` companion prescribed on the host.
- Session 2 ends here; the battery runs unattended while Paul is out for the day.

**State of knowledge at end of session**

- Baseline established (hybrid `-ngl 99 -ot "exps=CPU"`, `-nopo 1`): pp512 = 37.63 ± 3.69 t/s, tg128 = 5.15 ± 0.40 t/s; Config B (`-nopo 1 --no-host 1`) is flat at 37.29 ± 3.24 / 4.97 ± 0.45 — the repack lever is refuted or never engaged; enabling op_offload crashes with a generic `ROCm error` at ggml-cuda.cu:104 after the 1041.9 s load, cause unconfirmed (VRAM-exhaustion and 415-GiB-pinned-host hypotheses outstanding).
- Platform characterized: EPYC 7713 Zen3, 64C/128T, single socket, NPS1 (node 0: 1019408 MB), SMT pairs (i, i+64) confirmed; AVX2+FMA only (no AVX-512/VNNI); 8 CCDs at ~50 GB/s GMI2 read each, so ≥3–4 CCDs are needed to saturate DRAM.
- Memory ceiling measured, not assumed: STREAM with RFO correction converges on ~150 GB/s on the host and 152 GB/s in the container (73% of the 187.7 GB/s theoretical); bandwidth saturates at 16 spread threads (16→64 −4%, 16→128 −6%); the container is exonerated (cgroups: `cpu.max = max 100000`, `cpuset.cpus.effective = 0-127`, `memory.max = max`).
- The decode budget: 201 ms/token actual versus ~126–128 ms ideal (CPU experts 13.9 GB @ 152 GB/s ≈ 91–93 ms; GPU dense ~13.3 GB ≈ 33–35 ms) → ~73–75 ms/token unaccounted (~36%); prefill is separately CPU-compute-bound at 1.69 TFLOP/s.
- Corrected sizing: experts ≈ 4.92 bpw → 5.53 GiB per MoE layer, ~415 GiB resident; non-expert only ~13 GiB; ~107 GiB VRAM idle (17–18 expert layers would fit) — deliberately deferred by Paul until throughput is understood.
- Ruled out this session: NUMA misplacement, container throttling, HIP-graphs-off, ZenDNN as a factor (inert for 256-expert Q4_K), `GGML_CUDA_REGISTER_HOST` (dead code), `-sm tensor` (unsupported for glm-dsa), and MTP self-speculation (blk.78 TENSOR_SKIP).
- Leading open hypotheses, mapped to experiments: hybrid-split overhead (75 fork/joins, ~150 device boundaries, ~375 barriers, ~2,000 kernel launches per token) → the `-ngl 0` versus hybrid comparison, predicted ~5.5 t/s pure-CPU; CCD/fabric placement → the 16-spread versus 16-packed `-C` mask pair; clock throttling → turbostat on the host; op_offload OOM → the ascending ubatch ladder plus unified-memory escape hatch.
- Host kernel state: THP `always` + `defer+madvise`, C-states below C0 disabled (08:11–08:13); galactus-diag.sh v2 (Phase 0 inventory + fourteen benchmark phases B0–B12, one 17-minute model load each, no deadline, 6-hour per-phase hang guard, MODE=all|host|bench) launched in the container at ~08:46, with the host companion run prescribed.

