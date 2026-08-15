#!/usr/bin/env bash
#===============================================================================
# galactus-diag.sh   (v3)
#
# Unattended system inventory + llama.cpp benchmark battery for GLM-5.2 on
# Galactus (EPYC 7713 / 1 TB DDR4 / 4x Radeon Pro V620).
#
# WHAT CHANGED SINCE v2 (all four v2 phases B1-B4 crashed; here is why):
#   1. ZenDNN was shredding the graph and almost certainly causing the crashes.
#      REMOVED from the build (-DGGML_ZENDNN=OFF). PHASE 0 verifies it is gone.
#   2. -mmp 0 forced a 411 GiB hipHostMalloc PINNED allocation. That is why loads
#      took 14 minutes, why AnonHugePages was always 0, and why runs OOM'd.
#      => v3 uses -mmp 1 for almost everything. Loads become SECONDS.
#      => The battery is now many cheap invocations, not few expensive ones.
#   3. iommu=pt added to the host kernel cmdline. PHASE 0 verifies.
#
# Usage:
#   cd /root/STREAM            # so ./stream_c is found
#   chmod +x galactus-diag.sh
#   nohup ./galactus-diag.sh > /root/console.txt 2>&1 &
#
# Env overrides:
#   MODEL=<path to shard 1>   OUTDIR=/root/diag   LLAMA_SRC=/root/llama.cpp
#   SKIP_STREAM=1   SKIP_PREWARM=1   SKIP_SLOW=1   (SKIP_SLOW drops all -mmp 0 phases)
#   BENCH_TIMEOUT=10800   # per-invocation HANG GUARD, not a budget. 0 = off.
#===============================================================================

set -uo pipefail   # deliberately NOT -e: phases are allowed to fail and we log it.

MODEL="${MODEL:-/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf}"
MODEL_DIR="$(dirname "$MODEL")"
MODEL_SHARD2="${MODEL_SHARD2:-$MODEL_DIR/GLM-5.2-UD-Q4_K_XL-00002-of-00011.gguf}"
OUTDIR="${OUTDIR:-/root/diag-$(date +%Y%m%d-%H%M%S)}"
LLAMA_SRC="${LLAMA_SRC:-/root/llama.cpp}"
BENCH="${BENCH:-llama-bench}"
SKIP_STREAM="${SKIP_STREAM:-0}"
SKIP_PREWARM="${SKIP_PREWARM:-0}"
SKIP_SLOW="${SKIP_SLOW:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-10800}"

mkdir -p "$OUTDIR"; LOG="$OUTDIR/MAIN.log"
MON="$OUTDIR/monitors"; mkdir -p "$MON"
START_EPOCH=$(date +%s); PHASE_N=0; MONITOR_PIDS=()
LAST_RC=0

log()  { printf '%s\n' "$*" | tee -a "$LOG"; }
hr()   { log ""; log "================================================================================"; }
sec()  { hr; log "## $*"; log "================================================================================"; log ""; }
sub()  { log ""; log "---- $* ----"; }
elapsed() { local s=$(( $(date +%s) - START_EPOCH )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

cap() {
  local desc="$1"; shift
  local t0 t1 rc out
  log ""; log "\$ $*"; log "  # ${desc}"
  t0=$(date +%s); out=$(eval "$@" 2>&1); rc=$?; t1=$(date +%s)
  printf '%s\n' "$out" | sed 's/^/    /' | tee -a "$LOG"
  log "  [exit=${rc}  $((t1-t0))s]"
}

memsnap() {
  { echo "--- memsnap: $1 @ $(date -Is)"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|AnonHugePages|SwapFree|Mapped)' /proc/meminfo
  } | sed 's/^/    /' | tee -a "$LOG"
}

cleanup() {
  sec "TEARDOWN"
  for p in "${MONITOR_PIDS[@]:-}"; do [ -n "${p:-}" ] && kill "$p" 2>/dev/null; done
  cap "final memory" "free -h"
  cap "final VRAM"   "rocm-smi --showmeminfo vram 2>/dev/null || true"
  log ""
  log "PHASES ATTEMPTED : $PHASE_N"
  log "TOTAL RUNTIME    : $(elapsed)"
  log "ARTIFACTS        : $LOG"
  log "                   $MON/"
  log ""
  log "=== END OF RUN $(date -Is) ==="
}
trap cleanup EXIT

#===============================================================================
cat <<'BRIEF' | tee -a "$LOG"
################################################################################
#  GALACTUS / GLM-5.2 THROUGHPUT INVESTIGATION — BRIEFING (v3)                  #
#                                                                              #
#  If you are a language model reading this log with no prior context, this     #
#  section is the entire background. Everything below it is evidence.           #
################################################################################

## THE MACHINE ("Galactus")

  CPU      AMD EPYC 7713 — Zen3/Milan, 64 physical cores / 128 threads, single
           socket, 8 CCDs x 8 cores, 32 MB L3 per CCD.
           AVX2 + FMA. NO AVX-512, NO VNNI.
  RAM      1 TB DDR4, 8 channels, NPS1 (single NUMA node). VERIFIED.
  GPU      4x AMD Radeon Pro V620 (gfx1030 / RDNA2), 30704 MiB each = 122816 MiB.
           PCIe 4.0 x16 each (rocm-smi: "16.0GT/s x16"). VERIFIED.
           GTT pool = 497.8 GiB (= half of RAM, amdgpu default). REMEMBER THIS.
  Storage  Local ZFS on SATA SSD array. dd direct 1.5 GB/s, buffered 2.1 GB/s.
           NOT a bottleneck. VERIFIED.
  Host     Proxmox 7.0.6-2-pve. llama.cpp runs in an LXC container ("openwebui")
           with FULL capabilities and NO cgroup limits (cpu.max=max,
           cpuset=0-127, memory.max=max, nr_throttled=0). VERIFIED CLEAN.

  TOPOLOGY — VERIFIED from /sys/.../cache/index3/shared_cpu_list:
      CPUs  0-7  = CCD0     CPUs 32-39 = CCD4
      CPUs  8-15 = CCD1     CPUs 40-47 = CCD5
      CPUs 16-23 = CCD2     CPUs 48-55 = CCD6
      CPUs 24-31 = CCD3     CPUs 56-63 = CCD7
      CPUs 64-127 are the SMT siblings of 0-63 (cpu0 siblings = "0,64").
  => The -C masks used below are CORRECT.

## MEMORY BANDWIDTH — SETTLED

  STREAM, corrected for the RFO artifact (GCC turns Copy into memcpy with
  non-temporal stores, which pays no read-for-ownership; Scale/Add/Triad use
  ordinary vector stores and DO pay it, and STREAM does not count that traffic.
  Multiply Scale x1.5, Add/Triad x1.333. Proof: if Copy paid RFO its real
  traffic would be 227.7 GB/s, above the 204.8 GB/s theoretical ceiling.)

      threads:   8     16     32     48     64     96    128
      GB/s:    ~150   152    150    143    145    140    141     <- all four kernels agree

  ==> THE PLATFORM DELIVERS ~152 GB/s. This is 74% of theoretical for 8ch
      DDR4-3200 and is HEALTHY. It SATURATES AT 16 THREADS.
  ==> Packed (OMP_PROC_BIND=close) at 16 threads collapses to ~90 GB/s Copy /
      68 GB/s Triad. CCD SPREAD IS WORTH ~2x. Each Milan CCD reaches the IOD
      over one GMI2 link; you need 3-4+ CCDs active to saturate DRAM at all.
      llama.cpp does NOT spread threads this way by default.
  ==> Container STREAM == host STREAM (within 1%). The container costs nothing.

## THE MODEL — MEASURED, NOT ESTIMATED

  GLM-5.2, Unsloth UD-Q4_K_XL, 11 shards, 435.19 GiB, 753.86 B params.
  llama.cpp arch "glm-dsa" (LLM_ARCH_GLM_DSA).
    - 78 layers (blk.0..blk.77); first_k_dense_replace=3 => blk.3..blk.77 are
      the 75 MoE layers. 256 routed experts, 8 active. 1 shared expert.
    - MLA attention (kv_lora_rank=512, q_lora_rank=2048). 1M ctx.
    - blk.78 = MTP/NextN, flagged TENSOR_SKIP, NEVER ALLOCATED. --spec-type
      draft-mtp CANNOT work for glm-dsa. Do not suggest it.
    - -sm tensor is NOT supported for glm-dsa (throws). -sm row IS allowed.

  EXPERT TENSOR TYPES — VERIFIED by gguf_dump AND by the load log:
      ffn_gate_exps   1728 MiB   Q4_K    <- CPU_REPACK CAN handle this
      ffn_up_exps     1728 MiB   Q4_K    <- CPU_REPACK CAN handle this
      ffn_down_exps   2112 MiB   Q5_K    <- CPU_REPACK CANNOT (Q5_K unsupported)
      (exceptions: blk.8 has Q5_K gate/up + Q6_K down; blk.75/76/77 Q6_K down)
  => CPU_REPACK (AVX2 q4_K_8x8_q8_K) covers 3456 of 5568 MiB per layer = 62%
     of expert bytes. Not zero, not everything. ggml_repack_get_optimal_repack_type
     handles ONLY Q4_0, Q4_K, Q2_K, IQ4_NL.

  MEASURED BUFFER SIZES (from load_tensors, -ngl 99 -ot exps=CPU):
      ROCm0  4987.08 MiB   ROCm1  4431.34 MiB
      ROCm2  4431.34 MiB   ROCm3  4952.45 MiB    => 18.36 GiB on GPU
      ROCm_Host           420964.22 MiB          => 411 GiB of routed experts

  DECODE READS PER TOKEN (8 of 256 experts, all 75 MoE layers):
      13,125 MiB = 12.82 GiB = 13.77 GB from DDR4.

## THE BUDGET — AND THE HOLE

  Measured hybrid (-ngl 99 -ot exps=CPU -mmp 0 -t 64 -nopo 1, build 9942):
      pp512 = 37.6 t/s      tg128 = 5.15 t/s  (= 201 ms/token)

      CPU  13.77 GB routed experts @ 152 GB/s              ->   91 ms
      GPU  ~18.7 GB dense path, read SERIALLY one card at
           a time under -sm layer, ~400 GB/s effective     ->  ~47 ms
      -------------------------------------------------------------------
      ideal                                                   ~138 ms (7.2 t/s)
      actual                                                   201 ms (5.0 t/s)
      UNACCOUNTED                                              ~63 ms  (31%)

  *** GRAPH SPLITS = 155 PER TOKEN. VERIFIED via GGML_SCHED_DEBUG=1. ***
      Structure: ROCm_k (attention) -> CPU (ffn_norm + ffn_moe_topk) ->
      ROCm_k (ffn_moe_down) -> ... repeating for all 75 MoE layers, with device
      hops at layers 20/40/60. 6063 graph nodes.
      Payload crossing each boundary at tg: 24 KB. This is PURE LATENCY, not
      bandwidth: ~155 stream syncs + ~76 CPU threadpool fork/joins per token.
      63 ms / 155 splits = 0.4 ms per split. That is exactly what a sleeping
      64-thread pool costs to wake.

  *** FINDING THAT 63 ms IS THE POINT OF THIS SCRIPT. ***

## WHY EVERY v2 PHASE CRASHED (and what was done about it)

  CAUSE 1 — ZenDNN.  The v2 build had GGML_ZENDNN=ON. ZenDNN registers as an
  ACCEL backend. Its supports_op() correctly REJECTS the routed experts (256
  experts > its 32-expert cap; Q4_K unsupported) but it ACCEPTS Q8_0 MUL_MAT --
  and all 872 Q8_0 attention tensors are CPU-resident when -ngl 0. Result:
      -ngl 0:   sched_reserve: graph splits = 1088 (with bs=512)   <- SHREDDED
      -ngl 99:  sched_reserve: graph splits = 155                  <- normal
  Every crashed run printed the LIBXSMM banner (LIBXSMM is ZenDNN's JIT), and
  the crashes were SIGILL / SIGSEGV during the bs=512 graph reserve or at
  warmup. The one phase that survived (B0) used -n 1 --no-warmup and never
  built a 512-token graph, so ZenDNN never executed.
  ==> REBUILT WITH -DGGML_ZENDNN=OFF. PHASE 0.9 VERIFIES IT IS GONE.

  CAUSE 2 — the 411 GiB pinned host buffer.  With -mmp 0, make_cpu_buft_list()
  puts the GPU's PINNED HOST BUFFER ahead of CPU_REPACK in the priority list, so
  llama.cpp tries hipHostMalloc(420964 MiB). Consequences:
      * "load time = 858989 ms" -- 14 minutes, of which only ~5 is actual I/O.
      * AnonHugePages stayed at 0 kB in EVERY memsnap. Driver-pinned pages are
        not THP-eligible. THE HUGE-PAGE HYPOTHESIS WAS NEVER ACTUALLY TESTED.
      * GTT pool is 497.8 GiB. After one run took and released 411 GiB, the next
        run got "ggml_cuda_host_malloc: failed to allocate 420964.22 MiB of
        pinned memory: out of memory" and fell back to a plain CPU buffer.
      * And with -nopo 1 that pin buys NOTHING -- its only purpose is fast H2D
        for op_offload.
  ==> v3 USES -mmp 1 FOR ALMOST EVERYTHING. Under mmap the loader explicitly
      demotes the host buffer ("avoid using a host buffer when using mmap"), so
      there is no pin and the load is SECONDS. This is why v3 can afford ~15
      invocations where v2 could only afford 14.

## THE THREE MODEL CONFIGURATIONS (a "reload" happens only when these change)

  From cmd_params_instance::equal_mparams(): -m -ngl -ncmoe -sm -mg -ts -mmp
  -dio -dev --no-host -ot. EVERYTHING ELSE (-t -C --cpu-strict --poll -b -ub
  -p -n -d -fa -nopo -ctk -ctv) sweeps FOR FREE inside one load.

    M1 = -mmp 1                 experts in CPU_Mapped.  No pin. No THP. No repack.
                                LOADS IN SECONDS. The iteration workhorse.
    M2 = -mmp 0 --no-host 1     experts in CPU_REPACK (Q4_K gate+up = 62%) +
                                plain CPU (Q5_K down). ANONYMOUS memory, so
                                THP-eligible. No pin. THE PRODUCTION CANDIDATE.
    M3 = -mmp 0 (default)       experts in ROCm_Host, 411 GiB PINNED. Slow load.
                                The ONLY config with fast H2D for op_offload.

  op_offload fires from ANY host buffer (the sched checks
  ggml_backend_buffer_is_host), so it works under M1 too -- just with slower,
  bounce-buffered H2D. M3 exists to measure how much that costs.

## -C / --cpu-strict SEMANTICS (verified in ggml_thread_cpumask_next)

  --cpu-strict 0 (default): EVERY thread gets the full mask; the kernel migrates
                            them freely. This is current production behaviour.
  --cpu-strict 1          : thread i is pinned to the i-th SET BIT IN ASCENDING
                            CPU ORDER.
  parse_cpu_mask() reads an ordinary big-endian hex number, bit 0 = CPU 0.

  Because bits are consumed in ascending order, one mask means different things
  at different -t:
      mask                    -t 16                 -t 32
      0303030303030303   ->   8 CCDs, 2 thr/CCD     (WRAPS: IGNORE THAT ROW)
      0f0f0f0f0f0f0f0f   ->   4 CCDs, 4 thr/CCD     8 CCDs, 4 thr/CCD
      00000000ffffffff   ->   2 CCDs, 8 thr/CCD     4 CCDs, 8 thr/CCD

## OTHER SETTLED FACTS — DO NOT RE-INVESTIGATE

  * PCIe: all four cards are 16.0 GT/s x16 (PCIe 4.0). Not a constraint.
  * Storage: local ZFS, 1.5-2.1 GB/s. Not a constraint.
  * Container: no cgroup CPU/memory limits, full caps. Not a constraint.
  * NUMA: NPS1, single node. Not a constraint.
  * GGML_CUDA_REGISTER_HOST exists as an env var but
    ggml_backend_cuda_register_host_buffer() is EXPORTED-BUT-NEVER-CALLED from
    llama.cpp's src/. It does nothing. Do not suggest it.
  * GPU P2P: llama.cpp only enables it if GGML_CUDA_P2P is set (opt-in). Under
    -sm layer the ONLY cross-GPU traffic is three 24 KB tensors per token
    (l_out-19, l_out-39, l_out-59). P2P is IRRELEVANT here. It would only matter
    for -sm row. Deliberately not tested in this run.
  * HIP graphs are compiled in (GGML_HIP_GRAPHS defaults ON) and the
    "cc < AMPERE" gate does not fire on AMD. Kernel launch overhead is not the
    explanation.

## HOW TO READ THE RESULTS

   1. PHASE 0.9   *** IS ZenDNN GONE? *** If --list-devices still lists it, the
                  rebuild did not take and every crash will repeat.
   2. PHASE 0.7   IOMMU mode. Want "identity" (passthrough), not DMA/DMA-FQ.
   3. PHASE A     Does anything run at all now? A1 = plain, A2 = with -C masks.
   4. PHASE B     *** THE MAIN EVENT. *** -t x --poll sweep on the hybrid config.
                  STREAM saturates at 16 threads. Does llama.cpp?
                  If --poll 100 produces a step change, the 63 ms is threadpool
                  sleep/wake across the 155 splits and the fix is one flag.
   5. PHASE C     CCD spread at FIXED thread count. STREAM says this is worth 2x.
   6. PHASE D     -ngl 0 denominator. Reads 27+ GB/token, ZERO splits (now that
                  ZenDNN is gone). D vs B at matched -t IS the split overhead.
   7. PHASE F/G   op_offload. Prefill is CPU-COMPUTE-bound at 1.69 TFLOP/s. On
                  the GPUs it becomes PCIe-bound: ~410 GiB of expert weights per
                  graph eval over 4x PCIe4 x16, ~4 s per ubatch REGARDLESS of
                  ubatch size, so pp scales ~linearly with -ub.
                  Ceilings: ub 512 -> ~125 t/s; ub 2048 -> ~500 t/s. Now 37 t/s.
   8. PHASE I     M2 = the production candidate. CHECK: does "CPU_REPACK model
                  buffer size" appear, and does AnonHugePages finally climb?
   9. PHASE M     VRAM fill. ~104 GiB idle. Expected only ~1.25x. Lowest priority.

  Reference numbers:
      decode reads 13.77 GB/token (hybrid) or ~27 GB/token (-ngl 0)
      platform bandwidth: 152 GB/s  =>  91 ms (hybrid CPU part)
      current: 201 ms/token = 4.97 t/s ; 155 splits ; ~63 ms unaccounted
      current prefill: 37.6 t/s = 1.69 TFLOP/s on 64 Zen3 cores (AVX2, no VNNI)
################################################################################
BRIEF

log ""
log "RUN STARTED : $(date -Is)"
log "HOSTNAME    : $(hostname)"
log "OUTDIR      : $OUTDIR"
log "MODEL       : $MODEL"
log "DEADLINE    : NONE."
log "HANG GUARD  : ${BENCH_TIMEOUT}s per invocation (0 = off)."

#===============================================================================
sec "PHASE 0 — SYSTEM INVENTORY"
#===============================================================================

sub "0.1 kernel / OS / container"
cap "kernel"      "uname -a"
cap "distro"      "cat /etc/os-release 2>/dev/null | head -3"
cap "*** KERNEL CMDLINE — WANT amd_iommu=on iommu=pt ***" "cat /proc/cmdline"
cap "uptime"      "uptime"
cap "containerised?" "systemd-detect-virt 2>/dev/null; head -2 /proc/1/cgroup 2>/dev/null"
cap "cgroup cpu quota ('max 100000' = unlimited)" "cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none"
cap "cgroup cpuset (want 0-127)" "cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo none"
cap "cgroup memory (want 'max')"  "cat /sys/fs/cgroup/memory.max 2>/dev/null || echo none"
cap "cgroup throttling (nr_throttled must be 0)" "cat /sys/fs/cgroup/cpu.stat 2>/dev/null | grep -E 'nr_throttled|throttled_usec'"
cap "ulimit -l (max locked memory)" "ulimit -l"

sub "0.2 CPU topology (re-verify the -C masks)"
cap "lscpu"       "lscpu | grep -E 'Model name|Thread|Core|Socket|NUMA|MHz|L3'"
cap "ISA flags"   "grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512f|fma|bmi1|bmi2|sha_ni)$' | tr '\n' ' '"
cap "SMT siblings (want 0,64)" "cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list"
cap "*** CCD MAP — MUST BE 0-7 / 8-15 / ... OR THE -C MASKS ARE WRONG ***" \
    "for c in 0 8 16 24 32 40 48 56; do echo -n \"cpu\$c CCD: \"; cat /sys/devices/system/cpu/cpu\$c/cache/index3/shared_cpu_list; done"

sub "0.3 clocks / idle states"
cap "governor"    "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c"
cap "boost"       "cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null"
cap "idle states (1 = disabled)" "for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do echo -n \"\$(cat \$s/name 2>/dev/null): disable=\"; cat \$s/disable 2>/dev/null; done"
cap "current MHz spread" "grep 'cpu MHz' /proc/cpuinfo | awk '{s+=\$4;n++} END {printf \"avg %.0f MHz over %d cpus\n\", s/n, n}'"

sub "0.4 memory / THP"
cap "free"        "free -h"
cap "THP enabled" "cat /sys/kernel/mm/transparent_hugepage/enabled"
cap "THP defrag"  "cat /sys/kernel/mm/transparent_hugepage/defrag"
cap "*** AnonHugePages BASELINE (was ALWAYS 0 in v2 because of the pinned buffer) ***" "grep -E 'AnonHugePages|Hugepagesize' /proc/meminfo"
cap "swap (must be 0)" "swapon --show; grep -E 'SwapTotal|SwapFree' /proc/meminfo"
cap "DIMM population" "dmidecode -t 17 2>/dev/null | grep -E 'Locator|^[[:space:]]+Size|Configured Memory Speed' | grep -v 'No Module' || echo 'dmidecode unavailable (container) — STREAM already confirmed 8ch @ full speed'"

sub "0.5 NUMA"
cap "numactl --hardware" "numactl --hardware | head -6"

sub "0.6 GPUs"
cap "rocm-smi product" "rocm-smi --showproductname 2>/dev/null | grep -E 'Card Series|GFX Version'"
cap "*** VRAM + GTT POOL (GTT is the pinned-memory ceiling: was 497.8 GiB) ***" "rocm-smi --showmeminfo all 2>/dev/null | grep -E 'VRAM Total Memory|GTT Total Memory'"
cap "amdgpu gttsize param" "cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null"
cap "PCIe link (want 16.0GT/s x16)" "rocm-smi --showclocks 2>/dev/null | grep pcie"
cap "topology" "rocm-smi --showtopo 2>/dev/null | grep -A6 'Link Type'"

sub "0.7 IOMMU — you just added amd_iommu=on iommu=pt; VERIFY IT TOOK"
cap "*** IOMMU GROUP TYPE — want 'identity' (passthrough), NOT DMA/DMA-FQ ***" \
    "cat /sys/kernel/iommu_groups/*/type 2>/dev/null | sort | uniq -c || echo 'not readable from container — check on the Proxmox host'"
cap "iommu groups"  "ls /sys/kernel/iommu_groups 2>/dev/null | wc -l"
cap "dmesg iommu"   "dmesg 2>/dev/null | grep -i iommu | head -6 || echo 'dmesg not readable in container'"

sub "0.8 storage"
cap "mount"       "findmnt -T '$MODEL' 2>/dev/null || df -h '$MODEL'"
cap "shards"      "ls -la '$MODEL_DIR' | head -14"

sub "0.9 llama.cpp build  *** THE ZenDNN CHECK ***"
cap "build rev"   "cd '$LLAMA_SRC' 2>/dev/null && git log -1 --format='%H %cd %s'"
cap "*** GGML_ZENDNN MUST BE OFF ***" "grep -E 'GGML_ZENDNN|GGML_HIP:|GGML_HIP_GRAPHS|GGML_NATIVE|GGML_CUDA_NO_PEER_COPY|CMAKE_BUILD_TYPE|AMDGPU_TARGETS' '$LLAMA_SRC/build/CMakeCache.txt' 2>/dev/null"
cap "*** IF 'ZenDNN' APPEARS BELOW, THE REBUILD DID NOT TAKE AND EVERYTHING WILL CRASH AGAIN ***" "$BENCH --list-devices 2>&1 | tail -8"
cap "linked ggml libs (libggml-zendnn MUST NOT be linked)" "ldd \$(which $BENCH) 2>/dev/null | grep -i ggml"
cap "CPU backend variant (want haswell/znver, NOT plain x64)" "ls -la /usr/local/lib/libggml-cpu.so* 2>/dev/null"

sub "0.10 model layout (already settled; re-confirm the build sees the same file)"
GGUF_DUMP=""
for cand in "$LLAMA_SRC/gguf-py/gguf/scripts/gguf_dump.py" "$(command -v gguf-dump 2>/dev/null)"; do
  [ -n "$cand" ] && [ -e "$cand" ] && GGUF_DUMP="$cand" && break
done
if [ -n "$GGUF_DUMP" ]; then
  cap "expert tensor types (expect Q4_K gate/up, Q5_K down)" \
      "PYTHONPATH='$LLAMA_SRC/gguf-py' python3 '$GGUF_DUMP' '$MODEL_SHARD2' 2>&1 | grep -E 'blk\.(3|4|8)\.ffn_(up|gate|down)_exps'"
else
  log "  (gguf_dump.py not found; types already settled: Q4_K gate/up, Q5_K down)"
fi

sub "0.11 STREAM (re-baseline; container == host, ~152 GB/s corrected)"
if [ "$SKIP_STREAM" != "1" ] && [ -x ./stream_c ]; then
  log "  REMINDER: multiply Scale x1.5 and Add/Triad x1.333 for real DRAM traffic."
  for t in 8 16 32 64 128; do
    cap "STREAM t=$t SPREAD (all 8 CCDs)" \
        "OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=cores ./stream_c 2>&1 | grep -E 'Copy:|Triad:'"
  done
  for t in 16 32; do
    cap "STREAM t=$t PACKED (few CCDs) — EXPECT ~HALF" \
        "OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores ./stream_c 2>&1 | grep -E 'Copy:|Triad:'"
  done
else
  log "  (skipped)"
fi

#===============================================================================
sec "PHASE 1 — BACKGROUND MONITORS"
#===============================================================================
if command -v turbostat >/dev/null 2>&1; then
  turbostat --quiet --interval 10 --out "$MON/turbostat.txt" >/dev/null 2>&1 &
  MONITOR_PIDS+=($!); log "turbostat -> $MON/turbostat.txt"
  log "   *** READ Bzy_MHz. 2.6-3.0 GHz all-core = fine. ~2.0 GHz = BIOS"
  log "       determinism/cTDP throttle = a flat 30% tax on every number here. ***"
else
  log "turbostat NOT INSTALLED (needs CAP_SYS_RAWIO + MSRs; run it on the HOST)"
  ( while sleep 10; do
      echo "$(date -Is) avgMHz=$(grep 'cpu MHz' /proc/cpuinfo | awk '{s+=$4;n++} END{printf "%.0f", s/n}')"
    done > "$MON/cpumhz.txt" 2>&1 ) &
  MONITOR_PIDS+=($!); log "fallback MHz sampler -> $MON/cpumhz.txt"
fi

( while sleep 15; do
    echo "=== $(date -Is)"
    rocm-smi --showuse --showmemuse --showpower 2>/dev/null | grep -E 'GPU\[|Power|use|Memory'
  done > "$MON/rocm-smi.txt" 2>&1 ) & MONITOR_PIDS+=($!)
vmstat -t 10 > "$MON/vmstat.txt" 2>&1 & MONITOR_PIDS+=($!)
( while sleep 15; do
    echo "$(date -Is) $(grep -E '^(MemFree|Cached|AnonHugePages|Mapped)' /proc/meminfo | tr '\n' ' ')"
  done > "$MON/meminfo.txt" 2>&1 ) & MONITOR_PIDS+=($!)
log "rocm-smi / vmstat / meminfo samplers -> $MON/"
log "   *** AnonHugePages was 0 kB in EVERY v2 memsnap because of the 411 GiB"
log "       pinned buffer. Watch it during PHASE I (M2, -mmp 0 --no-host 1)."
log "       If it finally climbs, the THP hypothesis becomes testable at last. ***"

#===============================================================================
sec "PHASE 2 — PREWARM (pull the model into cache once)"
#===============================================================================
if [ "$SKIP_PREWARM" != "1" ]; then
  memsnap "before prewarm"
  cap "read all 11 shards once (435 GiB @ ~2 GB/s = a few minutes)" \
      "time cat $MODEL_DIR/*.gguf > /dev/null"
  memsnap "after prewarm"
  log "  If 'Cached' did not grow by ~435 GiB, ZFS is serving from ARC rather"
  log "  than the page cache. -mmp 1 loads will still be fast; just note it."
else
  log "  (skipped)"
fi

#===============================================================================
# Benchmark harness
#===============================================================================
COMMON="-fa 1 --progress -o md -v"
BENCH_ENV=""

bench() {
  local phase="$1"; shift
  local why="$1";  shift
  PHASE_N=$((PHASE_N+1))
  hr
  log "## $phase        [elapsed $(elapsed)]"
  log ""
  printf '%s\n' "$why" | sed 's/^/  /' | tee -a "$LOG"
  log ""
  memsnap "before $phase"
  local cmd="timeout --kill-after=180 $BENCH_TIMEOUT $BENCH -m '$MODEL' $COMMON $*"
  [ -n "$BENCH_ENV" ] && cmd="env $BENCH_ENV $cmd"
  log ""; log "\$ $cmd"
  local t0; t0=$(date +%s)
  eval "$cmd" 2>&1 | tee -a "$LOG"
  LAST_RC=${PIPESTATUS[0]}
  log ""
  log "[$phase  exit=$LAST_RC  took $(( ($(date +%s)-t0)/60 )) min]"
  if [ "$LAST_RC" -ne 0 ]; then
    log "!!! $phase FAILED (exit=$LAST_RC). Continuing."
    log "!!!   132 = SIGILL   139 = SIGSEGV   124 = timeout"
    log "!!!   If ZenDNN is still linked (see PHASE 0.9), that is the cause."
  fi
  memsnap "after $phase"
  BENCH_ENV=""
  sleep 15
}

MASK_ALL="ffffffffffffffffffffffffffffffff"   # all 128 logical CPUs
MASK_A="0303030303030303"   # t=16: 8 CCDs, 2/CCD   | t=32: WRAPS, ignore
MASK_B="0f0f0f0f0f0f0f0f"   # t=16: 4 CCDs, 4/CCD   | t=32: 8 CCDs, 4/CCD
MASK_C="00000000ffffffff"   # t=16: 2 CCDs, 8/CCD   | t=32: 4 CCDs, 8/CCD

#===============================================================================
bench "A1  SMOKE TEST — does anything run now that ZenDNN is gone?" \
"M1 (-mmp 1). Plain: no -C, no --cpu-strict. This is the closest thing to the
 one v2 phase that survived (B0), plus a real 512-token graph, which is exactly
 where B2/B3 died. If THIS crashes, ZenDNN is still linked (check PHASE 0.9) and
 nothing below will run.
 Also: watch the load time. Under -mmp 1 it should be SECONDS, not 14 minutes.
 And grep the load_tensors lines: you should see 'CPU_Mapped', NOT 'ROCm_Host'." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 32 -r 1

#===============================================================================
CPU_MASK_OK=1
bench "A2  SMOKE TEST — with -C / --cpu-strict (the flags every crashed run had)" \
"Same as A1 plus the pinning flags. B0 (survived) did NOT have them; B1-B4 (all
 crashed) DID. That correlation is probably coincidental -- B2/B3 died during
 graph_reserve, before any threadpool exists -- but prove it rather than assume.
 If this crashes and A1 did not, drop -C entirely and PHASE C is void." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 64 --poll 100 --cpu-strict 1 -C "$MASK_ALL" \
  -b 2048 -ub 2048 -p 512 -n 32 -r 1
[ "$LAST_RC" -ne 0 ] && CPU_MASK_OK=0 && log "!!! -C/--cpu-strict IS BROKEN. PHASE C will be skipped."

#===============================================================================
bench "B  *** THE MAIN EVENT *** — hybrid, thread x poll sweep" \
"M1. THE experiment. Decode only (-p 0) so it runs fast.

 STREAM saturates at 16 threads and DECAYS beyond. llama.cpp is being run with
 64. If tg peaks at 16-32 here, the extra 48 threads are buying nothing while
 paying a 64-wide barrier across 155 splits per token.

 --poll 100 keeps the threadpool SPINNING instead of futex-sleeping between the
 ~76 CPU splits per token. 63 ms unaccounted / 155 splits = 0.4 ms per split,
 which is exactly what waking a sleeping 64-thread pool costs.
 *** IF --poll 100 PRODUCES A STEP CHANGE, THE 63 ms IS FOUND AND THE FIX IS
     ONE FLAG. ***

 16 combos, ONE model load (-t and --poll are context params)." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 8,16,24,32,48,64,96,128 --poll 0,100 \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3

#===============================================================================
bench "B2  hybrid — prefill at the best thread counts" \
"Same load would be ideal but -p is a context param, so this is free anyway.
 Prefill is CPU-COMPUTE-bound (1.69 TFLOP/s on 64 Zen3 cores, AVX2, no VNNI),
 NOT bandwidth-bound, so it should behave completely differently from B:
 it should keep scaling with threads long after decode has flattened." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 16,32,48,64,96,128 --poll 100 \
  -b 2048 -ub 2048 -p 512 -n 0 -r 2

#===============================================================================
if [ "$CPU_MASK_OK" = "1" ]; then
bench "C  hybrid — CCD PLACEMENT at fixed thread count" \
"M1. Isolates Infinity-Fabric bandwidth from thread count: same threads, same
 everything, ONLY the CCD spread changes.

 STREAM's answer is unambiguous: at 16 threads, SPREAD = 152 GB/s Copy,
 PACKED = 90 GB/s Copy / 68 GB/s Triad. CCD spread is worth ~2x. Each Milan CCD
 reaches the IOD over one GMI2 link; 3-4+ CCDs must be active to saturate DRAM.
 llama.cpp does NOT spread threads this way by default.

 If llama.cpp shows the same, the right production config may be
 '-t 16 -C 0303030303030303 --cpu-strict 1', not '-t 64'.

 ROW MAP (bits are consumed in ascending CPU order):
     t=16  0303030303030303  -> 8 CCDs, 2/CCD
     t=16  0f0f0f0f0f0f0f0f  -> 4 CCDs, 4/CCD
     t=16  00000000ffffffff  -> 2 CCDs, 8/CCD
     t=32  0303030303030303  -> WRAPS. IGNORE THIS ROW.
     t=32  0f0f0f0f0f0f0f0f  -> 8 CCDs, 4/CCD
     t=32  00000000ffffffff  -> 4 CCDs, 8/CCD" \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 16,32 --cpu-strict 1 --poll 100 -C "$MASK_A,$MASK_B,$MASK_C" \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3
else
  log ""; log "## C  SKIPPED — -C/--cpu-strict crashed in A2."
fi

#===============================================================================
bench "D  -ngl 0 CPU-ONLY DENOMINATOR" \
"M1, everything on CPU. Reads ~27 GB/token with (now that ZenDNN is gone) ZERO
 graph splits, zero device syncs, zero GPU.

 *** In v2 this produced 'graph splits = 1088 (with bs=512)' because ZenDNN was
 claiming the 872 Q8_0 attention tensors and shredding the graph. If it still
 says 1088, ZenDNN is still linked. It should now say 1. ***

 *** D vs B AT MATCHED -t AND --poll IS THE SPLIT OVERHEAD. THAT IS THE 63 ms. ***
 At the measured 152 GB/s, 27 GB/token = 178 ms = 5.6 t/s. If -ngl 0 matches or
 beats the hybrid's 4.97 t/s, the four V620s are contributing NOTHING NET." \
  -ngl 0 -mmp 1 -nopo 1 \
  -t 16,32,48,64 --poll 0,100 \
  -b 2048 -ub 2048 -p 512 -n 64 -r 3

#===============================================================================
BENCH_ENV="GGML_SCHED_DEBUG=1"
bench "E  GRAPH SPLIT COUNT — re-verify 155 with ZenDNN gone" \
"-n 1 -r 1 --no-warmup so the dump is one graph, not sixty-four.
 Expect: 'sched_reserve: graph splits = 155'. If it changed, ZenDNN was
 affecting the hybrid graph too, not just -ngl 0." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 64 --poll 100 -b 2048 -ub 2048 -p 0 -n 1 -r 1 --no-warmup

#===============================================================================
bench "F  op_offload ON, ubatch ladder (M1 = pageable H2D)" \
"*** THE 3-10x ON PREFILL. ***
 op_offload streams CPU-resident weights to the GPU for any op whose batch dim
 >= 32 (GGML_OP_MUL_MAT_ID uses ne[2] = token count). Prefill stops being a
 1.69 TFLOP/s CPU GEMM and becomes a PCIe transfer: ~410 GiB of expert weights
 per graph eval spread over 4x PCIe4 x16, i.e. roughly 4 s per ubatch REGARDLESS
 of ubatch size. So pp scales ~LINEARLY with -ub.
     ub 512 -> ~125 t/s     ub 2048 -> ~500 t/s     current: 37 t/s

 Under M1 the weights are in a PAGEABLE mmap, so H2D is bounce-buffered and
 SLOWER than it could be. PHASE G repeats this with the pinned buffer to measure
 exactly what that costs. iommu=pt (just added) should help both.

 llama-bench prints rows INCREMENTALLY, so a crash at large -ub still leaves the
 small ones. The ladder is ordered small->large deliberately. -v is on: if it
 aborts, the real HIP error string is the line ABOVE the backtrace." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 \
  -t 64 --poll 100 -b 4096 -ub 128,256,512,1024,2048,4096 \
  -p 512 -n 0 -r 2

#===============================================================================
BENCH_ENV="GGML_CUDA_ENABLE_UNIFIED_MEMORY=1"
bench "G  op_offload ON + HIP MANAGED MEMORY (OOM escape hatch)" \
"GGML_CUDA_ENABLE_UNIFIED_MEMORY makes ggml_cuda_device_malloc() call
 cudaMallocManaged() instead of cudaMalloc(), permitting VRAM OVERSUBSCRIPTION.
 If F aborts on VRAM exhaustion, this should make it SURVIVE (slowly). If it
 STILL aborts, the abort is NOT an OOM and we are chasing a different bug.
 Diagnostic only — it also breaks the free-memory reporting the fitter uses." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 \
  -t 64 --poll 100 -b 2048 -ub 512,2048 \
  -p 512 -n 0 -r 2

#===============================================================================
bench "H  -sm row" \
"Under -sm layer only ONE card is live at a time, which is why the dense path
 costs ~47 ms/token instead of ~12. -sm row parallelises each layer across all
 four cards.
 CAVEAT: -sm tensor is NOT supported for glm-dsa and throws; -sm row IS allowed
 but may fail on MLA. A failure is itself a result.
 CAVEAT 2: row split does per-layer cross-device collectives, and GPU P2P is NOT
 enabled (llama.cpp requires GGML_CUDA_P2P, opt-in). So every collective is
 bounced through host memory. If -sm row looks promising here, THAT is when P2P
 becomes worth investigating." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 -sm row \
  -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 64 -r 3

#===============================================================================
BENCH_ENV="GGML_CUDA_GRAPH_OPT=1"
bench "I  GGML_CUDA_GRAPH_OPT=1" \
"Enables ggml-cuda's graph optimisation / concurrent stream-event launching (off
 by default). GLM-5.2's MLA attention is a long chain of tiny GEMVs at batch 1.
 HIP graphs are already active, so this is second-order.
 Compare against the -t 64 --poll 100 row of PHASE B." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 64 --poll 100 -b 2048 -ub 2048 -p 0 -n 64 -r 3

#===============================================================================
bench "J  FILL THE IDLE VRAM (fitter)" \
"18.36 GiB of 120 GiB VRAM is used; ~104 GiB is IDLE. Each MoE layer's routed
 experts are ~5.44 GiB, so ~17 of the 75 fit. Predicted: CPU reads drop from
 13.77 to ~10.6 GB/token => decode ~1.3x. Real, but not where the 63 ms is.

 DO NOT USE -ncmoe. It keeps experts of the FIRST N layers on CPU, and with
 --tensor-split unset llama.cpp apportions layers by free memory, so every
 GPU-resident expert layer piles onto ROCm3 and OOMs. The fitter computes -ngl,
 --tensor-split AND the per-layer overrides together.

 If the measured gain is MUCH larger than 1.3x, that is itself evidence about
 the split overhead and should be reported prominently." \
  -fitt 2048 -fitc 8192 -mmp 1 -nopo 1 \
  -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 128 -r 3

#===============================================================================
bench "K  decode at CONTEXT DEPTH" \
"Everything above measures decode from an EMPTY context. MLA keeps the KV cache
 small (44 MiB at 512 ctx), so degradation should be mild — confirm it." \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 \
  -t 64 --poll 100 -b 2048 -ub 2048 \
  -d 0,4096,16384,65536 -p 0 -n 64 -r 2

#===============================================================================
if [ "$SKIP_SLOW" = "1" ]; then
  log ""; log "## SKIP_SLOW=1 — skipping all -mmp 0 phases (L, M, N). Done."
  exit 0
fi

#===============================================================================
sec "SLOW PHASES — these use -mmp 0 and each load costs 5-15 minutes"
#===============================================================================

bench "L  *** THE PRODUCTION CANDIDATE *** M2 = -mmp 0 --no-host 1" \
"THIS IS THE CONFIG THAT SHOULD WIN.
 --no-host 1 removes the pinned host buffer from the CPU buffer-type priority
 list. Three things follow, none of which have EVER been true in any previous run:
   1. NO 411 GiB hipHostMalloc. No OOM, no 14-minute load.
   2. The experts land in ANONYMOUS memory => THP-ELIGIBLE for the first time.
   3. CPU_REPACK (AVX2 q4_K_8x8_q8_K) claims the Q4_K gate+up tensors = 62% of
      expert bytes. The Q5_K down tensors fall through to plain CPU.

 *** GREP THE load_tensors LINES: ***
     'CPU_REPACK model buffer size'  <- should be ~253 GiB (the Q4_K 62%)
     'CPU model buffer size'         <- should be ~158 GiB (the Q5_K/Q6_K rest)
     'ROCm_Host model buffer size'   <- MUST NOT APPEAR
 *** AND WATCH AnonHugePages IN monitors/meminfo.txt. It should climb toward
     ~400 GiB. It has been 0 kB in every run to date. ***

 NOTE: CPU_REPACK's buffer reports is_host = nullptr, so op_offload CANNOT fire
 for repacked weights. This config is therefore CPU-prefill only. That is the
 trade: repack + THP for decode, versus op_offload for prefill. You may end up
 running two different server configs." \
  -ngl 99 -ot "exps=CPU" -mmp 0 --no-host 1 -nopo 1 \
  -t 16,32,48,64 --poll 0,100 \
  -b 2048 -ub 2048 -p 512 -n 64 -r 3

cap "AnonHugePages after M2" "grep AnonHugePages /proc/meminfo"

#===============================================================================
bench "M  M3 = -mmp 0 default (411 GiB PINNED host buffer) + op_offload" \
"The ONLY config with fast, DMA-direct H2D for op_offload — which is the whole
 point of the pin. Now that iommu=pt is set, this is the config that should give
 the real prefill number.

 EXPECT: a 5-15 minute load and 'ROCm_Host model buffer size = 420964.22 MiB'.
 If it prints 'ggml_cuda_host_malloc: failed to allocate ... out of memory', the
 GTT pool (497.8 GiB) is fragmented from earlier runs — reboot the container and
 run this phase FIRST, or raise amdgpu.gttsize on the host.

 Compare pp against PHASE F (same thing over a pageable mmap). The delta IS the
 value of the pin." \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 \
  -t 64 --poll 100 -b 4096 -ub 512,1024,2048,4096 \
  -p 512 -n 0 -r 2

#===============================================================================
bench "N  M3 control — same load, op_offload OFF" \
"The original production config, for a clean apples-to-apples against everything
 above. This is the 37.6 t/s / 5.15 t/s baseline that started this whole
 investigation." \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 1 \
  -t 16,32,64 --poll 0,100 \
  -b 2048 -ub 2048 -p 512 -n 64 -r 3

#===============================================================================
sec "SUMMARY — READ IN THIS ORDER"
cat <<'SUMM' | tee -a "$LOG"

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
SUMM

log ""
log "All phases attempted. Total runtime: $(elapsed)"
