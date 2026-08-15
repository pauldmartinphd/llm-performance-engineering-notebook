## Session 3 — Monday, July 14, 2026 (mid-day) — v2 crash forensics and the v3 relaunch

### 10:22 — v2 battery output arrives: B0 alone survived

Paul pasted the output of the v2 diagnostic battery ("Here's the current status:"). Phase B0 succeeded; B1–B4 all crashed; B5 was still running with its log cut off at `done_getting_tensors:`.

Contents of the upload, as read from the logs:

- B0 succeeded: `sched_reserve: graph splits = 155`, 6063 graph nodes; the 420 GB pinned ROCm host buffer allocated; first-token throughput 1.59 t/s (cold start, not meaningful); `load time = 858989 ms` (~14.3 min).
- B1 died with "Illegal instruction" (SIGILL) at warmup; B2, B3, B4 died with segmentation faults (SIGSEGV). Every failed run contained:

```
ggml_cuda_host_malloc: failed to allocate 420964.22 MiB of pinned memory: out of memory
load_tensors:          CPU model buffer size = 420964.22 MiB
```

- The one successful run instead showed `load_tensors:    ROCm_Host model buffer size = 420964.22 MiB` (= 411 GiB pinned).
- `ulimit`: max locked memory = 8192 KB (8 MB RLIMIT_MEMLOCK); the container has CAP_IPC_LOCK.
- rocm-smi: GTT Total Memory = 534,463,750,144 B = 497.8 GiB per GPU (half of the 995.5 GiB system RAM — the amdgpu default gttsize; the gttsize parameter output was garbled/unset).
- B1 (`-ngl 0`) needed 429.5 GiB pinned (also cited in the logs as "B1 and B2 both failed at 439 GiB"); B3 failed at the identical 411.1 GiB that B0 had succeeded with; MemFree bounced back to 993 GiB between runs (~1 TB free).
- A `LIBXSMM_VERSION` banner appeared immediately before each crash; `LIBXSMM_TARGET: hsw`.
- B1 at `-ngl 0` with bs=512: `graph splits = 1088`; at bs=1: 1 split (~5.6 nodes per split at 1088).
- Compute buffers: B0 tiny at 0.4 MiB on ROCm; B4 (ubatch 512): 205 MiB on ROCm0/1/2, 435 MiB on ROCm3, 217 MiB host.
- Flags: B0 used `-ngl 99 -ot exps=CPU`, `-t 64 --poll 100`, no `-C`, no `--cpu-strict`, `-p 0` (graphs reserved only for n_tokens = 1 and 16); B1–B4 used `-C ffffffffffffffffffffffffffffffff` + `--cpu-strict` + `-p 512`; B5 ran without `-C`/`--cpu-strict`.
- Memsnaps: `AnonHugePages: 0 kB` in every snapshot — THP never applied.
- PCIe: `pcie clock level: 1 (16.0GT/s x16)` on all four cards.
- Storage: local ZFS; `dd` direct 1.5 GB/s, buffered 2.1 GB/s.
- Tensor type histogram: mostly Q8_0 and F32; expert tensors 150 Q4_K (gate+up, ~74 pairs plus extras) and 74 Q5_K (down); 872 Q8_0 attention tensors.
- GPU model buffers: 4987 + 4431 + 4431 + 4952 = 18.4 GiB.
- 96 IOMMU groups; `/proc/cmdline` had no IOMMU flags; Proxmox kernel 7.0.6-2-pve; systemd-boot (`initrd=\EFI\proxmox\...`).
- GPUs idle at 0 MHz, minimal power; CPU on the performance governor, boost enabled, max 3720 MHz; C-states disabled.

### 10:28 — Diagnosis: the 411 GiB pinned buffer plus ZenDNN; 155 splits confirmed; pivot to -mmp 1

Headline reading: B0 worked. B1–B4 all crashed. B5 is running now and will almost certainly join them. The number the battery was launched to obtain was in B0's log, verbatim:

```
sched_reserve: graph nodes  = 6063
sched_reserve: graph splits = 155
```

The structure matched the prediction exactly: `SPLIT #1: ROCm0` (attention), then `#2: CPU [ffn_norm-3, ffn_moe_topk-3]` → `#3: ROCm0 [ffn_moe_down-3]`, alternating for all 75 MoE layers, with device hops at layers 20/40/60. Roughly 76 CPU splits and 78 ROCm splits; the payload crossing each boundary is 24 KB at token generation (384 KB at batch of 16) — pure latency (155 stream syncs plus ~76 CPU fork/joins per token), not bandwidth.

Phase 0 of v2 closed four open questions:

- Expert tensor types (settled):

```
ffn_gate_exps   1728 MiB   Q4_K   <- repackable
ffn_up_exps     1728 MiB   Q4_K   <- repackable
ffn_down_exps   2112 MiB   Q5_K   <- NOT repackable
```

  Q6_K down tensors on blk.8/75/76/77. `CPU_REPACK` handles only Q4_0/Q4_K/Q2_K/IQ4_NL → covers 3456 of 5568 MiB per layer = 62% of expert bytes.
- Decode read per token nailed: 8 experts × (54 + 54 + 66) MiB × 75 layers = 12.75 GiB = 13.7 GB/token; at 152 GB/s → 90 ms. (Earlier working estimate before verification: 56.6 MB gate + 56.6 up + 69.2 down = 182.4 MB/layer → 13.68 GB/token → ~68 GB/s needed at ~5 t/s; per expert 6.75 + 6.75 + 8.25 = 21.75 MiB.)
- PCIe (settled): PCIe 4.0 x16 on all four (~25 GB/s per card, ~100 GB/s aggregate). Streaming 411 GiB of experts at that bandwidth ≈ 4.4 s per graph eval → an op_offload prefill ceiling near ~116 t/s.
- Storage (settled): local ZFS at 1.5 GB/s direct, 2.1 GB/s buffered — not a bottleneck.
- GPU-side weights (measured): 4987 + 4431 + 4431 + 4952 = 18.4 GiB; serialized under `-sm layer` at ~400 GB/s ≈ 49 ms. Budget: 90 + 49 = 139 ms ideal versus 201 ms actual → ~60 ms unaccounted.

**Observations**

- **Confirmed:** the premise of hypothesis H1 — 155 graph splits per token, alternating CPU/GPU, latency-dominated boundaries.
- **Correction:** the unaccounted decode gap is ~60 ms, not the 75 ms previously stated (working figures 62–63 ms, ~31% overhead, ~0.4 ms per split).
- Root cause, part one — the 411 GiB pinned buffer (`hipHostMalloc` page-locked `ROCm_Host` buffer). Three consequences: (1) 14-minute loads — of `load time = 858989 ms`, only ~5 min is I/O at 1.5 GB/s for 435 GiB; the other ~9 min is page-locking at ~780 MB/s effective pinning (B0 overall ~490 MB/s); (2) THP never applied — driver-pinned pages are not THP-eligible, hence `AnonHugePages: 0 kB` everywhere. **Correction:** "H5 has been untestable this entire time and I did not realise it." (3) B1–B4 crashes — the pin fails against the 497.8 GiB GTT pool, the loader falls back to a plain CPU buffer, then SIGILL/SIGSEGV follows. B0 is the only run in which the pin succeeded.
- **Hypothesis:** after B0 took and released 411 GiB, B3 failed at the identical size — "The pages aren't coming back" (driver leak / TTM accounting / GTT fragmentation).
- With `-nopo 1` the pinned buffer buys nothing; its sole purpose is fast H2D for op_offload.
- **Correction:** "ZenDNN is not inert at `-ngl 0`." It rejects the routed experts (256 experts, Q4_K) but accepts Q8_0 `MUL_MAT`, and all 872 Q8_0 attention tensors sit on the CPU at `-ngl 0` — hence `graph splits = 1088` (bs=512) versus 1 (bs=1). ZenDNN is a separate ACCEL backend, so every op it claims is a split boundary. **Hypothesis:** that shredded graph threw the SIGILL (LIBXSMM JITs; its banner printed on every abort). At `-ngl 99` attention is on GPU, ZenDNN sees nothing, and the graph is a clean 155.
- **Correction:** Paul rebuilt llama.cpp that very morning — `libggml-*.so.0.16.0` dated Jul 14 08:28 (files 08:28–08:32); `build: 657e01125 (10001)` versus the prior working `f84a51940 (9942)`, 59 commits back; the new build adds fused Gated Delta Net and Lightning Indexer paths for glm-dsa. Three variables moved at once (new build, THP=always, C-states off), and the script itself introduced `-C`/`--cpu-strict`.
- Supporting notes: the `-C` mask parse is safe (128 hex digits → 512 boolmask entries = GGML_MAX_N_THREADS; a 32-hex mask starts at index 127); earlier stable runs measured `--no-host 1` at 37.29/4.97 and default at 37.63/5.15; a plain 439 GB malloc/free test left MemFree unchanged; the crash signals are SIGILL/SIGSEGV, not SIGKILL (so not the OOM killer); the 1088-split `-ngl 0` graph "seems worth reporting upstream."
- **Decision:** kill the battery and pivot to mmap:

```bash
pkill -f diag.sh; pkill -f llama-bench
```

  With `-mmp 1` the loader explicitly demotes host buffers under mmap — no pinned allocation; local ZFS at 2 GB/s maps the model in seconds; 15-minute iterations become 1-minute iterations.
- **Decision:** bisect the crash first (two runs, five minutes):

```bash
# A: no masks
llama-bench -m … -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 -fa 1 \
  -t 64 --poll 100 -b 2048 -ub 2048 -p 512 -n 64 -r 1 --progress

# B: same + the masks that every crashed run had (and B0 did not)
llama-bench -m … -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 -fa 1 \
  -t 64 --poll 100 --cpu-strict 1 -C ffffffffffffffffffffffffffffffff \
  -b 2048 -ub 2048 -p 512 -n 64 -r 1 --progress
```

  Then the thread/poll sweep:

```bash
llama-bench -m … -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 1 -fa 1 \
  -t 16,24,32,48,64,96,128 --poll 0,100 \
  -b 2048 -ub 2048 -p 0 -n 64 -r 3 --progress
```

  Then the production candidate `-mmp 0 --no-host 1` — anonymous memory (THP-eligible) plus `CPU_REPACK` on the Q4_K gate/up tensors, with no 411 GiB pin. Confirm from the load log:

```
load_tensors: CPU_REPACK model buffer size = …   <- Q4_K gate+up (62% of expert bytes)
load_tensors:       CPU model buffer size = …   <- Q5_K down + the rest
```

  and check `grep AnonHugePages /proc/meminfo` mid-run (it should be non-zero for the first time).
- **Decision:** two build/boot items — rebuild with `-DGGML_ZENDNN=OFF` (it cannot touch the experts, it registers as an ACCEL device, it destroys the `-ngl 0` graph, and it is linked rather than dlopen'd, so the .so cannot simply be moved aside); and add `amd_iommu=on iommu=pt` to the Proxmox host kernel command line before chasing op_offload (100 GB/s of aggregate H2D DMA through a translating IOMMU is not free).

### 10:29 — Paul asks about ROCm PCIe peer-to-peer

"One other thing, is there something i need to do to enable ROCm PCIe card to card communication?"

### 10:32 — P2P triage: irrelevant under -sm layer; probe written; parked

Answer: for the current bottleneck, P2P is nearly irrelevant — and it may not be available on gfx1030 at all. From B0's scheduler dump, under `-sm layer` the only cross-GPU traffic is three tensors:

```
SPLIT #36:  ROCm1  inputs: [l_out-19 (24K)]
SPLIT #77:  ROCm2  inputs: [l_out-39 (24K)]
SPLIT #118: ROCm3  inputs: [l_out-59 (24K)]
```

72 KB per token; even bounced through host memory at 1 GB/s that is ~70 µs of the 201 ms token. P2P also does nothing for op_offload (host→device DMA, not device→device). It matters only for `-sm row`, whose per-layer cross-device collectives (~78 layers × 3 peers ≈ a few hundred cross-device syncs per token) would otherwise each bounce device→host→device.

llama.cpp behavior, verified in `ggml-cuda.cu`:

```c
if (getenv("GGML_CUDA_P2P") != nullptr) {
    ...
    cudaDeviceCanAccessPeer(&can_access_peer, id, id_other);
    if (can_access_peer) cudaDeviceEnablePeerAccess(id_other, 0);
}
```

Opt-in by presence of the environment variable (not its value). `cudaMemcpyPeerAsync` handles every cross-device tensor copy (line 2364); the build has `GGML_CUDA_NO_PEER_COPY:BOOL=OFF` (compiled in), so without peer access it stages through host memory ("You're getting correct results via a bounce today"). The build also has `GGML_HIP_NO_VMM:BOOL=ON`, so `GGML_CUDA_P2P=1` would take the simple `cudaDeviceEnablePeerAccess` route.

Probe program supplied:

```c
// p2p.c
#include <hip/hip_runtime.h>
#include <stdio.h>
int main(void) {
    int n; hipGetDeviceCount(&n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        if (i == j) continue;
        int can = 0; hipDeviceCanAccessPeer(&can, i, j);
        printf("GPU%d -> GPU%d : %s\n", i, j, can ? "P2P OK" : "NO P2P");
    }
}
```

```bash
hipcc p2p.c -o p2p && ./p2p
```

(`rocm_bandwidth_test` gives the same answer plus a device-to-device bandwidth matrix.)

Host-side prerequisites, all on the Proxmox host rather than the container:

```bash
cat /sys/kernel/iommu_groups/*/type | sort | uniq -c
# DMA / DMA-FQ = translate;  identity = passthrough
```

```bash
# on the host
echo 'amd_iommu=on iommu=pt' >> /etc/kernel/cmdline   # merge into the existing line
proxmox-boot-tool refresh && reboot
```

```bash
lspci -vv -s 03:00.0 | grep -i 'Memory at.*prefetchable'
```

Want `[size=32G]`, not `[size=256M]`; if small, enable Above 4G Decoding and Resizable BAR in BIOS (`lspci -vv` returned nothing useful inside the container — run on the Galactus host).

**Observations**

- **Hypothesis** (flagged uncertain): RDNA2/gfx1030 may not support ROCm P2P over PCIe at all — solid on CDNA/MI-series, historically absent or broken on Navi, which is why `GGML_CUDA_NO_PEER_COPY` exists.
- Topology facts: `rocm-smi --showtopo` reports Link Type PCIE, Hops 2, weight 40 for every pair — GPU → host bridge → GPU; no XGMI (the V620 has none); four separate root complexes (buses `03`, `83`, `86`, `c3`) on the EPYC Milan IOD fabric.
- EPYC root ports implement ACS, forcing P2P TLPs upstream through the IOD fabric (works, slower than a switch path). **Decision:** leave ACS alone; `pcie_acs_override` breaks IOMMU isolation and buys nothing.
- **Decision:** order of operations — (1) run the `hipDeviceCanAccessPeer` probe (if `NO P2P`, topic closed); (2) add `amd_iommu=on iommu=pt` at the next host reboot regardless (the same flag matters for op_offload's ~100 GB/s aggregate H2D DMA); (3) only if P2P is available, run `-sm row` with and without `GGML_CUDA_P2P=1`. Nothing here is on the critical path — parked until the `-mmp 1` bisect is done.

### 10:57 — Directives: full rerun; IOMMU flags added; ZenDNN-free rebuild

Paul chose a clean slate rather than the incremental bisect: "Just put it all into the script. I'm going to delete the output and rerun from scratch. Ignore the p2p stuff for now then. I also just rebooted the container before running. I'll rerun all tests including the diagnostics." He also reported two system changes: "I added amd_iommu=on iommu=pt." and "I rebuilt with -DGGML_ZENDNN=OFF."

### 11:03 — v3 script written: galactus-diag.sh (16 phases, three model configurations)

The battery was rewritten from scratch: `-mmp 1` makes loads take seconds instead of 14 minutes, so the battery is now many cheap invocations instead of few expensive ones. The script was delivered as `/mnt/user-data/outputs/galactus-diag.sh` (artifact saved 11:05:36 ET), with the launch:

```bash
cd /root/STREAM
nohup ./galactus-diag.sh > /root/console.txt 2>&1 &
```

Pre-write crash re-analysis (a correction of the earlier `-C` suspicion): B2 and B3 died during `graph_reserve` at `n_tokens=512` — before any threadpool or mask exists — so `-C` could not be responsible there. B0 only reserved graphs for n_tokens = 1 and 16 (`-p 0`); B1–B5 had `-p 512`, which clamps n_ubatch to 512 and forces the 512-token graph where the crashes occurred. B1 hit SIGILL at warmup (16 threads); B4 cleared sched_reserve (155 splits) and then hit SIGSEGV at warmup. Common denominator: ZenDNN/LIBXSMM.

Model-size accounting from the tensor dump and load logs: blk.8 (higher precision) = 6744 MiB; blk.75–77 (Q6_K down) = 5976 MiB each; the remaining 71 standard layers = 5568 MiB each; total ≈ 420 GiB, matching the measured 420,964 MiB. Per-token expert read ≈ 13.1 GiB ≈ 13.77 GB. Budget: CPU 13.77 GB @ 152 GB/s = 90.6 ms; GPU 18.7 GB @ ~400 GB/s ≈ 47 ms; ideal 138 ms (7.25 t/s) versus actual 201 ms (4.97 t/s) → ~63 ms unaccounted ≈ 0.4 ms per split.

The three model configurations, made explicit (a "reload" happens only when these change):

| | flags | expert buffer | pinned | THP | repack |
|---|---|---|---|---|---|
| **M1** | `-mmp 1` | `CPU_Mapped` | no | no | no |
| **M2** | `-mmp 0 --no-host 1` | `CPU_REPACK` + `CPU` | no | **yes** | **62%** |
| **M3** | `-mmp 0` | `ROCm_Host` | 411 GiB | no | no |

**Observations**

- **Hypothesis:** ZenDNN's `supports_op` accepts Q8_0 MUL_MAT at bs=512 (rejects at bs=1), executes a real GEMM through LIBXSMM, and crashes — this fits all four v2 deaths. Removing ZenDNN should fix all of them; the script bisects defensively anyway. Stated read: "`-C` was never the problem... ZenDNN accepting the 872 Q8_0 attention tensors at batch 512 fits all four deaths. Your rebuild should have fixed it. A2 proves it either way."
- Design: `-mmp 1` for eleven of the sixteen phases; the five `-mmp 0` phases are quarantined at the end behind `SKIP_SLOW=1`. Crash handling: Phase 0.9 checks `--list-devices` and `ldd` for ZenDNN; Phases A1/A2 bisect (A1 plain, A2 with `-C`/`--cpu-strict`); Phase C auto-skips if A2 fails.
- M2 (Phase L) is the production candidate — never once run to date. It is the only configuration with anonymous memory (THP finally testable; `AnonHugePages` has read `0 kB` in every memsnap so far) plus AVX2 repack on the Q4_K gate/up tensors. **Prediction:** Phase L should print `CPU_REPACK model buffer size` ≈ 253 GiB alongside `CPU model buffer size` ≈ 158 GiB.
- Phase B is "where the money is": 16 combos, one load, decode only. STREAM saturates at 16 threads and decays past it; production has been running 64. 63 ms ÷ 155 splits = 0.4 ms per split — what waking a sleeping 64-thread pool costs. **Hypothesis (H1 test):** "If `--poll 100` produces a step change, the whole thing was a flag."
- Phase D is the denominator and a ZenDNN receipt. **Prediction:** it printed `graph splits = 1088` last time; it should now print `1`.
- Practical notes: a prewarm reads 435 GiB at 2.1 GB/s ≈ 3.5 min; there is a double-caching risk (page cache + ZFS ARC = up to 870 GiB); under mmap, `--no-host` is redundant (mmap demotes the host buffer to plain CPU, so `CPU_REPACK` is never selected under mmap — to be checked empirically in `load_tensors` output); lxcfs virtualizes `/proc/meminfo` inside a Proxmox LXC; the CPU masks were computed from the CCD topology.
- The battery started at 11:08:13 (OUTDIR `/root/diag-20260714-110813`). Its complete machine log is transcribed as Session 4 below; the dialogue continued in parallel.

### 11:56 — Two upload attempts arrive empty; triage and digest commands

Paul attached the in-progress results file twice ("How does this look now?", "This is where it is"); both uploads arrived with no content and an empty `/mnt/user-data/uploads/`. **Dead end:** "The upload came through empty... Twice now, so it's not a fluke." Two candidate causes: the file too large for the attachment path, or genuinely zero bytes. Triage commands supplied:

```bash
D=$(ls -dt /root/diag-*/ | head -1); echo "$D"
ls -la "$D"
wc -l "$D/MAIN.log"
```

If MAIN.log were 0 bytes, the script died before its first `log()` call (likely `mkdir -p "$OUTDIR"` failing or `tee` missing from `$PATH`); if it has content, it is a size problem — build a digest instead:

```bash
{
  sed -n '1,/^RUN STARTED/p' "$D/MAIN.log"          # the briefing
  grep -nE '^## |^\$ |^\| |exit=|graph splits|load_tensors:|load time|CPU_REPACK|ROCm_Host|CPU_Mapped|AnonHugePages|Illegal|Segmentation|error|ZenDNN|iommu' "$D/MAIN.log"
} > /root/diag-digest.txt
wc -l /root/diag-digest.txt
```

```bash
D=$(ls -dt /root/diag-*/ | head -1)
ls -la "$D/MAIN.log"
```

```bash
grep -nE '^## |^\$ |^\| |exit=|graph splits|load time|load_tensors:|CPU_REPACK|ROCm_Host|CPU_Mapped|AnonHugePages|Illegal|Segmentation|ZenDNN|iommu_groups' "$D/MAIN.log" > /root/digest.txt
wc -c /root/digest.txt
```

If even the digest proved too big, the four greps that decide everything else:

```bash
grep -A3 'list-devices'        "$D/MAIN.log" | head -12   # is ZenDNN gone?
grep    'graph splits'          "$D/MAIN.log"              # 155 hybrid, 1 at -ngl 0?
grep -E 'load_tensors:.*model buffer' "$D/MAIN.log" | sort -u
grep -B2 -A30 '^## B '          "$D/MAIN.log"              # the -t x --poll table
```

### 11:57 — results.txt arrives (244k lines, mid-Phase B)

On the third attempt the file landed: results.txt, 244k lines, 6.8 MB — the run still in progress, the log ending mid-Phase B. Paul: "HEre you go."

### 12:00 — Partial v3 read: both crash causes dead; H1 dead; the CPU is at the memory wall; C = 90 ms

Assessment: a very good run; let it finish — but the answer to the main question is already visible.

Phase B table as of the partial log (tg t/s):

| `-t` | `--poll 0` | `--poll 100` |
|---|---|---|
| 8 | 3.89 | 3.90 |
| 16 | 5.27 | 5.28 |
| **24** | **5.53** | **5.54** |
| 32 | 5.53 | 5.46 |
| 48 | 5.49 | 5.48 |
| 64 | 5.40 | 4.84 ± 1.01 |
| 96 | 2.76 | 2.76 |

Implied CPU bandwidth versus STREAM, from the model `t_total = C + t_cpu(n)` with C constant in thread count:

| `-t` | ms/token | implied CPU BW | STREAM |
|---|---|---|---|
| 8 | 256.4 | 83 GB/s | ~150 |
| 16 | 189.4 | 139 GB/s | 152 |
| 24 | 180.5 | **152 GB/s** | 152 |
| 32 | 180.8 | **152 GB/s** | 150 |
| 48 | 182.1 | **150 GB/s** | 143 |
| 64 | 185.2 | **145 GB/s** | 145 |

**Observations**

- **Confirmed:** both crash causes are dead — `A1 exit=0`, `A2 exit=0`. `ldd` shows no `libggml-zendnn`; `--list-devices` shows only ROCm0–3; the benchmark backend column reads plain `ROCm` (previously `ROCm,ZenDNN`). "ZenDNN was the crash." `-C`/`--cpu-strict` was never the problem — A2 ran clean with the full 128-bit mask. (Phase 0.9's `GGML_ZENDNN:BOOL=ON` in CMakeCache.txt is stale text from an old build directory; `ldd` is authoritative.)
- **Confirmed:** `-mmp 1` worked as designed — `CPU_Mapped model buffer size`, no `ROCm_Host`, no 411 GiB pin. A1 took 25 min (cold; wall ~24.7 min, page-faulting 435 GiB from ZFS); A2 took 1 minute (cached). IOMMU passthrough took: 96 `identity` groups.
- Load-path detail: the prewarm read ran at 1.4 GB/s over 435 GiB, yet the Linux page cache did not grow (377 MB before and after) — the ZFS ARC holds the data instead (visible as `SReclaimable`/`Slab`, not `Cached`), which is why A2 loaded in a minute. The cumulative load-time prints (44 s, 65 s, 81 s … up to 579 s) are misleading; `llama_perf_context_print` values are all zeros (llama-bench does not populate them). 16,380 distinct CUDA graph IDs (HIP graphs active and reused). VRAM barely touched: ~5.2 GiB of 30 GiB per card; compute buffers tiny (205 MiB max). pp512 = 34.65 t/s at t=64. Build 657e01125 confirmed.
- **Refuted (H1, dead end):** `--poll` does nothing — "Identical at every thread count. The gap is not threadpool sleep/wake across the 155 splits. Scratch that hypothesis entirely."
- The CPU side is perfect. Plateau across t=24…48 means t_cpu is saturated: `t_cpu(24) = 13.77 GB ÷ 152 GB/s = 90.6 ms` → **C = 180.5 − 90.6 = 90 ms**. From 24 threads up, llama.cpp's Q4_K/Q5_K expert path runs at full platform memory bandwidth — even reproducing STREAM's own decay at 48/64. Below 24 threads it is compute-limited (AVX2 unpack). "There is nothing left to win on the CPU. It is at the memory wall."
- Therefore 90 of the 180 ms is the GPU side; the GPUs only remove ~87 ms of CPU work (the 13.5 GB dense path they hold) — "roughly a wash."
- **Prediction (Phase D):** `-ngl 0` reads ~27 GB/token at 152 GB/s = 178 ms = 5.6 t/s with zero splits. If it lands there, it beats the hybrid's 5.54 and the GPUs are net-negative in their current role.
- **Correction:** filling VRAM is upgraded to a double win (from the 1.25× quoted the previous day): only 18.4 GiB of 120 GiB is in use, compute buffers peak at 205 MiB; moving ~17 MoE layers' experts to the cards cuts CPU bytes 13.77 → 10.6 GB/token (90 → 70 ms) and removes those layers' CPU splits (155 → ~121). Phase J will measure it. (Working figures: overhead 43 → 33.6 ms, GPU +7.75 ms, total ~158 ms → 6.3 t/s, +14%.) `-sm row` likewise matters more than previously said (the dense path is currently serialized one card at a time; 47 → 12 ms would give ~145 ms → 6.9 t/s, +25%).
- **Decision:** settings effective immediately — `-t 24`; drop `--poll` (noise); never `-t 96` (2.76 t/s, a 2× collapse from SMT sibling pairs stalling the barrier).
- Housekeeping: the log carries 232,110 lines of `CUDA Graph id … reused` spam (from `-v` plus HIP-graph debug). On completion:

```bash
D=$(ls -dt /root/diag-*/ | head -1)
grep -vE 'CUDA Graph id|CUDA graph' "$D/MAIN.log" > /root/diag-clean.log
wc -c /root/diag-clean.log
```

- Phases still ahead: D (denominator), F (op_offload), H (`-sm row`), J (VRAM), L (repack + THP).

### 12:02 — Log-spam stripping agreed; the run stays alive

Paul offered to parse out the `CUDA Graph id … reused` lines before sending. Agreed: strip them at source — they're 232k lines of pure noise and they'll only get worse.

```bash
D=$(ls -dt /root/diag-*/ | head -1)
grep -vE 'CUDA Graph id [0-9]+ (reused|created|updated)|CUDA graph' "$D/MAIN.log" > /root/diag-clean.log
wc -lc /root/diag-clean.log
```

If still too big, a payload-only digest (keeps every result table, exit code, buffer placement, split count, and crash):

```bash
grep -nE '^## |^\$ |^\| |^\[.*exit=|graph splits|graph nodes|model buffer size|compute buffer size|AnonHugePages|load time =|Illegal|Segmentation|error|abort|CPU_REPACK|ROCm_Host|CPU_Mapped|SPLIT #' "$D/MAIN.log" > /root/diag-digest.log
wc -lc /root/diag-digest.log
```

**Observations**

- **Decision:** "Don't kill it to grab the log" — files are written incrementally, so a partially-written MAIN.log can be grepped at any time. Most-wanted phases: D (does `-ngl 0` beat 5.54 and prove the GPUs net-negative?), then F, H, J.

### 12:04 — Paul asks for a maximum-TPS estimate

"Can you guess how high I can get the TPS on this model with all possible optimizations? Or not enough info yet. It's okay to be wrong. I'm merely curious"

### 12:06 — Ceiling estimate: decode 7–8 t/s realistic (hard ceiling ~11); prefill 200–800 t/s

Assessment: enough for decode; not enough for prefill — that hinges entirely on whether op_offload survives, which is still unmeasured. The wall: `t_total = C + t_cpu(n)`; the plateau at t=24–48 fixes `t_cpu(24) = 13.77 GB ÷ 152 GB/s = 90.6 ms`, hence C = 90 ms, and the constant reproduces the entire curve including STREAM's decay at 48/64. "A token is 90 ms of GPU + 90 ms of CPU, and the two never overlap" — the graph is strictly sequential (GPU attention → CPU experts → GPU down, ×75). The CPU half already runs at 152 GB/s; the only fix is to read fewer bytes.

Decode lever table:

| lever | mechanism | Δ |
|---|---|---|
| `-t 24` instead of 64 | already measured | 5.40 → 5.54 |
| **Fill VRAM** (17 of 75 expert layers) | CPU bytes 13.77 → 10.65 GB → 70 ms; also deletes 34 of the 155 splits | −25 ms |
| **`-sm row`** | dense path stops being serialized one card at a time | −25 ms, **if it works at all** |
| `GGML_CUDA_GRAPH_OPT` | launch gaps across ~4,500 nodes | −0 to 5 ms |

Bits-per-weight, the lever with real headroom (decode t/s scales inversely with expert bpw):

| quant | expert bpw | GB/token | t_cpu | est. tg (VRAM filled) |
|---|---|---|---|---|
| **UD-Q4_K_XL** (current) | ~4.92 | 13.77 | 90 ms | 6.5 |
| UD-Q3_K_XL | ~3.5 | 9.8 | 64 ms | **~8.5** |
| UD-Q2_K_XL | ~2.7 | 7.6 | 50 ms | **~9.5** |

Prefill, if op_offload works (expert GEMMs move to the V620s; prefill becomes a PCIe transfer of ~440 GB of expert weights per graph eval over 4× PCIe 4.0 x16 in parallel ≈ 5 s per ubatch regardless of ubatch size, so pp scales almost linearly with `-ub`):

| `-ub` | est. pp |
|---|---|
| 512 | ~95 t/s |
| 2048 | ~370 t/s |
| 4096 | ~750 t/s |

**Observations**

- **Prediction:** decode with high confidence 6.3–6.8 t/s (VRAM fill alone — the double win); ~7.5–8 t/s if `-sm row` works ("Coin flip. MLA + row split is not a well-trodden path and it may simply throw."); hard ceiling on this hardware/quant ~11 t/s (CPU expert read driven to zero — impossible, since 411 GiB of experts never fits 120 GiB of VRAM). "Call it 7 t/s realistic, 8 t/s if things break your way. A 25–45% improvement, and then you are against physics."
- "Dropping to Q3 buys you more than every software optimization in this thread combined. Whether that's an acceptable trade on a 744B model is your call." Platform alternative: Genoa/Turin with 12-channel DDR5 at ~400 GB/s → CPU half 34 ms → 12–15 t/s.
- Prefill "is where the money is" — a different bottleneck (compute, not bandwidth). Current 34.65 t/s = 1.69 TFLOP/s on 64 Zen3 cores (AVX2, no VNNI). Compute buffers of 205 MiB leave enormous VRAM headroom for higher `-ub`; `iommu=pt` now in place makes the DMA numbers plausible. "That's a 6–20× on prefill versus a 1.3× on decode." If Phase F comes back clean, it is the biggest result of the investigation.
- **Hypothesis** (stated as the weak point): the 90 ms GPU constant decomposes as ~47 ms serialized dense-weight reads + ~30 ms HIP stream syncs across 155 boundaries + ~13 ms kernel-launch gaps — "a decomposition I inferred, not measured." If mostly sync, `-sm row` will not help and 6.5 t/s is the ceiling; if mostly serialized read, row split reaches 8. Phase H settles it.
- Supporting figures: VRAM budget 120 GiB − non-expert − KV − compute ≈ 92 GiB for experts ≈ 16–17 layers at ~5.4 GiB per MoE layer; with 17 layers on GPU, GPU reads 21.8 GB/token (+7.8 ms at 400 GB/s → ~95 ms), CPU 10.65 GB → 70 ms, total ~165 ms = 6.1 t/s (+10%). **Correction** (of the 12:00 Phase-D prediction): counting the dense path at ~18.7 GB/token, CPU-only would read 32.5 GB/token → ~214 ms → 4.7 t/s — worse than the hybrid, so the GPUs do help (the hybrid saves ~34 ms, 16%). GPU effective bandwidth today ~208 GB/s across four cards ≈ 10% utilization (latency-bound, serialized). Kernel-level estimate: 6063 nodes, ~4500 on GPU; HIP-graph launch ~2 µs/kernel ≈ 9 ms; ~4.2 MB average read per kernel at 400 GB/s ≈ 10 µs ≈ 47 ms compute → ~56 ms; measured 90 ms → ~34 ms unaccounted ≈ 155 syncs × ~200 µs ≈ 31 ms (an earlier alternative put it at 155 × 2 × 30 µs = 9.3 ms; 277 µs/split observed). MLA attention ≈ 25 tiny GEMVs/layer × 78 layers. Aggressive-quant sizes: IQ2/Q2_K_XL ≈ 200 GB, IQ1 ≈ 150 GB; uniform Q4_K = 4.5 bpw → 12.6 GB/token → 83 ms. PCIe prefill arithmetic: 410 GiB ≈ 440 GB, ~110 GB/card; 32 GB/s theoretical, ~20 GB/s realistic → 5.5 s/microbatch → 745 t/s @ ub 4096, 372 @ 2048, 93 @ 512; `-ub 8192` potentially ~1500 t/s; with VRAM filled, only 58 of 75 layers stream (~340 GB → ~4.3 s per microbatch).

### 12:07 — Paul asks about speculative decoding

"What about for future speculative decoding with flash or MTP?"

### 12:11 — Speculative decoding: structurally weak on sparse MoE; MTP is the right draft but llama.cpp will not load it

Assessment: enough to answer, and the answer is counterintuitive — it runs against the usual advice. On a dense model, verifying B tokens costs the same as one (weights read once). Not true for top-8-of-256 MoE with the experts on CPU — with independent routing, distinct experts touched across B tokens:

```
E(B) = 256 × (1 − (1 − 8/256)^B)

E(1)=8   E(2)=15.8   E(3)=23.3   E(4)=30.5   E(5)=37.5
```

Near-linear, so the CPU expert read (half the token time) scales almost linearly with speculation depth. What is free: the 90 ms GPU path reads its 18.7 GB of dense weights once regardless of batch size, and the 155 stream syncs are batch-independent — "That constant is the entire prize." Verified in source: llama.cpp builds the verify batch as `[last_accepted, draft_0 … draft_{N-1}]` — N drafts = batch of N+1 (`common_speculative.cpp:294,353`); expected tokens per cycle `α = (1 − p^(N+1))/(1 − p)`.

With a 10 ms/token GPU-resident draft, p = 0.75, independent routing (pessimistic bound):

| N | verify | cycle | α | t/s |
|---|---|---|---|---|
| 0 | 181 ms | 181 | 1.00 | **5.52** (matches measurement) |
| 1 | 268 | 278 | 1.75 | 6.29 |
| 2 | 353 | 373 | 2.31 | 6.19 |
| 3 | 435 | 465 | 2.73 | 5.87 |
| 4 | 515 | 555 | 3.05 | **5.50** (worse than no speculation) |

"Deep speculation makes it slower." With realistic ~30% routing overlap: +26–30%, peaking at N=2, still degrading past N=3.

MTP is the right draft, for two reasons: (1) nearly free — `blk.78` is 5.5 GiB against ~100 GiB of idle VRAM; pinned to a card, drafting costs ~3 ms/token instead of 10–20; (2) no second model in RAM, no vocabulary mismatch. But it does not work today — from `src/models/glm-dsa.cpp`:

```c
for (int i = 0; i < n_layer_all; ++i) {
    int flags = 0;
    if (i >= n_layer) {
        // skip all tensors in the NextN layers
        // TODO @ngxson : TENSOR_NOT_REQUIRED was a hack, need to remove it later
        flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
    }
```

`blk.78` is never allocated; `--spec-type draft-mtp` (whatever it does for DeepSeek-V4-Flash) is a no-op for `glm-dsa`.

**Prediction** — MTP (p ≈ 0.85, draft ≈ 3 ms), 30% routing overlap, stacked on the VRAM fill:

| N | t/s |
|---|---|
| 0 | 6.1 |
| 1 | 8.6 |
| **2** | **9.7** |
| 3 | ~9.5 |

Honest bottom line:

| config | est. tg |
|---|---|
| today | 5.5 |
| + VRAM fill + `-t 24` | ~6.5 |
| + `-sm row` (if it works) | ~7.5 |
| + MTP spec, N=2 | **~9.5–10** |

**Observations**

- **Dead end (for now):** MTP drafting on GLM-5.2 — a known gap with an upstream TODO marker; plausible that it lands. "Speculation is worth roughly the same as everything else combined — but only shallow, only with a nearly-free draft, and only once someone deletes eight lines in `glm-dsa.cpp`."
- A separate draft model must be small, dense, fully GPU-resident, and share GLM-5.2's 154,880-token vocabulary; a MoE draft needing CPU offload goes backwards. Whether a suitable GLM-5.2 sibling exists (e.g. a "Flash" variant) is unknown.
- Repeated caveat: the 90 ms decomposition (~47 ms serialized dense reads + ~43 ms sync overhead) is inferred, not measured; if mostly sync, `-sm row` will not help and speculation amortizes less. Phase H settles it.
- Supporting figures: MTP blk.78 = 256 experts, 9.66 B params; its experts on CPU cost ~174 MB per draft token ≈ 1.1 ms; NextN embed_tokens + shared_head ≈ 1.5 GB; shared_head 951 M params Q6_K ≈ 0.77 GB ≈ 5 ms GPU-resident; draft cost 5–10 ms (experts on CPU) versus ~2–3 ms (block on GPU). MTP N=1 estimate: draft 3 ms, verify 237–268 ms, α 1.75 (GLM reportedly 80–90% first-token acceptance) → 6.9–7.7 t/s (+25–40%); with GPU residency α 1.85 → 7.7–8 t/s. Draft-model speeds: a 3B draft ≈ 40–60 t/s (~20 ms/token, too slow); needed is 0.5–1B Q8 at 80–120 t/s (~10 ms). Simple framing: speculation amortizes the GPU dense 90 → 30 ms/token, total 180 → 120 ms/token. VRAM-fill scaling factor on CPU time ≈ 0.774 (13.77 → 10.65 GB); the B=1 VRAM-filled baseline ≈ 165 ms ≈ 6 t/s.

### State of knowledge at end of session

- The v2 crashes are fully root-caused: ZenDNN (built in with `GGML_ZENDNN=ON`) accepted Q8_0 MUL_MATs at bs=512 and crashed in LIBXSMM, and the 411 GiB `hipHostMalloc` pinned buffer (`420964.22 MiB`) failed against the 497.8 GiB GTT pool after B0 released it. Both causes eliminated (rebuild with `-DGGML_ZENDNN=OFF`; `-mmp 1`), and proven dead by v3's A1/A2 (exit=0, exit=0).
- `-C`/`--cpu-strict` exonerated; the crash signals were SIGILL/SIGSEGV, never SIGKILL.
- `amd_iommu=on iommu=pt` added to the host and confirmed effective: all 96 IOMMU groups now `identity`.
- `-mmp 1` cut iteration cost from ~14 minutes to seconds-to-one-minute per load (A1 25 min cold from ZFS, A2 1 min warm).
- Hypothesis H1 (`--poll`) is dead: identical throughput at every thread count. Best decode 5.54 t/s at t=24 (180.5 ms/token); t=96 collapses to 2.76 t/s.
- The CPU expert path saturates DDR4 at 152 GB/s from t=24 up; the token model is `t_total = C + t_cpu(n)` with C = 90 ms of GPU-side time per token — the new optimization target.
- Ceiling estimates: decode 6.3–6.8 t/s from the VRAM fill; 7.5–8 t/s if `-sm row` works; ~11 t/s hard ceiling; prefill 200–800 t/s if op_offload works (unmeasured).
- Speculative decoding is structurally weak on top-8-of-256 MoE (expert reads scale near-linearly with verify batch); shallow MTP (N=2) would give ~9.5–10 t/s but llama.cpp's glm-dsa loader skips blk.78.
- The v3 battery (16 phases) is running as `diag-20260714-110813`; phases D, F, H, J, L are the ones that matter.

---

