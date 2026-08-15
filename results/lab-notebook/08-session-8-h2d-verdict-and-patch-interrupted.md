## Session 8 — Monday evening, July 14 – Wednesday, July 16, 2026 — The H2D verdict and the patch, interrupted

### 19:25 — Four-stream H2D: 65.7 GB/s; the copy-mechanism diagnosis reverses itself

Paul pasted the probe result:

```
20 warnings generated when compiling for host.
4 concurrent H2D: 65.7 GB/s aggregate
root@openwebui:~#
```

65.7 GB/s across four concurrent host→device streams — 3× the 22 GB/s single-link offload. The fabric is not the wall.

| | measured |
|---|---|
| single-link H2D (current offload) | 22 GB/s |
| 4 concurrent H2D | 65.7 GB/s |
| 4 concurrent GPU↔GPU peer | 49.4 GB/s |

The options first framed as: round-robin (copies still sequential on the host thread → streaming stays ~22 GB/s aggregate, but copy l+1 overlaps GEMM l on an idle card; the predicted bound ~130 t/s; does not tap the 65.7) versus a harder async host→device rewrite (pinned source, `hipMemcpyAsync` on each backend's stream, event-gate before compute launch; streaming approaches 65.7 GB/s, the 15 s term collapses toward ~5 s; ~200–250 t/s). Then a web search ("llama.cpp op_offload async host to device copy scheduler pull request 2026") upended the premise.

**Observations**

- **Correction (third revision of the copy mechanism):** the offload copy is already async. Issue #20757, citing current line numbers, shows the selective expert copy in `ggml_backend_sched_compute_splits()` copies used expert sub-rows CPU→GPU via `ggml_backend_tensor_set_async()` — not the blocking `tensor_copy` path. The wrong function had been quoted (`ggml_backend_tensor_copy_async`). The "blocking serialized copy" diagnosis had been pointing at the wrong copy site.
- Reading the function directly: the expert copy is `ggml_backend_tensor_set_async` on the split's own stream (line 48), already asynchronous, already copying only used experts — but `ggml_backend_synchronize(input_backend)` precedes the copy loop (line 6) and the ids read (line 25) forces another sync. Each split: sync → read ids → async-copy its experts → next split sync → and so on.
- **Hypothesis (revised):** "The copies are already async. They're serialized only because they all target one card. Spreading the target across four cards lets four async streams run concurrently — and your 65.7 GB/s H2D measurement is the ceiling that unlocks." The simple round-robin patch is the high-value patch; the hard async rewrite is largely unnecessary. **Prediction:** ~180–220 t/s honest projection, with the caveat that the per-split `ggml_backend_synchronize(input_backend)` may cap the gain below the full 65.7.
- Upstream findings: Issue #20757 (two-tier GPU+RAM expert cache for MoE offload, seeking a C++ contributor) and Issue #18530 (`GGML_OP_OFFLOAD_MIN_BATCH` configurability — relevant to the DFlash batch-32 offload cliff). Also surfaced: Issue #22065 (GTT offloading) and Issue #20697 (disk-based context checkpointing, `--cache-disk`). Sources listed for the search: llama.cpp server README, llama-server manpages, llama-cpp-pydist, a tech-insider tutorial, and the four issues. Histogram check re-issued:

```bash
grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
```

### 19:32 — The exact two-edit patch delivered

Paul: "Okay, give the exact patch to make then and I'll apply and test it now." The exact anchors were pulled — the struct is `calloc`'d (line 1738), so the new field zero-initializes; `bool op_offload;` at line 819 is the insertion anchor.

**Edit 1** — file `ggml/src/ggml-backend.cpp`, line 819. Find:

```c
    bool op_offload;
```

Replace with:

```c
    bool op_offload;
    unsigned off_rr; // round-robin cursor for op_offload target (fallback when layer index unparsable)
```

**Edit 2** — same file, line ~919 (the only match for `1.off`). Find:

```c
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
```

Replace with:

```c
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                // Distribute offloaded ops across all eligible GPU backends instead of
                // always taking the first. With a large CPU-resident MoE this stops
                // every offloaded expert matmul from serializing onto backend 0's
                // single PCIe link. Key on the layer index so a layer's gate/up/down
                // land on the same device (avoids extra peer hops); fall back to a
                // round-robin cursor if the name can't be parsed.
                int n_elig = 0;
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        n_elig++;
                    }
                }
                if (n_elig > 0) {
                    int il = -1;
                    int key;
                    if (src->name[0] != '\0' && sscanf(src->name, "blk.%d.", &il) == 1 && il >= 0) {
                        key = il % n_elig;
                    } else {
                        key = (int)(sched->off_rr++ % (unsigned)n_elig);
                    }
                    int seen = 0;
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            if (seen == key) {
                                SET_CAUSE(tensor, "1.off");
                                return b;
                            }
                            seen++;
                        }
                    }
                }
            }
```

Design notes: `src` is the weight tensor (loop variable at line 909); its name is `blk.N.ffn_*_exps.weight`, so `sscanf(src->name, "blk.%d.", &il)` extracts the layer; `n_elig` is computed first because the eligible-backend set varies per op — the key must be modulo the actual eligible count, not `src_backend_id`.

Build:

```bash
cd /root/llama.cpp
cmake --build build -j64 && cmake --install build
```

Test 1 (histogram, gates everything):

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_patched.txt 2>&1

grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
```

```bash
grep -c '## SPLIT' /root/sched_patched.txt   # was 1494 total; should be similar
```

Test 2 (throughput, only if Test 1 equalized):

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```

Live watch:

```bash
watch -n1 'rocm-smi --showuse | grep -E "GPU\[|use"'
```

**Observations**

- **Prediction:** baseline histogram ROCm0 731 / ROCm1 133 / ROCm2 175 / ROCm3 147, ~1,186 GPU splits; success = four ROCm counts ~290 each. Throughput baseline 104.97; plausible landing ~170–220; if the histogram equalizes but throughput reaches only ~140, the per-split `ggml_backend_synchronize(input_backend)` at line ~1588 is why.
- `-mmp 0` chosen for pinned host memory (async H2D from pinned pages hits the measured 65.7 GB/s; pageable would bounce-buffer). Fallback if the pinned allocation OOMs (411 GiB into the 498 GiB GTT pool): `-mmp 1` for a functional number first.

### 19:37 — The build breaks on ZenDNN

Paul pasted the build output:

```
CMake Error at /root/llama.cpp/build/_deps/zendnn-prefix/src/zendnn-stamp/zendnn-build-Release.cmake:49 (message):
  Command failed: 2
   '/usr/bin/cmake' '--build' '/root/llama.cpp/build/_deps/zendnn-prefix/build' '--target' 'zendnnl'
gmake[2]: *** [ggml/src/ggml-zendnn/CMakeFiles/zendnn.dir/build.make:89: _deps/zendnn-prefix/src/zendnn-stamp/zendnn-build] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2658: ggml/src/ggml-zendnn/CMakeFiles/zendnn.dir/all] Error 2
gmake: *** [Makefile:146: all] Error 2
```

The log also showed `Built target llama-common-base`, `Built target ggml-base`, `Built target vulkan-shaders-gen`, `Built target cpp-httplib`, and a UI download from b10001 (npm not found, HF download succeeded). Diagnosis: the ZenDNN subproject broke, nothing to do with the patch — `ggml-backend.cpp` compiled (`Built target ggml-base`). The same ZenDNN had been removed four days earlier for SIGILL crashes and was established as inert for GLM-5.2 (rejects 256-expert `MUL_MAT_ID` and Q4_K). "Kill it."

```bash
cd /root/llama.cpp
cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1030 \
  -DGGML_ZENDNN=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j64 && cmake --install build
```

If the cache conflicts:

```bash
cd /root/llama.cpp
rm -rf build
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j64 && cmake --install build
```

Confirmation commands:

```bash
# patch is still in the source
grep -n 'round-robin cursor for op_offload' /root/llama.cpp/ggml/src/ggml-backend.cpp

# ZenDNN really is out of the binary
ldd $(which llama-bench) | grep -i zendnn    # should print NOTHING
llama-bench --list-devices 2>&1 | tail -6    # should show only ROCm0-3, no ZenDNN
```

**Observations**

- **Dead end (build):** ZenDNN subproject failure; **Hypothesis:** a `git pull` reset a cached CMake variable or a fresh clone re-enabled it.

### 19:51 — Patch confirmed at line 820; histogram output missing

```
root@openwebui:~/llama.cpp# grep -n 'round-robin cursor for op_offload' /root/llama.cpp/ggml/src/ggml-backend.cpp
820:    unsigned off_rr; // round-robin cursor for op_offload target (fallback when layer index unparsable)
```

Both edits in, build succeeded; Paul ran the histogram command but the paste cut off before its output. Re-requested:

```bash
grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
```

```bash
grep -c '## SPLIT' /root/sched_patched.txt
```

```bash
llama-bench --list-devices 2>&1 | tail -6
```

Pre-registered outcomes: all four ~290 with count ~1,186 → works; ROCm0 still ~731 → did not take; spread but count exploded (2,000+) → layer-keying not co-locating. The answer came back empty:

```
root@openwebui:~/llama.cpp# grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
root@openwebui:~/llama.cpp# grep -c '## SPLIT' /root/sched_patched.txt
0
```

Zero splits captured despite `-v` — the flag that had produced the 31,795-line dump. Debugging commands:

```bash
wc -l /root/sched_patched.txt
tail -20 /root/sched_patched.txt
```

```bash
ls -la $(which llama-bench)
ls -la /root/llama.cpp/build/bin/llama-bench
```

```bash
llama-bench --list-devices 2>&1 | tail -6
ldd $(which llama-bench) | grep -i zendnn
```

**Observations**

- Three possibilities: crashed (a patch bug — if the inner `seen == key` match never fires, the block falls through toward `return -1`); still ZenDNN; or ran fine but the model failed to load.

### 19:52 — The log tail: an OOM at 4322.38 MiB on device 0

```
root@openwebui:~/llama.cpp# grep -c '## SPLIT' /root/sched_patched.txt
0
root@openwebui:~/llama.cpp# wc -l /root/sched_patched.txt
2326 /root/sched_patched.txt
root@openwebui:~/llama.cpp# tail -20 /root/sched_patched.txt
model has unused tensor blk.78.ffn_gate_inp.weight (size = 6291456 bytes) -- ignoring
create_tensor: loading tensor blk.78.exp_probs_b.bias
model has unused tensor blk.78.ffn_gate_exps.weight (size = 1811939328 bytes) -- ignoring
model has unused tensor blk.78.ffn_down_exps.weight (size = 2214592512 bytes) -- ignoring
model has unused tensor blk.78.ffn_up_exps.weight (size = 1811939328 bytes) -- ignoring
model has unused tensor blk.78.ffn_gate_shexp.weight (size = 13369344 bytes) -- ignoring
model has unused tensor blk.78.ffn_down_shexp.weight (size = 13369344 bytes) -- ignoring
model has unused tensor blk.78.ffn_up_shexp.weight (size = 13369344 bytes) -- ignoring
model has unused tensor blk.78.nextn.eh_proj.weight (size = 80216064 bytes) -- ignoring
model has unused tensor blk.78.nextn.enorm.weight (size = 24576 bytes) -- ignoring
model has unused tensor blk.78.nextn.hnorm.weight (size = 24576 bytes) -- ignoring
model has unused tensor blk.78.nextn.shared_head_norm.weight (size = 24576 bytes) -- ignoring
done_getting_tensors: tensor 'token_embd.weight' (q8_0) (and 225 others) cannot be used with preferred buffer type ROCm_Host, using CPU instead
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4322.38 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate ROCm0 buffer of size 4532346240
llama_model_load: error loading model: unable to allocate ROCm0 buffer
llama_model_load_from_file_impl: failed to load model
llama_bench: error: failed to load model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

The cause was asserted as fact: a 4.3 GiB allocation failing means ROCm0 is already full from a previous run that did not release VRAM — almost certainly a llama-bench process from crashed/killed runs. Check-and-clear commands:

```bash
# what's holding VRAM?
rocm-smi --showmeminfo vram
rocm-smi --showpids

# any zombie llama processes?
ps aux | grep -E 'llama|bench' | grep -v grep
```

```bash
pkill -9 -f llama-bench
```

```bash
rocm-smi --gpureset -d 0    # or reset all: rocm-smi --gpureset
```

```bash
sudo modprobe -r amdgpu && sudo modprobe amdgpu
```

```bash
rocm-smi --showmeminfo vram | grep -E 'GPU|Used'
```

**Observations**

- Model facts fixed by the tail: blk.78 (the unused nextn block) — ffn_gate_exps 1811939328 B, ffn_down_exps 2214592512 B, ffn_up_exps 1811939328 B, the three shexp tensors 13369344 B each, nextn.eh_proj 80216064 B; token_embd.weight is q8_0; "and 225 others" = the 225 expert tensors (75 MoE layers × 3).

### 19:53 — Paul rejects the stranded-VRAM claim

Paul: "the card is already full from a previous run that didn't release its VRAM. That has never happened before and I don't believe it." The epistemics conceded — it had been stated as fact when it's a hypothesis, and "it's never happened before" is a real data point against it — a re-read of the failure (the 225 expert tensors failing over from `ROCm_Host` to CPU suggested the buffer-type resolution had changed between runs, the only changes being the patch and the reconfigure) led to a request for a direct measurement:

```bash
rocm-smi --showmeminfo vram
```

**Observations**

- **Correction:** hypothesis restated as hypothesis; three decisive outcomes pre-registered: all near-empty → the OOM is a pinned-buffer path from the reconfigure; ROCm0 holding GiB with no process → stranded, clearable; a process holding it → `--showpids` names it.

### 19:53 — rocm-smi: all four cards 28–29 GiB used — a live resident server

```
GPU[0]		: VRAM Total Memory (B): 32195477504
GPU[0]		: VRAM Total Used Memory (B): 29351837696
GPU[1]		: VRAM Total Memory (B): 32195477504
GPU[1]		: VRAM Total Used Memory (B): 28229672960
GPU[2]		: VRAM Total Memory (B): 32195477504
GPU[2]		: VRAM Total Used Memory (B): 28236787712
GPU[3]		: VRAM Total Memory (B): 32195477504
GPU[3]		: VRAM Total Used Memory (B): 29281198080
```

All four cards ~28–29 GiB used of ~30 — a model currently loaded across all four GPUs. "I was right that the cards are full, wrong that it's stranded." The container is named `openwebui`; the obvious candidate was a live `llama-server` resident with the model loaded, serving the web UI. The benchmark tried to grab 4.3 GiB on top of a nearly-full card and lost.

```bash
rocm-smi --showpids
ps aux | grep -E 'llama-server|llama-cli|llama' | grep -v grep | grep -v bench
```

**Observations**

- **Correction:** "My 'previous run didn't release' guess was wrong in mechanism — this isn't a leak, it's a running service."

### 19:54 — Container reset; the post-reset checklist

Paul offered "I can reset the entire container." Endorsed — the reset kills the resident server and empties the cards — with three checks staged for afterwards:

```bash
rocm-smi --showmeminfo vram | grep Used
```

```bash
grep -c 'round-robin cursor for op_offload' /root/llama.cpp/ggml/src/ggml-backend.cpp
ls -la $(which llama-bench)
```

```bash
llama-bench --list-devices 2>&1 | tail -6    # ROCm0-3 only, no ZenDNN
```

then the histogram block again (success = ROCm0 falling 731 → ~290 with the others risen).

**Observations**

- **Decision:** reset the whole container rather than hunt and kill the resident process — fastest route to empty cards, at the cost of the running service.

### 19:55 — Cards clear at 17 MB; a Vulkan backend appears

```
root@openwebui:~# rocm-smi --showmeminfo vram | grep Used
GPU[0] : VRAM Total Used Memory (B): 17182720
GPU[1] : VRAM Total Used Memory (B): 17182720
GPU[2] : VRAM Total Used Memory (B): 17182720
GPU[3] : VRAM Total Used Memory (B): 17182720
```

17 MB per card — just framebuffer; all four empty, no auto-restart yet. The binary checks:

```
root@openwebui:~# ls -la $(which llama-bench) /root/llama.cpp/build/bin/llama-bench 2>&1
-rwxr-xr-x 1 root root 17952 Jul 14 19:41 /root/llama.cpp/build/bin/llama-bench
-rwxr-xr-x 1 root root 17952 Jul 14 19:41 /usr/local/bin/llama-bench
root@openwebui:~# llama-bench --list-devices 2>&1 | tail -6
  ROCm2: AMD Radeon Pro V620 (30704 MiB, 30618 MiB free)
  ROCm3: AMD Radeon Pro V620 (30704 MiB, 30618 MiB free)
  Vulkan0: AMD Radeon Pro V620 (RADV NAVI21) (30704 MiB, 30687 MiB free)
  Vulkan1: AMD Radeon Pro V620 (RADV NAVI21) (30704 MiB, 30687 MiB free)
  Vulkan2: AMD Radeon Pro V620 (RADV NAVI21) (30704 MiB, 30687 MiB free)
  Vulkan3: AMD Radeon Pro V620 (RADV NAVI21) (30704 MiB, 30687 MiB free)
```

The binary matches the build (both 17952 bytes, Jul 14 19:41, on PATH) — but the reconfigure had picked up a Vulkan backend absent from earlier builds: ROCm0–3 and Vulkan0–3, the same four physical cards enumerated twice. Not fatal, but a variable: `sched->n_backends` is larger, the round-robin `n_elig` count changes, and numbers will not read against the 4-backend baseline if the scheduler sees 8 GPU backends. Requested:

```bash
llama-bench --list-devices 2>&1
```

plus the histogram block, with a Vulkan-aware variant of the check (the original `sed` pattern only captures `ROCm[0-9]|CPU` and would silently drop Vulkan splits):

```bash
grep '## SPLIT' /root/sched_patched.txt | grep -oE 'ROCm[0-9]|Vulkan[0-9]|CPU' | sort | uniq -c
```

**Observations**

- **Decision (pending):** if Vulkan appears in the split accounting, rebuild with `-DGGML_VULKAN=OFF`.
- The thread never saw the patched histogram output directly in this session; the day's work ends here.

*The session resumes the following morning, Tuesday, July 15.*

### 10:36 — ZenDNN PR #23414: "Will I see an improvement with zendnn now?"

Paul sent an empty file attachment plus a link to llama.cpp PR #23414 and the question. The PR was fetched: "ggml-zendnn : add Q8_0 quantization support by z-sachin" — adds Q8_0 matmul to ZenDNN (previously F32/BF16 only). Headline benchmarks from the PR: Mixtral-8x7B Q8_0 prompt processing +90% to +213%, token gen −0.14%; gemma 31B Q8_0 prompt processing +68% to +115%, token gen −0.35%. Every gain is prompt processing; decode is flat or slightly negative — ZenDNN accelerates compute-bound prefill GEMM and does nothing for bandwidth-bound decode.

Verdict — does not apply to GLM-5.2, three reasons: (1) Q8_0 only, and GLM-5.2's experts are Q4_K/Q5_K (the 411 GiB that dominates prefill compute); (2) GLM-5.2's Q8_0 attention already runs on GPU (`-ngl 99`) and ZenDNN is a CPU backend — only an `-ngl 0` regime would feed it; (3) at that time an open PR, not merged. "No, you won't see an improvement." Keep `-DGGML_ZENDNN=OFF` (it broke the build twice and caused SIGILL crashes). The unfinished histogram was re-flagged:

```bash
grep '## SPLIT' /root/sched_patched.txt | grep -oE 'ROCm[0-9]|Vulkan[0-9]|CPU' | sort | uniq -c
```

**Observations**

- **Prediction:** no improvement for GLM-5.2 from ZenDNN Q8_0; where it would matter is a Q8_0 dense model run on the CPU (`-ngl 0` or heavy offload), which could nearly double prefill on the 64-core EPYC.

### 10:37 — A bare merge notice

Paul pasted "Pull request successfully merged and closed / You're all set — the branch has been merged." Which PR was unclear from the notice — the ZenDNN Q8_0 PR #23414 (nothing changes; keep it off) or Paul's own round-robin patch sent upstream (would be significant, but doubted) — so the number or title was requested, reiterating the standing state: cards cleared to 17 MB, patched binary of 19:41 confirmed, a Vulkan backend that should not be there, and the histogram in /root/sched_patched.txt still unseen.

**Observations**

- Ambiguity left open pending the PR number; the histogram grep was repeated once more.

### 10:38 — The full PR #23414 benchmark table; the verdict holds

Paul pasted the PR's benchmark tables (config: threads = 96, type_k = bf16, type_v = bf16).

Llama-3.1-8B-Instruct Q8_0 (Prompt Size | GGML_CPU_Q8_0 t/s | ZenDNN_Q8_0 t/s | Gain):

```
256    472.28   730.87   54.75%
512    450.86   832.48   84.64%
768    446.81   864.52   93.49%
1024   439.58   800.15   82.03%
2048   405.07   778.34   92.15%
tg128   33.08    33.14    0.18%
```

Mixtral-8x7B Q8_0:

```
256    156.09   297.67   90.70%
512    156.63   389.44   148.64%
768    156.76   417.38   166.25%
1024   154.70   438.73   183.60%
2048   150.11   470.41   213.38%
tg128   20.95    20.92    -0.14%
```

gemma4 31B Q8_0:

```
256    116.05   195.02   68.05%
512    112.53   229.12   103.61%
768    111.96   239.02   113.49%
1024   110.93   238.03   114.58%
2048   106.37   222.32   109.01%
tg128    8.50     8.47    -0.35%
```

gemma-4-26B-A4B-it Q8_0:

```
256    570.87   597.84   4.72%
512    581.80   666.18   14.50%
768    588.67   683.91   16.18%
1024   574.79   684.13   19.02%
2048   562.26   642.08   14.20%
tg128   33.96    33.83    -0.38%
```

This confirmed the merge was #23414, now on master. The tg128 column is the tell: +0.18%, −0.14%, −0.35%, −0.38% — decode nothing across the board. "Your decode is 6.01 t/s against a DDR4 wall — this PR moves it by zero." Prefill does not benefit either: GLM-5.2's experts are Q4_K gate/up plus Q5_K down (411 GiB, from the gguf dump) while the PR is Q8_0-only, and GLM-5.2's Q8_0 attention already lives on GPU. Where it would earn its keep: a Q8_0 dense coding model run on the CPU or under heavy offload (the Llama-8B and gemma-31B rows, +80–115%) — roughly double prompt processing on the 64-core EPYC; a separate llama.cpp build for that use.

**Observations**

- **Confirmed:** ZenDNN Q8_0 PR #23414 merged upstream; irrelevant to GLM-5.2 on this machine. Keep `-DGGML_ZENDNN=OFF`.
- Histogram request repeated a third time (ROCm0 was 731; success ~290; prefill 105 → ~180 hoped).

### 10:39 — Paul reveals the plateau: PP ~120 with the patches

Paul: "I already handled that in another thread. the short answer is that PP got up to about 120 and didn't improve further with the patches." (That other thread's work is documented in full as Session 9.) The result matched the corrected physics — the predicted bound, initially resisted: the expert copies are async but the per-split `ggml_backend_synchronize(input_backend)` serializes them on the host thread, so distributing the target across four cards buys overlap (copy l+1 hides behind GEMM l) but not parallel transfer; predicted ceiling ~130 t/s. ~120, plateauing, is within noise of the bound. The 65.7 GB/s aggregate H2D headroom is real but host-thread syncs prevent tapping it.

| | pp |
|---|---|
| where it started | 37.6 |
| unclamped ubatch (the `-p` fix) | 104.97 |
| + round-robin patch | ~120 |

3.2× total. The `-p` clamp fix was the big win; the patch added ~15% on top.

**Observations**

- **Confirmed:** the ~130 bound; the cap is host-thread copy serialization, not a flag or card fix.
- **Decision (investigation closed):** decode 6.01 t/s at the 152 GB/s DDR4 wall, every software lever dead — only fewer bytes (smaller quant, ruled out) or more bandwidth (Genoa, at absurd RAM prices) moves it. Prefill ~120 t/s, 3.2× banked, capped by host-thread copy serialization. Build config: `-DGGML_ZENDNN=OFF`, fits VRAM, `-t 32`, `-ub 8192`, op_offload on (free for decode).
- The next increment — making copies concurrent by removing per-split syncs — is a real scheduler rewrite, ~120 → maybe ~200 (an 8K prompt from 68 s to 41 s), estimated a week in delicate code. Issue #20757 is where the maintainers discuss the copy subsystem.

### 10:44 — The frontier gap pinned: ~10x

Paul: "As of today I'm about 10x slower than claude or chatgpt ai, right? not bad. In the future the gap will close to 5x slower probably." Hosted decode rates were checked (Artificial Analysis): Claude Opus ~58 t/s, GPT-5.6 ~69 t/s, reasoning-tier median ~78. Galactus at 6.01 → ~10x on decode vs Opus, ~11–13x vs the faster ones. End-to-end on an 8K prompt:

| | prefill (TTFT) | decode 3K tokens | total |
|---|---|---|---|
| Opus (hosted) | ~2 s | ~52 s | ~54 s |
| Galactus today (120 pp, 6 tg) | ~68 s | ~500 s | ~9.5 min |

~10.5x end-to-end; worse for agentic loops (a 10-turn loop is ten prefills of growing context), felt ~12–15x. "~10x is fair for a single chat turn. The real number is 10-15x."

**Observations**

- **Hypothesis (on the future 5x):** on the current box ~10x → maybe ~7x if DFlash lands (decode 6 → ~11, gated on an external draft). Reaching 5x needs DFlash and the prefill rewrite (~120 → ~200) and the frontier not pulling away (hosted decode 58 → 90+ in a year is plausible). Genoa/Turin at ~450 GB/s: decode 6 → ~9, plus DFlash ~16 = ~4–5x frontier, but DDR5 alone is +$10k+ at 2026 prices. Realistic software ceiling ~7x; 5x is a next-build target.

### 10:46 — The bar restated: 5x of today's frontier

Paul clarified he meant 5x slower than today's frontier as a fixed target, ballparking frontier PP ~1000 and token gen ~50. The yardstick was adjusted (hosted decode ~58–78 t/s — the 50 a touch low; prefill high-hundreds to ~1000+): 5x of frontier = decode ~10–15 t/s, prefill ~200 t/s.

| lever | decode | needs |
|---|---|---|
| today | 6.0 | — |
| DFlash spec | ~11 | a GLM-5.2 draft model to exist (not Paul's to make) |
| Genoa/DDR5 (~450 GB/s) | ~9 | ~$10k+ RAM |
| both | ~16 | both of the above |

**Observations**

- Decode hits the 5x bar only with DFlash, new memory, or both; software alone caps ~6–7 without a draft. Prefill can hit 5x-of-frontier without buying anything: the scheduler copy-rewrite (removing per-split syncs, four streams concurrent against 65.7 GB/s) takes ~120 → ~200.
- Most likely on current hardware and available software: prefill ~150–200, decode ~6–7 — ~5x on prefill, still ~10x on decode.

### 10:47 — DFlash as a category-level bet

Paul: "Right, and dflash is certain too. It's not here yet, and maybe not in this model. But it will land and it will run for glm5.2 or better class models eventually." The bet held up with two refinements: (1) the dependency is a draft trained against a model that fits the machine's shape — the real bet is that the local-inference ecosystem stays healthy enough to ship drafts for popular open frontier models (z-lab already did it for Qwen); (2) the machine gets better at speculation over time for free — verify-pass cost is dominated by the sparse-MoE expert read, and rising VRAM capacity plus the trend toward more/smaller/sparser experts both shrink it.

| | today | the bet matures |
|---|---|---|
| prefill | ~120 | ~200 (scheduler rewrite, Paul's to do) |
| decode | 6 | ~11–15 (spec lands + more VRAM-resident experts) |
| vs. frontier | ~10x | ~5x, plausibly better |

Caveat: the frontier moves (hosted decode ~58 → ~120 in two years), so "5x slower" is 5x of a faster target. What was bought is the hard, expensive, appreciating part — 1 TB of memory bandwidth and 122 GB of VRAM with lanes to feed it; the speculation layer comes free on someone else's training budget.

**Observations**

- **Hypothesis (endorsed):** the bet is about ecosystem vitality, not physics — speculative decoding is standard infrastructure, and a DFlash-or-successor draft for a locally runnable frontier-open-MoE-class model (GLM-5.2, GLM-6, or whatever 700B-open ships in 18 months) is a good bet.

### 10:49 — "5x slower than fast is still usable": the reading-speed threshold

Paul: "I'm saying, claude feels pretty fast today. If it's 5x slower that's still usable." The arithmetic holds: Claude at ~50–60 t/s is already faster than anyone reads; 5x slower ≈ 11 t/s ≈ ~330 words/minute, still at or above reading pace — barely noticeable for anything read as it streams, visceral only when not reading (bulk generation, agentic loops, code scrolled past). The threshold effect: the human floor for interactive generation is ~7–10 t/s; the frontier at 55 is far over it, Galactus at 6 sits just under (why it feels sluggish), and the matured ~11–15 crosses it. "The jump that matters isn't closing to 5x — it's crossing from below reading speed to above it." The asterisk is prefill/TTFT: a 60-second wait before the first token still annoys, which is why the prefill rewrite (~120 → ~200) matters more to felt experience than the last few decode t/s.

**Observations**

- Closing framing on the record: "you don't need frontier speed, you need above-reading-speed." A local 744B-class model streaming as fast as its owner reads, entirely on owned hardware, is a genuinely good way to run.

*The session closes the following day, Wednesday, July 16.*

### 14:41 — /export

Paul issued `/export`; the transcript closed with the exporter footer ("Powered by Claude Exporter").

### State of knowledge at end of Session 8

- Four-stream pinned host→device bandwidth measured at 65.7 GB/s aggregate (vs 22 GB/s single link and 49.4 GB/s four-pair peer): the fabric was never the wall.
- The copy-mechanism diagnosis settled on its third revision: MoE expert copies already go through `ggml_backend_tensor_set_async` (async, sparse via the `used_ids` bitset); they serialize because every split targets ROCm0. The exact two-edit layer-keyed patch (unsigned `off_rr` at line 819/820; distribution block at ~919) was delivered, applied (confirmed at line 820), and built after a ZenDNN break forced a `-DGGML_ZENDNN=OFF` reconfigure.
- The first patched histogram run OOMed at a 4322.38 MiB allocation on device 0 — traced not to the patch but to the live resident openwebui llama-server holding ~28–29 GiB on every card; a container reset cleared all four to 17 MB.
- The reconfigure resurrected a Vulkan backend (Vulkan0–3, RADV NAVI21) alongside ROCm0–3 — flagged as a confound for `n_backends` and the split accounting; the patched histogram was never seen inside this session.
- ZenDNN Q8_0 PR #23414 merged upstream (+54.75% to +213.38% prompt processing on Q8_0 models, tg128 flat/negative); irrelevant to GLM-5.2 (Q4_K/Q5_K experts, Q8_0 attention already GPU-resident). `-DGGML_ZENDNN=OFF` stays.
- Paul reported from a separate thread that prefill plateaued at ~120 with the patches — within noise of the corrected ~130 bound. Tally: prefill 37.6 → 104.97 (ubatch unclamp) → ~120 (patch), 3.2× total; decode 6.01 t/s at the DDR4 wall. Investigation declared closed on this box.
- Frontier framing: ~10x slower than hosted decode today (~10.5x end-to-end on an 8K prompt: ~54 s vs ~9.5 min); the realistic software path is ~7x, with 5x requiring DFlash (an ecosystem bet — no GLM-5.2 draft exists) and the scheduler copy rewrite; the usability threshold that matters is reading speed, ~7–10 t/s.

---

