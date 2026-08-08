## Session 9 — Tuesday, July 21, 2026 — Building and benchmarking the patch

This session is the separate thread in which the scheduler patch was actually built and benchmarked end to end. The transcript (`llamacpp_patch.md`, exported 07:37 ET) carries no timestamps; entries below follow sequence order. A second analysis stream, whose output Paul pasted in, appears throughout. Context artifact: "Claude State Export.zip" (saved 7/21 06:53 ET) contains the openwebui system prompt and knowledge files Paul prepared for local-model use — the workload the patched machine was being tuned to serve.

### Entry 1 — The brief: the measured split pin and round-robin patch v1

Paul opened with "Consider the following observation and proposed patch. Will this patch work, does it make sense, and will it dramatically improve my prefill?" and pasted the review's package: the measured split distribution —

```
Splits:   ROCm0  731   (49%)
          ROCm2  175
          ROCm3  147
          ROCm1  133
          CPU    308
```

— ROCm0 owning 731 of 1,186 GPU splits (62%), "exactly the `return b`=0 concentration, confirmed at runtime, no longer inferred"; the node-level counts dismissed as a red herring (resident attention nodes); 731 splits on ROCm0 = 731 host→device transfers serialized onto one card's link = the 22 GB/s single-link bottleneck the ubatch ladder measured; the everything-agrees checklist (splits 62% ✓, copy path blocking/serialized ✓, streaming 22 GB/s = one link ✓, peer fabric 49 GB/s healthy ✓, flag tricks cannot move it ✓); the named dead end (the original `src[1]` activation-routing diff is a no-op because the activation is unassigned when the heuristic runs); and patch v1 — the per-op round-robin block for `ggml/src/ggml-backend.cpp` around line 919 with the `int off_rr;` struct field (identical to the version recorded in Session 7 at 19:11), plus the build/verify sequence:

```bash
cd /root/llama.cpp
cmake --build build -j64 && cmake --install build

# re-run the exact split-distribution check that just confirmed the pin
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_patched.txt 2>&1
grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
```

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```

**Observations**

- Baseline: 104.97 pp8192 t/s. **Hypothesis (pasted review):** round-robin moves streaming from one link to four; the 15 s streaming term drops to ~4–5 s; path to ~250–350 t/s.
- Stated caveats: round-robin ignores locality (peer hops affordable at 49 GB/s aggregate); the copy is still blocking (async within a card would need a separate harder patch); decode never triggers offload at batch 1 — 6.01 t/s untouched.

### Entry 2 — First analysis: the histogram will equalize, but the copies are believed blocking

The patch was assessed as aimed at the right place — it will compile and equalize the split histogram — but the 15 s → 4–5 s claim was rejected: host→device copies cannot take the async path (`cpy_tensor_async` returns false unless both backends are GPU), so every expert-weight transfer goes through the synchronous fallback in `ggml_backend_sched_compute_splits`, blocking the single scheduler thread; copies to different cards cannot overlap regardless of destination. What the patch does buy: the fallback synchronizes the destination before copying, so with targets spread the destination is idle and copy l+1 overlaps GEMM l — while the GEMMs remain a strict serial chain across layers.

**Observations**

- **Prediction (bound):** 8192 tokens at 104.97 t/s ≈ 78 s; perfectly hiding 15 s of streaming yields ~63 s ≈ ~130 t/s. "Worth having, but not 250–350."
- **Decision (design seed):** prefer keying on the layer — `sscanf(src->name, "blk.%d.", &il); pick = il % n_elig` with a cursor fallback — keeping each layer's gate/up/down local and stable across ubatches; per-op round-robin would add two extra multi-hundred-MB peer hops per layer at ub 8192 and roughly double the split count. Make `off_rr` unsigned; the struct is calloc'd in `ggml_backend_sched_new` (keep the grep). Expect ROCm1–3 to newly allocate about one layer's expert staging; compare total split count before/after; decode untouched (`offload_op` requires batch ≥ 32).
- Suggested pre-test for the harder async patch: `hipMemcpyAsync` from pinned host to all four cards on four streams from one thread — if the root complex caps aggregate H2D near 22–40 GB/s rather than ~88, no scheduler change reaches the target. (This became the 65.7 GB/s measurement.)
- Caveat on these claims: the `cpy_tensor_async` and calloc details date from mid-2025 knowledge; confirm against the actual checkout.

### Entry 3 — The 65.7 GB/s measurement and the pasted review's self-correction

Paul returned with the probe result and the pasted review's follow-up. Headline: four concurrent host→device streams sustain 65.7 GB/s, 3× the 22 GB/s single-link offload.

| | measured |
|---|---|
| single-link H2D (current offload) | 22 GB/s |
| 4 concurrent H2D | 65.7 GB/s |
| 4 concurrent GPU↔GPU peer | 49.4 GB/s |

"The gap between 22 and 65.7 is the prize. The blocking serial copy path leaves 2/3 of your H2D bandwidth on the floor." The pasted review's initial framing: round-robin → ~130 (the serial-copy limit); an async H2D patch (pinned source, `hipMemcpyAsync` per-backend stream, event-gate) is what the 65.7 unlocks → ~200–250 t/s — "The 65.7 says the hard patch is worth writing. If it had come back ~30, I'd have told you to ship round-robin and stop." The pasted review then self-corrected after a search: Issue #20757, citing current line numbers, shows the selective expert copy in `ggml_backend_sched_compute_splits()` moves used expert sub-rows CPU→GPU via `ggml_backend_tensor_set_async()` — not the blocking path. Its revised mechanism: the copies are already async on the split's own stream and already sparse; the serialization is that every split targets ROCm0, so all 731 async copies queue back-to-back on one card's stream — "the async-ness is wasted because there's only one destination." Third-revision hypothesis: the simple round-robin patch is the high-value patch; ~180–220 t/s honest projection; the hard async rewrite largely unnecessary. Upstream context: Issue #20757 (two-tier GPU+RAM expert cache, wants a C++ contributor — "your round-robin + benchmark is a cleaner, smaller, immediately-shippable contribution") and Issue #18530 (`GGML_OP_OFFLOAD_MIN_BATCH` configurability). The instruction: run the histogram check first (`grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c`; ROCm0 should drop 731 → ~290), with expectation "well above ~130, plausibly 180+."

**Observations**

- **Correction (pasted review, third revision of its mechanism):** blocking copy (wrong); round-robin gives only overlap, ~130 (wrong, assumed blocking); truth — the copies are already async, serialized only by the single target.
- **Prediction (pasted review):** round-robin, the simple already-drafted patch, is the high-value patch; ~180–220 t/s honest projection; the hard async rewrite largely unnecessary. Residual risk flagged but unbounded: the per-split `ggml_backend_synchronize(input_backend)` syncs — not boundable from a source read.

### Entry 4 — Upstream master read directly; the ids-read data dependency; a pre-registered null

Rather than adjudicate between recollections, the sources were fetched — Issue #20757 and raw `ggml-backend.cpp` from master — and the saved file grepped for `used_ids|MUL_MAT_ID|set_async|synchronize` and `need_new_split|GGML_SCHED_MAX_SPLIT_INPUTS|prev_ids_tensor`. Verdict: the pasted review's line-level reading is accurate; its conclusion is not.

The binding constraint is a data dependency, not the copy path. In the selective-copy block, per expert weight: wait on the destination, sync the CPU backend, then — once per distinct ids tensor — `ggml_backend_tensor_get_async(ids_backend, ...)` followed by `ggml_backend_synchronize(ids_backend)`, a host-blocking drain of whichever GPU produced the routing ids. Only then are the `tensor_set_async` copies enqueued. The ids for layer *l* come from layer *l*'s router, which needs attention *l*, which needs layer *l−1*'s expert GEMM, which followed layer *l−1*'s copies. Copies → GEMM → attention → ids sync → copies: strictly serial, on any number of cards, under any assignment policy. "Four destinations give you four idle streams that each wait their turn." The pasted review's unbounded caveat "is in fact decidable from the code: it is the whole effect."

The enabling patch was sketched — a large-batch bypass of the ids gate. At ub 8192, selectivity saves nothing (the fraction of experts untouched ≈ e^(−8192k/N) ≈ 0):

```c
const int64_t n_sel = node->src[2]->ne[0] * node->src[2]->ne[1]; // top_k * n_tokens
if (n_sel >= 8 * n_expert) {
    copy_experts(0, n_expert - 1);   // hoist the lambda above the ids read
} else {
    /* existing ids path — decode untouched */
}
```

**Observations**

- **Prediction (pre-registered):** build round-robin alone → the histogram equalizes while prefill stays ≈105 t/s. "If it jumps well above ~115, my chain analysis is wrong somewhere and I will re-derive."
- Synergy argument: with a single target, the per-input destination wait drains the same card running the previous GEMM — run-ahead collapses to zero even with the bypass; with layer-keyed distribution the destination card last worked several layers ago, the wait is free, and copies genuinely prefetch on idle links toward the 65.7 GB/s ceiling. "The two patches are synergistic; neither is sufficient alone."
- **Correction (of the pasted review):** issue #20757 is closed, not live.
- Caveats: true async H2D on HIP requires a pinned source — the load log must show expert tensors in a `ROCm_Host` buffer with `-mmp 0`; no throughput number projected without decomposing the remaining ~63 s; confirm the tree matches master via `grep -n "used_ids\|prev_ids_tensor" ggml/src/ggml-backend.cpp`.
- **Decision (order):** for upstream attribution, run round-robin alone first — "its predicted null result is itself evidence for the dependency analysis."

### Entry 5 — The final Edits 1 and 2, applied and built

The final patch text as built (anchors confirmed: struct calloc'd at line 1738; `bool op_offload;` at line 819).

**Edit 1 — add the field.** File `ggml/src/ggml-backend.cpp`, line 819. Find:

```c
    bool op_offload;
```

Replace with:

```c
    bool op_offload;
    unsigned off_rr; // round-robin cursor for op_offload target (fallback when layer index unparsable)
```

**Edit 2 — the routing change.** Same file, line ~919 (only match for `1.off`). Find:

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

Build:

```bash
cd /root/llama.cpp
cmake --build build -j64 && cmake --install build
```

Test 1 — split concentration (gates everything):

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

Test 2 — the number (only if Test 1 equalized):

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```

```bash
watch -n1 'rocm-smi --showuse | grep -E "GPU\[|use"'
```

Paul's report closing the turn: "I did this. What else should I d now?" — patch applied and built, no results yet posted.

**Observations**

- Before: ROCm0 731, ROCm1 133, ROCm2 175, ROCm3 147; want ~290 each.
- Stated expectation (carried from the pasted review's framing): "Plausible landing ~170–220," with the line ~1588 sync named as the cap risk. If `-mmp 0` OOMs on the pinned allocation (411 GiB into the 498 GiB GTT pool, sometimes fragmented by earlier runs), fall back to `-mmp 1`.

### Entry 6 — Edit 3 drafted: the ids-gate bypass

With Tests 1/2 pending, the readout table was issued — ROCm0 still ~731 → the patch did not take (check `ldd $(which llama-bench) | grep ggml`); equalized at ~105 → the ids-gated chain binds, proceed to Edit 3; equalized at ~125–145 → destination-wait removed, chain partially binding, still proceed; equalized ≥170 → the chain analysis is wrong somewhere, profile before further patches — and the pasted review's caveat was corrected: `ggml_backend_synchronize(input_backend)` syncs the CPU backend, cheap; the expensive sync is `ggml_backend_synchronize(ids_backend)` inside the ids read, draining the GPU that just ran the router.

**Edit 3 — skip the ids read at prefill batch sizes.** Rationale: at ub 8192 with top-k routing, P(expert unused) ≈ e^(−top_k·n_tokens/n_expert) ≈ 0 — the ids read saves no bandwidth and only imposes the serializing sync; decode (n_sel = top_k) keeps the selective path. First confirm the tree matches master (`grep -n "prev_ids_tensor" ggml/src/ggml-backend.cpp`; the anchor occurs once, in `ggml_backend_sched_compute_splits`). Find:

```c
                    if (ids_tensor != prev_ids_tensor) {
```

Replace with:

```c
                    // At prefill-sized batches every expert is selected by some token
                    // (P(unused) ~ e^(-top_k*n_tokens/n_expert)), so the ids tell us
                    // nothing worth waiting for -- but reading them forces a
                    // host-blocking synchronize on the backend that produced them,
                    // chaining this copy behind the previous layer's expert GEMM.
                    // Skip the read and mark all experts used: the copy can then be
                    // issued immediately and overlap compute on other devices.
                    if (ids_tensor->ne[0] * ids_tensor->ne[1] >= 8 * n_expert) {
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        memset(used_ids.data(), 0xFF, used_ids.size() * sizeof(ggml_bitset_t));
                        prev_ids_tensor = ids_tensor;
                    } else if (ids_tensor != prev_ids_tensor) {
```

**Observations**

- All-set bits make the existing grouping loop emit one contiguous `copy_experts(0, n_expert - 1)` — identical bytes to a full-tensor copy, no lambda changes; extra bits in the last bitset word are harmless; add `#include <cstring>` if `memset` is undeclared.
- Extra checks specified after rebuild: rerun Test 1 (histogram stays equalized, split count similar), then Test 2; a decode regression check at `-n 64 -p 0` against baseline 6.01 (the distribution fires only at batch ≥ 32; the Edit 3 condition is false at batch 1); a pinning check — the load log should show expert tensors in `ROCm_Host` with `-mmp 0`.

### Entry 7 — The Vulkan confound: stale CMake cache

Paul reported a Vulkan appearance during the rebuild. Diagnosis: most likely a stale CMake cache — `cmake -B build` on an existing `build/` reuses `build/CMakeCache.txt`, whose variables are sticky; if `GGML_VULKAN` was ever ON in that directory it stays on (it defaults OFF in ggml, pointing at the cache). Alternatives to rule out: a different `llama-bench` on PATH, or dynamic backend loading picking up a stray `libggml-vulkan.so`.

```bash
grep -i vulkan build/CMakeCache.txt         # cached ON?
which -a llama-bench                          # more than one on PATH?
ldd $(which llama-bench) | grep -i vulkan     # does the binary link vulkan?
ls /usr/local/lib*/libggml*vulkan* build/bin/libggml*vulkan* 2>/dev/null  # stray .so?
```

Clean fix:

```bash
rm -rf build
cmake -B build -DGGML_HIP=ON -DGGML_VULKAN=OFF \
  -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j64 && cmake --install build
```

**Observations**

- Why it gates the test: if Vulkan registers devices it enumerates the same four AMD GPUs the HIP backend claims — changing `sched->n_backends` (the exact quantity the patch keys on: `src_backend_id == sched->n_backends - 1` and the modulo over eligible backends) and potentially double-counting devices, so "ROCm0 731" stops meaning what it meant. A histogram showing only `ROCm*` labels is mild evidence Vulkan is listed but idle on the offload path. The stale-binary case is the more consequential: the patch may not be the binary under test.

### Entry 8 — Test 1: the histogram equalizes at 285/300/294/292

```
root@openwebui:~# grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
    308 CPU
    285 ROCm0
    300 ROCm1
    294 ROCm2
    292 ROCm3
```

Distribution even (285–300, ~5% spread); ROCm0 dropped 731 → 285. The split count did not balloon: 1171 GPU splits vs the original 1186; 1479 total vs 1494. "Layer-keying did exactly what it was supposed to — spread the target without fragmenting."

**Observations**

- "Hold the champagne": both theories predicted equalization; the histogram is necessary but not discriminating. The pre-registered claim is that distribution alone equalizes the histogram while prefill stays ≈105 — the ub 8192 ladder is the arbiter.
- The histogram shows only ROCm0..3 + CPU — five backends, CPU last, exactly `n_backends = 5` — so this run's accounting is trustworthy; clear the cache later regardless.
- Gating question posed: is Edit 3 compiled into this binary, or only Edit 2? (The histogram cannot distinguish — Edit 3 changes when copies issue, not which card.)

### Entry 9 — Test 2, Edit 2 alone: 105.71 — the pre-registered null confirmed

Paul ran the throughput test (Vulkan still present in this build):

```
root@openwebui:~# llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro V620 (RADV NAVI21) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
(... same for devices 1–3 ...)
```

| model | size | params | backend | ngl | threads | n_batch | n_ubatch | fa | ot | mmap | test | t/s |
| --- | ---: | ---: | --- | --: | --: | --: | --: | --: | --- | ---: | ---: | ---: |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm,Vulkan | 99 | 32 | 8192 | 8192 | 1 | exps=CPU | 0 | pp8192 | 105.71 ± 0.56 |

105.71 vs the 104.97 baseline — no movement (the gap inside ±0.56). Paul confirmed via the follow-up question that this binary carried Edit 2 only.

**Observations**

- **Confirmed (pre-registered):** "Distribution alone did not touch prefill." The pasted review's 170–220 projection is falsified; the binding constraint is the ids-read sync chaining each layer's copies behind the previous layer's GEMM, not the copy destination. "Four idle streams that each wait their turn are still serial."
- Vulkan is now confirmed active (`Found 4 Vulkan devices`, backend column `ROCm,Vulkan`), double-mapping the same four physical cards; the Test 1 histogram (only ROCm splits) suggests it idle on the offload path — "lean toward 105.71 being a valid ROCm-only number," but not certified. Clean rebuild ordered (`rm -rf build`; `-DGGML_HIP=ON -DGGML_VULKAN=OFF -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release`).
- Verification grep offered for the Edit 3 question: `grep -n 'ids_tensor->ne\[0\] \* ids_tensor->ne\[1\] >= 8' ggml/src/ggml-backend.cpp`.

### Entry 10 — The serial-chain bound; Edit 3 ordered into the build

"Edit 2 only, and the number didn't move — that confirms the pre-registered prediction and kills the distribution-unlocks-parallelism theory." The bound restated from dependency structure: a single ubatch of 8192 makes the layers a strict serial chain; exactly one layer computes at any instant on the one card that owns it; four cards take turns by layer — compute does not parallelize across them (the pasted review's "GEMM spreads across four cards' shaders" is wrong for this workload; that needs multiple ubatches in flight). The only recoverable time is the H2D streaming term (~15–17 s of the ~77.5 s pass) hidden under compute via prefetch. Cap for the whole approach (distribution + Edit 3): roughly 130–135 t/s, full stop; above that requires parallelizing compute (pipelining or tensor parallel). Edit 3 restated in condensed form:

```c
// find: if (ids_tensor != prev_ids_tensor) {
// replace with:
                    if (ids_tensor->ne[0] * ids_tensor->ne[1] >= 8 * n_expert) {
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        memset(used_ids.data(), 0xFF, used_ids.size() * sizeof(ggml_bitset_t));
                        prev_ids_tensor = ids_tensor;
                    } else if (ids_tensor != prev_ids_tensor) {
```

(The condensed form of the same code as the full Edit 3 in Entry 6; the bypass branch precedes the existing selective path, which is left unchanged.)

**Observations**

- **Decision:** keep Edit 2 in — "Edit 3 lets the host run ahead, Edit 2 spreads those run-ahead copies onto idle links so they hit your 65.7 GB/s. Neither works alone."
- **Prediction (pre-registered for Edit 2+3):** prefill breaks above ~115, lands low-to-mid 120s, ceiling ~135. If it stays ~105, the ids-sync was not binding (a residual destination-wait synchronize at line ~1573, or unprofiled) — stop patching, take rocprof/per-split timings.
- Build hygiene: the clean `-DGGML_VULKAN=OFF` rebuild; optional control sequence (Vulkan-off + Edit 2 only → expect ~105 → proves Vulkan idle → then add Edit 3) versus folding both into one build.

### Entry 11 — "Alright, it built. What's the test"

The arbiter:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```

Baselines to beat: 104.97 stock, 105.71 Edit 2 alone. Read before trusting the number: (1) the banner must have no `ggml_vulkan: Found 4 Vulkan devices` and the backend column must read `ROCm`, not `ROCm,Vulkan`; (2) the load log must show `ROCm_Host buffer size` carrying the bulk of the weights (a plain `CPU buffer` means pageable, `tensor_set_async` staging through a bounce buffer synchronously — a false negative); (3) the t/s. Decode correctness check:

```bash
llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 -t 32 -p 0 -n 64 -r 2 -o md
```

**Observations**

- Interpretation grid: Edit 2+3 in and above ~115 = chain theory holds (expect low-to-mid 120s, ceiling ~135); still ~105 with pinning confirmed = the chain theory is dead alongside the pasted review's — trace. Decode baseline 6.01: "If that moved, something fired that shouldn't have."

### Entry 12 — Discovery while waiting: -ot silently disables pipeline parallelism

Paul asked, "Is there anything else that we can do in the code to improve prompt processing?" `src/llama-context.cpp`, fetched from master, showed:

```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer_all &&
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
    cparams.offload_kqv &&
    !model.has_tensor_overrides();   // <-- -ot trips this
```

`-ot` is a tensor override, so `pipeline_parallel` is false, `ggml_backend_sched_new(..., parallel=false, ...)` sets `n_copies = 1`, and no events are created (verifiable by the load log missing `pipeline parallelism enabled`). Minor consequence: with no events, the destination wait at `ggml-backend.cpp:1573` is a host-blocking `ggml_backend_synchronize(split_backend)` instead of a stream-ordered `event_wait`. Major consequence: the benchmark is single-ubatch by construction (`-p 8192 -ub 8192`) and structurally cannot pipeline — hence the ~135 bound. A real 32k prompt is four ubatches, which can pipeline across four cards (ubatch 1 enters layer 0 while ubatch 0 is at layer 40) — genuine compute parallelism, with compute ~60 s of the 77.5 s pass. Plan: measure `-p 32768 -ub 8192` and `-p 16384`; if flat at ~105, attack the condition (comment out `!model.has_tensor_overrides()`).

**Observations**

- Caveats: multi-ubatch multiplies streaming (nothing persists between passes; 4 ubatches = 4× the 17 s — still a paper win since compute is ~78% of the pass); `n_copies=4` quadruples staging (~4 GiB/layer → ~16 GiB/card vs 30), with an auto-fallback at line 619; unknown why upstream excludes tensor overrides (an adjacent `TODO: move these checks to ggml_backend_sched` suggests conservatism — "test it, don't assume it").
- Hardware floor recorded from the Vulkan banner: `matrix cores: none` — gfx1030 is RDNA2, no WMMA, no MFMA; the ~60 s of compute is shader dot-products, "no software fixes it."
- **Dead end:** the two-tier expert cache of #20757 is a decode optimization — at prefill every expert is used and the hit rate collapses. `--n-cpu-moe N` (~15–20% off streaming) is redundant if Edit 3 hides streaming.

### Entry 13 — pp8192 = 119.36 ± 0.12: fastest yet, attribution pending

"First some results. Then you tell me what to do next."

```
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB): (4x AMD Radeon Pro V620, gfx1030, VMM: no, Wave Size: 32, VRAM: 30704 MiB)
```

| model | size | params | backend | ngl | threads | n_batch | n_ubatch | fa | ot | mmap | test | t/s |
| --- | ---: | ---: | --- | --: | --: | --: | --: | --: | --- | ---: | ---: | ---: |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm | 99 | 32 | 8192 | 8192 | 1 | exps=CPU | 0 | pp8192 | 119.36 ± 0.12 |

Vulkan gone (backend `ROCm`). 119.36 — the fastest prefill so far, +13.7% over stock. But the run bundled the Vulkan removal with the source change; attribution had to wait ("Vulkan removal alone is entirely capable of producing +13"). What survives regardless: 104.97 vs 105.71 was a Vulkan-matched pair, so "distribution alone does nothing" holds within that regime. Paul confirmed via the follow-up that this build carried Edit 2 + Edit 3.

**Observations**

- **Confirmed (provisionally):** 119.36 sits above the pre-registered ~115 threshold, just under the predicted low-to-mid 120s — "consistent with the chain theory — which is exactly why I'm not claiming it yet." The `-p 32768`/`-p 16384` scaling runs were queued regardless.

### Entry 14 — Accounting at 119.36; the pipeline patch proposed

At 119.36 the pass is 68.6 s (down from 78.0 s). Against ~17 s streaming + ~60 s compute floor, full hiding would give ~61 s ≈ ~134 t/s — so ~7–8 s of streaming remains exposed, ~15 t/s left in the copy path, and then compute is the wall. **Hypothesis:** the remaining exposure is due to `n_copies = 1` (no events), leaving the destination wait at `ggml-backend.cpp:1573` as a host-blocking `ggml_backend_synchronize(split_backend)` that throttles run-ahead — caused by `-ot` tripping `!model.has_tensor_overrides()`.

Run 1 (free, no rebuild) — long-prompt scaling:

```bash
llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 16384,32768 -n 0 -r 2 -o md
```

Run 2 — the patch, `src/llama-context.cpp`, drop the last condition:

```cpp
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv;
            // !model.has_tensor_overrides();   // <-- -ot tripped this
```

Verify: the load log prints `pipeline parallelism enabled` and `sched copies = 4`; then rerun pp8192 (the win is narrow — events allow deeper run-ahead, maybe 119 → high 120s/low 130s) and pp32768 (the real test).

**Observations**

- Failure modes pre-registered: `n_copies=4` quadruples staging (~4 GiB/layer → ~16 GiB vs 30 GiB/card) — it fails safe at line 619, logging `compute buffer allocation failed, retrying without pipeline parallelism`; the fallback is a rebuild with `-DGGML_SCHED_MAX_COPIES=2` (the `#ifndef` at ggml-backend.cpp:764). Check that pp8192 output is still coherent, not just fast.

### Entry 15 — pp16384 = 86.11; the attention decomposition; pp32768 predicted at 55.3

The first scaling row arrived (pp32768 still running):

| model | size | params | backend | ngl | threads | n_batch | n_ubatch | fa | ot | mmap | test | t/s |
| --- | ---: | ---: | --- | --: | --: | --: | --: | --: | --- | ---: | ---: | ---: |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm | 99 | 32 | 8192 | 8192 | 1 | exps=CPU | 0 | pp16384 | 86.11 ± 0.10 |

**Correction (own test design):** "flat at ~119 means serial compute" was a bad test — prefill throughput drops with length regardless, because attention is O(n²); 86.11 at 16k is expected scaling. Decomposition (linear terms double, causal attention quadruples; pair count n²/2: 33.6M → 134.3M):

```
T(8192)  = 8192/119.36  =  68.63s  =  S + A + C
T(16384) = 16384/86.11  = 190.27s  = 2S + 4A + 2C
T(16384) - 2·T(8192) = 53.01s = 2A   →   A = 26.5s
```

Attention = 26.5 s of 68.6 s = 39%; streaming + expert GEMM = 42.1 s (collinear — separable only by the 22 GB/s measurement pinning S ≈ 17 s, hence expert GEMM ≈ 25 s). **Correction:** the earlier ~60 s "compute floor" had lumped attention in; attention is the larger half. Strategic consequence: attention share 39% at 8k, 56% at 16k, 72% at 32k — "everything we've done to the copy path decays in value as your context grows"; the pipeline patch becomes more valuable (it parallelizes attention), with pp16384 pipeline ON vs OFF (86.11 = OFF baseline) as the clean test.

**Observations**

- **Prediction (pre-registered):** the model reports `glm-dsa` — if it were a sparse-attention variant the exponent would differ from 2. pp32768 discriminates: quadratic predicts 55.3 t/s; meaningfully above means the decomposition is redone.

### Entry 16 — pp32768 = 55.76: the decomposition validated at 0.8 percent error

| model | size | params | backend | ngl | threads | n_batch | n_ubatch | fa | ot | mmap | test | t/s |
| --- | ---: | ---: | --- | --: | --: | --: | --: | --: | --- | ---: | ---: | ---: |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm | 99 | 32 | 8192 | 8192 | 1 | exps=CPU | 0 | pp16384 | 86.11 ± 0.10 |
| glm-dsa ?B Q4_K - Medium | 435.19 GiB | 753.86 B | ROCm | 99 | 32 | 8192 | 8192 | 1 | exps=CPU | 0 | pp32768 | 55.76 ± 0.03 |

55.3 predicted, 55.76 actual — 0.8% error. This settles glm-dsa: attention scales quadratically; DSA "is not saving you anything." Refit across three points (every combination agrees within 1%):

```
T(8192)  =  68.63s  =  L +  A
T(16384) = 190.27s  = 2L + 4A
T(32768) = 587.66s  = 4L + 16A
                      →  A ≈ 26.3s,  L ≈ 42.3s   (at 8192 scale)
```

| prompt | attention | expert streaming |
|---|---|---|
| 8k | 38% | 25% |
| 16k | 55% | 18% |
| 32k | 72% | 12% |

At 32k, deleting streaming entirely buys 13%. The real long-prompt number: 55.76 t/s means a 32k prompt costs ten minutes. Pipeline modeling (perfect 4-stage):

```
4S + (16A + 4C)/4  =  68 + 130.5  =  198.5s  →  ~165 t/s   (vs 55.76 today)
```

**Observations**

- **Confirmed:** the two-component decomposition (A ≈ 26.3 s quadratic attention, L ≈ 42.3 s linear terms at 8192 scale). Structural limit noted: streaming and expert GEMM are both linear and perfectly collinear — "no number of prompt-length points can ever separate them"; only the 22 GB/s measurement pins S ≈ 17 s.
- The ~165 is a ceiling, not a forecast (ramp, stage imbalance, and streaming overlap eat into it), and it concentrates at long context (pp8192 = one ubatch → nothing beyond the event destination-wait).
- **Decision:** apply the llama-context.cpp change, rebuild, rerun; confirm `pipeline parallelism enabled` + `sched copies = 4`; watch for the allocation-failed fallback (`-DGGML_SCHED_MAX_COPIES=2`); verify pp8192 still ~119 and text coherent (a safety condition is being bypassed).

### Entry 17 — The graph-reuse trap; the pre-registered prediction table

Asked "What should I expect the values to end up being," the saved upstream source was re-read first — one thing could invalidate the whole prediction — and found at `llama-context.cpp:1328`, the graph-reuse path:

```cpp
if (!graph_reuse_disable && res->can_reuse(gparams)) {
    // with pipeline parallelism, the previous graph_compute_async may still be running...
    if (cparams.pipeline_parallel) {
        ggml_backend_sched_synchronize(sched.get());   // <-- full barrier, all backends
    }
```

Prefill ubatches have identical shapes, so `can_reuse` fires from ubatch 2 onward — with pipeline_parallel on, a full drain of every backend per ubatch. Reuse and pipelining are mutually exclusive by design: `cur_copy` only advances inside `ggml_backend_sched_alloc_graph` (line 1873), which the reuse path skips, so the n_copies buffers never rotate and the sync is the only correctness guard. The pipeline patch alone therefore gains nothing (a barrier per ubatch, 4× staging spent); it additionally needs `LLAMA_GRAPH_REUSE_DISABLE=1`. (`ggml_backend_sched_reset` at line 1826 does not synchronize, so the non-reuse path is clean.)

| test | now | pipeline patch only | patch + `LLAMA_GRAPH_REUSE_DISABLE=1` |
|---|---|---|---|
| pp8192 | 119.36 | 125–135 | 125–135 |
| pp16384 | 86.11 | ~86, unchanged | 120–145 |
| pp32768 | 55.76 | ~56, unchanged | 105–125 |

Why pp32768 gains only ~2×, not the ~165: **Correction —** ubatch imbalance was unaccounted. Attention per ubatch is 1×, 3×, 5×, 7× of A, so the four ubatches cost 51.6 s, 104.2 s, 156.8 s, 209.4 s; the pipeline overlaps the first three while the last traverses all four stages alone:

```
makespan ≈ (51.6 + 104.2 + 156.8)/4 + 209.4 ≈ 288s   →  ~114 t/s
```

The tail worsens with context — pipelining pays less at 64k than at 32k.

**Observations**

- Caveats: reuse-disable costs a full graph rebuild and re-split per ubatch (~1479 splits); n_copies=4 with 4 ubatches multiplies staging and activation memory (watch for `compute buffer allocation failed, retrying without pipeline parallelism`; fall back to `-DGGML_SCHED_MAX_COPIES=2`); two upstream safety conditions are bypassed at once — verify coherence, not just speed.

### Entry 18 — Pipeline results: 86.02 unchanged; then n_copies = 4 OOMs to "sched copies = 1"

Without the env var, pp16384 came back 86.02 ± 0.00 — unchanged against 86.11 (both repetitions identical), matching the middle-column prediction: the pipeline patch alone does nothing at multi-ubatch. But neither the patch's presence nor the env var could be confirmed from that run (llama-bench suppresses the load log), so one command was issued to verify all three states plus the number:

```bash
LLAMA_GRAPH_REUSE_DISABLE=1 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 16384 -n 0 -r 2 -o md -v 2>&1 | \
  grep -iE "pipeline parallelism|sched copies|graph reuse|allocation failed|pp16384"
```

Wanted: `graph reuse disabled`, `pipeline parallelism enabled`, `sched copies = 4`, no `compute buffer allocation failed`; prediction with all four confirmed: 120–145 at pp16384 (pp32768 dropped deliberately — pp16384 answers in one fifth the wall clock; the 32k prediction of 105–125 is tail-limited by the 209 s solo traversal). The result:

```
llama_context: graph reuse disabled
llama_context: pipeline parallelism enabled
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
sched_reserve: reserve took 522.47 ms, sched copies = 1
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |    8192 |     8192 |   1 | exps=CPU              |    0 |         pp16384 |         86.12 ± 0.08 |
```

Both flags took, but the reserve OOMed on `n_copies=4` and fell back: `sched copies = 1` means pipelining never ran. 86.12 is the identical serial path measured a third time (86.11, 86.02, 86.12). Fix — halve the copies:

```bash
cmake -B build -DGGML_HIP=ON -DGGML_VULKAN=OFF -DGGML_SCHED_MAX_COPIES=2 \
  -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j64 && cmake --install build
```

(Watch configure for `Manually-specified variables were not used by the project: GGML_SCHED_MAX_COPIES`; if present, edit the `#define` at `ggml-backend.cpp:765` from 4 to 2 directly. Want `sched copies = 2`, no allocation failure.)

**Observations**

- **Confirmed:** the middle column of the prediction table (patch-only ≈ unchanged).
- **Prediction:** pp16384 120–145 at full strength (two copies is all two ubatches need); pp32768 85–100, degraded from 105–125 (two overlapped pairs: ~117 s first pair + ~249 s second ≈ 366 s ≈ 90 t/s; still ~1.6× over 55.76). If ~86 with `sched copies = 2` confirmed → the reuse-path analysis is wrong → trace. If n_copies=2 also OOMs → "pipelining is simply unavailable on 30 GiB cards with 4 GiB of expert staging per layer, and 119 / 86 / 56 is where this configuration lands."

### Entry 19 — sched-copies mechanics; the guard vindicated; PR triage

Paul asked "What does sched copies = 2 do." The mechanics: it is the pipeline depth — how many ubatches in flight. For every tensor crossing a device boundary (a split input), the scheduler allocates `n_copies` duplicate buffers (`ggml-backend.cpp:1358–1369`); each graph evaluation claims the next copy round-robin (`cur_copy = next_copy; next_copy = (next_copy + 1) % n_copies` at line 1873) and `compute_splits` reads/writes only that index (line 1562), so ubatch k writes buffer k mod N while k−1 still reads k−1 mod N. With `n_copies = 1` there is one buffer per input — ubatch k+1 would overwrite what k reads — so the scheduler must drain the device first: the host-blocking `ggml_backend_synchronize(split_backend)` at line 1573 "that has been serializing you all along." Multiple copies remove the hazard and populate the `events[backend][copy]` array (line 811), turning the destination wait into a stream-ordered `ggml_backend_event_wait`. The cost is linear in VRAM (as then understood): N× every split-input buffer, and split inputs include staged expert weights at ~4 GiB/layer — which is what blew the compute buffer at n_copies=4. Dropping 4 → 2 gives up depth (at most two stages at once, ~2× compute parallelism instead of 4×): pp16384 unaffected (two ubatches need two buffers), pp32768 drops to 85–100.

Paul then asked the sharper question: "There seems to be a tradeoff here that might prevent this patch from landing upstream then? Or am I wrong." The concession: "You're right, and your OOM is the evidence."

**Observations**

- **Correction (of an earlier claim):** `!model.has_tensor_overrides()` had been called "conservative rather than principled" — wrong; the fallback vindicates it. `-ot` turns split inputs from small activations into multi-GB weight tensors; `n_copies` replicates every one; 4× ~4 GiB/layer does not fit in 30 GiB. Over-broad (it denies 80 GiB cards) but not arbitrary.
- The harder blocker: `LLAMA_GRAPH_REUSE_DISABLE=1` is not shippable — reuse exists because rebuilding per token wrecks decode; the conflict is structural (`split_graph` bakes `cur_copy` into node pointers at line 1374; a reused graph cannot rotate buffers).
- Upstream-shaped sketch: the scheduler replicates every split input n_copies times — free for activations, catastrophic for offloaded weights, and weights do not need replication (only one ubatch occupies layer L at a time). In the loop at 1358–1369, allocate one copy when `src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS` and point every copy slot at it; the reuse conflict wants a separate fix (condition reuse on graph type: prefill rebuilds and pipelines, decode reuses and syncs). "Design reasoning from reading, not a measurement."
- **Decision (PR triage):** Edit 2 measured zero alone and ignores locality (could regress on a weaker peer fabric) — not a PR by itself. Edit 3 has the cleanest story ("at prefill batch sizes the ids read buys nothing and costs a device drain") — write that up first. Nothing is upstream-ready without a clean before/after.

### Entry 20 — n_copies = 2 also OOMs; the input/output-flag discovery closes pipelining

```
llama_context: graph reuse disabled
llama_context: pipeline parallelism enabled
sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism
sched_reserve: reserve took 516.42 ms, sched copies = 1
```

Before chasing memory, the CMake option was confirmed real — `set(GGML_SCHED_MAX_COPIES "4" CACHE STRING "ggml: max input copies for pipeline parallelism")` in `ggml/CMakeLists.txt` (confirmable via `grep GGML_SCHED_MAX_COPIES build/CMakeCache.txt`) — and then the reason no setting above 1 can ever fit surfaced. `ggml-backend.cpp:1363`:

```c
if (sched->n_copies > 1) {
    ggml_set_input(tensor_copy);
    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
}
```

Marking every split-input copy input+output disables ggml-alloc's buffer reuse for all of them. At n_copies=1 the flags are never set, so galloc overlaps staging — one layer's expert buffer at a time, ~4 GiB/card, fits. At n_copies=2 every offloaded layer's staging is simultaneously live: ~23 layers per card × ~4.3 GiB × 2 ≈ ~200 GiB per card, against 30.

**Observations**

- **Correction (own model):** "the cost isn't 2× or 4×. It's (layers per card) × n_copies, and it was never going to fit at any setting above 1... your two identical OOMs were telling you so while I was blaming VRAM headroom."
- **Confirmed (upstream guard):** `!model.has_tensor_overrides()` "prevents a *guaranteed* OOM for every `-ot`/`--cpu-moe` user on any hardware. I've now called that condition unprincipled twice and been wrong both times. It's correct as written."
- **Dead end:** pipeline parallelism under `-ot` on 30 GiB cards, at any n_copies > 1. The sharpened upstream idea — do not set the input/output flags on WEIGHTS-usage split inputs and allocate one copy for them (each device's stream serializes its own work; the flags protect host-written inputs racing the GPU, a hazard streamed weights on the device's own stream do not have) — would make pipelining affordable for CPU-MoE offload, but is allocator-adjacent and subtle: `test-backend-ops` and coherence matter more than the benchmark.
- **Decision point posed:** "accept 119 / 86 / 56 and stop, or write that allocator change."

### Entry 21 — Rollback to Edits 2+3; attribution locked at +14.4; revert verified at 119.29

Paul: "Let's roll back to just patch 2. give me instructions for backing out the other patches." After disambiguating the numbering ("Patch 3" had been used for both Edit 3 and the llama-context change) and an explicit choice, Paul elected to keep both Edit 2 and Edit 3 — the 119.36 configuration — and back out only the pipeline experiment:

```bash
cd /root/llama.cpp
git diff > ~/moe-offload-edits.patch    # save everything first
git checkout src/llama-context.cpp
git diff --stat                          # should now list ONLY ggml/src/ggml-backend.cpp
unset LLAMA_GRAPH_REUSE_DISABLE

cmake -B build -DGGML_SCHED_MAX_COPIES=4 -DGGML_HIP=ON -DGGML_VULKAN=OFF \
  -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j64 && cmake --install build
```

(Leaving `LLAMA_GRAPH_REUSE_DISABLE` exported would cost a full graph rebuild per token — a decode regression.) Verification, wanting the diagnostics absent:

```bash
llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md -v 2>&1 | \
  grep -iE "pipeline parallelism|sched copies|graph reuse|allocation failed|pp8192"
```

A side note for the write-up flagged a bookkeeping discrepancy — a claimed "+20–30" was larger than the +14.4 from the only matched pair (104.97 → 119.36, both Vulkan-present). Paul settled it: "No, your math was correct. It's 14.4." That locked the attribution — the Vulkan removal contributed ~0, so 104.97 was a valid baseline and the `ROCm,Vulkan` column was cosmetic (the Test 1 histogram, with zero Vulkan splits, had said so). **The record: 104.97 stock; 105.71 distribution alone (nothing); 119.36 distribution + ids-gate bypass. Edit 3 delivered all 14.4; Edit 2 contributed nothing on its own but was the precondition** — "without somewhere to run ahead *to*, removing the ids sync buys nothing." The result lands inside the pre-registered 115–135 band; the chain theory holds: the binding constraint was `ggml_backend_synchronize(ids_backend)` draining the GPU that produced the routing ids. The rollback verification run:

```
sched_reserve: reserve took 422.59 ms, sched copies = 1
glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |    8192 |     8192 |   1 | exps=CPU              |    0 |          pp8192 |        119.29 ± 0.19 |
```

**Observations**

- **Confirmed:** 119.29 ± 0.19 vs 119.36 ± 0.12 — identical within noise; the gain survived the revert intact. Independent confirmation: reserve 422 ms vs 516/522 ms during the pipeline attempts (the failed reserve-and-retry cycle gone). All four signals correct (no pipeline/reuse/allocation lines, `sched copies = 1`).
- **Prediction (recorded, not measured):** keeping Edit 3 alone would give ~105, not ~119 — the edits are synergistic; without Edit 2 every offloaded op targets ROCm0 and the run-ahead immediately blocks on that one card's destination wait.
- Outstanding: the decode check (expect 6.01 quoted at the time) — it also catches a leaked `LLAMA_GRAPH_REUSE_DISABLE`.

### Entry 22 — The PR draft; decode settled at 5.51 → 5.51; the transcript moves off-topic

Paul: "Write me a pull request message... For places where you need measurements from my other session, put a placeholder telling me to paste in." `PR-moe-offload-prefill.md` was written to a local outputs path (`…/outputs/PR-moe-offload-prefill.md`), initially with eight placeholders (grep `[PASTE`), two flagged as mattering most: pp16384/pp32768 on unpatched master (every long-context number in the thread is patched — no "before" column), and tg64 with the patch (the decode-unaffected claim then an argument, not a measurement). Framing: lead with distribution-alone-does-nothing to pre-empt the reviewer's first question; include the pipeline-parallelism finding as a closing note stating the `!has_tensor_overrides()` guard is correct; review will land on the 8× threshold and the locality trade, both flagged open.

Paul then supplied the patched decode measurement:

```
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |   1 | exps=CPU              |    0 |            tg64 |          5.51
```

adding "(This is correct because layer offloading was used to get 6tps)."

**Observations**

- **Correction:** the PR had quoted 6.01 in the master decode row — an apples-to-oranges comparison; 6.01 came from a different layer-offload placement, not `-ot "exps=CPU"`. The file was edited: the master row became a placeholder to be filled with the same `-ot "exps=CPU"` placement, 6.01 removed entirely, and the inert-at-batch-1 reasoning added — Distribution: `ggml_backend_offload_op()` returns false below batch 32, so `n_elig == 0` and the new block is skipped entirely; ids bypass: `ids_tensor->ne[0] * ids_tensor->ne[1]` is `top_k × 1` at batch 1, far below `8 · n_expert`, so the selective path is unchanged. Fill procedure via stash cycle:

```bash
git stash && cmake --build build -j64 && cmake --install build
llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 -t 32 -p 0 -n 64 -r 2 -o md
git stash pop && cmake --build build -j64 && cmake --install build
```

- **Prediction, then Confirmed:** the master number would be exactly 5.51 (both edits provably unreachable at batch 1). Paul: "I already have that from yesterday. It was 5.51" — unpatched master, same placement. "Decode is measured, not argued: 5.51 → 5.51, exactly as predicted." The PR's decode table became master 5.51 / patched 5.51, "No change, as expected — both edits are unreachable at batch 1 by construction."
- **Decision (PR state at session end):** placeholder list trimmed to six — (1) ROCm/HIP version, (2) llama.cpp commit hash, (3) pp16384/pp32768 on unpatched master, (4) `test-backend-ops` result, (5) an output-coherence check (llama-cli, any prompt), (6) single-GPU and non-MoE multi-GPU sanity runs. "Only one of them needs a rebuild: master pp16384/pp32768." No PR was submitted within this transcript.
- The transcript then shifts off-topic — an ASUS TUF GAMING B550M-PLUS / ECC / Ryzen 5950X build discussion unrelated to Galactus — and the patch narrative ends here.

### State of knowledge at end of Session 9 — final tally of the investigation

- **Prefill: 37.6 → 104.97 → 119.36 t/s (pp8192).** The ubatch unclamp (`-p` fix) delivered 104.97; the two-edit scheduler patch delivered 119.36 ± 0.12 (+13.7%, pass 78.0 s → 68.6 s); the revert-and-verify run reproduced it at 119.29 ± 0.19. Where the configuration lands across context: 119 (pp8192) / 86 (pp16384) / 56 (pp32768).
- **Decode: ~5.5–6.0 t/s at the DDR4 wall.** tg64 = 5.51 unpatched → 5.51 patched with the same `-ot "exps=CPU"` placement (both edits provably unreachable at batch 1); the earlier 6.01 belonged to a different layer-offload placement. The 152 GB/s DDR4 read of CPU-resident experts remains the decode ceiling; only a smaller quant (rejected) or new memory moves it, with DFlash-class speculation (~11 t/s) pending an ecosystem-supplied draft model.
- **What was proven about the scheduler.** The `return b` first-fit in `ggml_backend_sched_backend_id_from_cur` concentrates offloaded ops on backend 0 — measured, not inferred: 731 of 1,186 GPU splits (62%) on ROCm0, equalized to 285/300/294/292 by the layer-keyed patch without split-count inflation (1171 vs 1186 GPU; 1479 vs 1494 total).
- **Distribution alone is a null result:** Edit 2 by itself measured 105.71 ± 0.56 against 104.97 — confirming the pre-registered prediction and falsifying the 170–220 projection. The copies were already async (`ggml_backend_tensor_set_async`, sparse via `used_ids`); the binding serializer was the ids-read data dependency — `ggml_backend_synchronize(ids_backend)` draining the GPU that produced the routing ids, chaining each layer's copies behind the previous layer's expert GEMM.
- **Edit 3 (ids-gate bypass at n_sel ≥ 8·n_expert) delivered the entire +14.4 t/s, with Edit 2 as its precondition** — run-ahead requires idle destination links. Neither edit works alone; both are inert at decode by construction.
- **Prefill decomposition validated to 0.8%:** pp32768 predicted 55.3, measured 55.76 ± 0.03. Attention A ≈ 26.3 s (quadratic — glm-dsa's DSA is not sub-quadratic here), linear terms L ≈ 42.3 s at 8192 scale (S ≈ 17 s streaming pinned by the 22 GB/s measurement; expert GEMM ≈ 25 s). Attention's share grows 38% → 55% → 72% at 8k/16k/32k, so the copy-path win decays with context.
- **Pipeline parallelism is a dead end on this hardware:** `-ot` trips `!model.has_tensor_overrides()`; bypassing it plus `LLAMA_GRAPH_REUSE_DISABLE=1` OOMed identically at n_copies=4 and n_copies=2 (reserve 522.47 / 516.42 ms, `sched copies = 1`; post-revert 422.59 ms) because `n_copies > 1` sets input/output flags that disable ggml-alloc reuse — staging is layers × copies ≈ ~200 GiB/card against 30. The upstream guard is correct as written; a WEIGHTS-usage single-copy allocator change is the identified upstream-shaped fix.
- **Deliverable:** the kept configuration is Edits 1+2+3 in `ggml/src/ggml-backend.cpp` (diff saved to `~/moe-offload-edits.patch`; build flags `-DGGML_SCHED_MAX_COPIES=4 -DGGML_HIP=ON -DGGML_VULKAN=OFF -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release`), plus the drafted `PR-moe-offload-prefill.md` with six placeholders outstanding.
- Net of the whole investigation: prefill 3.2× (37.6 → 119.36), decode held at the memory wall (~5.5–6.0 t/s), and a measured, mechanistic account of why — with the remaining headroom (concurrent copies, ~200 t/s prefill; speculation, ~11 t/s decode) identified and priced.
