# llama.cpp scheduler patch — multi-GPU MoE offload for prefill

Three edits to `ggml/src/ggml-backend.cpp`. On Galactus they raised GLM-5.2 prefill from **104.97 to 119.36 t/s** (pp8192, +13.7%), on top of the free 2.6× from unclamping `n_ubatch`. Decode does not change. The edits act only at prefill batch sizes.

**The problem.** With a large MoE in system RAM and `-ot exps=CPU`, the llama.cpp scheduler offloads every large expert matmul to backend 0. On Galactus, 731 of 1,186 GPU splits went to ROCm0, and three cards stayed idle. The expert-weight copies are already asynchronous (`ggml_backend_tensor_set_async`). Two things still serialized them. First, they all queued onto one card's stream. Second, a per-split `ggml_backend_synchronize(ids_backend)` blocked the host to read the routing ids, which chained each layer's copy behind the previous layer's GEMM.

**The fix.** Edit 2 spreads the offload target across all eligible GPUs, keyed on the layer index, so each layer's gate/up/down stay on one card. Edit 1 adds a fallback cursor for Edit 2. Edit 3 skips the ids read at prefill batch sizes, where every expert is used, so the read saves no bandwidth and only adds the blocking synchronize. The async copies can then issue at once and overlap compute across the cards.

> **Source and cautions.** The line numbers are from llama.cpp master, about July 21, 2026. Confirm your tree before you patch:
> ```bash
> grep -n "prev_ids_tensor\|used_ids" ggml/src/ggml-backend.cpp   # Edit 3 anchor, one hit in compute_splits
> grep -n "bool op_offload;"          ggml/src/ggml-backend.cpp   # Edit 1 anchor
> ```
> **Edit 2 alone changes nothing.** We predicted this before the run, and the run confirmed it: 105.71 vs 104.97 baseline. The copies were already async; the ids read (Edit 3) was the true serializer. Edits 2 and 3 together produce the gain. Edit 2 alone gives an equal split histogram with unchanged throughput. Check the histogram to confirm the mechanism, and the throughput to confirm the gain.

---

## Edit 1 — add the round-robin fallback field

`ggml/src/ggml-backend.cpp`, about line 819. The struct is created with `calloc`, so the field starts at zero with no explicit assignment.

**Find:**
```c
    bool op_offload;
```
**Replace with:**
```c
    bool op_offload;
    unsigned off_rr; // round-robin cursor for op_offload target (fallback when layer index unparsable)
```

## Edit 2 — spread the offload target across eligible GPUs

Same file, about line 919. This is the only match for the `"1.off"` cause string.

**Find:**
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
**Replace with:**
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
`src` is the weight tensor, with the name `blk.N.ffn_*_exps.weight`, so `sscanf(src->name, "blk.%d.", &il)` reads the layer number. Compute `n_elig` first, because the set of eligible backends can change per op, and the key must use the modulo of the actual eligible count.

## Edit 3 — skip the ids read at prefill batch sizes

Same file, in `ggml_backend_sched_compute_splits`. If `memset` is undeclared, add `#include <cstring>`.

**Find:**
```c
                    if (ids_tensor != prev_ids_tensor) {
```
**Replace with:**
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
The all-set bits make the existing grouping loop emit one contiguous `copy_experts(0, n_expert - 1)`. This copies the same bytes as a full-tensor copy, and it needs no change to the lambda. Extra bits in the last bitset word do no harm; the loop stops at `n_expert`.

---

## Build and check

```bash
cd /path/to/llama.cpp
cmake --build build -j$(nproc) && cmake --install build
```

**Check 1 — did the concentration break up? Do this check first.**
```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > sched_patched.txt 2>&1

grep '## SPLIT' sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
grep -c '## SPLIT' sched_patched.txt   # split count should stay near the pre-patch total, not grow
```
Before: ROCm0 731, ROCm1 133, ROCm2 175, ROCm3 147. After: about equal, near 290 each. If ROCm0 is still about 731, the patch did not take effect. Confirm that `ldd $(which llama-bench) | grep ggml` points at your rebuilt library. Do not measure throughput yet.

**Check 2 — the number. Do this check only if Check 1 gave an equal histogram.**
```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```
`-mmp 0` pins the host buffer. Async H2D from pinned pages reaches the multi-stream bandwidth limit; pageable memory uses a bounce buffer and stays lower. Confirm the load log shows the expert tensors in a `ROCm_Host` buffer. If `-mmp 0` runs out of memory on the pinned allocation, use `-mmp 1` for a working number first.

**Regression check (one run):** decode `-n 64 -p 0` should not change. Round-robin acts only at batch 32 or more, and Edit 3's condition is false at batch 1.

## Galactus results

| Build | pp8192 | pp16384 | pp32768 | tg64 |
|---|---|---|---|---|
| Stock, ub 8192 | 104.97 ± 0.53 | — | — | 5.51 |
| Edit 2 only (distribution) | 105.71 ± 0.56 (no change) | — | — | — |
| **Edit 2+3** | **119.36 ± 0.12** | 86.11 ± 0.10 | 55.76 ± 0.03 | 5.51 |

Split histogram after the patch: ROCm0 731 → 285; distribution 285/300/294/292 plus CPU 308.

## Upstream

This relates to [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757) (a two-tier GPU+RAM expert cache for MoE offload). It touches the same code (`compute_splits`, the selective expert copy). An upstream PR for this patch is an open item; the text above is its basis. If you take this to a PR, the two contributions are the layer-keyed distribution and the batch-size-gated ids bypass. State the no-change result for Edit 2 alone, so reviewers see why Edit 3 is required.
