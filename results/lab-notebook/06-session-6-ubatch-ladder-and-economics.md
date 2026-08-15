## Session 6 — Monday, July 14, 2026 (late afternoon) — The ubatch ladder and the economics of the build

### 15:29 — The purchase thesis, scored against four justifications

While the v4 ladder ran, Paul stated his original purchase thesis: the GPUs were bought "entirely for the prefill, KV cache and attention layers/shared experts and to run small models in VRAM (e.g. like tiny vision model, maybe 27B fast coding model etc)" — at 2026 prices he knew he would never fit a large MoE in VRAM. The message cut off mid-thought at "And"; scored against the four stated purposes, the buy came out vindicated on all four.

1. **Prefill.** GPU prefill at 40 t/s vs the CPU's 20. The first ladder row proved the mechanism: 19.8 s per ubatch, 441 GB of experts moved, 22.3 GB/s — essentially line rate for PCIe 4.0 x16. A plumbing limitation, not a card limitation.
2. **Attention + KV cache.** The `-ngl 0` run was the receipt: the CPU needs ~167 ms for the dense path (MLA projections, shared experts, output head); the GPUs do it in ~48 ms of actual reading. MLA compresses the KV cache to ~90 KB/token — 44 MiB at 512 context, well under 12 GiB at 128K — sitting in VRAM at 512 GB/s instead of DDR4 at 152 GB/s.
3. **Shared experts.** `ffn_*_shexp` fires on every token across all 75 layers; `-ot "exps=CPU"` correctly does not match it (`shexp` does not contain the substring `exps`). Q8_0, ~12 MiB per layer, permanently GPU-resident.
4. **Small models in VRAM.** 122 GiB of VRAM; a 27B coder at Q8 is ~29 GiB — four could be held simultaneously. A 27B dense model tensor-parallelizes (unlike GLM-5.2): 27B at Q8 across four cards = 60–100+ t/s.

**Observations**

- **Prediction (later corrected):** `-ub 8192` should land near 300 t/s; fixing `return b` puts prefill near 800–1,200 t/s.
- **Correction (self):** "I kept evaluating these cards as GLM-5.2 accelerators... I've been benchmarking the wrong hypothesis for two messages." With 411 GiB of experts and no consumer or prosumer configuration reaching that, spending on VRAM bandwidth would have been irrational; the experts always live in DDR4, and DDR4 was always the wall.
- Swapping in 3090s would buy 5% and cost 26 GiB of capacity; MI100s at 1,229 GB/s HBM2 would be pure waste.

### 15:30 — Component costs disclosed; the $9,050 build, itemized

Paul supplied the remaining line items: "$600 for CPU, $600 for motherboard, $250 for case, $400 for PSU and $5600 for RAM." With the GPUs at $1,600 after tax and shipping, the build totals $9,050.

| component | cost | share |
|---|---|---|
| RAM (1 TB DDR4) | $5,600 | 62% |
| 4× V620 | $1,600 | 18% |
| EPYC 7713 | $600 | 7% |
| Motherboard | $600 | 7% |
| PSU | $400 | 4% |
| Case | $250 | 3% |

Supporting analysis recorded alongside the table:

- Parameter accounting: routed experts ~22.65B, shared experts ~2.83B, attention ~10B, dense layers 0.68B, embeddings/output 1.9B → ~38.1B active (matching the claimed 40B). Memory layout per token: 13.77 GB of experts read from DDR4, 19.7 GB of dense weights in VRAM; GPU buffers total 18.36 GiB; total footprint ~33.5 GB/token.
- 62% of the build went to the bottleneck component: "91 of every 181 ms is the CPU reading experts out of that DDR4 at 152 GB/s." Correct spending — without 1 TB the model does not load.
- The CPU is the sleeper: $600 for 64 cores when the decode sweep says only 24 are needed (decode peaks at t=24–32 and decays past it); what was actually bought is 8 memory channels and 128 PCIe lanes, both load-bearing. The motherboard delivers 8 channels plus four x16 slots — 89 GB/s aggregate host-to-device bandwidth.
- Genoa/SP5 counterfactual: 12 channels of DDR5 ≈ 400 GB/s vs 152. Model: `90 ms (GPU, unchanged) + 13.77 GB ÷ 400 GB/s = 124 ms = 8.1 t/s` — +35% decode. Dual-socket DDR4 EPYC 7003 would be 16 channels ≈ 410 GB/s theoretical, but MoE decode cannot be NUMA-partitioned (the router picks experts dynamically); cross-socket Infinity Fabric ~64 GB/s/direction on Milan caps scaling at ~1.4×, not 2×.
- Mac Studio M3 Ultra comparison: 512 GB unified at 819 GB/s, ~$10,000; ~33 t/s theoretical, 20–25 t/s realistic under llama.cpp Metal, ~15–18 t/s quoted as the practical verdict — 2.5–3× decode but wrong for this owner (near-zero context room in 512 GB, no PCIe, no GPUs, no upgrades, no small-model hosting, a sealed appliance).

**Observations**

- **Decision:** "Good build, and the GPUs were the best-value line item." The next dollar should not go to hardware — the `return b` heuristic in `ggml-backend.cpp` is twenty lines.

### 15:34 — 2026 RAM price research: DDR4 emphatically the right call

At 15:32 Paul directed: "Go check out 2026 RAM prices and then answer for me." A search ("DDR4 server RAM price 2026 shortage DRAM price surge") returned live figures.

- Live server-RAM tracker (updated twelve days prior): best DDR4 $5.15/GB (Samsung 32GB); best DDR5 $30.94/GB (SK Hynix 32GB) — DDR5 roughly 6× DDR4. A 64 GB DDR5 RDIMM was ~$255 in Q3 2025 and crossed $900 a year later, $14.06/GB.
- Paul paid $5.47/GB ($5,600 / 1 TB) — essentially spot for DDR4 today.

| | this build | Genoa/DDR5 equivalent |
|---|---|---|
| RAM | 1 TB DDR4 — $5,600 | 768 GB DDR5 RDIMM @ $14.06/GB — $10,800 |
| CPU | EPYC 7713 — $600 | EPYC 9004 — ~$2,000 |
| Board | SP3 — $600 | SP5 — ~$1,000 |
| GPUs / PSU / case | $2,250 | $2,250 |
| Total | $9,050 | ~$16,050 |

+77% of total build cost for +35% decode — and that is the cheap 768 GB configuration; matching 1 TB of DDR5 RDIMM alone runs $14,400, putting the build near $21,000.

- Appreciating asset: DDR4 ECC RDIMM rose ~60–80% from early 2025 to Q1 2026; Taiwanese makers are raising 8Gb DDR4 contract prices up to 50% in Q3 2026 vs Q2. Samsung, Micron, and SK Hynix have largely exited DDR4; only Nanya and Winbond produce it, below demand. SK Hynix calls 2027 the "worst year," with the crunch lasting to 2030. "You could not rebuild this machine today for $9,050."

GPU re-scoring on the decode-governing metric, cost per unit memory bandwidth:

| | cost | bandwidth | $/GB-per-sec |
|---|---|---|---|
| 1 TB DDR4 | $5,600 | 152 GB/s | $36.84 |
| 4× V620 (as used, one card live) | $1,600 | 512 GB/s | $3.13 |
| 4× V620 (aggregate installed) | $1,600 | 2,048 GB/s | $0.78 |

The GPUs are 12× better value than the RAM on the metric that governs decode, 47× on bandwidth physically owned. Two software bottlenecks keep the rest idle: `llm_arch_supports_sm_tensor()` returns false for `GLM_DSA` (MLA blocks tensor parallelism, leaving 1,536 GB/s of installed VRAM bandwidth idle), and `return b` in `ggml-backend.cpp` (every offloaded op goes to card 0, three of four PCIe 4.0 x16 links idle during prefill).

**Observations**

- **Correction (self):** "DDR4 was emphatically the right call, and it isn't close. I was wrong to keep floating Genoa."
- **Decision:** "So: buy nothing" — not DDR5, not Genoa, not better GPUs. The next gain is twenty lines in `ggml-backend.cpp`.
- Sources cited (10): Unibetter RAM Shortage 2026 Guide; PCSP Refurbished DDR4; datacenterdisk memory-chip-shortage-2026; Tom's Hardware RAM price index 2026; Club386 DDR4 +50%; Gizmochina DDR4 +50% Q3 2026; wccftech memory shortages DDR4 +50%; bacloud When Will RAM Prices Drop; VersaLogic Supply Chain Brief; ramexchange Why Are RAM Prices Increasing.

### 15:34 — A result announced but not received; reference predictions staged

Paul wrote "I have a new result. Is this starting to match your model?" but no output came through. Retrieval commands were staged and the linear model pre-registered against the single standing row (ub 512 → 25.90: 19.8 s per ubatch, 22.3 GB/s on ROCm0's single link — per-ubatch cost ~100% transfer, ~0% compute).

```bash
D=$(ls -dt /root/diag-v4-*/ | head -1)
grep -E '^\| glm-dsa' "$D/MAIN.log"
```

| `-ub` | ubatches per 8192 | predicted pp |
|---|---|---|
| 512 | 16 | 25.9 ✓ |
| 1024 | 8 | ~52 |
| 2048 | 4 | ~103 |
| 4096 | 2 | ~185 |
| 8192 | 1 | ~300 |

**Observations**

- The `-nopo 1` control rows should sit flat near 35 regardless of ubatch.
- Three failure signatures pre-registered: flat ~26 across all ubatches → not PCIe throttling; rows beating linear → copies overlapping compute; OOM at 4096/8192 → ROCm0 carrying everything, with `-ts 0,1,1,1` as the fix.

### 15:36 — Rows one and two land: 25.90 and 41.68; the two-term model appears

At 15:34:56 Paul pasted the log (`cat diag-v4-20260714-150115/MAIN.log | grep pp`). The dump also carried standing context: `=> tg128 = 6.01 t/s  pp512 = 40.11 t/s`; `tg 5.52 vs 5.53. pp 34.64 vs 34.65. The AVX2`; `-sm row` impossible ("device ROCm0 does not support split buffers"); `-sm tensor` not supported for glm-dsa; the annotated `return b; // <-- FIRST supporting backend. Always 0.` at ggml-backend.cpp:919-925; and the CPU_Mapped buffer sizes (46166.88 MiB, then 8× 46303.78 MiB, then 22067.47 MiB).

```
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |      512 |   1 | exps=CPU              |    0 |          pp8192 |         25.90 ± 0.36 |
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |     1024 |   1 | exps=CPU              |    0 |          pp8192 |         41.68 ± 0.12 |
```

The linear model predicted ~52 at ub 1024; the measurement was 41.68. Back-solving per-ubatch cost:

```
ub  512:  25.90 t/s  ->  8192/25.90 = 316.3 s  ÷ 16 ubatches = 19.77 s/ubatch
ub 1024:  41.68 t/s  ->  8192/41.68 = 196.5 s  ÷  8 ubatches = 24.57 s/ubatch
```

```
t_ubatch = A + B·ub
19.77 = A + B·512
24.57 = A + B·1024
B = 9.38 ms per token
A = 14.97 s per ubatch
```

- Fixed term = weight streaming: 437.8 GB ÷ 14.97 s = 29.2 GB/s. PCIe 4.0 x16 is 31.5 GB/s theoretical — 93% of line rate from a pageable mmap.
- Variable term = expert GEMM: 45.3 GFLOP/token ÷ 9.38 ms = 4.83 TFLOP/s — one V620 doing all 75 layers, ~12% of the card's fp16 peak (256 skinny GEMMs, 32 tokens per expert at ub 1024).

Revised curve with asymptote:

| `-ub` | predicted pp |
|---|---|
| 512 | 25.9 ✓ |
| 1024 | 41.7 ✓ |
| 2048 | ~60 |
| 4096 | ~77 |
| 8192 | ~89 |
| ∞ | 107 (hard ceiling) |

**Observations**

- **Correction:** "I told you ~300 at ub 8192. The real number is ~89" — roughly 2.2× on prefill, not 7×. The cost per ubatch is not constant; it rose 24% when the ubatch doubled.
- Both terms are single-card artifacts (one x16 link, one GPU), both traceable to `return b`. Fixed, both divide by four: `t_ubatch = 3.75 s + 2.34 ms·ub` → ub 4096 → 307 t/s, ub 8192 → 357 t/s, asymptote → 427 t/s. "From ~89 to ~350. A 4×."
- **Prediction (watch item):** ub 2048 near 60 confirms the linear fit and a ~107 ceiling; meaningfully above 60 means GEMM efficiency is improving and the curve keeps climbing.

### 15:38 — Row three: 62.64; marginal cost per token is falling

Paul pasted the third row at 15:37:01.

```
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |     2048 |   1 | exps=CPU              |    0 |          pp8192 |         62.64 ± 0.13 |
```

```
ub  512:  25.90  ->  19.768 s/ubatch
ub 1024:  41.68  ->  24.569 s/ubatch
ub 2048:  62.64  ->  32.694 s/ubatch
```

Linear extrapolation from the first two points said 59.9; the measurement was 62.64, 4.6% above — the GEMM grows more efficient as the batch deepens.

```
512 -> 1024:   4.801 s / 512  tokens  =  9.377 ms/tok  =  4.83 TFLOP/s
1024 -> 2048:  8.125 s / 1024 tokens  =  7.935 ms/tok  =  5.71 TFLOP/s
```

+18% effective compute rate as tokens-per-expert doubled from 16 to 32 (ub 4096 = 128 tokens/expert; ub 8192 = 256). The fixed term held: streaming ≈ 15–16 s → 437.8 GB ÷ 15.5 s = 28.2 GB/s (~90% of line rate).

| `-ub` | pp8192 |
|---|---|
| 512 | 25.90 ✓ |
| 1024 | 41.68 ✓ |
| 2048 | 62.64 ✓ |
| 4096 | ~85 |
| 8192 | ~105–110 |

Error progression on the ub 8192 prediction:

| pass | predicted @ ub 8192 | error source |
|---|---|---|
| first pass | ~460 | assumed all four cards stream in parallel — then found `return b` |
| second pass | ~300 | one card, but treated as pure PCIe, ignored the GEMM |
| linear fit (2 pts) | ~89 | too low — missed the improving GEMM efficiency |
| now (3 pts) | ~105–110 | |

```
                  today (ROCm0 only)      with return b fixed
streaming term         15.5 s                  3.9 s
compute term         7.9 ms/tok              2.0 ms/tok
pp @ ub 4096            ~85                    ~335
pp @ ub 8192           ~105                    ~400
```

**Observations**

- Projected gain over the 40 t/s baseline: ~2.6×; an 8K prompt drops from 3.4 minutes to 78 seconds — from a flag.
- Ceiling estimate: infinite ubatch on one V620 caps at ~150–250 t/s; four cards → 600–1,000.
- **Prediction (watch item):** ub 4096 near 85 means the efficiency curve is behaving; clearing 95 means the GEMM is fattening faster and the ceiling is higher.

### 15:42 — "So how can I get all four cards to do this together" — the offending code and three options

Paul asked the direct question at 15:41:51. The responsible code was produced and the options laid out.

```c
// ggml/src/ggml-backend.cpp:916-929
if (tensor->op != GGML_OP_ROPE && src->buffer != NULL &&
    src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
    int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
    // check if a backend with higher prio wants to offload the op
    if (sched->op_offload && src_backend_id == sched->n_backends - 1 &&
        ggml_backend_buffer_is_host(src->buffer)) {
        for (int b = 0; b < src_backend_id; b++) {
            if (ggml_backend_supports_op(sched->backends[b], tensor) &&
                ggml_backend_offload_op(sched->backends[b], tensor)) {
                SET_CAUSE(tensor, "1.off");
                return b;              // <-- always ROCm0
            }
        }
    }
    SET_CAUSE(tensor, "1.wgt%d", i);
    return src_backend_id;
}
```

The loop picks the first backend that can do the job; backend 0 always can for CPU weights, so all 225 expert tensors per ubatch cross ROCm0's link and all 75 layers of GEMM run on ROCm0.

- **Option 1 (no code):** `-dev` reorders the whole model but backend 0 is still one card — "cannot fix this. Don't waste a run on it." What could help is making ROCm0 a pure streaming engine:

```bash
llama-bench -m ... -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 0 -nopo 0 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2
```

ROCm0 holds zero model layers (the full 30.7 GiB becomes staging buffer), ROCm1–3 carry attention; `-mmp 0` pinned memory is worth ~+15% on a 100% PCIe path. **Prediction:** "Expect maybe 110–125. Not the fix."

- **Option 2 (the fix):** route `GGML_OP_MUL_MAT_ID` by `src[1]` (the routed activations, already on the layer's home GPU) instead of `src[0]` (the expert weight on the CPU) — a ~20-line block walking `tensor->src[j]` for a GPU-resident source, with the old behaviour as fallback.

| | today | patched |
|---|---|---|
| streaming | 15.5 s (1 link) | 3.9 s (4 links) |
| GEMM | 7.9 ms/tok (1 card) | 2.0 ms/tok (4 cards) |
| pp @ ub 4096 | ~85 | ~335 |
| pp @ ub 8192 | ~105 | ~400 |

```bash
cd /root/llama.cpp
git checkout -b offload-home-device
# edit ggml/src/ggml-backend.cpp
cmake --build build -j64 && cmake --install build
```

Verification plan: `GGML_SCHED_DEBUG=1` — splits should cycle `ROCm0 … ROCm1 … ROCm2 … ROCm3`; `rocm-smi` should show all four cards busy.

- **Option 3:** send it upstream — every large-MoE `--cpu-moe` multi-GPU user (DeepSeek, Kimi, Qwen3-235B, GLM) silently gets one card's prefill.

**Observations**

- **Decision (order of operations):** (1) `-ts 0,1,1,1` tonight, expecting ~110–125; (2) patch and re-run the ladder → ~335–400; (3) submit a PR.

### 15:44 — Rows four and five: 84.62 and 104.97; the model closes

Paul pasted the final two rows at 15:44:33.

```
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |     4096 |   1 | exps=CPU              |    0 |          pp8192 |         84.62 ± 3.58 |
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |    0 |          pp8192 |        104.97 ± 0.53 |
```

"104.97. I predicted 105–110. The ladder is complete and the model is closed."

```
ub  512:  25.90  ->  19.768 s   ( 4.83 TFLOP/s marginal)
ub 1024:  41.68  ->  24.569 s   ( 4.83)
ub 2048:  62.64  ->  32.694 s   ( 5.71)
ub 4096:  84.62  ->  48.351 s   ( 6.09)
ub 8192: 104.97  ->  78.041 s   ( 6.24)
```

The marginal compute rate climbed 4.83 → 6.24 TFLOP/s as tokens-per-expert went 16 → 256, now flattening (GEMM saturated). Final two-term fit across a 16× range of ubatch:

```
t_ubatch = 14.9 s  +  7.72 ms × ub
      437.8 GB over    one V620 doing
      ROCm0's x16      all 75 layers of
      = 29.4 GB/s      expert GEMM
      = 93% of         @ ~6.2 TFLOP/s
      line rate
```

**40.11 → 104.97 t/s — a 2.6× on prefill, for a flag.** An 8K prompt drops from 3m24s to 78 seconds. The underlying bug: `-p 512` had silently clamped `n_ubatch` for four days. Both remaining terms belong to one card, both to `return b`:

| | today | patched |
|---|---|---|
| streaming | 14.9 s (one x16 link) | 3.7 s (four links) |
| GEMM | 7.72 ms/tok (one V620) | 1.93 ms/tok (four) |
| pp @ ub 8192 | 104.97 | ~415 |

Production configuration proposed:

```bash
llama-server -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -fitt 8192 -fitc 65536 --no-mmap -fa 1 \
  -t 32 -b 8192 -ub 8192 \
  --jinja --host 0.0.0.0
```

**Observations**

- `-nopo` stays default — op_offload never fires at batch 1 (`ne[2] = 1 < 32`), so prefill gets 105 t/s and decode stays untouched at 6.01.
- Two caveats: the fitter fills VRAM to within `-fitt` of full and does not model op_offload staging, so it may starve ROCm0 of ~5.4 GiB (hence `-fitt 8192`, not 2048); `--no-mmap` gives pinned host memory and pushes 29.4 GB/s toward line rate.
- **Decision:** two things found in the codebase — the scheduler pathology (costing every multi-GPU MoE offloader four cards' worth of prefill) and the `llm_arch_supports_sm_tensor` MLA exclusion. "Write it up. This one's worth sending."

### 15:52 — Decode re-confirmed: 6.01 t/s, declared done

Paul asked "And what is my token gen again?" The answer: 6.01 t/s, from Phase J, the fitter, at `-t 64`.

| config | tg |
|---|---|
| where it started | 5.15 |
| best hybrid, `-t 24` | 5.54 |
| fitter (106 GiB VRAM filled) | 6.01 |
| no GPUs at all (`-ngl 0`) | 3.87 |

Decode ceiling equation, fitting every decode measurement to within 3%: `t_token = 90 ms (GPU dense path + 155 splits) + CPU_expert_bytes ÷ 152 GB/s`. With the fitter holding ~16 expert layers, the CPU reads 10.8 GB/token from DDR4 → 71 ms → 161 ms total → 6.2 t/s.

**Observations**

- **Prediction:** ~2% more available — Phase J ran `-t 64` but decode peaks at 24–32 threads (5.53–5.54 vs 5.40 at 64), so fitter plus the right thread count ≈ 6.1–6.2.
- **Dead ends (decode levers exhausted):** `--poll` (nothing), `--cpu-strict` (harmful), `CPU_REPACK` (engaged perfectly, zero gain), THP (never applies), `-sm row` (split buffers deleted from llama.cpp), `-sm tensor` (MLA models excluded), `GGML_CUDA_GRAPH_OPT` (nothing). Even every expert in VRAM (impossible — 411 GiB vs 122) yields only 11 t/s.
- The asymmetry:

| | before | now | gain |
|---|---|---|---|
| prefill | 37.6 | 104.97 | 2.8× |
| decode | 5.15 | 6.01 | 1.17× |

- The only decode lever left is fewer bytes: Q3_K_XL → ~7.7 GB/token from the CPU → ~7 t/s (a quality call). `return b` still holds a projected 4× on prefill (105 → ~415).

### 15:55 — DFlash speculative decoding analyzed; Q3 rejected

Paul: "What will dflash get me for speculative execctuion on it? And no, Q3 is not worth it." The verify-pass cost was calibrated from the fitter run (6.01 t/s = 166.4 ms/token, 90 ms GPU constant; the 76.4 ms figure is measured):

```
t_verify(B) = 90 ms  +  76.4 ms × E(B)/8
E(B) = 256 × (1 − 0.96875^B)    <- distinct experts touched by B tokens
```

With 8-of-256 routing, adjacent tokens barely share experts, so a verify batch of B reads roughly B× the expert bytes. DFlash model (p ≈ 0.9 back-solved from the claimed 8× on Qwen3; one block-diffusion draft pass ≈ 20 ms; verify batch = N+1; `α = (1 − p^(N+1))/(1 − p)`):

| block | verify | α | t/s |
|---|---|---|---|
| — | 166 ms | 1.00 | 6.01 |
| 2 | 218 | 1.90 | 8.0 |
| 3 | 269 | 2.71 | 9.4 |
| 4 | 317 | 3.44 | 10.2 |
| 5 | 364 | 4.10 | 10.7 |
| 6 | 409 | 4.69 | 10.9 |
| 8 | 497 | 5.70 | 11.0 (peak) |
| 12 | 655 | 7.18 | 10.6 |
| 15 | 796 | 8.15 | 10.0 |

**Observations**

- **Prediction:** ~11 t/s from 6.01, an 83% gain (assumes ~30% expert-routing overlap between adjacent tokens; strictly independent routing → ~8.8 t/s; unmeasured for GLM-5.2).
- Optimal block is 6–8, not DFlash's trained 15 (10.0 vs 11.0); set `--spec-draft-n-max 6`.
- Cliff at batch 32: `GGML_OP_OFFLOAD_MIN_BATCH` defaults to 32; a verify batch of 32+ triggers op_offload and streams the entire 344 GB expert set over ROCm0's single link:

```
B = 31:  CPU path,  ~1.9 s
B = 32:  op_offload fires,  14.9 s + 7.72 ms×32  =  15.2 s      <- 8× worse
```

Mitigation: set `GGML_OP_OFFLOAD_MIN_BATCH=1024` so op_offload only fires for real prefill (a block of 15 gives B=16, safely under).

- **Dead end (blocker):** there is no `GLM-5.2-DFlash` draft model. DFlash needs a block-diffusion network trained against the specific target, converted with `--target-model-dir`; z-lab shipped drafts for Qwen, nobody has trained one for a 744B target, and Paul cannot make one. "Watch the model hub."
- Combined outlook:

| | today | + `return b` patch | + DFlash |
|---|---|---|---|
| prefill | 104.97 | ~415 | ~415 |
| decode | 6.01 | 6.01 | ~11 |

An 8K prompt with a 3K response: 9.6 minutes today → 5.3 minutes with both, against ~1 minute for hosted Opus.

### 18:10 — The first patch draft, against master

After a pause, Paul at 18:09:14: "Alright, generate the patch for me here and I'll paste it over to the server. Tell me what edits to make to what files. I'll do this off of the master branch on github. And then tell me what string to run to test it." The exact current master lines were pulled and the patch issued: file `ggml/src/ggml-backend.cpp`, function `ggml_backend_sched_backend_id_from_cur`, the `if (sched->op_offload && ...)` block around line 919 (the only match for `1.off` in the file).

Find:

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
                // Prefer the device that already holds this op's activations, so
                // that offloaded weight streaming and compute spread across all
                // devices instead of concentrating every H2D transfer and every
                // matmul onto backend 0. Falls back to the original first-fit.
                int off_backend_id = -1;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    const struct ggml_tensor * asrc = tensor->src[j];
                    if (asrc == NULL || asrc == src) {
                        continue;
                    }
                    if (asrc->buffer == NULL || ggml_backend_buffer_is_host(asrc->buffer)) {
                        continue;
                    }
                    int asrc_backend_id = ggml_backend_sched_backend_from_buffer(sched, asrc, tensor);
                    if (asrc_backend_id >= 0 && asrc_backend_id < src_backend_id) {
                        off_backend_id = asrc_backend_id;
                        break;
                    }
                }
                if (off_backend_id >= 0 &&
                    ggml_backend_supports_op(sched->backends[off_backend_id], tensor) &&
                    ggml_backend_offload_op(sched->backends[off_backend_id], tensor)) {
                    SET_CAUSE(tensor, "1.off.act");
                    return off_backend_id;
                }
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
```

One block, one file; all callees already in scope, no new includes. For `GGML_OP_MUL_MAT_ID`: `src[0]` = expert weight (CPU), `src[1]` = routed activations (layer's home GPU); the old first-fit stays as fallback.

Build:

```bash
cd /root/llama.cpp
git fetch origin && git checkout master && git pull
git checkout -b offload-to-activation-device
# make the edit above
cmake --build build -j64 && cmake --install build
```

If the build directory is stale (reconfigure, keep ZenDNN off):

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_ZENDNN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j64 && cmake --install build
```

Verify routing to four cards:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 2>&1 | grep -E 'ffn_moe.*ROCm|1\.off' | head -40
```

The ladder, directly comparable:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 2048,4096,8192 -p 8192 -n 0 -r 2 \
  -o md --progress
```

Watch the cards live:

```bash
watch -n1 'rocm-smi --showuse --showmemuse | grep -E "GPU\[|use"'
```

| `-ub` | measured (today) | patched target |
|---|---|---|
| 2048 | 62.64 | ~150–190 |
| 4096 | 84.62 | ~270–310 |
| 8192 | 104.97 | ~360–415 |

**Observations**

- **Prediction:** both terms of `t_ubatch = 14.9 s + 7.72 ms × ub` divide by ~four if the patch works; not a clean 4× (PCIe P2P activation hops plus scheduler bookkeeping) — 3–3.5× realistic.
- If `-mmp 0` OOMs on load (pinning ~411 GiB), drop to `-mmp 1` for the functional test.

### 18:11 — "Before I do this, are you absolutely certain" — the walk-back

Paul asked for certainty before touching the server. It could not be given: the diagnosis was confident, but the patch was not certain to be correct — right to stop.

- **Proven (measured, five points, tight fit):** prefill cost `14.9 s + 7.72 ms × ub`; streaming at 93% of one x16 line rate; something serializes onto one card.
- **Read from source, high confidence:** the `return b` loop picks backend 0; `MUL_MAT_ID` has `src[0]` = weight, `src[1]` = activations.
- **Inference, not verified:** that walking `src[]` for a GPU-resident activation and returning that backend spreads the load without breaking routing or correctness.

Three genuine unknowns:

1. **Hypothesis (unverified, load-bearing):** where `src[1]` actually lives at scheduling time. The function runs during graph split assignment, before placement is finalized; `src[1]` may not have a `buffer` yet, in which case the check dereferences null or the branch skips and the patch is an expensive no-op. "I did not trace the assignment order. This is the load-bearing assumption and I never verified it."
2. Forcing the op onto a different device than its weight may create a copy the scheduler places badly (GEMM to ROCm2, stream the weight to ROCm2 and bounce the activation ROCm1→ROCm2 — net worse).
3. Whether upstream already fixed or reworked this.

Three routes to certainty, cheapest first:

```bash
git -C /root/llama.cpp log --oneline -20 -- ggml/src/ggml-backend.cpp
```

plus an issue-tracker search for "cpu-moe multi-gpu prefill" or "op_offload tensor split"; then a zero-code mechanism test — does forcing expert layers onto specific cards via `-ot` already spread the offload:

```
-ot "blk\.(3|4|5|6)\.ffn_.*_exps\.=ROCm1,blk\.(7|8|9|10)\.ffn_.*_exps\.=ROCm2,exps=CPU"
```

and finally instrumentation: the `GGML_SCHED_DEBUG=2` check is not optional — all-`ROCm0` output means assumption (1) is wrong.

**Observations**

- **Decision:** run the `-ts 0,1,1,1` and `-ot` pinning experiments first (one load each, no compilation) — confirm or kill the mechanism, then patch. "I gave you that patch with more confidence than I'd earned. The diagnosis is solid. The fix is a hypothesis."

### 18:12 — Correction: what the -ot test actually measures; the full command

Paul quoted the mechanism test back and asked for the entire command. The framing needed correcting first: pinning expert layers to ROCm1/ROCm2 does not test op_offload — it tests the opposite. Pinned experts become resident on the GPU — no CPU→GPU streaming for them — which is the `-fitt` fitter path already measured at pp 40, not the op_offload path at pp 105. What the test does answer: whether these cards can run expert GEMM on more than one device at once, and whether `-ot` places tensors where told — a precondition for the patch being meaningful, not proof it routes correctly.

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.(3|4|5|6)\.ffn_.*_exps\.=ROCm1,blk\.(7|8|9|10)\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 \
  -p 8192 -n 0 -r 1 \
  -o md --progress -v 2>&1 | tee /root/ot-placement-test.txt
```

Post-run checks:

```bash
grep -E 'load_tensors:.*ROCm[0-9].*buffer size' /root/ot-placement-test.txt
```

(ROCm1 and ROCm2 should each show ~44 GiB more than ROCm0/ROCm3 if the rules landed), and the number against the baseline `ub 4096 → 84.62`. Watch live:

```bash
watch -n1 'rocm-smi --showmemuse | grep -E "GPU\[|Memory"'
```

**Observations**

- **Correction:** the cleaner mechanism test remains `-ts 0,1,1,1` (ROCm0 as pure scratch, directly attacking the concentration); if only one is run, run that. Read the `-ot` test as "resident experts on multiple cards," not "spread offload."

### 18:13 — File upload fails again; session ends mid-handoff

Paul attempted a file upload ("> File:") that arrived empty. Nothing had landed in the uploads directory — the only file present was `results.txt`, the 10 MB v4 log from earlier, timestamped 18:34 — the same failure mode as before. Retrieval commands were left standing:

```bash
# 1. did placement take? ROCm1/ROCm2 should be ~44 GiB larger than ROCm0/ROCm3
grep -E 'load_tensors:.*ROCm[0-9].*buffer size' /root/ot-placement-test.txt

# 2. the number
grep -E '^\| glm-dsa' /root/ot-placement-test.txt
```

and, if it crashed:

```bash
tail -20 /root/ot-placement-test.txt
```

**Observations**

- **Dead end (transport):** second consecutive empty document upload.
- Four pre-registered outcomes against the `ub 4096 → 84.62` baseline: above 84.62 = placement took and the patch premise is sound; OOM on load = ROCm1/2 too full (30 GiB cards); unchanged with placement confirmed = resident experts do not move prefill, the bottleneck is elsewhere; ROCm1/2 not larger = `-ot` not honoring the rules.

### State of knowledge at end of Session 6

- Build cost fixed at $9,050; RAM $5,600 (62%, $5.47/GB against a 2026 DDR4 spot of $5.15/GB and DDR5 at $30.94/GB). Genoa/DDR5 counterfactual ≈ $16,050–$21,000 for +35% decode. Verdict: DDR4 was right; buy nothing.
- The ubatch ladder is complete: pp8192 = 25.90 / 41.68 / 62.64 / 84.62 / 104.97 t/s at ub 512/1024/2048/4096/8192. Prefill improved 40.11 → 104.97 (2.6×) purely by unclamping `n_ubatch`; the four-day-old cause was `-p 512` silently clamping it.
- Two-term prefill model closed: `t_ubatch = 14.9 s + 7.72 ms × ub` — 437.8 GB streamed at 29.4 GB/s (93% of one PCIe 4.0 x16 line) plus one V620 doing all expert GEMM at ~6.2 TFLOP/s. Both terms are single-card artifacts of `return b` at ggml-backend.cpp:919-925.
- Decode confirmed at 6.01 t/s (fitter, 106 GiB VRAM filled); declared done against the 152 GB/s DDR4 wall; every software lever exhausted; only Q3_K_XL (~7 t/s, rejected) or new memory moves it.
- DFlash projects ~11 t/s (83%) with optimal block 6–8 and a hard cliff at verify batch B=32 (`GGML_OP_OFFLOAD_MIN_BATCH`); blocked — no GLM-5.2 draft model exists.
- A first patch (route offloaded ops to the activation's device, tag `1.off.act`) is drafted against master, projected 360–415 t/s at ub 8192, but its load-bearing assumption — that `src[1]` has a resolved buffer at scheduling time — is explicitly unverified.
- Certainty was walked back on request; the agreed order is zero-code mechanism tests (`-ot` pinning, `-ts 0,1,1,1`) before any compilation.
- The `-ot` placement test has been issued (tee to `/root/ot-placement-test.txt`); its output has not yet been seen — the file upload failed twice.

---

