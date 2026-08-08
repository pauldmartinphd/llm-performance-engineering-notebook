## Session 5 — Monday, July 14, 2026 (afternoon) — Interpreting v3; the untested ubatch regime

The v3 battery was still running (11:08–14:25) through the first half of this session; the dialogue proceeded in parallel, returning to the battery's output when the complete results arrived at 14:34.

### 12:13 — Paul asks about DFlash speculation

"What about for dflash speculation on this model? I know it's not merged yet but assume it is fully implemented for this exercise."

### 12:15 — DFlash: merged upstream, but the verify-side tax rules; ~12.5 t/s best case

dflash postdated the model's training, so this was researched rather than guessed. **Correction** (of the question's premise): `draft-dflash` is already on master — `docs/speculative.md` documents it in the `--spec-type` list; "The blocker isn't merge status." DFlash produces an entire block of draft tokens in a single forward pass (block diffusion) and injects the target model's hidden states into the draft model's attention; the draft is a small block-diffusion model trained for a specific target (e.g. `z-lab/Qwen3-4B-DFlash` for `Qwen/Qwen3-4B`), ~1–2 GB. Two wins over EAGLE3/MTP: a block of 15 costs one pass instead of fifteen, and hidden-state injection pushes acceptance to p ≈ 0.9 (back-solved from the claimed 8× on Qwen3).

Key analysis: "Both of those are draft-side wins. Your bottleneck is the verify side. DFlash does not touch it." The batching premise is only half true on this machine — the GPU half (90 ms: 18.7 GB dense weights + 155 syncs) is batch-independent and amortizes; the CPU half (90 ms: 13.77 GB of routed experts from DDR4) scales near-linearly with the verify batch:

```
E(B) = 256 × (1 − 0.96875^B)
B:  1     3     5     8     12     16
E:  8    23.3  37.5  57.4  81.2  102.1     distinct experts touched
```

At block 16: 102 experts read instead of 8 — 12.8× the bytes for at most 16 tokens. Model parameters: G = 90 ms, DDR4 at 152 GB/s, one draft pass ≈ 20 ms, p = 0.9, `α = (1 − p^(N+1))/(1 − p)`, verify batch B = N+1, ~30% routing overlap assumed:

| block B | E(B) | CPU read | verify | cycle | α | **t/s** |
|---|---|---|---|---|---|---|
| 1 (none) | 8 | 91 ms | 181 | 181 | 1.00 | 5.5 |
| 3 | 18.7 | 212 | 302 | 322 | 2.71 | 8.4 |
| 5 | 28.7 | 325 | 415 | 435 | 4.10 | 9.4 |
| **8** | 42.6 | 482 | 572 | 592 | 5.70 | **9.6** |
| 12 | 59.2 | 670 | 760 | 780 | 7.18 | 9.2 |
| 16 | 73.9 | 837 | 927 | 947 | 8.15 | 8.6 |

Stacked projection (VRAM fill = CPU bytes × 0.77, and `-sm row` taking G from 90 to ~55):

| | t/s |
|---|---|
| today | 5.5 |
| + `-t 24`, VRAM fill, `-sm row` | ~7.5 |
| + **DFlash, block 8** | **~12.5** |

What would truly unlock it — experts in VRAM, so that verify(B) ≈ verify(1):

| | expert size | fits in 120 GiB? |
|---|---|---|
| UD-Q4_K_XL (current) | 411 GiB | no |
| UD-Q2_K_XL | ~226 GiB | no |
| REAP-504B (128 of 256 experts) @ Q4 | ~205 GiB | no |
| REAP-504B @ Q2 | ~113 GiB | **barely** |

**Observations**

- **Decision/recommendation:** the optimum block is ~5–8; DFlash's trained block size of 15 makes it worse here; `--spec-draft-n-max 6` or similar would be wanted.
- **Prediction:** ~12.5 t/s with DFlash block 8 stacked on the other optimizations — "roughly 2.3× over today, and about 30% better than MTP" (the earlier MTP estimate was ~9.7). Drafting is only 20 ms of a ~600 ms cycle here.
- **Dead end (the actual blocker):** no `GLM-5.2-DFlash` draft model exists; DFlash requires one trained specifically against the target (converted with `--target-model-dir` to inherit tokenizer/embeddings), and z-lab has only shipped Qwen drafts. "For a 744B model that is not a weekend project."
- Secondary cost: hidden-state injection means capturing auxiliary layers from the target; with layers split across 4 GPUs plus CPU, that is another cross-device hop per draft step (small, 24 KB) on a critical path that already has 155 syncs.
- **Hypothesis** (flagged least-confident): the 30% routing-overlap assumption. If routing is near-independent, all numbers drop ~20% and the optimal block shrinks to 4. Measurable by instrumenting `ggml_compute_forward_mul_mat_id` to count distinct expert IDs per batch.
- Working-figure detail: at p=0.9 the verify-batch sweep gave batch 5 = 7.66 t/s, batch 3 = 7.27, batch 4 = 7.89; with 30% overlap batch 4 ≈ 8.5, batch 5 ≈ 8.3, batch 9 ≈ 8.9; with the row-cut G ≈ 55, peak ≈ 12.6 t/s at batch 8. A llama.cpp MoE/hybrid replay caveat applies to recurrent/SSM hybrids, not MLA models like GLM-5.2.
- Sources cited: github.com/amanning3390/dflash-llama; llama.cpp PR #22105 (add DFlash support, ruixiang63); llama.cpp Discussion #21569; inventivehq.com; DataCamp; xhinker.medium.com; allenkuo.medium.com; llama.cpp docs/speculative.md.

### 12:19 — Paul compares 12.5 t/s to hosted models

"12.5 tps is about 25% the speed of Claude/ChatGPT? That's *really* good."

### 12:20 — Reality check: 21% of Opus 4.8; the honest near-term ceiling is ~13%; prefill is what is felt

Artificial Analysis figures were checked rather than eyeballed:

| model | output t/s |
|---|---|
| Claude Opus 4.8 (max) | 58.5 |
| Claude Opus 4.7 (max) | 49.6 |
| GPT-5.6 Sol (high) | 69.2 |
| median, reasoning tier | ~78 |

12.5 ÷ 58.5 = 21%; against Opus 4.7, exactly 25% — "You nailed it." But with caveats. **Correction:** "12.5 is not on the menu" — it requires the nonexistent GLM-5.2-DFlash draft. What is actually on the table:

| | decode | % of Opus 4.8 |
|---|---|---|
| today | 5.5 | 9% |
| `-t 24` + VRAM fill | ~6.5 | 11% |
| + `-sm row`, if it works | ~7.5 | 13% |
| + DFlash (hypothetical) | 12.5 | 21% |

"~13% is the honest near-term ceiling." And decode t/s is not what a user feels — prefill is: at the current 34.65 t/s, an 8K-token prompt means four minutes before the first token (Opus: a couple of seconds); even if op_offload lands at ~300–500 t/s prefill, an 8K prompt is 16–27 s of dead air. Reasoning-model verbosity multiplies the tax (Opus 4.8 flagged "very verbose" — 120M output tokens on the Intelligence Index versus a 60M average). End-to-end, 8K prompt → 3K output including thinking:

| | TTFT | generation | **total** |
|---|---|---|---|
| Opus 4.8 hosted | ~2 s | 52 s | **~1 min** |
| Galactus, near-term (7.5 t/s, 300 t/s prefill) | 27 s | 400 s | **~7 min** |
| Galactus, with DFlash | 16 s | 240 s | **~4.5 min** |

"So 5–8× slower end to end, not 4×."

**Observations**

- What remains remarkable: GLM-5.2 sits in Artificial Analysis's frontier tables alongside Opus 4.8 and GPT-5.6, running "in your basement on used server parts, at roughly human reading speed (~250 wpm ≈ 5–6 tok/s)."
- **Decision (emphasis):** "Phase F is the phase that matters. Decode is a 1.3× fight against physics. Prefill is a 10× fight against a flag that's currently crashing." (Phase F was running at this moment.)
- Sources: artificialanalysis.ai (GPT-5.6 landing page; GPT-5.6 Sol high/medium/max; GPT-5.6 Terra max; Claude Opus 4.8; Claude Opus 4.7; Opus 4.6 versus GPT-5 comparison), callsphere.ai, tokencalculator.com.

### 12:45 — Paul asks why op_offload is not landing

"Why isn't op_offload landing?"

### 12:49 — Scheduler source read: every offloaded op goes to ROCm0; the copies are blocking

Rather than speculate, the scheduler was read. Two structural findings, both previously held wrong; neither proven to be the crash, but together they explain it, and they change the payoff.

**Finding 1 — every offloaded matmul goes to ROCm0.** `ggml-backend.cpp:919-925`, verbatim:

```c
if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {
        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
            SET_CAUSE(tensor, "1.off");
            return b;                              // <-- FIRST supporting backend
        }
    }
}
```

Backends are ordered `[ROCm0, ROCm1, ROCm2, ROCm3, CPU]`; `b = 0` always wins. All 225 expert tensors (75 MoE layers × 3 tensors) for all layers are copied to ROCm0, which also does all 75 layers of expert GEMMs while ROCm1–3 sit idle. **Correction** (of an earlier claim): the earlier "410 GiB spread over four cards in parallel" was wrong. It's 408 GiB onto one card, over one x16 link.

**Finding 2 — the copies are blocking.** `ggml_backend_cuda_cpy_tensor_async()` bails when the source is not HIP, verbatim:

```c
if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
    return false;
}
```

So `ggml_backend_tensor_copy_async()` falls through to, verbatim:

```c
// an async copy would normally happen after all the queued operations on both backends are completed
// to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
ggml_backend_synchronize(backend_src);
ggml_backend_synchronize(backend_dst);
ggml_backend_tensor_copy(src, dst);          // BLOCKING
```

"225 blocking H2D transfers and 450 full backend syncs per ubatch, with zero overlap against compute." PCIe time and GPU time are strictly additive.

Revised prefill ceiling (408 GiB of expert weights per graph eval, one x16 Gen4 link at ~25 GB/s, plus ~6 s of expert GEMM on a single V620, no overlap):

| `-ub` | prefill |
|---|---|
| 512 | **29 t/s** — worse than today's 34.65 |
| 1024 | 58 |
| 2048 | 117 |
| 4096 | **~170–230** |

**Observations**

- **Correction:** "Not the 750–1000 I quoted. Still a 5–7× — but it only pays above `-ub 1024`, and it needs 4096 to be worth the trouble."
- The vice: "Small ubatch is safe and useless. Large ubatch is valuable and fatal." At ub 4096, ROCm0 holds its 4987 MiB of model weights plus the streamed expert copies plus every offloaded layer's MoE activations, while ROCm1–3 idle at ~5 GiB each of a 30 GiB card. "op_offload was designed for 'one GPU, a few layers don't fit.' You are asking it to be a four-GPU streaming engine for a model that is 97% CPU-resident."
- **Decision:** a fix worth trying, verbatim:

```bash
-ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -nopo 0 -b 4096 -ub 4096
```

  `-ts 0,1,1,1` gives ROCm0 zero model layers and its full 30.7 GiB as offload scratch; ROCm1–3 carry attention for all 78 layers. Cost: three hidden-state hops per layer (100 MB each at ub 4096, ~15 GB/ubatch of inter-GPU traffic) versus 438 GB of weight streaming — "cheap."
- Still unknown: the actual HIP error string from v2's op_offload crash; `ggml-cuda.cu:104` is the generic `ggml_cuda_error()` abort and llama-bench swallowed the message — could be OOM, could be an illegal address from the MMQ kernel with 256 experts on gfx1030. Phase F was running at that moment with `-v` and a small-to-large ubatch ladder.
- Working details: op_offload fires only when `ne[2]` (n_tokens) ≥ 32, so it never fires during token generation (n_tokens = 1); each expert tensor is 1728–2112 MiB, one layer's three ≈ 5.44 GiB; 225 copies ≈ ~408 GiB per ubatch, ~16.7–17.5 s of transfer at ~25 GB/s; the ub-512 estimate is 29–31 t/s; measured compute buffers with `-nopo 1` were small (e.g. 205 MiB on ROCm0 at ub 2048); pipeline parallelism is disabled by `-ot exps=CPU`, so n_copies = 1; `GGML_SCHED_MAX_SPLIT_INPUTS` = 10.

### 12:51 — Paul asks whether to keep the battery running

"So just continue to let it run?"

### 12:52 — Decision: let it run; phase triage; the -ts 0,1,1,1 test queued for afterward

**Decision:** "Yes. Let it run. Nothing left in it is wasted." Reloads now cost ~1 minute (warm ZFS ARC plus `-mmp 1`). The phase-value table:

| phase | value | why |
|---|---|---|
| D (`-ngl 0`) | high | The denominator. If it beats 5.54, the four V620s are net-negative in their current role. Also confirms ZenDNN is gone (should print `graph splits = 1`, not 1088). |
| F (op_offload ladder) | highest | The HIP error string. Four rounds of guessing end here. |
| G (unified memory) | high, if F crashes | Distinguishes OOM from not-OOM. |
| J (fitter / VRAM) | high, upgraded | Only mechanism that puts expert weights on ROCm1–3. |
| H (`-sm row`) | high | Biggest lever on the 90 ms GPU constant. |
| L (`--no-host 1`) | high | Does `CPU_REPACK` appear? Does `AnonHugePages` move off zero? |
| C, E, I | moderate/low | C matters more now that t=24 is the optimum. E just re-confirms 155. |
| K (depth) | low, and slow | At 34.65 t/s prefill, `-d 65536` is 31 min per rep, ×2 reps — an hour for one predictable row. Kill the script when it reaches K if watching. |
| M, N | moderate | `-mmp 0`, so 15-min loads each. M is op_offload with the pinned buffer. |

To queue up after the battery (not in the script), verbatim:

```bash
llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 24 -b 4096 -ub 1024,2048,4096 \
  -p 512 -n 0 -r 2 -v --progress
```

**Observations**

- Phase J upgraded because op_offload dumps everything on card 0: the fitter distributes expert layers across all four cards, and every layer the fitter parks in VRAM is a layer op_offload does not stream down card 0's link.
- `-ts 0,1,1,1` rationale: ROCm0 zero layers, 30.7 GiB free as offload scratch; "If F crashes at ub 2048/4096 and this survives, that's the answer." "Check back after F."
- (Retrospective note: the queued command as written pairs `-p 512` with `-ub 1024,2048,4096` — the very pairing proven at 14:37/14:49 to clamp n_ubatch to 512.)
- Working details: Phase B was nearly done at this moment (one data point at t=128 remaining); the fitter and `-ot` are mutually exclusive, and `-ts` is disabled by the fitter; fitter plus op_offload compose — e.g. 17 expert layers in VRAM cut streaming from 408 GiB to 315 GiB.

### 14:34 — Complete v3 results delivered

Paul attached the finished results.txt: "Here are the complete results." (The battery had ended at ~14:25; the results.txt artifact was saved at 14:33:44 ET.)

### 14:37 — Full v3 interpretation: the fitter wins; the model validates; four hypotheses dead; op_offload never actually tested

A complete run, and it changes several conclusions — including two earlier ones.

**Winner — Phase J (fitter): tg128 = 6.01, pp512 = 40.11**, the best numbers in the entire battery. VRAM fill, verbatim from the log:

```
ROCm0 27636.72 MiB    ROCm2 26645.77 MiB
ROCm1 26636.33 MiB    ROCm3 27696.32 MiB     => 106 GiB, spread evenly
tensor blk.11.ffn_gate_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm2
```

+8.5% decode and +16% prefill over the best hybrid. "Use it." (Working notes: the fitter run used `-t 64`; `-t 32` measured ~2.5% better elsewhere, so combining might give ~6.15 t/s. The fitter also tried to spill some expert tensors to the ROCm_Host pinned buffer, which reported zero size — that fallback did not work as intended. ~16 MoE layers placed on GPU ≈ 88 GiB of expert weights.)

**The performance model is now fully validated:** `t_token = 90 ms (GPU + 155 splits) + CPU_expert_bytes ÷ 152 GB/s`

| config | CPU bytes | predicted | **measured** |
|---|---|---|---|
| hybrid, `-ot exps=CPU` | 13.77 GB | 181 ms → 5.5 | **5.53** |
| fitter (~16 layers on GPU) | 10.8 GB | 161 ms → 6.2 | **6.01** |
| `-ngl 0` | 27 GB + CPU attention | 258 ms → 3.9 | **3.87** |

**Four hypotheses, all dead:**

- **Dead end:** `CPU_REPACK` engaged — and gave exactly nothing. Verbatim:

```
load_tensors: CPU_REPACK model buffer size = 255744.00 MiB     (60.7% — I predicted 62%)
load_tensors:        CPU model buffer size = 165220.22 MiB     (the Q5_K down tensors)
```

  Repack fired on the Q4_K gate/up tensors; result tg 5.52 versus 5.53, pp 34.64 versus 34.65. "The AVX2 `q4_K_8x8_q8_K` kernel is not faster than the standard one on Zen3. `--no-host` is dead. Delete it."
- **Dead end:** `-sm row` is impossible. Verbatim: `llama_model_load: error loading model: device ROCm0 does not support split buffers` — a hard fail at load. "The 90 ms GPU constant cannot be attacked this way. That was my biggest remaining lever and it doesn't exist."
- **Dead end:** `--poll` — zero effect at every thread count, every config. `--cpu-strict`/`-C` — no gain and harmful: t=16 strict gives 3.09 versus 5.27 unpinned. "The OS already places threads well. Drop both." `GGML_CUDA_GRAPH_OPT`: 5.40 versus 5.40.
- **Dead end:** THP never applied — `AnonHugePages: 0 kB` even in Phase L with anonymous memory. "Untested and probably untestable."

**op_offload never actually ran — a script bug**, self-attributed. Verbatim from the logs:

```
llama_context: n_batch  = 512
llama_context: n_ubatch = 128 / 256 / 512
```

"`-p 512` clamps `n_ctx` to 512, which clamps `n_batch`, which clamps `n_ubatch`." Every row at `-ub 1024/2048/4096` silently ran at 512 — "which is exactly why they're identical (25.94 / 25.95 / 25.87)." "I flagged this failure mode in my second reply — '`-p 512` with `-ub 4096`: the ubatch is effectively 512' — and then wrote it straight into the script. That's on me." op_offload was therefore tested only in the regime predicted to lose:

| `-ub` | measured | prediction |
|---|---|---|
| 512, pageable (`-mmp 1`) | 25.77 | ~29 |
| 512, pinned (`-mmp 0`) | 29.72 | ~29 |
| CPU baseline | 34.65 | — |

"The model was right. Pinning is worth +15%. And the regime where op_offload wins was never touched." The one experiment left, verbatim:

```bash
llama-bench -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -fitt 2048 -fitc 8192 -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 4096 -ub 4096 -p 4096 -n 0 -r 2 -v --progress
```

`-p 4096` so the ubatch is not clamped; `-mmp 0` for the pin (+15%); the fitter so only ~59 layers stream instead of 75.

Where things stand:

| | before | now |
|---|---|---|
| decode | 5.15 | **6.01** |
| prefill | 37.6 | **40.11** |

Production config, verbatim:

```bash
llama-server -m ... -fitt 2048 -c 65536 -fa 1 -t 32 --no-mmap
```

"No `--poll`, no `-C`, no `--cpu-strict`, no `--no-host`, no `-ncmoe`, no `-ot`, no `-sm row`."

**Observations**

- **Correction** (of an earlier hypothesis): `-ngl 0` at 3.87 settles it — the GPUs are worth +43%, not net-negative. That earlier hypothesis was wrong. The CPU is bad at MLA — it needs ~167 ms for the dense path that the GPUs do in 90. (Implied CPU-only bandwidth ~105 GB/s at `-ngl 0`, not 152.)
- **Prediction:** the queued experiment should stream ~344 GB per ubatch onto ROCm0's single x16 link ≈ 14 s + ~5 s GEMM → ~200 t/s versus 40 today. "A 5×. That is the last real prize."
- **Decision:** "decode is done." The ceiling is `90 + bytes/152`; even with every expert in VRAM (impossible) that is 11 t/s. The only remaining decode lever is fewer bytes — Q3_K_XL → CPU reads ~7.7 GB → ~7 t/s (working figure 141 ms → ~7.1 t/s). "Prefill is where the 5× is."
- Additional numbers recovered in this analysis: Phase F did not crash (the v2 crash "was ZenDNN," not op_offload); Phase M pinned op_offload ~29.7 t/s, still below the 34.65 baseline; Phase L (`--no-host 1`) nearly identical to M1, tg ~5.4–5.5; the `GGML_CUDA_GRAPH_OPT` control 5.40 t/s, no gain; Phase B2 prefill scaling 12.81 @ t=16 → peak 32.92 @ t=96 → 32.48 @ t=128, with B2's t=64 30.73 versus A1's 34.65 at the same `--poll 100` attributed to variance; Phase D tg peaks 3.87 @ t=32 and collapses to 1.44 @ t=64 (1.4–1.5 across t=48–64), with the hybrid showing the analogous pathology at t=128 (1.29–1.31); Phase C strict-t16 cluster 3.09–3.14 (~41% penalty versus non-strict 5.27, regardless of mask), the wrapped-mask case effectively zero (**dead end**), t=32 strict 8-CCD 5.53 versus 4-CCD 5.07 (9%) and no gain over OS scheduling; Phase H exited with the load error; Phase K exit code 134 = SIGABRT in state save/restore (`state_seq_set_data`) — a llama.cpp bug with CUDA graphs/ROCm, not actionable (**dead end** for `-d`); CPU_REPACK allocation ~250 GiB versus the standard CPU buffer ~161 GiB.

### 14:44 — Paul requests a new script

"Alrighty, right me a new script." [sic — "write me"]

### 14:49 — v4 written: galactus-prefill.sh — op_offload at real ubatch, dead hypotheses removed

The `-p` clamp proven from source, verbatim:

```c
llama-bench.cpp:1247   cparams.n_ctx = n_prompt + n_gen + n_depth;
llama-context.cpp:239  cparams.n_batch  = std::min(cparams.n_ctx, params.n_batch);
llama-context.cpp:241  cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch);
```

"`-p 512` → `n_ctx = 512` → `n_batch = 512` → `n_ubatch = 512`. Every op_offload row above 512 was a lie." The v4 script was created (v3 cleared first, syntax-checked, delivered as `/mnt/user-data/outputs/galactus-prefill.sh` — half the size of v3, because most of it is now settled). Launch, verbatim:

```bash
nohup ./galactus-prefill.sh > /root/console-v4.txt 2>&1 &
```

**Observations**

- **Correction (the fix):** every `-p` in v4 is ≥ the largest `-ub` in its phase. Phase A1 uses `-p 8192` with a ladder up to `-ub 8192`; A2 uses `-p 16384`. A1 design: `-nopo` and `-ub` are both context params, so all ten rows (control plus treatment) come from one model load.
- **Prediction:** the `-nopo 1` control rows stay flat at ~35 t/s; the `-nopo 0` rows should climb ~29 → 58 → 117 → 230 → 460. "If they do, that's the 5–10×. If they stay pinned at 26, the model is wrong and we instrument `ggml_backend_tensor_copy()` next."
- Deleted from v4, all listed in its briefing under "SETTLED AND DEAD — DO NOT RETEST" with the measurement that killed each: `--poll`, `-C`/`--cpu-strict`, `--no-host`/`CPU_REPACK`, `-sm row`, `GGML_CUDA_GRAPH_OPT`, THP, `-ncmoe`, `-d`, ZenDNN, STREAM, and the DIMM/PCIe/NUMA/cgroup inventory. The 232k lines of `CUDA Graph id … reused` spam are now filtered at source.
- New phases: **B** — `-ts 0,1,1,1` (ROCm0 zero model layers → full 30.7 GiB offload scratch; ROCm1–3 carry attention; "If A1 OOMs at large ubatch, this is the fix"); **C** — pinned buffer (at clamped ub 512 the pin was +15%, 29.72 versus 25.77; since op_offload cost is ~100% PCIe and pageable H2D bounces at roughly half rate — **Prediction:** could be +40–80% on op_offload rows; worth the 15-minute load); **E2** — confirms op_offload never fires at batch 1 (`ne[2] = 1 < 32`), so it is free for decode; if E and E2 match, production leaves `-nopo` at default; **I** — replaces `-d` with `-pg`, sidestepping the `state_seq_set_data` SIGABRT.
- What to watch during A1: `monitors/rocm-smi.txt` — "If op_offload is live, ROCm0 should be pegged while ROCm1–3 idle" (the `return b` behavior made visible).
- Design details: the fitter's `-fitt` is sweepable but modifies tensor_split/tensor_buft_overrides, so each `-fitt` value gets its own invocation; the fitter does not reserve staging VRAM for op_offload, so it needs a large margin (e.g. `-fitt 12288` ≈ 12 GiB/card) or is skipped for op_offload runs; op_offload does not affect decode (below the 32-token threshold), so fitter plus op_offload compose in production; ~16 loads total, estimated 4–6 hours; ladder-row timing estimates ub 1024 ≈ 5.5 min, 2048 ≈ 2.7 min, 4096 ≈ 1.4 min; llama-bench requires `-b` ≥ `-ub`, so both are 8192 for the top rung; ub 16384 was considered (est. 964 t/s versus 482 at 8192) but `-p 16384` slow rows cost 21 min each, hence the A1/A2 split; later phases include a prefill-thread re-sweep, a decode fitter `-t` sweep 16–64 at `-b 2048`, a `-fitt` margin sweep, a context-versus-experts `-fitc` sweep, and Phase H (v4) — the final production config, pp and tg together via `-pg`.

### 15:02 — Paul asks whether the GPUs are worthless

"Can you tell me what's up with my GPUs? Are they just totally worthless?"

### 15:05 — GPUs measured at +55%; 11% bandwidth utilization; both parallelism modes blocked by MLA

"No. They're worth +55% — that's measured, not inferred":

| | tg |
|---|---|
| `-ngl 0` (no GPUs at all) | 3.87 |
| `-ot exps=CPU` (GPUs hold the dense path) | 5.53 |
| fitter (GPUs hold dense + 16 expert layers) | **6.01** |

"Pull the cards and you lose a third of your throughput." The number showing something is wrong: "The GPUs read 19.7 GB per token and take 90 ms to do it. That's an effective 219 GB/s — against 2,048 GB/s of installed bandwidth. You are using 11% of what you paid for."

Reason 1 — under `-sm layer`, layers 0–19 sit on ROCm0, 20–39 on ROCm1, and so on; the graph is processed sequentially, so each card works a quarter of the token and idles three quarters: "Four cards, one card's worth of bandwidth."

Reason 2 — both parallelism modes are closed, and not because of AMD. `-sm row` is gone from llama.cpp entirely — `ggml_backend_cuda_reg_get_proc_address()` no longer exports the split-buffer entry point, verbatim:

```c
static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0)             { ... }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) { ... }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0)  { ... }
    if (strcmp(name, "ggml_backend_get_features") == 0)          { ... }
    return nullptr;                     // no ggml_backend_split_buffer_type
}
```

"That's why you got `device ROCm0 does not support split buffers` — llama.cpp deleted row-split for CUDA and HIP alike and replaced it with real tensor parallelism" (`comm_allreduce_tensor`). But tensor parallelism refuses this model, verbatim:

```c
bool llm_arch_supports_sm_tensor(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM_DSA:      // <-- you
            return false;
```

Every model on that list uses MLA — the compressed KV latent does not shard across devices. "So the thing keeping your four cards from working in parallel is GLM-5.2's attention design — not ROCm, not RDNA2, not AMD."

Reason 3 — at batch 1 the cards do GEMVs, 0% compute utilization (each V620: 72 CUs, ~40 TFLOP/s fp16). "You are using four graphics cards as expensive, slow DRAM." Prefill/op_offload is the one regime where their compute means anything — which is what v4 is measuring.

Would better GPUs help? Barely. Four 3090s (936 GB/s each, ~1.8× the bandwidth):

```
GPU weight read   19.7 GB / ~750 GB/s   =  26 ms   (was 48)
split overhead    155 syncs             =  42 ms   (unchanged)
CPU experts       13.77 GB / 152 GB/s   =  91 ms   (unchanged)
                                          -------
                                           159 ms  =  6.3 t/s
```

"6.3 vs your 6.01. Five percent, for four new cards" — still stuck in `-sm layer` (the MLA restriction is architectural), and with less total VRAM (96 GiB versus 122). "Your GPUs are not the bottleneck. Your DDR4 is." 91 of every 181 ms is the CPU reading experts at 152 GB/s.

The cards hold "27% of a model that is 3.6× too big for them." Given a model that fits:

| | total @ Q4 | fits in 122 GiB? | est. tg |
|---|---|---|---|
| GLM-5.2 | 435 GiB | no (3.6× over) | 6.0 |
| DeepSeek-V4-Flash | **~78 GiB** | **yes** | **30–60** |
| GLM-4.5-Air (106B-A12B) | ~60 GiB | yes | 40–70 |

**Observations**

- Established fact: Paul already runs DeepSeek-V4-Flash on this machine — at 7.16 t/s, with `--cpu-moe` pushing every expert into RAM by choice. At Q4 it fits entirely in VRAM: 13B active at ~4.9 bpw ≈ ~8 GB/token; at a single card's 512 GB/s that is 16 ms. "You are leaving something like a 5–8× on the table on a model you already have on disk."
- Honest summary: the V620s are mediocre AI cards (RDNA2, no matrix cores, second-tier ROCm support, 512 GB/s unremarkable in 2026), but none of that is what is hurting; they are earning +55% doing the only job available.

### 15:09 — Paul: the GPUs were bought for prompt processing

"I was more hoping that they'd improve my prompt processing though"

### 15:11 — Prefill is exactly where the GPUs should win: 80× compute, PCIe-bound; the twenty-line patch

"Your instinct is right, and it's the one thing this entire investigation still hasn't measured." Prefill is compute-bound: the CPU runs it at 1.7 TFLOP/s (64 Zen3 cores, AVX2, no VNNI). Each V620 delivers ≈ 35 TOPS int8 via `V_DOT4_I32_I8`; four ≈ ~140 TOPS — about 80× the CPU. (Working figures: V620 peak int8 ~73.7 TOPS at 2.0 GHz boost, 30–40 effective; CPU cross-check 512 tokens / 13.7 s × 45.3 GFLOP/token ≈ 1.69 TFLOP/s; the fitter's 40 t/s ≈ 1.8 TFLOP/s.)

Why they have not: "`-p 512` clamped `n_ubatch` to 512, and at ubatch 512 op_offload loses to the CPU by design. That's the whole point of v4." Compute was never going to be the ceiling anyway — prefill sees every expert:

```
E(B) = 256 × (1 − 0.96875^B)     B=512 → 255.99996
```

`ggml_backend_tensor_copy()` moves the whole tensor regardless, so every ubatch drags 438 GB of expert weights across PCIe (344 GB with the fitter holding 16 layers resident). The two scheduler findings set the ceiling — all offloaded ops go to ROCm0 (one x16 link, 25 GB/s, not 100), and the copies are blocking (PCIe and compute strictly additive). Arithmetic, verbatim:

```
per ubatch, with the fitter:
  stream 344 GB ÷ 25 GB/s (ONE card)         = 13.8 s
  expert GEMMs on ROCm0 alone @ ~35 TOPS     =  4.2 s   (at ub 4096)
                                               ------
                                               ~19 s
```

**Prediction:**

| `-ub` | est. prefill |
|---|---|
| today (CPU) | **40** |
| 2048 | ~120 |
| 4096 | **~210** |
| 8192 | ~310 |
| 16384 | ~420 |

"A 5–8×. That's what v4 will tell you, and it's real. An 8K prompt goes from 3.4 minutes to 25 seconds." What the hardware could do if both implementation choices were fixed:

```
  stream 344 GB ÷ 100 GB/s (FOUR cards)      =  3.4 s
  expert GEMMs across four cards             =  1.0 s
  overlapped instead of serialized           =  max(3.4, 1.0) = 3.4 s
                                                4096 ÷ 3.4  ≈  1,200 t/s
```

"Roughly a 6× beyond what llama.cpp will give you today, and a 30× over your CPU. Your four PCIe 4.0 x16 links are 100 GB/s of aggregate bandwidth and llama.cpp uses one of them."

The patch is small — two places:

```c
for (int b = 0; b < src_backend_id; b++) {
    if (ggml_backend_supports_op(...) && ggml_backend_offload_op(...)) {
        SET_CAUSE(tensor, "1.off");
        return b;                    // <-- always 0. Should prefer src[1]'s device.
    }
}
```

(`ggml/src/ggml-backend.cpp:919-925` — return the backend that already holds the op's activation source, the layer's home GPU, instead of the first supporting backend), and `ggml_backend_cuda_cpy_tensor_async()` in `ggml/src/ggml-cuda/ggml-cuda.cu`, which returns `false` for a host source and forces the blocking fallback — pinned host memory can be DMA'd asynchronously, "a missed case, not a hardware limit."

**Observations**

- "Neither is exotic. The first is the bigger win and it's maybe twenty lines." Landing it upstream would fix this for everyone running large MoE across multiple GPUs.
- Summary: decode — GPUs +55%, near their ceiling, DDR4 is the wall. Prefill — "they're the whole answer... llama.cpp is currently feeding them through a straw — one of four straws, with the valve closed while they chew."
- Working figures: the four-card-spread patch estimates ~930 t/s at ub 4096 without overlap, ~1200 with; the fitter-only no-patch path ≈ 205–216 t/s; asymptotic single-card ceiling ≈ 660 t/s at very large ubatch; an RPC-backend workaround was considered and rejected (the scheduler would still pick the first backend); Q3_K_XL would cut streaming ~29%.

### 15:21 — First v4 row arrives: pp8192 = 25.90 at ub 512

Paul ran, on the server, verbatim:

```
root@openwebui:~# cat diag-v4-20260714-150115/MAIN.log  | grep pp
```

and pasted the output (key lines verbatim, with interleaved v4-briefing lines included):

```
      => tg128 = 6.01 t/s     pp512 = 40.11 t/s
                         tg 5.52 vs 5.53. pp 34.64 vs 34.65. The AVX2
  -sm row                IMPOSSIBLE: "device ROCm0 does not support split buffers".
  -sm tensor             Not supported for glm-dsa (llm_arch_supports_sm_tensor
                         (anonymous memory). Never applied. Untestable.
                         with --tensor-split unset llama.cpp apportions layers by
  TWO STRUCTURAL FACTS, both verified in ggml-backend.cpp / ggml-cuda.cu:
  (1) ALL OFFLOADED OPS GO TO ROCm0.  ggml-backend.cpp:919-925:
              if (supports_op(backends[b], t) && offload_op(backends[b], t))
                  return b;          // <-- FIRST supporting backend. Always 0.
  H   The final production config, pp and tg together.
$ cd '/root/llama.cpp' && git log -1 --format='%H %cd %s'
   HIP error string will appear immediately ABOVE any backtrace.
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
load_tensors:   CPU_Mapped model buffer size = 46166.88 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 46303.78 MiB
load_tensors:   CPU_Mapped model buffer size = 22067.47 MiB
resolve_fused_ops: resolving fused Gated Delta Net support:
resolve_fused_ops: resolving fused Lightning Indexer support:
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |    8192 |      512 |   1 | exps=CPU              |    0 |          pp8192 |         25.90 ± 0.36 |
resolve_fused_ops: resolving fused Gated Delta Net support:
```

The v4 run directory `diag-v4-20260714-150115` shows the battery started at 15:01:15. First A1 row: ngl 99, n_batch 8192, n_ubatch 512, fa 1, exps=CPU, nopo 0 — **pp8192 = 25.90 ± 0.36 t/s**. CPU_Mapped buffers: 46166.88 + 8 × 46303.78 + 22067.47 MiB.

### 15:22 — Calibration point: the clamp is gone; 22.3 GB/s ≈ PCIe 4.0 x16 line rate

"That first row is the calibration point, and it's better news than it looks."

```
| ngl 99 | n_batch 8192 | n_ubatch 512 | exps=CPU | nopo 0 | pp8192 | 25.90 ± 0.36 |
```

"`n_batch = 8192`. Not 512. The clamp is gone — this is the first honest op_offload measurement in the entire investigation." Per-ubatch cost:

```
8192 tokens ÷ 25.90 t/s = 316 s
316 s ÷ 16 ubatches of 512 = 19.8 s per ubatch
```

Checked against the streaming model (441 GB of expert weights per graph eval onto ROCm0's single link):

```
441 GB ÷ 19.8 s = 22.3 GB/s
```

"That is essentially line rate for PCIe 4.0 x16 — from a *pageable* mmap, no less. So the per-ubatch cost is ~100% PCIe transfer. Compute is noise. The model isn't just right, it's tight." The 19.8 s is constant per ubatch, so the ladder should be nearly linear.

**Prediction:**

| `-ub` | ubatches for 8192 | predicted pp |
|---|---|---|
| 512 | 16 | **25.9** (measured) |
| 1024 | 8 | ~52 |
| 2048 | 4 | ~103 |
| 4096 | 2 | ~185 |
| 8192 | 1 | ~300 |

(The 4096/8192 rows bend slightly below linear as the single-card expert GEMM registers — roughly 5 s at ub 4096, 10 s at 8192.) "Against your current 40 t/s, ub 8192 is a 7×. An 8K prompt drops from 3.4 minutes to 27 seconds."

Watch commands supplied, verbatim:

```bash
D=$(ls -dt /root/diag-v4-*/ | head -1)
grep -E '^\| glm-dsa' "$D/MAIN.log" | awk -F'|' '{print $6, $7, $10, $12, $13}'
```

```bash
watch -n2 'rocm-smi --showuse --showmemuse | grep -E "GPU\[|use"'
```

"ROCm0 should be pegged while 1, 2 and 3 sit near zero. That's `return b` ... made visible on the wire."

**Observations**

- Timing expectations: the op_offload rows come fast (ub 4096 ≈ 45 s, ub 8192 ≈ 30 s); the slow rows are the `-nopo 1` CPU controls (~8 min each — the CPU does not care about ubatch); the whole phase ≈ an hour.
- Headline: "You're getting 22.3 GB/s. You have 89 GB/s installed. Fix `return b` so the offload goes to the layer's home GPU instead of always card 0, and the same 4096-token ubatch costs 5 s instead of 20: ~800 t/s instead of ~185. The 7× is free today. The 30× is a twenty-line patch."
- Cross-checks: the CPU baseline at ub 512 is 14.6 s per ubatch versus op_offload's 19.8 s — a 36% slowdown, matching the 25.90-versus-34.65 ratio; pageable H2D at 22.2–22.3 GB/s exceeded the 10–15 GB/s expected for pageable transfers; pinned should approach ~26 GB/s practical of ~32 theoretical.

### 15:26 — Paul asks whether $1,600 for the four GPUs was a good deal

"So say I paid $350 per GPU. Would ~$1600 including tax and shipping for these four GPUs have been a good deal?"

### 15:27 — Verdict: yes, with the receipt; the next money goes to DDR5 — but the free patch first

"Yes. And unusually, you now have the receipt." What the $1,600 measurably bought:

| | no GPUs (`-ngl 0`) | the V620s (fitter) | + op_offload (projected) |
|---|---|---|---|
| decode | 3.87 t/s | **6.01** | 6.01 |
| prefill | 20.04 t/s | **40.11** | **~300** |

"+55% decode, and 2× prefill today rising to ~15× once op_offload runs at a real ubatch. For a model that is 435 GiB against 122 GiB of VRAM. That's $12.50/GiB of PCIe 4.0 x16-attached VRAM." (Working figure ≈ $13/GiB, ~$400 per 32 GB card with tax.)

The obvious "better" purchase would have been worse — four 3090s (1.8× bandwidth, CUDA, ~75% more money, ~$2,800):

```
GPU weight read   19.7 GB ÷ ~750 GB/s   =  26 ms   (was 48)
split overhead    155 syncs             =  42 ms   (unchanged)
CPU experts       13.77 GB ÷ 152 GB/s   =  91 ms   (unchanged)
                                          -------
                                           159 ms  =  6.3 t/s   vs your 6.01
```

"Five percent. And they'd have *less* VRAM (96 vs 122 GiB), so fewer resident experts, which claws most of that back." Prefill would not improve either — op_offload is ~100% PCIe-bound (the first v4 row measured 22.3 GB/s ≈ line rate), so tensor-core compute idles. "Same story for MI100s at 1229 GB/s HBM2. Double the price, ~zero gain." (Working figures: MI100 = 32 GB HBM2, CDNA1, used $700–1,000 each, 4× = $3–4k.)

**Observations**

- Three findings make VRAM-capacity-per-dollar the correct metric, none knowable in advance: decode is DDR4-bound, not VRAM-bandwidth-bound (HBM would have been wasted money); tensor parallelism is blocked by MLA, not by AMD (`llm_arch_supports_sm_tensor()` returns false for GLM_DSA and DEEPSEEK2/32/4 — every MLA model, every vendor); prefill is PCIe-bound, so lane count and PCIe generation matter, "which you have four of."
- The ROCm tax "turned out to be near zero here": ZenDNN was Paul's own build flag; split buffers were deleted for CUDA too; `return b` is vendor-agnostic. "You didn't actually hit an AMD-specific wall."
- Where the cards genuinely cost: RDNA2 has no matrix cores (would matter only if op_offload became compute-bound — it will not; it is PCIe-stuck); poor resale (odd cloud-gaming parts); "a day of your life debugging, though most of that was llama.cpp, not the cards."
- **Decision:** the next $1,600 should not go to GPUs. "The DDR4 is the wall" — 91 of every 181 ms is the CPU reading experts at 152 GB/s. Twelve channels of DDR5 on Genoa/Turin ≈ 450 GB/s → the expert read drops 91 → 24 ms:

```
90 ms (GPU, unchanged) + 24 ms  =  114 ms  =  8.8 t/s     (+46%)
```

  (Working figure: a used Genoa system — EPYC 9004 32/64-core, board, 12× 64 GB DDR5 — estimated $4–6k in 2026.) But before spending anything: "`return b` is free." Twenty lines in `ggml-backend.cpp` take prefill from ~300 to ~800–1,200 t/s using the hardware already in the chassis — a bigger return than any purchase on the list.
- Caveat recorded: the used-market prices are recollection, not research; the relative argument rests on Paul's own measurements.

### State of knowledge at end of session

- The performance model is validated across three configurations: `t_token = 90 ms (GPU + 155 splits) + CPU_expert_bytes ÷ 152 GB/s` — predicted 5.5 / 6.2 / 3.9 against measured 5.53 (hybrid) / 6.01 (fitter) / 3.87 (`-ngl 0`).
- Decode is declared done at 6.01 t/s (fitter, 106 GiB VRAM filled); the hard ceiling on this quant is ~11 t/s; the only remaining decode lever is fewer bytes (Q3_K_XL → ~7 t/s) or a DDR5 platform (~8.8 t/s, +46%).
- Production configuration: `llama-server -m ... -fitt 2048 -c 65536 -fa 1 -t 32 --no-mmap` — no `--poll`, `-C`, `--cpu-strict`, `--no-host`, `-ncmoe`, `-ot`, or `-sm row`.
- Four hypotheses are formally dead: CPU_REPACK (engaged, zero gain), `-sm row` (removed from llama.cpp; MLA also blocks `-sm tensor`), `--poll`/`--cpu-strict` (nothing/harmful), THP (never applied, probably untestable).
- v3 never actually tested op_offload above ubatch 512: `-p 512` clamps n_ctx → n_batch → n_ubatch (`llama-bench.cpp:1247`, `llama-context.cpp:239/241`) — a script bug flagged and then reintroduced by its author.
- Two structural scheduler facts set the op_offload ceiling: all offloaded ops go to the first supporting backend (`return b` → ROCm0, one x16 link at ~25 GB/s instead of 100 GB/s aggregate), and CPU→GPU copies are blocking (no overlap with compute).
- v4 (`galactus-prefill.sh`, run directory `diag-v4-20260714-150115`) launched at 15:01:15 to measure the untested regime; its first row — pp8192 = 25.90 ± 0.36 at n_ubatch 512, n_batch 8192 — back-calculates to 19.8 s per ubatch = 22.3 GB/s, essentially PCIe 4.0 x16 line rate from a pageable mmap; predicted ladder ~52 / ~103 / ~185 / ~300 t/s at ub 1024/2048/4096/8192.
- The GPUs are measured at +55% (3.87 → 6.01) while running at 11% of installed bandwidth (219 of 2,048 GB/s); a `return b` patch (~20 lines) plus async pinned copies would take prefill toward ~800–1,200 t/s on existing hardware.
- The $1,600 GPU purchase is judged good ($12.50/GiB of PCIe-attached VRAM; four 3090s would yield only 6.3 versus 6.01 t/s decode with less VRAM); DFlash speculation would reach ~12.5 t/s but no GLM-5.2 draft model exists; hosted-model gap: ~9% of Opus 4.8 decode today, ~13% honest near-term ceiling, 5–8× slower end-to-end.
- Open at session close: the v4 ubatch ladder (rows above 512 still pending at 15:27).

