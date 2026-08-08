## Session 1 — Sunday, July 13, 2026 (evening) — Background and plan

*Conversation "Applying concepts to Galactus with GLM5.2", opened 7/13/2026 22:43. This session is background, architecture research, sizing, and planning; no commands were executed on Galactus. All figures in this session are either quoted from the pasted source material or are estimates, flagged as such.*

### 22:43 — Opening question and source material: the tensor-offload post

Paul opened by pasting an approximately one-year-old r/LocalLLaMA post — "Don't Offload GGUF Layers, Offload Tensors! 200%+ Gen Speed? Yes Please!!!" by skatardude10 — and asked: "Explain to me what this is talking about and how I can apply it to Galactus for a model like GLM5.2 (us the unsloth Q4_XL version of this model as your example)". Everything below in this entry is quoted background from the paste, not a measurement on Galactus.

**The post's central claim.** A QwQ merge at IQ4_M went from 3.95 t/s (59 of 65 layers on GPU) to 10.61 t/s (all 65/65 layers on GPU) by restricting selected FFN tensors to CPU — at the same VRAM use; the OP was inspired by a post running Qwen3 235B on a single 3060 12GB at 6 t/s. The technique is only relevant when VRAM forces some layers onto CPU; if the model already fits on GPU there is nothing to gain. Mechanism claimed: a layer contains attention tensors (small, GPU-heavy, benefit from parallelization) and FFN tensors (very large, basic matmuls that a CPU can handle); `--overridetensors` (koboldcpp) / `-ot` (llama.cpp) pins individual tensors to CPU by regex.

Winning command (10.61 T/s):

```bash
python ~/koboldcpp/koboldcpp.py --threads 10 --usecublas --contextsize 40960 --flashattention --port 5000 --model ~/Downloads/MODELNAME.gguf --gpulayers 65 --quantkv 1 --overridetensors "\.[13579]\.ffn_up|\.[1-3][13579]\.ffn_up=CPU"
```

Result line: `[18:44:54] CtxLimit:39294/40960, Amt:597/2048, Init:0.24s, Process:68.69s (563.34T/s), Generate:56.27s (10.61T/s), Total:124.96s`

Layer-offload baseline (3.95 T/s):

```bash
python ~/koboldcpp/koboldcpp.py --threads 6 --usecublas --contextsize 40960 --flashattention --port 5000 --model ~/Downloads/MODELNAME.gguf --gpulayers 59 --quantkv 1
```

Result line: `[18:53:07] CtxLimit:39282/40960, Amt:585/2048, Init:0.27s, Process:69.38s (557.79T/s), Generate:147.92s (3.95T/s), Total:217.29s`

**Tensor selection details from the post.** The OP targeted ffn_up (mostly IQ4_XS in his file) because his ffn_down varied IQ4_XS–Q5/Q8. Example GGUF table rows (QwQ-32B Q3_K_M): `blk.1.ffn_down.weight [27 648, 5 120] Q5_K`; `blk.1.ffn_gate.weight [5 120, 27 648] Q3_K`; `blk.1.ffn_norm.weight [5 120] F32`; `blk.1.ffn_up.weight [5 120, 27 648] Q3_K` — offloading a Q5 ffn_down to CPU saves more VRAM than a Q3 ffn_up/gate. His every-other-Q4 versus every-third-any-quant choice changed speed by only 0.4 tokens/second. Regex recipes given: all FFN up `"\.\d+\.ffn_up=CPU"`; every other `"\.\d*[13579]\.ffn_up=CPU"`; every third `"\.\d*[0369]\.ffn_up=CPU"`; up+gate `"\.\d+\.(ffn_up|ffn_gate)=CPU"` — the stated goal being to target as few tensors as possible while still offloading all layers, keeping VRAM maxed. Thread advice in the post: threads = physical cores − 1 (12C/24T → `--threads 11`), though the OP later noted 6 threads of 24 measured best (measurable, not substantial). The post claims whole-layer offload uses the same memory as tensor offload "but sucks way more", and wishes llama.cpp would do this automatically. Caveat from commenter Caffeine_Monster: gains appear mostly on hardware where the CPU is the bottleneck, and there is a penalty (PCIe memory transfer) for non-concurrent tensors; the OP concurred the penalty is the memory bottleneck on the PCI bus but argued whole CPU layers are worse.

**Community data points included in the paste** (recorded exactly, for later reference):

- DrVonSinistro: dual Xeon E5-2690 v4, 256GB DDR4, 60GB VRAM (2x P40 + 1x A2000), Qwen3 235B IQ4_XS: 2.9 → 4.2 t/s with 95/95 layers offloaded.
- PDXSonic: 128GB DDR4 / 4x P100: ~4.3 T/s on the Q2K.
- 3750gustavo: 30B at 10 t/s on 8GB VRAM, 16k context, 4-bit, no KV cache quant or flash attention.
- farkinga (the inspiration post): 88GB Q2 quant of Qwen3 235B at 6 tps on a 16GB GPU.
- Caffeine_Monster: recommends ik_llama.cpp (github.com/ikawrakow/ik_llama.cpp) for big MoE offloads; self-hosts full-precision DeepSeek V3/R1 on large DDR5 plus a few 3090s; says mainline llama.cpp prompt processing is the bottleneck for big MoE.
- Mkengine: Qwen3-30B-A3B — ik_llama 9.2 t/s versus mainline llama.cpp 10.7 t/s with a Qwen3-0.6B draft model (ik_llama lacks speculative decoding).
- a_beautiful_rhind: 7 t/s (mainline) versus 14 t/s (ik) on 235B; prompt processing about the same; for dense models, mainline wins.
- Lissanro: EPYC 7763 64-core, DeepSeek R1/V3 UD-Q4_K_XL, 8 tokens/s with selective tensor overrides and full context cache on four 3090s, using ik_llama.cpp (claims ~2x mainline for R1/V3, comparable to ktransformers).
- TheRealGentlefox: 10 → 18 tk/s on Qwen3 30B A3B with `--overridetensors "blk\.([0-9]*[02468])\.ffn_.*_exps\.=CPU"`.
- RampantSegfault (gemma3 27b IQ4_XS, 16GB card, 16k ctx): baseline 46 layers 6.86 t/s; `\.\d*[0369]\.(ffn_up|ffn_gate)=CPU` at 99 layers 7.76 t/s; `\.\d*[03689]\.(ffn_up|ffn_gate)=CPU` at 99 layers 6.96 t/s; `\.\d*[0369]\.(ffn_up|ffn_down)=CPU` at 99 offload 8.02 t/s and 7.95 t/s; `\.\d*[0-9]\.(ffn_up)=CPU` at 99 offload 6.4 t/s; `\.(5[6-9]|6[0-3])\.(ffn_*)=CPU` at 55 offload 7.6 t/s; `\.(5[3-9]|6[0-3])\.(ffn_*)=CPU` at 99 layers 10.4 t/s. Net 6.86 → 10.4 t/s (blank chat, empty context).
- An LM Studio user: qwen 30BA3B ~11 tok/s with 8 layers offloaded, ~9.5 with none; 16GB file, 6GB card.
- dampflokfreund: 3 → 11 token/s, 30B A3B on a 2060 laptop.
- shenglong: quotes the Unsloth docs — use `-ot ".ffn_.*_exps.=CPU"` to offload all MoE layers to CPU; `-ub 1` was the missing piece for his model.
- esuil: laptop 3050 4GB VRAM, 12B Mistral Nemo (41 layers, 16k ctx): 16–18 layers fit without override → 24–25 with; gains 10–25% depending on context, never worse.
- the-proudest-monkey: Qwen3-235B-A22B-UD-Q2_K_XL, dual 3090 + Ryzen 7900 + 64GB DDR5: 47/95 layers → ~9 t/s; all layers except randomly selected (ffn_up|ffn_down|ffn_gate) tensors → 12.5 t/s.
- Monkey_1505: PP 20 → 64 t/s after tensor sorting; separately, batch-size tuning with MoE matters 30–40%: bs 256 → 50 t/s, 128 → 64 t/s, 64 → 45 t/s, 32 → 30 t/s (theory: smaller batches activate fewer experts per batch; sweet spot ~64–128 for Qwen3 30B A3B). Second finding: on a mobile dGPU, offloading only the first 3 layers' gate/up/down tensors to CPU even when the model fully fits took PP from 30 t/s to 170 t/s on Qwen 14B (`--overridetensors "blk\.[0-3]\.(ffn_gate|ffn_up|ffn_down)\.weight=CPU"`); works only on Qwen 14B, not 4B or Llama 3.1 8B (~5x PP speedup; GPU compute choke on frontloaded heavy matrices).
- Sidran (Vulkan backend, 32GB RAM, 8GB VRAM, Qwen3 30B A3B UD Q4_K_XL): baseline ~12 t/s at 15/48 layers; best +2 t/s via `"\.(16|24|28|4[0-7])\.(ffn_down_exps|ffn_up_exps|ffn_gate_exps)\.weight=CPU"` at 25/48 layers; update: `"\.ffn_(down|gate|up)_exps\.weight=CPU"` gives ~+1 t/s with half the VRAM free, all 48 layers offloaded at 12288 ctx — could run near-full context (30720) on 8GB. Suspects something "stuck" with Vulkan.
- ilintar: Qwen3 30B MoE Q3_K_L on 10GB VRAM with `(up_exps|down_exps)=CPU`.
- ffpeanut15: on RTX3060 mobile, offloading merely 4 layers cuts performance by 70%.
- Chromix_: threads = cores−1 gives only a tiny gain; selecting the minimum cores needed (avoiding compute/memory-latency binding) plus OS-level pinning to real cores is faster; alternating offloaded layers adds a GPU↔CPU transfer each alternation in theory.
- Other Q&A in the thread: an MoE explanation (fewer active params; choose which experts sit on CPU versus GPU); henk717 notes 32B Q4_K_S just fits on 24GB; skatardude10's hypothetical: 12 tensors per layer, 8 run best on GPU, 4 are huge but CPU-tolerable — layer offload = CPU inference bottleneck, tensor offload = memory-bandwidth loading spread more evenly.

sammcj: Qwen3 235B IQ3_M at ~7.6 tk/s on 48GB VRAM using `--override-tensor '([4-9]+).ffn_.*_exps.=CPU'`; his full command (note it uses `([3-8]+)`):

```bash
/app/llama-server
      --port 9045 --flash-attn --slots --metrics -ngl 99
      --cache-type-k q8_0 --cache-type-v q8_0
      --no-context-shift
      --ctx-size 32768
      --n-predict 32768
      --temp 0.5 --top-k 20 --top-p 0.95 --min-p 0 --repeat-penalty 1.05 --presence-penalty 2.0
      --jinja --reasoning-format deepseek
      --model /models/Qwen3-235B-A22B.i1-IQ3_M.gguf
      --threads 23
      --threads-http 23
      --cache-reuse 256
      --main-gpu 0
      --tensor-split 0.5,0.5
      --override-tensor '([3-8]+).ffn_.*_exps.=CPU'
```

Impossible_Ground_15: 3090+4090 (48GB), 192GB DDR5, 9950X3D; was 4.4 tk/s, now 6–7 tk/s after adding `--no-kv-offload`:

```bash
llama-server.exe -m "C:\models\Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf" -ngl 99 -c 16384 --override-tensor "([4-9]+).ffn_.*_exps.=CPU" --ubatch-size 512 --batch-size 512 --flash-attn --prio 2 --threads 15 --slots --alias llamacpp --verbose-prompt --host 0.0.0.0 --port 9331 --cache-reuse 256 --reasoning-format deepseek --jinja --split-mode layer --log-timestamps --log-colors --metrics --mlock --verbosity 1
```

prompt_seeker: 5700X + 128GB DDR4-3200 + RTX3090, 32B Q4_K_M. Setting1, 53/65 layers (VRAM 23.10GB): `./llama-server -fa -m AI-45/Smoothie-Qwen3-32B.i1-Q4_K_M.gguf -ngl 53 -c 32768 --mlock --no-mmap -b 1024 -ub 1024`. Setting2, ffn_up→CPU (VRAM 23.18GB): same with `-ngl 99 ... -ot "ffn_up=CPU"`. Results by input tokens: 25 → pp 39.42 / tg 6.86 versus pp 30.05 / tg 6.86; 3909 → pp 632.50 / tg 6.26 versus pp 620.03 / tg 6.71; 14181 → pp 545.32 / tg 2.89 versus pp 571.25 / tg 6.53 — tensor offload holds tg at long context.

Old_Cantaloupe_6558: 6 cores, 32GB DDR4-2133, 3060 12GB, Qwen3-30B-A3B-UD-Q8_K_XL: 13/48 layers 6.5 tps → all 48 layers 10 tps:

```bash
../llama.cpp/build/bin/llama-cli -m ./Qwen3-30B-A3B-UD-Q8_K_XL.gguf -ngl 13 -c 40960 -fa -t 5 -b 256 -ub 256 --temp 0.7 --top-k 40 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.1 -f ./prompt.txt
../llama.cpp/build/bin/llama-cli -m ./Qwen3-30B-A3B-UD-Q8_K_XL.gguf -ngl 48 -ot "blk\.(0?[2-9]|1[2-9]|2[1-9]|3[1-9]|4[1-7])\.ffn_.*_exps\.=CPU" -c 40960 -fa -t 5 -b 256 -ub 256 --temp 0.7 --top-k 40 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.1 -f ./prompt.txt
```

dopey_se: Tesla P100 loads Qwen3-30B-A3B at 19.12 tokens/second:

```bash
llama-server -m /models/Qwen3-30B-A3B-BF16/Qwen3-30B-A3B-Q8_0.gguf -c 19456 -ngl 100 -b 4096 --temp 0.6 --top-p 0.95 --min-p 0 --top-k 20 --no-mmap -n 38912 --flash-attn -ot '([4-9]+).ffn_.*_exps.=CPU'
```

thkitchenscientist (T5810, 14-core, 96GB RAM, RTX2060 12GB; ~10.5GB VRAM occupied — a counter-example where layer and tensor split are identical):

```bash
./llama-cli -m ~/models/Qwen3-32B-Q4_K_M.gguf -ngl 0 --threads 27 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5
```

7.3/2.6 t/s (CPU only);

```bash
./llama-cli -m ~/models/Qwen3-32B-Q4_K_M.gguf -ngl 30 --threads 27 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5
```

12.9/4.3 t/s (layer split);

```bash
./llama-cli -m ~/models/Qwen3-32B-Q4_K_M.gguf -ngl 99 --override-tensor "ffn_up=CPU,ffn_down=CPU" --threads 27 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5
```

12.5/4.3 t/s (tensor split).

**Observations**

- The post's core idea: `-ngl` is block-granular, but tensors differ wildly in value-per-byte of VRAM; regex-level tensor placement can keep every layer's attention on GPU while evicting only bulk FFN weight.
- The claimed gains span 0% (thkitchenscientist) to ~170% (the OP's dense-model case); the counter-examples and the PCIe caveat suggest results depend heavily on where the starting configuration sits.
- The question for this investigation: what, if anything, of this applies to Galactus running GLM-5.2 under `--cpu-moe`.

### 22:54 — Research pass: GLM-5.2 architecture, llama.cpp offload semantics, sizing, and the plan

A long research pass (web searches plus llama.cpp source reads) preceded the answer. Research-side notes: one GitHub API rate limit was hit (worked around via raw.githubusercontent.com fetches), and huggingface.co was fetched via web_fetch rather than the shell. The findings, sizing arithmetic, and the resulting plan follow.

**Hardware context (recalled from prior work).** Galactus: 4x AMD Radeon Pro V620 — gfx1030-class RDNA2 (Navi 21), 32GB GDDR6 each, ~128 GB total VRAM, ~122 GiB usable; AMD EPYC 7713 (64-core); 1TB DDR4; total system memory 1.12 TB (1TB DDR4 + 122GB VRAM). Known optimal thread count from prior tuning: `-t 64` (established on Qwen3.5-397B-A17B). EPYC 7713 theoretical fp32 peak ~5.1 TFLOP/s; 8-channel DDR4-3200 stated as sustaining ~160 GB/s (an assumption that Session 2 would revisit). Prior baseline on this box: DeepSeek-V4-Flash at 7.16 tokens/second with `--cpu-moe`.

**GLM-5.2 identification (post-cutoff model, searched).** 744B parameters, 40B active per token, MoE; released June 2026 by Zhipu/Z.ai; 1M context window; arch string `glm_moe_dsa` (`GlmMoeDsaForCausalLM` upstream; `glm-dsa` in llama.cpp). DSA = DeepSeek-style sparse attention with an indexer selecting top-k tokens; most layers share a single indexer — only ~1 in 4 layers are full indexer layers; 57 of the 78 layers rely on the shared indexer with top-k frequency 4.

**Architecture from config.** 78 layers (blk.0–blk.77); first 3 dense (`first_k_dense_replace=3`); MoE layers blk.3–blk.77 (75 layers); 256 routed experts, 8 active per token; 1 shared expert (always active); hidden_size 6144; moe_intermediate_size 2048; MLA attention, kv_lora_rank=512; blk.78 is an MTP/NextN (multi-token prediction) block. An early estimate (~89 MoE layers, ~437GB experts, ~4.9GB/layer, ~18 layers fitting) was superseded once the real config was found (78 layers, 75 MoE).

**Quant file.** Unsloth UD-Q4_K_XL GGUF: 467 GB across 11 shards (~434.9 GiB); average ~4.94 bpw. Cited: cedric on X — Unsloth's dynamic UD-Q4_K_XL and UD-Q5_K_XL are "essentially lossless"; dynamic 4-bit may be the sweet spot.

**Compatibility.** Stock llama.cpp originally could not load GLM-5.2 GGUFs (missing indexer tensors on non-full layers), but indexer tensors are now optional in current llama.cpp — the reported HF bug is resolved. Support came via PR #19460 "model: support GLM MoE DSA arch (NOTE: indexer is not yet supported)" (ngxson); the feature request was Issue #24730. llama.cpp tensor names: routed experts `blk.N.ffn_gate_exps.weight` / `ffn_down_exps` / `ffn_up_exps`; shared expert `blk.N.ffn_gate_shexp.weight` etc.; a single global tensor-name map covers all architectures.

**llama.cpp flag semantics verified in source.**

- `--cpu-moe` / `-cmoe` ≡ `-ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"` (all routed experts to CPU).
- `--n-cpu-moe N` / `-ncmoe N` keeps routed experts of the **first N layers** on CPU — verified in `common/arg.cpp`, which emits `blk\.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU` for i ∈ [0, N).
- `-ot` mechanics, verified in `llama-model-loader.cpp`: comma-separated `pattern=buffer_type` pairs; matching uses `std::regex_search` (substring semantics, ECMAScript syntax); first match wins with an early break — catch-all `=CPU` rules must come last. An unknown buffer type makes it list the valid ones; `--list-devices` also available.
- Buffer/device names by backend: `ROCm0..ROCm3` on HIP/ROCm builds, `Vulkan0..Vulkan3` on Vulkan, `CUDA0..` on CUDA (via the `GGML_CUDA_NAME` macro: "ROCm" when HIP, "MUSA" for MUSA, else "CUDA").
- `--fit on` is the default in current master (`fit_params` defaults true): it auto-computes `-ngl`, `--tensor-split`, and per-layer expert overrides from measured free VRAM, packing layers back-to-front (later layers on GPU first), densely packing regular layers and partially offloading MoE layers, handling fractional layers.
- Critical verified behavior (`common/fit.cpp`): the fitter aborts (logged warning, graceful fallback) the moment the user sets `-ngl`, `--tensor-split`, `-ot`, or `-ncmoe` — fit and manual placement are mutually exclusive. This explains HF reports of `--fit on` "not respecting VRAM limits" when other flags were set.
- `--fit-target <MiB>` (default 1024 MiB headroom per device) and `--fit-print` exist; exposed via the `llama-fit-params` binary (`tools/fit-params/fit-params.cpp`). `common_memory_breakdown_print` gives a per-device memory table at load.
- With `tensor_split` unset and the fitter disabled, llama.cpp apportions layers across devices by free memory — approximately equal layer counts on 4 identical V620s — and the split is computed before and without knowledge of `-ot` overrides.
- llama.cpp warns: `tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance`.
- `llama-bench` supports `-t`, `-fa`, `-mmp`, and sweepable `-ncmoe` (comma list) and `-ot` groups. Separator inversion: in llama-bench, commas separate benchmark configurations and semicolons separate rules within one configuration — the opposite of llama-server/llama-cli.
- TENSOR_SKIP semantics verified: skipped tensors are marked unused, return nullptr, never allocated — blk.78's ~6 GB of expert weights cost zero RAM/VRAM (with or without mmap); the load log prints `model has unused tensor ... -- ignoring`. Resident model ≈ 459–460 GB (467 GB minus blk.78: experts ~5–6 GB, attention ~0.15 GB, embedding/projection ~1.6–2 GB).
- Also noticed as new in master: the `--fit` machinery and KV unified mode.

**Sizing estimates** (parameter counts × assumed bpw — flagged as estimates, not a GGUF dump):

- Routed experts per MoE layer = 3 × 256 × 6144 × 2048 ≈ 9.66 B params (9.664 B); × 75 MoE layers ≈ 724.8 B ≈ 97% (97.4–97.5%) of the ~744 B total. Everything else ≈ 19 B params (embeddings ~0.95 B + output ~0.95 B ≈ 1.9 B combined; dense-layer FFN ~0.68 B; shared experts across MoE layers ~2.83 B; MLA attention assuming q_lora_rank ~1536 DeepSeek-V3-style — unverified).
- Per-MoE-layer expert bytes at UD-Q4_K_XL mixed precision: ~5.0–6.0 GiB (cross-checks: 5.06 GiB at 4.5 bpw, ~5.4 GiB at 4.8 bpw; trace range 5.0–6.2 GiB, average ~5.4; experts total ~405–415 GiB against the 434.9 GiB file). Non-expert total ~20–30 GiB (an earlier pass said 15–20 GiB).
- KV cache (MLA compressed; llama.cpp caches the compressed latent): kv_lora_rank 512 + rope dim 64 (assumed, unverified) = 576 elements/token/layer at f16 → ~90 KB/token across 78 layers → ~2.8–2.9 GiB at 32K ctx; ~5.6 GiB at 64K; ~11.2–11.5 GiB at 128K; ~23 GB at 256K; q8_0 KV halves these.
- Compute buffers: ~8–12 GiB total across 4 devices (scales with ubatch and expert count); output logits buffer at ubatch 2048 f32 ≈ 1.27 GB.
- VRAM budget as given:

```
122  total
- ~25  non-expert weights (all 78 layers)
- ~5-8 KV cache (MLA compressed; small — read the exact figure from the log)
- ~10  compute buffers + per-device fragmentation across 4 cards
= ~80-90 GiB available for routed experts
→ ~14-16 of 75 MoE layers on GPU
```

i.e., ~17–21% of expert weights (~20% of expert traffic) moved off the DDR4 bus. Candidate `-ncmoe` values considered: `-ncmoe 63` (last 15 MoE layers 63–77 on GPU), `-ncmoe 62` (16 layers, 62–77, ~88 GiB — tight), conservative `-ncmoe 66` (12 layers).

**Throughput models.**

- PCIe traffic at decode: 59 CPU-resident MoE layers → 59 GPU↔CPU round trips per token; at batch 1 the payload is ~24 KB each way (negligible); at batch 2048 it balloons to ~3 GB per direction per layer (meaningful at PCIe 4.0).
- CPU compute, prefill: with 256 experts top-8, a large batch activates essentially all experts → dense GEMM ≈ 35.6 GFLOP/token across 59 CPU layers ≈ 73 TFLOP per 2048-token batch; at a practical 1–3 TFLOP/s quantized GEMM ≈ ~36 s.
- CPU bandwidth, decode: 8 active experts touch ~185 MB/layer/token; 59 CPU layers → ~10.9 GB/token; at ~160 GB/s → ~14.7 tokens/s CPU-side ceiling.

**The multi-GPU trap** (flagged as "the part that will bite"). The fitter aborts on any of `-ngl`/`-ts`/`-ot`/`-ncmoe`; with the fitter off and `-ts` unset, layers split approximately equally: ~`0–19 → GPU0`, `20–39 → GPU1`, `40–58 → GPU2`, `59–77 → GPU3`, computed before and without knowledge of `-ot`. So `-ncmoe 62` would place 16 heavy expert layers (~85 GiB) on GPU3 alone — instant OOM.

**Recommended plan.** Step 1 — try the fitter first:

```bash
llama-fit-params \
  -m ~/models/GLM-5.2-GGUF/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -c 65536 -fa on --fit-print on
```

then, if the projection looks sane, serve with no manual placement flags:

```bash
llama-server \
  -m ~/models/GLM-5.2-GGUF/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -c 65536 -fa on -t 64 --no-mmap \
  -b 2048 -ub 2048 --jinja \
  --temp 1.0 --top-p 0.95 --min-p 0.01
```

Tune headroom with `--fit-target <MiB>` (default 1024 MiB/device) rather than reaching for `-ot`. `--no-mmap` "is not optional here" (llama.cpp's own warning; 1 TB RAM available).

Step 2 — manual placement if the fitter underperforms. First get buffer names (`ROCm0..3` on HIP, `Vulkan0..3` on Vulkan):

```bash
llama-server --list-devices
```

Then place experts per-device, drawing from each device's own layer range (keeps a layer's experts on the same card as its attention — avoids cross-GPU hidden-state hops); verify the actual ranges in the load log first:

```bash
-ngl 99 \
-ot "blk\.1[6-9]\.ffn_(up|down|gate)_exps=ROCm0,\
blk\.3[6-9]\.ffn_(up|down|gate)_exps=ROCm1,\
blk\.5[5-8]\.ffn_(up|down|gate)_exps=ROCm2,\
blk\.7[4-7]\.ffn_(up|down|gate)_exps=ROCm3,\
ffn_(up|down|gate)_exps=CPU"
```

Mechanics (verified in `llama-model-loader.cpp`): comma-separated rules, `std::regex_search` (substring, ECMAScript), first match wins — the catch-all `=CPU` must come last. This is 4 expert layers per card (~21–24 GiB) plus the non-expert share, KV, and compute buffers — tight on a 32 GiB card.

Step 3 — sweep with llama-bench (`-ncmoe` sweep lists; `-ot` sweepable groups; separator inversion noted):

```bash
llama-bench -m .../GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -t 64 -fa on -mmp 0 -p 2048 -n 128 \
  -ot "<config A rules;joined;by;semicolons>,<config B rules>"
```

Also: re-sweep threads — the 64-thread optimum was established on Qwen3.5-397B-A17B, and GLM-5.2's different expert geometry (8 × 2048 intermediate) may shift it. Larger `-b`/`-ub` (2048–4096) amortize CPU-offload cost for MoE but grow compute-buffer VRAM.

**Caveats stated.** Per-layer expert sizes and non-expert totals are parameter counts × assumed bpw, not a GGUF dump; ground truth comes from `llama-gguf-hash`/`gguf_dump.py` or the per-device memory breakdown llama.cpp prints at load. The KV figure depends on the MLA rope dimension, which was not verified — the load log gives it exactly.

**Sources.** ~47 links cited. Technically load-bearing: unsloth/GLM-5.2-GGUF (UD-Q4_K_XL tree + benchmark discussion #3), Unsloth GLM-5.2 docs, zai-org/GLM-5.2 config.json (plus GLM-5/5.1/4.5/4.6/4.7 configs), llama.cpp PR #19460 (GLM MoE DSA support; indexer not yet supported), llama.cpp Issue #24730 (GLM 5.2 support request), llama.cpp Discussion #13154 (`-ot` documentation), llama.cpp Discussion #18049 (automation for GPU layers/tensor split/overrides with MoE optimizations — the fit machinery), ubergarm/GLM-5.1-GGUF discussion #8 (draft DSA PR), the Doctor-Shotgun llama.cpp MoE offload guide (HF blog + gist), ik_llama.cpp hybrid CPU/GPU docs, cedric_chee on UD-Q4_K_XL/Q5_K_XL being essentially lossless, vLLM/SGLang/NVIDIA GLM-5.2 recipe pages, and GLM-5.2 release coverage (June 2026, 1M-token context, open-weight; REAP-pruned variants pipenetwork GLM-5.2-REAP50-Q3_K_M and 0xSero GLM-5.2-REAP-504B noted).

**Observations**

- **Correction:** some guides claim `-ncmoe` counts from the highest layers; reading `common/arg.cpp` settled it — it keeps the experts of the *first* N layers on CPU.
- **Correction:** "CUDA1"-style device names were assumed initially; verified that HIP builds name devices `ROCm0..ROCm3`.
- **Correction (of the post's framing):** the 200%+ headline came from a dense model with whole blocks evicted to CPU; that regime does not exist on Galactus, which already runs `--cpu-moe`. "Not 200%. Anyone promising that number is describing a different starting point than yours."
- The post's insight is a year old and already absorbed into llama.cpp (`--cpu-moe`/`-cmoe`, `--n-cpu-moe`, the default-on `--fit` auto-fitter). The remaining opportunity: `--cpu-moe` leaves ~90 GB of Galactus's 122 GB VRAM idle.
- Two structural consequences: blk.78 costs nothing (TENSOR_SKIP; expect `model has unused tensor ... -- ignoring` in the load log); the shared expert stays on GPU automatically (`ffn_*_shexp` does not match `_exps` regexes). Warning recorded: a lazy `-ot "ffn_.*=CPU"` would sweep the shared expert and router onto CPU and cost real throughput, since those fire on every token.
- **Hypothesis:** Paul's observed DeepSeek-V4-Flash tg (7.16 t/s) sits below the ~14.7 t/s bandwidth ceiling computed for this class of workload, so something besides expert-weight traffic is also limiting throughput.
- **Prediction:** prefill ~57 tokens/s for a 2048-token batch on CPU ("aligns with the DeepSeek-V4-Flash numbers"); decode CPU-side ceiling ~14.7 tokens/s at the assumed ~160 GB/s.
- **Prediction:** ~14–16 of 75 MoE layers fit in VRAM; moving 16 layers cuts per-token CPU expert bytes ~21% → ~1.27× on the CPU-bound portion; realistically a 15–25% uplift, i.e. ~1.2–1.3× on tg, similar or slightly better on pp.
- **Decision:** `-ncmoe` by itself is not usable on this 4-GPU box (would OOM GPU3 at ~85 GiB). Plan: (1) fitter first via `llama-fit-params`; (2) manual per-device `-ot` placement as fallback, starting at 3 expert layers per card and walking up; (3) `llama-bench` sweeps of `-ncmoe`/`-ot`/threads.

**State of knowledge at end of session**

- GLM-5.2 geometry established: 78 layers (blk.0–77), first 3 dense, 75 MoE with 256 routed experts (8 active) plus 1 shared expert; hidden 6144, moe_intermediate 2048; MLA attention with kv_lora_rank=512; blk.78 is an MTP/NextN block that llama.cpp TENSOR_SKIPs at zero memory cost.
- File: Unsloth UD-Q4_K_XL, 467 GB in 11 shards (~434.9 GiB), ~4.94 bpw average; routed experts ≈ 97% of parameters, ~5.0–6.0 GiB per MoE layer (estimate, not a GGUF dump).
- Galactus has ~122 GiB usable VRAM; `--cpu-moe` leaves ~90 GiB idle; ~14–16 expert layers estimated to fit after non-expert weights, KV, and compute buffers.
- llama.cpp semantics pinned from source: `-ncmoe` counts the first N layers; the auto-fitter is default-on but aborts if `-ngl`/`-ts`/`-ot`/`-ncmoe` is set; `-ot` is substring-regex, first match wins; devices are ROCm0–3; llama-bench inverts the comma/semicolon separators.
- `-ncmoe` alone is unusable here: default splits would pile ~85 GiB of experts onto GPU3 (30.7 GiB card).
- Predicted uplift from filling VRAM: ~1.2–1.3× on tg over the `--cpu-moe` baseline — not the post's 200%.
- Theoretical CPU-side numbers on the table (later revised): pp ~57 t/s at batch 2048; tg ceiling ~14.7 t/s at an assumed ~160 GB/s.
- No commands executed on Galactus yet; prior reference point is DeepSeek-V4-Flash at 7.16 t/s with `--cpu-moe`.

