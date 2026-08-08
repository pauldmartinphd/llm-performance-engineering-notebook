# Galactus Lab Notebook — GLM-5.2 Throughput Investigation

**Period covered:** Sunday, July 13, 2026 – Tuesday, July 21, 2026
**Investigator:** Paul Martin
**Subject:** Characterizing and improving the inference throughput of GLM-5.2 (753.86 B-parameter mixture-of-experts model) running under llama.cpp/ROCm on the server "Galactus"
**Compiled:** July 24, 2026, from the original conversation exports, benchmark logs, and the automated diagnostic run log. All timestamps are US Eastern Time. Nothing in this notebook is reconstructed from memory; every command and number is taken from the primary records listed in the source table below.

---

## Headline results

| Metric | Start (7/14, 07:00) | End (7/21) | Change |
|---|---|---|---|
| Prefill, pp8192-class workload | 37.63 t/s (pp512, clamped ubatch) | 119.36 ± 0.12 t/s (pp8192, patched llama.cpp, ub 8192) | ≈ 3.2× |
| Decode (short context) | 5.15 t/s (tg128, -t 64) | 6.01 t/s (tg128, fitter config; 5.51–5.54 t/s in the -ot exps=CPU config) | +17% (config-dependent) |
| Platform memory bandwidth | unknown; NUMA suspected | 152 GB/s measured (STREAM, RFO-corrected), saturating at 16 threads | settled |
| Root cause of prefill ceiling | unknown | op_offload scheduler pins all MoE offload work to ROCm0 (731 of 1,186 GPU splits) + per-split expert-ids synchronization | proven, patched |
| Decode ceiling | unknown | DDR4 bandwidth wall: ~13.8 GB/token ÷ 152 GB/s ≈ 91 ms of the ~180 ms token budget | settled |

## System under test

| Component | Detail |
|---|---|
| CPU | AMD EPYC 7713 (Zen 3 "Milan"), 64 cores / 128 threads, single socket, 8 CCDs × 8 cores, 32 MB L3 per CCD. AVX2 + FMA; no AVX-512, no VNNI |
| RAM | 1 TB DDR4 — 8 × 128 GB 3DS RDIMMs, 8 channels, rated 2666 MT/s, configured (overclocked) to 2933 MT/s. NPS1 (single NUMA node) |
| GPUs | 4 × AMD Radeon Pro V620 (gfx1030, RDNA2), 30,704 MiB VRAM each = 122,816 MiB total; PCIe 4.0 ×16 each; VMM: no |
| Storage | Local ZFS on SATA SSD array; dd direct 1.5 GB/s, buffered 2.1 GB/s (verified not a bottleneck) |
| Host | Proxmox (kernel 7.0.14-4-pve at the time of the v3 run); llama.cpp runs in LXC container "openwebui" (Debian 13 "trixie") with full capabilities and no cgroup limits |
| GTT pool | 497.8 GiB (half of RAM, amdgpu default) — relevant to pinned-allocation failures |
| llama.cpp | Baseline: build f84a51940 (9942), backend ROCm+ZenDNN. From 7/14 mid-day: build 657e01125 (10001), rebuilt with -DGGML_ZENDNN=OFF. From 7/21: build 10001 plus the two-edit scheduler patch developed in Session 9 |

## Model under test

GLM-5.2 (Zai Org), Unsloth UD-Q4_K_XL quantization: 11 GGUF shards, 435.19 GiB on disk, 753.86 B parameters, ~4.96 bits per weight. llama.cpp architecture `glm-dsa`. 79 blocks: blk.0–2 dense, blk.3–77 MoE (75 layers), blk.78 MTP/NextN (TENSOR_SKIP — never allocated, so `--spec-type draft-mtp` cannot work). 256 routed experts, 8 active per token, 1 shared expert. MLA attention (kv_lora_rank 512, q_lora_rank 2048), DSA lightning indexer, 1 M context. Expert tensors per MoE layer: ffn_gate_exps 1728 MiB Q4_K, ffn_up_exps 1728 MiB Q4_K, ffn_down_exps 2112 MiB Q5_K (exceptions: blk.8 Q5_K/Q6_K; blk.75–77 Q6_K down). Decode reads ≈ 13.77 GB per token in the hybrid config; ≈ 27 GB per token CPU-only.

## Conventions used in this notebook

Entries are ordered by wall-clock time and grouped into nine work sessions. `### HH:MM — title` marks an entry; commands appear verbatim in fenced blocks; benchmark rows are reproduced as llama-bench printed them. Inline labels mark the epistemic status of statements at the time they were made: **Hypothesis**, **Prediction**, **Confirmed**, **Refuted**, **Dead end**, **Decision**, **Correction**. "STREAM, RFO-corrected" means Scale ×1.5 and Add/Triad ×4/3 to account for read-for-ownership traffic that STREAM does not count (Copy is compiled to non-temporal stores and needs no correction). The v3 diagnostic run (Session 4) executed unattended from 11:08 to 14:25 on 7/14 while the dialogue of Sessions 3 and 5 continued; its phases are presented as a block in wall-clock position, with per-phase times reconstructed from the log's elapsed stamps.

## Primary sources

| File (as kept in `Galactus Testing/`) | Saved (ET) | Content |
|---|---|---|
| Claude-Applying concepts to Galactus with GLM5.2.md | exported 7/16 14:45 | Main conversation, 7/13 22:43 – 7/16 14:41 (Sessions 1–3, 5–8) |
| galactus_triad.txt | 7/14 08:02 | STREAM thread sweep, 16–128 threads, bare-metal host |
| galactus_dmidecode_memory.txt | 7/14 08:05 | DIMM population: 8 × 128 GB, 2666 rated / 2933 configured |
| galactus_kernel_tuning.txt | 7/14 08:11 | THP always + defer+madvise; C-states disabled |
| galactus-diag.sh | 7/14 11:05 | v3 diagnostic script (16 phases) as launched at 11:08 |
| results.txt | 7/14 14:33 | MAIN.log of diag-20260714-110813 (the full v3 run, 175,463 lines) |
| newtest.txt | 7/14 18:13 | Failed -ot placement llama-bench log (comma-parse OOM) |
| Claude State Export.zip | 7/21 06:53 | openwebui system prompt + knowledge files (context setup) |
| llamacpp_patch.md | exported 7/21 07:37 | Patch-development session transcript (Session 9) |

---
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

**Decode bandwidth.** tg128 = 5.15 t/s = 194 ms/token. Per token: 8 experts × 3 matrices × 6144 × 2048 ≈ 302 M params per MoE layer; × 75 CPU-resident layers at ~4.5–5.0 bpw = ~13–14 GB/token → effective ~65–72 GB/s (byte figures ±15% due to the mixed-precision UD layout). Against 204.8 GB/s theoretical for 8-channel DDR4-3200 and "typically 150–170 GB/s achievable" (a claim corrected at 07:56): "You are at roughly a third of peak." Causes proposed, in order: (1) missing repack; (2) NUMA — `--physcpubind` sets no memory policy, and with `-mmp 0` weights fault in under the default local-node policy; if the BIOS were NPS2/NPS4, a large fraction of 435 GiB could sit on the wrong node. "Run `numactl --hardware` first." If >1 node: prepend `numactl --interleave=all` and add `--numa numactl`. Also: confirm 0–63 are distinct physical cores via `lscpu -e | head -70` or `cat /sys/devices/system/cpu/cpu1/topology/thread_siblings_list`.

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

Want 8 or 16 populated DIMMs at `Configured Memory Speed: 3200 MT/s`. If 4 DIMMs, or 2666/2400 MT/s: peak is 102 or 170 GB/s rather than 204.8, and 69 GB/s effective is an ordinary 65–70% — "no software bug exists and this whole line of inquiry closes."

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

"All four converge on ~150 GB/s. That is 73% of the 204.8 GB/s theoretical for eight channels of DDR4-3200 — a healthy, normal figure, and it confirms all eight channels are populated and running at speed. Skip the dmidecode; the memory subsystem is fine." Caveat (marked): the NT/RFO reading is inference from the shape of the numbers, not verified in the binary — but Copy at 151.8 GB/s is a hard floor either way, and llama.cpp weight reads are pure-read, which on Milan typically runs above Triad: "call the ceiling ~150–170 GB/s."

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

The full sweep was decoded. The RFO artifact is provable from the numbers: if Copy paid RFO, its real traffic would be 227.7 GB/s — above the 204.8 GB/s theoretical ceiling, which is impossible. So GCC turned the Copy loop into `memcpy`, glibc used non-temporal stores (threshold ~192 MB), and Copy's counted bytes equal its actual bytes; Scale/Add/Triad keep ordinary vector stores and pay uncounted RFO. Applying the corrections (Copy ×1.0, Scale ×1.5, Add/Triad ×1.333):

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
- Memory ceiling measured, not assumed: STREAM with RFO correction converges on ~150 GB/s on the host and 152 GB/s in the container (73% of the 204.8 GB/s theoretical); bandwidth saturates at 16 spread threads (16→64 −4%, 16→128 −6%); the container is exonerated (cgroups: `cpu.max = max 100000`, `cpuset.cpus.effective = 0-127`, `memory.max = max`).
- The decode budget: 201 ms/token actual versus ~126–128 ms ideal (CPU experts 13.9 GB @ 152 GB/s ≈ 91–93 ms; GPU dense ~13.3 GB ≈ 33–35 ms) → ~73–75 ms/token unaccounted (~36%); prefill is separately CPU-compute-bound at 1.69 TFLOP/s.
- Corrected sizing: experts ≈ 4.92 bpw → 5.53 GiB per MoE layer, ~415 GiB resident; non-expert only ~13 GiB; ~107 GiB VRAM idle (17–18 expert layers would fit) — deliberately deferred by Paul until throughput is understood.
- Ruled out this session: NUMA misplacement, container throttling, HIP-graphs-off, ZenDNN as a factor (inert for 256-expert Q4_K), `GGML_CUDA_REGISTER_HOST` (dead code), `-sm tensor` (unsupported for glm-dsa), and MTP self-speculation (blk.78 TENSOR_SKIP).
- Leading open hypotheses, mapped to experiments: hybrid-split overhead (75 fork/joins, ~150 device boundaries, ~375 barriers, ~2,000 kernel launches per token) → the `-ngl 0` versus hybrid comparison, predicted ~5.5 t/s pure-CPU; CCD/fabric placement → the 16-spread versus 16-packed `-C` mask pair; clock throttling → turbostat on the host; op_offload OOM → the ascending ubatch ladder plus unified-memory escape hatch.
- Host kernel state: THP `always` + `defer+madvise`, C-states below C0 disabled (08:11–08:13); galactus-diag.sh v2 (Phase 0 inventory + fourteen benchmark phases B0–B12, one 17-minute model load each, no deadline, 6-hour per-phase hang guard, MODE=all|host|bench) launched in the container at ~08:46, with the host companion run prescribed.

# Galactus / GLM-5.2 Performance Investigation — Lab Notebook, Part 2 of 3

Continues from Part 1, which ended with the v2 diagnostic battery launched on the morning of July 14, 2026. Hardware under test: "Galactus" — AMD EPYC 7713 (64C/128T Zen3), 1 TB DDR4-2933 8-channel, 4× AMD Radeon Pro V620 (gfx1030, 30.7 GiB each), llama.cpp ROCm build, LXC container "openwebui" on Proxmox. Model under test: GLM-5.2, Unsloth UD-Q4_K_XL, 435.19 GiB, 753.86 B parameters.

---

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

# Galactus / GLM-5.2 Performance Investigation — Lab Notebook, Part 3 of 3

Machine: "Galactus" — AMD EPYC 7713 (64C/128T, Zen 3), 1 TB DDR4-2933 (8-channel, ~152 GB/s), 4× AMD Radeon Pro V620 (gfx1030, 30.7 GiB VRAM each, 122 GiB total), llama.cpp ROCm build, LXC container "openwebui" on Proxmox. Model: GLM-5.2 (glm-dsa), Unsloth UD-Q4_K_XL, 435.19 GiB, 753.86 B parameters, 11-shard GGUF at `/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf`.

Part 2 ended mid-afternoon on July 14 with the v4 diagnostic (the ubatch ladder) running. Standing numbers at that point: decode 6.01 t/s (fitter configuration), prefill 40.11 t/s (pp512), and the first v4 row, pp8192 = 25.90 t/s at ub 512.

---

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

## Session 7 — Monday, July 14, 2026 (evening) — Multi-card placement experiments and the split histogram

### 18:13 — newtest.txt named; first read of an incomplete log

Paul pointed at the log by name ("Okay, its newtest.txt") before the file itself arrived. What was reachable showed the run had not finished — the log stopped mid-load at `blk.78` skip messages, with no result row, no `load_tensors:` buffer summary, no error, no exit marker. Two findings from the partial capture:

1. The `-ot` rules were accepted verbatim:

```
tensor blk.3.ffn_gate_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm1
tensor blk.3.ffn_down_exps.weight (2112 MiB q5_K) buffer type overridden to ROCm1
... blk.4, 5, 6 → ROCm1
```

Placement mechanism confirmed: `-ot` puts expert tensors exactly where pointed. "That was the precondition for the patch to mean anything, and it holds." Missing, however: the blk.7–10 → ROCm2 lines never appear.

2. The load was slow and unintentionally pinned:

```
done_getting_tensors: tensor 'token_embd.weight' cannot be used with preferred buffer type ROCm_Host, using CPU instead
```

`ROCm_Host` is the pinned buffer — a ~14-minute pinned load rather than the seconds-long mmap load intended. Proposed re-run:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.(3|4|5|6)\.ffn_.*_exps\.=ROCm1,blk\.(7|8|9|10)\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 \
  -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot2.txt
```

```bash
grep -E 'overridden to ROCm2' /root/ot2.txt | head        # did blk.7-10 land on ROCm2?
grep -E '^\| glm-dsa' /root/ot2.txt                        # the number
```

**Observations**

- Baseline to beat: `ub 4096 → 84.62` (all experts on CPU).
- **Prediction (branches):** above 84.62 → multi-card expert compute is real and the patch premise confirmed; unchanged → resident experts do not help prefill, do not patch; OOM → ROCm1/ROCm2 too full — informative in itself.
- Caveat restated: this tests resident experts (necessary but not sufficient for the patch, which routes streamed experts).

### 18:14 — newtest.txt attached: the OOM and the broken regex

The file arrived. **Artifact — `newtest.txt` (saved 18:13:59 ET):** the failed `-ot` llama-bench log. Its header reads `llama-bench: benchmark 1/3: starting` — the comma-separated `-ot` rules had been parsed as three separate benchmark configurations, so configuration 1 carried only the ROCm1 rule and lost its `exps=CPU` catch-all; the load then attempted to allocate 157272.74 MiB on ROCm1 and died out of memory. (The full anatomy of that parse emerged over the next several entries.)

```
allocating 157272.74 MiB on device 1: cudaMalloc failed: out of memory
failed to allocate ROCm1 buffer of size 164912416512
```

157 GiB onto ROCm1's 30 GiB card. The first diagnosis targeted the regex: `blk\.(3|4|5|6)` is unanchored — `blk.3` is a substring of `blk.34`, and `regex_search` matches anywhere — so the rule swept roughly 40 layers (blk.3/4/5/6 plus blk.30–39, blk.40–49, blk.53, blk.63, and so on) onto ROCm1, attempted 157 GiB, and died before reaching the ROCm2 rule (hence zero ROCm2 matches, and why the earlier capture showed only blk.3–6). Fixed with a literal-dot anchor:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.[3-6]\.ffn_.*_exps\.=ROCm1,blk\.[7-9]\.ffn_.*_exps\.=ROCm2,blk\.10\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 \
  -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot3.txt
```

```bash
grep -c 'overridden to ROCm1' /root/ot3.txt    # want exactly 12  (4 layers × 3 tensors)
grep -c 'overridden to ROCm2' /root/ot3.txt    # want exactly 12
grep -E '^\| glm-dsa' /root/ot3.txt            # the result
```

**Observations**

- **Dead end:** the OOM exposed a bug in the test design, not the machine; it does not prove multi-card resident experts fail — the experiment never ran.
- **Correction (self):** "I've now handed you two broken `-ot` regexes in a row" — the regex_search-matches-anywhere warning had been written into Paul's own diagnostic script three versions earlier.
- Expected footprint: 4 layers per card ≈ 22 GiB of experts each plus ~5 GiB attention share and streaming scratch, inside 30 GiB.

### 18:16 — ot3: "failed to load model," a six-line log

Paul ran ot3 verbatim. Output: the four-device init block (Total VRAM 122816 MiB; 4× AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM 30704 MiB each), then:

```
llama_bench: error: failed to load model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

with an empty result table. The real error line was requested:

```bash
grep -E 'out of memory|failed to allocate|overridden to ROCm|error loading' /root/ot3.txt | head -20
```

```bash
grep -E 'cudaMalloc|out of memory|failed to allocate ROCm|buffer of size|overridden to' /root/ot3.txt
```

and staged a two-layer-per-card fallback that "cannot OOM":

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.3\.ffn_.*_exps\.=ROCm1,blk\.4\.ffn_.*_exps\.=ROCm1,blk\.5\.ffn_.*_exps\.=ROCm2,blk\.6\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot4.txt
```

```bash
grep -c 'overridden to ROCm1' /root/ot4.txt   # must be exactly 6
grep -c 'overridden to ROCm2' /root/ot4.txt   # must be exactly 6
grep -E '^\| glm-dsa' /root/ot4.txt
```

**Observations**

- **Dead end:** second consecutive load failure, error stage unknown pending the tee'd log; placement math checked out this time (`[7-9]` = 3 layers + `10` = 1 → 4 layers per card).
- **Prediction (branches):** `allocating ~157000 MiB` again → the regex still over-matching; a smaller number (~25–30 GiB) → placement correct but 8 resident layers genuinely do not fit.

### 18:17 — The six-line log reinterpreted: parse failure, not allocation

`cat /root/ot3.txt` showed the entire log was the init block, the error line, and the empty table header — no placement lines, no allocation messages. Course corrected: this was not the 157 GiB over-match — the run died earlier, at argument parsing; the `-ot` string never took effect. Isolation test with a single rule:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.3\.ffn_.*_exps\.=ROCm1,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot5.txt
```

```bash
grep -cE 'overridden to ROCm1' /root/ot5.txt   # want 3
grep -E '^\| glm-dsa|error|out of memory' /root/ot5.txt
```

```bash
llama-bench --list-devices
```

**Observations**

- **Correction:** the failure stage moved from allocation to parsing; the four-clause `-ot` string was suspected of hitting a parse limit or malformed-rule rejection.
- Open question flagged: does `-ot ...=ROCm1` even parse on this build (newtest.txt had shown "overridden to ROCm1," so it parsed before); stray characters from retyping also flagged.

### 18:18 — Paul forces the lookup; the comma/semicolon inversion found in the parser

Paul: "Well can you go look this up?  I think you're supposed to use | | | or , , ,". The llama-bench parser, read directly, gave the root cause — `-ot` uses two delimiter levels:

- comma `,` separates benchmark configurations — each comma-group becomes a separate run (line 901: `strcspn(value, ",")`);
- semicolon `;` separates rules within one configuration (line 930: `strcspn(override_group, ";")`).

The earlier comma string had therefore been parsed as three separate benchmark runs, each with one rule. Run 1 put blk.3–6 (unanchored, ~40 layers) on ROCm1 with nothing on CPU — the other experts defaulted to VRAM — hence the 157 GiB OOM recorded in newtest.txt. "Commas made three broken configs instead of one correct config." This is the inverse of `llama-server`/`llama-cli`, where `-ot` rules are comma-separated; that exact inversion had been noted in the v3 script briefing and the command then written with commas anyway — the same mistake twice. Device names come from `ggml_backend_buft_name(buft)` (line 897); a wrong name produces `error: unrecognized buffer type` plus a valid list (line 947).

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.[3-6]\.ffn_.*_exps\.=ROCm1;blk\.[7-9]\.ffn_.*_exps\.=ROCm2;blk\.10\.ffn_.*_exps\.=ROCm2;exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot6.txt
```

```bash
grep -c 'overridden to ROCm1' /root/ot6.txt   # 12  (blk.3,4,5,6 × 3 tensors)
grep -c 'overridden to ROCm2' /root/ot6.txt   # 12  (blk.7,8,9,10 × 3 tensors)
grep -E '^\| glm-dsa' /root/ot6.txt
```

**Observations**

- **Correction (root cause, from source):** the comma/semicolon inversion, not a parse limit.
- Expected to load without OOM (~22 GiB experts per card plus attention and scratch, inside 30 GiB). Baseline to beat: 84.62.

### 18:19 — ot6: "failed to create context"

Paul ran ot6 verbatim; the same four-device init block, then:

```
llama_bench: error: failed to create context with model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

The changed error read as progress — "failed to create context" happens after the model loads; the semicolons fixed parsing, placement worked, and the run died building the compute context. Requested detail:

```bash
grep -E 'overridden to ROCm|error|failed|out of memory|cudaMalloc|buffer' /root/ot6.txt | head -40
```

**Prediction:** with expert layers resident on ROCm1/ROCm2 and op_offload on (`-nopo 0`) at `-ub 4096`, there is no room for the compute buffer — the weights fit; the working memory for a 4096-token batch does not. Fix step 1, drop the ubatch:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.[3-6]\.ffn_.*_exps\.=ROCm1;blk\.[7-9]\.ffn_.*_exps\.=ROCm2;blk\.10\.ffn_.*_exps\.=ROCm2;exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 2048 -ub 2048 -p 2048 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot7.txt
```

Baseline at ub 2048: 62.64. Fix step 2 if 2048 still fails — halve the resident layers:

```
-ot "blk\.3\.ffn_.*_exps\.=ROCm1;blk\.4\.ffn_.*_exps\.=ROCm1;blk\.5\.ffn_.*_exps\.=ROCm2;blk\.6\.ffn_.*_exps\.=ROCm2;exps=CPU"
```

**Observations**

- Insight flagged as central: resident experts and op_offload scratch compete for the same VRAM — the same ceiling would apply to the patched multi-card path. The `-ts 0,1,1,1` idea (ROCm0 kept empty as scratch) reiterated.

### 18:20 — The grep returns only the error line; two structural hypotheses

Paul's grep of ot6.txt returned exactly one line — the error itself; no "overridden to ROCm" lines at all, though in ot3.txt placement lines had appeared during load. The unfiltered view was requested:

```bash
wc -l /root/ot6.txt
sed -n '6,40p' /root/ot6.txt
```

and a narrowing grep:

```bash
grep -iE 'overridden|split|assigned to device|compute buffer|graph|context|reserve|cudaMalloc|out of memory|GGML_ASSERT|abort' /root/ot6.txt
```

**Observations**

- **Hypothesis (branch 1):** placement lines present → compute-buffer OOM as predicted; `-ub 2048` fixes it.
- **Hypothesis (branch 2):** no placement lines → a structural failure: `-ot` puts layer 3–10 experts on ROCm1/2 while those layers' attention stays on ROCm0 (the default `-ngl` split) — the graph builder may reject a layer spread across devices. Noted as informative for the patch, which separates weight location from run location in the same way.

### 18:24 — ot7 at ub 2048 runs: pp2048 = 51.74 — the negative result

The experiment finally executed (build 657e01125 (10001)):

```
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |     2048 |   1 | blk\.[3-6]\.ffn_.*_exps\.=ROCm1;blk\.[7-9]\.ffn_.*_exps\.=ROCm2;blk\.10\.ffn_.*_exps\.=ROCm2;exps=CPU |          pp2048 |         51.74 ± 0.00 |
```

pp2048 = 51.74 with 8 expert layers resident on ROCm1/ROCm2, against the all-CPU baseline of 62.64 at ub 2048: resident experts on multiple cards made prefill worse, by 17%. The `-ub 4096` context failure was thereby confirmed as compute-buffer OOM — the cards are tight.

**Observations**

- **Hypothesis (later retracted):** the 8 resident layers force activations to hop ROCm0→ROCm1, compute, and hop back, per layer per token, through the host bridge (`Hops: 2`, no XGMI) — "that is exactly what my patch would trigger, at scale."
- **Decision (later reversed):** "Do not build the patch."
- Standing best reaffirmed: ub 8192, all experts CPU, op_offload on → 104.97 t/s; "the single-card concentration that I thought was a bug to fix is, on this hardware, close to optimal... The 105 t/s you already have is, most likely, the ceiling." `-ts 0,1,1,1` still noted as a different mechanism worth one run.

### 18:25 — Paul pushes back; the P2P retraction

Paul: "Can you look into anything I can do here.  I have four cards.  Prefill should be faster and parallelizable.  I also think we should look into that p2p routing maybe?" The source, read again, forced a retraction of the earlier explanation: op_offload doesn't do GPU-to-GPU copies at all. In op_offload the weight is on the host, so the transfer is host→device; `GGML_CUDA_P2P` gates GPU↔GPU only (lines 334-348) and is irrelevant to the offload path. What the `-ot` experiment actually did: op_offload fires only for CPU-resident weights (`ggml_backend_buffer_is_host(src->buffer)`, line 919); the resident experts on ROCm1 are not host, so those 8 layers ran as ordinary resident-GPU matmuls while their attention stayed on ROCm0 — the hidden state crossed ROCm0→ROCm1 and back via `cudaMemcpyDeviceToDevice` (lines 754-756), which does need peer access, not enabled. The experiment created GPU↔GPU traffic through the resident-weight path, not the offload path. It tested the wrong mechanism.

Proposed tests, in order — first a 30-second P2P capability probe:

```bash
cat > /tmp/p2p.c <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
int main(void){int n;hipGetDeviceCount(&n);
for(int i=0;i<n;i++)for(int j=0;j<n;j++){if(i==j)continue;
int c=0;hipDeviceCanAccessPeer(&c,i,j);
printf("GPU%d->GPU%d: %s\n",i,j,c?"YES":"no");}return 0;}
EOF
hipcc /tmp/p2p.c -o /tmp/p2p && /tmp/p2p
```

then whether `GGML_CUDA_P2P=1` changes the existing offload number (source says it should not; confirm):

```bash
GGML_CUDA_P2P=1 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 -o md
```

then the still-undone `-ts 0,1,1,1` run. Both follow-ups were issued with tee targets:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 -o md 2>&1 | tee /root/ts.txt
```

```bash
GGML_CUDA_P2P=1 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 -o md
```

**Observations**

- **Correction:** "the `-ot` result does not actually condemn the patch... the experiment was not a valid test of the patch, and I called it decisive when it wasn't." The patch is untested, not disproven.
- Re-verified on build 657e01125: the offload path is host→device only; the copy falls through to the blocking path (lines 514-517) because `cpy_tensor_async` returns false for a CPU source. The bottleneck is serialization, not peer bandwidth.
- Self-assessment on the record: three wrong or oversold mechanism claims this session. The measured truth: prefill is host→device-bound, serialized, one card — 22 GB/s against ~89 GB/s of installed lane bandwidth.

### 18:26 — hipcc fails on a .c file; workarounds

The probe would not compile:

```
failed to execute:/opt/rocm-7.2.0/lib/llvm/bin/clang++  -O3 --driver-mode=g++ -O3 --hip-link  -x c /tmp/p2p.c -o "/tmp/p2p"
```

(establishing ROCm 7.2.0 on the box). Diagnosis: a hipcc quirk — `-x c` fights `--driver-mode=g++`. Fixes offered:

```bash
cp /tmp/p2p.c /tmp/p2p.hip
hipcc /tmp/p2p.hip -o /tmp/p2p && /tmp/p2p
```

```bash
/opt/rocm-7.2.0/lib/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx1030 \
  /tmp/p2p.c -o /tmp/p2p \
  -L/opt/rocm-7.2.0/lib -lamdhip64 && /tmp/p2p
```

No-compile alternatives:

```bash
rocm-smi --showtopo
```

```bash
rocm-bandwidth-test -t   # topology + peer access matrix
```

Recalled hardware fact: the earlier `--showtopo` showed `Link Type: PCIE, Hops: 2` for the pairs; the single needed bit is whether any GPU pair reports peer access possible.

**Observations**

- **Dead end (tooling):** hipcc mis-detects `.c` files; the `.hip` extension or an explicit `-x hip --offload-arch=gfx1030 -lamdhip64` invocation works around it. ROCm version on the box established: 7.2.0.

### 18:27 — P2P capability: all twelve directions YES

Paul reported "2 warnings generated when compiling for host," then:

```
GPU0->GPU1: YES
GPU0->GPU2: YES
GPU0->GPU3: YES
GPU1->GPU0: YES
GPU1->GPU2: YES
GPU1->GPU3: YES
GPU2->GPU0: YES
GPU2->GPU1: YES
GPU2->GPU3: YES
GPU3->GPU0: YES
GPU3->GPU1: YES
GPU3->GPU2: YES
```

**Observations**

- **Refuted:** every pair: YES. That kills the "cards can't talk to each other" claim outright. The patch is not dead — untested. It was also now unknown why the `-ot` experiment lost, the P2P explanation being disproven.
- Capability is not bandwidth: four cards on separate root complexes through the EPYC IOD fabric could peer anywhere from 3 GB/s to 25 GB/s. Bandwidth probes queued:

```bash
rocm-bandwidth-test 2>/dev/null | head -60 || echo "not installed"
```

```bash
cat > /tmp/pbw.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#define N (256*1024*1024)
int main(){
  hipSetDevice(0); hipDeviceEnablePeerAccess(1,0);
  void *a,*b; hipSetDevice(0); hipMalloc(&a,N);
  hipSetDevice(1); hipMalloc(&b,N);
  hipSetDevice(0);
  for(int w=0;w<3;w++) hipMemcpyPeer(b,1,a,0,N);
  hipDeviceSynchronize();
  hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s);
  for(int i=0;i<20;i++) hipMemcpyPeer(b,1,a,0,N);
  hipEventRecord(e); hipEventSynchronize(e);
  float ms; hipEventElapsedTime(&ms,s,e);
  printf("GPU0->GPU1 peer: %.1f GB/s\n", 20.0*N/1e9/(ms/1000.0));
  return 0;
}
EOF
hipcc /tmp/pbw.hip -o /tmp/pbw && /tmp/pbw
```

- **Decision thresholds:** ≥15 GB/s → peer traffic cheap, tensor-parallel expert compute viable, "path to ~300–400 t/s prefill is real"; 3–8 GB/s → careful per-scheme testing; near host-memory rate → little advantage. The `-ts` run (ts.txt) and `GGML_CUDA_P2P=1` control (p2ptest.txt) were re-issued verbatim in parallel.

### 18:29 — Single-pair peer bandwidth: 16.2 GB/s; ts.txt missing

Paul: "GPU0->GPU1 peer: 16.2 GB/s" — above the ≥15 threshold, "not the 3-8 GB/s host-relay penalty I feared." Four-card expert compute back on the table on measured evidence. Two unknowns remained: aggregate peer bandwidth with all four cards peering simultaneously through the shared IOD (could scale to ~64 GB/s or contend down to ~20), and how much peer traffic each scheme generates. Patch math sketch: spread 438 GB of expert streaming across four host→device links; peer cost = activation hops — at 16 GB/s, a 100 MB activation per layer ≈ ~6 ms, across 75 layers ≈ ~450 ms of peer traffic per token if every layer hops. The queued results were requested:

```bash
grep -E '^\| glm-dsa' /root/ts.txt
```

```bash
grep -E '^\| glm-dsa' /root/p2ptest.txt
```

(~105 expected for the P2P control, since offload is host→device.) The response:

```
root@openwebui:~# grep -E '^\| glm-dsa' /root/ts.txt
grep: /root/ts.txt: No such file or directory
```

The queued commands had not run. A check was proposed and the commands re-issued:

```bash
ls -la /root/ts.txt /root/p2ptest.txt 2>&1
```

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ts.txt
```

```bash
grep -E '^\| glm-dsa|error|out of memory|failed' /root/ts.txt
```

plus the four-pair aggregate peer probe (concurrent pairs {0→1},{1→0},{2→3},{3→2}, async, 256 MiB, 20 iterations):

```bash
cat > /tmp/pbw4.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#define N (256*1024*1024)
int main(){
  int pairs[4][2]={{0,1},{1,0},{2,3},{3,2}};
  void *src[4],*dst[4]; hipStream_t st[4];
  for(int k=0;k<4;k++){int s=pairs[k][0],d=pairs[k][1];
    hipSetDevice(s); hipDeviceEnablePeerAccess(d,0); hipMalloc(&src[k],N);
    hipStreamCreate(&st[k]);
    hipSetDevice(d); hipMalloc(&dst[k],N);}
  for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]);
    for(int w=0;w<3;w++) hipMemcpyPeerAsync(dst[k],pairs[k][1],src[k],pairs[k][0],N,st[k]);}
  for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]); hipStreamSynchronize(st[k]);}
  hipSetDevice(0); hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s,0);
  for(int i=0;i<20;i++) for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]);
    hipMemcpyPeerAsync(dst[k],pairs[k][1],src[k],pairs[k][0],N,st[k]);}
  for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]); hipStreamSynchronize(st[k]);}
  hipSetDevice(0); hipEventRecord(e,0); hipEventSynchronize(e);
  float ms; hipEventElapsedTime(&ms,s,e);
  printf("4 concurrent pairs: %.1f GB/s aggregate\n", 4.0*20.0*N/1e9/(ms/1000.0));
  return 0;
}
EOF
hipcc /tmp/pbw4.hip -o /tmp/pbw4 && /tmp/pbw4
```

**Observations**

- **Prediction (`-ts` branches):** above 105 → ROCm0 was VRAM-contended and headroom exists; flat at 105 → the blocking serialized host→device copy is the wall; OOM → drop to ub 4096. Baseline: 104.97.
- **Prediction (pbw4 thresholds):** near 4×16 = 64 GB/s → the fabric scales and the patch has headroom; contending to ~20 → four cards streaming would choke each other.

### 18:31 — -ts 0,1,1,1 fails to create context: self-defeating by design

Paul ran the ts command verbatim; the same init block, then:

```
llama_bench: error: failed to create context with model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

The detail was requested:

```bash
grep -iE 'out of memory|cudaMalloc|failed to allocate|buffer of size|compute buffer|reserve' /root/ts.txt
```

and identified the design flaw in its own experiment: `-ts 0,1,1,1` puts zero model layers on ROCm0, packing all 78 attention layers onto ROCm1/2/3 (26 each) — but op_offload still sends every streamed expert to ROCm0 (the `return b` = backend 0 behaviour). ROCm1–3 cannot fit 26 attention layers each plus the ub 8192 compute buffer, so context creation fails. "`-ts 0,1,1,1` is self-defeating: it frees the wrong card." A fallback was staged to at least get a number:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 4096 -ub 4096 -p 4096 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ts2.txt
```

**Observations**

- **Correction (own experiment design):** the third flag-level failure in a row.
- **Decision:** all three experiments (`-ot`, `-ts`, the retry) hit the same wall — "op_offload's ROCm0 hardcoding means no `-ts`/`-ot` juggling can spread the streaming load. Only code can." Flag-level experiments declared exhausted. Remaining: (a) the pbw4 aggregate number, the deciding fact; (b) the patch itself, only if justified, tested functionally at low ubatch first.

### 18:31 — Aggregate peer bandwidth: 49.4 GB/s

```
22 warnings generated when compiling for host.
4 concurrent pairs: 49.4 GB/s aggregate
```

49.4 GB/s aggregate against 16.2 single-pair: the fabric scales at ~76% efficiency across concurrent transfers and does not collapse to ~20. Measured-facts scoreboard at this point: peer access YES in all 12 directions; single-pair peer 16.2 GB/s; four-pair aggregate 49.4 GB/s; current prefill = one card, one link, 22 GB/s host→device, serialized.

Before writing the patch, the mechanism was verified — does `src[1]` (the activation) have a resolved buffer at graph-assignment time? The source answered no: lines 1047-1050 walk the nodes calling `backend_id_from_cur` without pre-assigning sources — the src-assignment block at lines 1052-1066 is `#if 0` dead code. `src[1]` therefore has `tensor_backend_id == -1` when the heuristic runs; the original activation-routing patch would read `asrc->buffer` as null and fall through to `return b` = ROCm0. A silent no-op. The scheduler's own design comment (line 1074): "ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known" — pass 2 (line 1078+) expands GPU assignments to adjacent nodes, but the offload heuristic at line 919 fires in pass 1 and hardcodes backend 0 before pass 2 runs.

A no-code diagnostic was ordered first:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 2>&1 | grep -E 'ffn_moe|split|ROCm[0-9]|off' | head -60 > /root/sched.txt
wc -l /root/sched.txt
head -60 /root/sched.txt
```

**Observations**

- **Prediction (later corrected):** not a clean 4× to 415, but "250-350 t/s is plausible, versus 105 today."
- **Confirmed (against own patch):** "it confirms my patch is broken" — the drafted activation-routing patch cannot work as written.
- **Decision:** the correct patch is narrower — either defer (return -1, let pass-2 expansion place the op with its layer; risk of landing on CPU) or round-robin the target backend (cruder, predictable).

### 18:35 — sched.txt is empty; the debug format discovered

The diagnostic produced `0 /root/sched.txt`. Zero lines meant grep mismatch or uncaptured output, not "no splits." Unfiltered capture proposed:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_full.txt 2>&1
wc -l /root/sched_full.txt
grep -icE 'split|SPLIT|backend|ROCm|CPU' /root/sched_full.txt
```

```bash
grep -inE 'split #|## SPLIT|Split|node.*backend|=.*ROCm|=.*CPU|1\.off|1\.wgt' /root/sched_full.txt | head -40
```

Source reading established the format: `## SPLIT #N: <backend>` plus per-node lines `node #N (op): name [BACKEND cause]` with cause tags `1.off` / `1.wgt` — and that the dump goes through `GGML_LOG_DEBUG`, filtered unless log verbosity is debug. Fix — add `-v`:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_full.txt 2>&1
echo "total lines:"; wc -l /root/sched_full.txt
echo "split headers:"; grep -c '## SPLIT' /root/sched_full.txt
echo "which backends own the splits:"; grep '## SPLIT' /root/sched_full.txt | grep -oE 'ROCm[0-9]|CPU' | sort | uniq -c
echo "where the offloaded (1.off) ops landed:"; grep '1.off' /root/sched_full.txt | grep -oE '\[(ROCm[0-9]|CPU)' | sort | uniq -c
```

with `GGML_SCHED_DEBUG=2 GGML_LOG_LEVEL=4 llama-bench ...` as the fallback if `-v` alone did not surface it.

**Observations**

- **Hypothesis:** zero lines because the debug log is suppressed, not absent.

### 18:44 — Without -v: 10 lines; llama-bench's null log callback found

Paul ran the no-`-v` version: 10 lines total, 3 grep matches — just device-init lines. Reading the tool's source settled it: llama-bench installs a null log callback (`llama_null_log_callback`, line 2143) that discards all ggml logging unless `-v` is set; even the "slow copy" warning that fires on the blocking offload path goes through `GGML_LOG_DEBUG` (line 490) and is swallowed. "Not a flag problem — the tool suppresses it by design." The needed run, verbatim:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_full.txt 2>&1

wc -l /root/sched_full.txt
grep -c '## SPLIT' /root/sched_full.txt
grep '1.off' /root/sched_full.txt | grep -oE '\[ROCm[0-9]|\[CPU' | sort | uniq -c
```

**Observations**

- Caveat noted: at `-ub 512` the batch is 512, above the 32-token `GGML_OP_OFFLOAD_MIN_BATCH`, so offload should fire; if the `1.off` grep is empty, bump to `-p 2048 -ub 2048` and re-grep.

### 19:08 — With -v: 31,795 lines, 1,494 splits; the 1.off grep comes back empty

```
31795 /root/sched_full.txt
1494
```

The dump landed; the third command (the `1.off` grep) printed nothing — the pattern was wrong, not offload absent (the bracket format in source is `[%5.5s %8.8s]`, so the cause tag may render padded or truncated). Follow-up queries:

```bash
# what do the node lines actually look like? show a MoE expert node
grep -m5 'ffn_moe\|_exps' /root/sched_full.txt

# what cause tags exist, and how often?
grep -oE '1\.[a-z]+' /root/sched_full.txt | sort | uniq -c

# which backends own the 1494 splits?
grep '## SPLIT' /root/sched_full.txt | grep -oE 'ROCm[0-9]|CPU' | sort | uniq -c
```

**Observations**

- **Correction:** the empty `1.off` grep reflects a wrong pattern (the bracket format in source is `[%5.5s %8.8s]`, so the cause tag may render padded or truncated), not an absence of offload. The third command is the runtime answer: if ~1,400 of the 1,494 splits are ROCm0, the diagnosis holds; if evenly spread, the theory is wrong and the 105 ceiling has a different cause.

### 19:09 — Partial grep results: name fragments, and the ROCm_Host insight

After a stray keystroke ("v"), Paul pasted partial results. The first grep showed tensor lines:

```
tensor blk.3.ffn_gate_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm_Host
create_tensor: loading tensor blk.3.ffn_gate_exps.weight
tensor blk.3.ffn_down_exps.weight (2112 MiB q5_K) buffer type overridden to ROCm_Host
create_tensor: loading tensor blk.3.ffn_down_exps.weight
tensor blk.3.ffn_up_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm_Host
```

The cause-tag grep (run twice, same output):

```
    576 1.attn
     56 1.exp
    564 1.ffn
      1 1.gguf
     40 1.indexer
```

Reinterpreted: the `1.xxx` tokens are tensor name fragments (`blk.1.attn…`, `blk.1.ffn…`), not scheduler cause tags. Key observation: `buffer type overridden to ROCm_Host` — pinned, page-locked host RAM; with `-ot exps=CPU` the experts live in the pinned host buffer, on no GPU. Direct queries issued for the actual question:

```bash
# the actual question: of 1494 splits, which backend runs each?
grep '## SPLIT' /root/sched_full.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c

# and the node-level backend column (5th field in "[BACKEND cause]")
grep -oE '\[(ROCm[0-9]|CPU) ' /root/sched_full.txt | sort | uniq -c
```

**Observations**

- Clean measured facts restated while leveling with Paul: prefill 104.97 t/s at ub 8192 (2.6× over start); the offload copy blocking and serialized (source, lines 514-517); peer bandwidth healthy (16 GB/s single, 49 GB/s aggregate); every VRAM-layout flag trick hits the same wall. "Squeezing past 105 requires a code change to the scheduler."
- **Prediction (branches):** splits ~1,400 on ROCm0 → write the patch knowing it is real; distributed → the bottleneck is the blocking copy itself, and the fix is the harder `cpy_tensor_async` change.

### 19:10 — The split distribution: ROCm0 owns 731 of 1,494

```
root@openwebui:~# grep '## SPLIT' /root/sched_full.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
    308 CPU
    731 ROCm0
    133 ROCm1
    175 ROCm2
    147 ROCm3
root@openwebui:~# grep -oE '\[(ROCm[0-9]|CPU) ' /root/sched_full.txt | sort | uniq -c
  18811 [ROCm0 
  14188 [ROCm1 
  18922 [ROCm2 
  15874 [ROCm3
```

"The diagnosis holds, and now it's measured." ROCm0 owns 731 splits — 49% of all 1,494, 62% of the 1,186 GPU splits — versus ~133/175/147 for the other three cards. The node-level counts (18811/14188/18922/15874) are a red herring: they count resident attention nodes, distributed normally by the layer split. The splits expose op_offload: 731 splits on ROCm0 = 731 host→device transfers serialized onto one card's link = the measured 22 GB/s single-link bottleneck. Everything-agrees checklist: splits ROCm0 62% ✓; copy path blocking/serialized (source) ✓; streaming 22 GB/s = one link ✓; peer fabric 49 GB/s aggregate, healthy ✓; flag tricks cannot move it, the pin is in code ✓.

**Decision:** the patch is justified by measurement. The original activation-routing diff being a no-op, the correct narrow change is round-robin. Patch v1 — file `ggml/src/ggml-backend.cpp`, find around line 919:

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
                // Round-robin the offload target across all eligible GPU backends
                // instead of always taking the first. With a large CPU-resident
                // MoE this stops every offloaded expert matmul from serializing
                // onto backend 0's single PCIe link.
                int n_elig = 0;
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        n_elig++;
                    }
                }
                if (n_elig > 0) {
                    int pick = sched->off_rr % n_elig;
                    int seen = 0;
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            if (seen == pick) {
                                sched->off_rr++;
                                SET_CAUSE(tensor, "1.off");
                                return b;
                            }
                            seen++;
                        }
                    }
                }
            }
```

Second edit — add to `struct ggml_backend_sched {`:

```c
    int off_rr; // round-robin cursor for op_offload target selection
```

Zero-initialized if the struct is calloc'd (verify: `grep -n 'calloc\|ggml_backend_sched_new' ggml/src/ggml-backend.cpp`; if not, add `sched->off_rr = 0;` in init). Build and verify placement moved:

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

Then the number:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```

**Observations**

- **Prediction (histogram):** before, ROCm0 731 and others ~150; if the patch works, all four ~290 each. If ROCm0 is still 731, the patch did not take effect — do not measure throughput.
- **Prediction (throughput, later corrected):** the fitted model says a path to ~250–350 against the 104.97 baseline. Caveats: round-robin ignores locality (peer hops affordable at 49 GB/s aggregate); the copy is still blocking (async within a card is a separate, harder `cpy_tensor_async` patch); decode never triggers offload at batch 1 — 6.01 t/s untouched.

### 19:23 — A counter-analysis arrives

Paul pasted a detailed technical review from a separate analysis session (referred to below as the pasted review). Its claims, condensed:

- The patch is aimed at the right place, will compile, and should equalize the split histogram — but will not deliver 15 s → 4–5 s streaming, because that assumes four links transfer in parallel and the confirmed blocking copy path prevents it. `cpy_tensor_async` in the CUDA/HIP backend returns false unless both backends are GPU, so every expert-weight transfer goes through the synchronous fallback in `ggml_backend_sched_compute_splits`, blocking the single scheduler thread. The loop is sequential: copies to ROCm1 and ROCm2 cannot overlap regardless of destination. Round-robin alternates the destination link; it does not aggregate them. Streaming stays ~22 GB/s and ~15 s total.
- What the patch does buy: the fallback synchronizes the destination backend before copying — today each weight copy waits for ROCm0 to finish the previous expert GEMM. With targets spread, the destination is idle, so copy l+1 overlaps GEMM l. The GEMMs remain a sequential chain across layers (layer l+1 depends on layer l through attention), so four cards do not parallelize compute for a single ubatch either.
- Bound from the fitted decomposition: 8192 tokens at 104.97 t/s ≈ 78 s; perfectly hiding a 15 s streaming term yields ~63 s ≈ ~130 t/s. "Worth having, but not 250–350."
- The dramatic version requires concurrent copies (async host→device with a pinned source, `memcpyAsync` on each backend's own stream, an event before compute launch, or a copy-thread pool). Before writing that: a ten-minute standalone test — `hipMemcpyAsync` from pinned host to all four cards on four streams from one thread. If the root complex or RAM bandwidth caps aggregate H2D near 22–40 GB/s rather than ~88, no scheduler change can reach the target.
- On the patch itself: the struct is calloc'd in `ggml_backend_sched_new` in every known version (keep the grep step); make `off_rr` unsigned. Per-op round-robin splits gate/up/down of one layer across three cards — two extra multi-hundred-MB peer hops per layer at ub 8192 and roughly double the split count. Prefer keying on the layer: `sscanf(src->name, "blk.%d.", &il); pick = il % n_elig`, falling back to the cursor if parsing fails — same size, keeps each layer's three expert matmuls local, stable across ubatches, needs no struct field.
- Expect each of ROCm1–3 to newly allocate roughly one layer's expert staging in its compute buffer; check VRAM headroom on first run. Compare total split count before/after, not just shares. Decode untouched (`offload_op` requires batch ≥ 32). Caveat on its own claims: the `cpy_tensor_async` rejection and calloc details date from mid-2025 — confirm the fallback still performs the destination sync before the copy in this checkout, "that stall is where this patch's entire gain comes from."

**Observations**

- **Hypothesis (pasted review):** the realistic bound for round-robin is ~130 t/s, not 250–350 — the single blocking host thread prevents parallel transfer; the patch's entire gain is overlap of copy l+1 with GEMM l.
- **Decision seed (pasted review):** key the assignment on the layer index rather than a per-op cursor.

### 19:23 — Verification, adoption of layer-keying, and the h2d4 gate

The counter-analysis was verified against the current tree: lines 516-518 — `ggml_backend_tensor_copy_async` hits the blocking fallback for host→device (the async path at lines 508-509 needs both ends GPU); the split loop (~line 1583) does `ggml_backend_synchronize(split_backend)` before the copy when there is no event — each expert copy waits for that card to finish its previous GEMM; copy and compute fully serialize on ROCm0.

A new source finding altered the magnitude: a comment at line ~1587 — "when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used" — llama.cpp already sparse-copies offloaded experts (only the ~8-per-token actually routed, gated by the `used_ids` bitset). The "438 GB per ubatch" was the dense figure; the real transfer is far smaller, and the measured 22 GB/s is moving only active experts. This does not change the ~130 bound (derived from the measured 78 s wall clock, which already reflects sparse copying), but the streaming term is smaller than 15 s, so the overlap win is more modest — "possibly closer to ~120 than 130." The layer-keyed revision was adopted:

```c
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                int n_elig = 0;
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) n_elig++;
                }
                if (n_elig > 0) {
                    int il = -1, key;
                    if (tensor->src[0] && sscanf(tensor->src[0]->name, "blk.%d.", &il) == 1 && il >= 0) {
                        key = il % n_elig;                 // co-locate a layer's gate/up/down
                    } else {
                        key = (int)(sched->off_rr++ % (unsigned)n_elig);   // fallback
                    }
                    int seen = 0;
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            if (seen == key) { SET_CAUSE(tensor, "1.off"); return b; }
                            seen++;
                        }
                    }
                }
            }
```

(`unsigned off_rr` per the overflow note; the `blk.%d.` sscanf is on `src[0]`, the weight, whose name carries the layer.) Everything was gated on the aggregate host→device probe — the pbw4 probe measured GPU↔GPU, not four concurrent host→device streams:

```bash
cat > /tmp/h2d4.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#define N (256*1024*1024)
int main(){
  void *h; hipHostMalloc(&h,N);            // pinned source
  void *d[4]; hipStream_t s[4];
  for(int k=0;k<4;k++){hipSetDevice(k);hipMalloc(&d[k],N);hipStreamCreate(&s[k]);}
  for(int k=0;k<4;k++){hipSetDevice(k);hipMemcpyAsync(d[k],h,N,hipMemcpyHostToDevice,s[k]);}
  for(int k=0;k<4;k++){hipSetDevice(k);hipStreamSynchronize(s[k]);}
  hipSetDevice(0);hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);
  hipEventRecord(a,0);
  for(int i=0;i<20;i++)for(int k=0;k<4;k++){hipSetDevice(k);hipMemcpyAsync(d[k],h,N,hipMemcpyHostToDevice,s[k]);}
  for(int k=0;k<4;k++){hipSetDevice(k);hipStreamSynchronize(s[k]);}
  hipSetDevice(0);hipEventRecord(b,0);hipEventSynchronize(b);
  float ms;hipEventElapsedTime(&ms,a,b);
  printf("4 concurrent H2D: %.1f GB/s aggregate\n",4.0*20.0*N/1e9/(ms/1000.0));
  return 0;}
EOF
hipcc /tmp/h2d4.hip -o /tmp/h2d4 && /tmp/h2d4
```

**Observations**

- **Correction (of the earlier prediction):** the earlier claim — streaming 15 s → 4-5 s across four links → ~300 t/s — was wrong; it assumes parallel transfer, which the single blocking host thread prevents. The pasted review's bound stands: ~63 s → ~130 t/s. Spread across cards is not parallel across cards.
- **Decision:** layer-keying adopted as strictly better — co-locates a layer's gate/up/down, halves peer traffic, stable across ubatches.
- **Decision (thresholds):** "This number decides the whole endeavor. ~80+ → the async-copy patch is worth writing and the ceiling is high. ~40 → round-robin's ~130 is near the wall and the harder patch buys little. ~22 → the bottleneck was never the scheduler, it's your H2D fabric, and nothing in software moves it."
- The session ends with the h2d4 probe not yet run.

### State of knowledge at end of Session 7

- The zero-code mechanism tests all failed to move the needle or to run: three `-ot` command failures (unanchored regex; comma-vs-semicolon delimiter inversion, documented in the `newtest.txt` artifact's 157272.74 MiB ROCm1 OOM; compute-buffer OOM at ub 4096), then the one that ran — resident experts on ROCm1/ROCm2 at ub 2048 — measured pp2048 = 51.74 vs the 62.64 baseline, 17% worse.
- The 51.74 negative result was initially read as condemning the patch, then retracted: the experiment exercised the resident-weight peer-copy path, not op_offload's host→device path. The patch is untested, not disproven.
- Peer fabric measured: capability YES in all 12 directions; single-pair 16.2 GB/s; four-pair aggregate 49.4 GB/s (~76% scaling). Current offload streaming: ~22 GB/s, one link, blocking, against ~89 GB/s of installed lane bandwidth.
- `-ts 0,1,1,1` is self-defeating (op_offload hardcodes ROCm0 as target; freeing ROCm0 overloads ROCm1–3 with 26 attention layers each and the context fails to build). Flag-level experiments are exhausted; only code can spread the streaming load.
- The runtime split histogram (31,795-line GGML_SCHED_DEBUG dump, obtainable only with `-v` because llama-bench installs a null log callback): CPU 308 / ROCm0 731 / ROCm1 133 / ROCm2 175 / ROCm3 147 — ROCm0 owns 62% of the 1,186 GPU splits. The `return b` pin is confirmed at runtime, no longer inferred.
- The original activation-routing patch is a proven no-op (`src[1]` unassigned at heuristic time; the src-assignment block is `#if 0` dead code). Round-robin patch v1 was drafted, then revised to layer-keyed v2 (`sscanf(src->name, "blk.%d.")`) after the pasted counter-analysis.
- Expected gain corrected downward: not 250–350 t/s but ~130 (possibly ~120), because the single blocking host thread prevents parallel transfer; llama.cpp already sparse-copies only used experts.
- The h2d4 four-stream pinned host→device probe is queued as the decider (~80+ / ~40 / ~22 GB/s thresholds). Standing numbers: prefill 104.97 t/s, decode 6.01 t/s.

---

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
