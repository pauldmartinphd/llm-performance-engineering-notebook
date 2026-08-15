# Reproducing this on your own hardware

You do not need Galactus's exact parts. You need the same workload shape: a large MoE model with the routed experts in system RAM and the dense path on one or more GPUs. Follow the steps in this order to find the speed limit and raise it.

## 0. Prerequisites
- llama.cpp, built for your GPU backend (ROCm here; CUDA, Metal, and Vulkan use the same scheduler logic).
- A large MoE GGUF that does not fit in VRAM, so the experts must live in RAM.
- STREAM (`stream.c`), compiled for your core count.

## 1. Find your memory-bandwidth limit (this limits decode)
Run STREAM across a thread sweep. Apply the RFO correction (Scale ×1.5, Add/Triad ×4/3; Copy usually needs no correction — confirm it does not exceed your theoretical limit). This gives your decode limit. See [experiments/galactus-diag.sh](experiments/galactus-diag.sh) for the exact command, and [hardware/galactus/galactus_triad.txt](hardware/galactus/galactus_triad.txt) for a sample of the output.

Predict decode before you measure it: `t/token ≈ C + bytes_per_token ÷ bandwidth`. `bytes_per_token` is the active-expert size at your quant. `C` is your GPU-side constant, about 90 ms on Galactus; measure it once and reuse it. If your measured decode is far below this prediction, a setting is wrong. Fix it before you tune further.

## 2. Baseline both phases correctly
```bash
llama-bench -m <model> -ngl 99 -ot "exps=CPU" -fa 1 \
  -t <physical-cores/2..physical-cores> -b 8192 -ub 8192 -p 8192 -n 128 -r 2 -o md
```
Set `-ub` directly. `-p` limits `n_ubatch`, so a low `-p` limits prefill without warning. Sweep the thread count. Expect a maximum near half your physical cores, and a large drop when you use SMT siblings.

## 3. See where the offload goes
```bash
GGML_SCHED_DEBUG=2 llama-bench -m <model> -ngl 99 -ot "exps=CPU" -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > sched.txt 2>&1
grep '## SPLIT' sched.txt | sed -E 's/.*: (GPU?[0-9]|CPU|ROCm[0-9]|CUDA[0-9]).*/\1/' | sort | uniq -c
```
If one GPU holds most of the splits, the [prefill patch](patches/prefill/README.md) applies to you. If the offload is already balanced, the patch will not help. This is why you check first.

## 4. Apply the patch (if step 3 justified it)
Follow [patches/prefill/README.md](patches/prefill/README.md). Run step 3 again first, and confirm the histogram is now equal. Then run step 2 again for the throughput. The histogram confirms the mechanism. The throughput confirms the gain.

## 5. Add speculative decode for the last decode gains
- GLM family: `--spec-type draft-mtp --spec-draft-n-max 2`.
- DeepSeek-V4-Flash: `--spec-type draft-dspark --spec-draft-n-max 3`, with the block-5 drafter in VRAM.
- Sweep `n-max` from 1 to 5. Expect a maximum at 2 or 3, and a drop by 5 from the verify cost on a top-k-of-many MoE. Leave `p-min` off on technical prose.
- Measure through the `llama-server` JSON timings, or with `script -q`. Do not pipe `llama-cli` to a file; it drops the timing lines.

## What to record
Use the CSV schema in [results/data/](results/data/): date, experiment, configuration (with the exact flags and build), metric, value, source. The configuration column is what makes a number reproducible. A t/s figure without its configuration has no value. Record the dead ends too. The [refuted-hypotheses table](takeaways/refuted-hypotheses.md) saved more time than any single gain.
