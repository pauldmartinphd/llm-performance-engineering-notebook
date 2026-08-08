# What transfers to your system

Every result in this repo comes from one machine (Galactus; see [../README.md](../README.md) and [../specs/hardware.md](../specs/hardware.md)). This page sorts the results by how far they carry. It tells you which parts to copy, which to measure again, and which to ignore as ours.

The whole project has one workload shape: **a large MoE model with the routed experts in system RAM (`-ot exps=CPU` or `--cpu-moe`) and the dense path — attention, shared experts, and the KV cache — on GPUs.** If this is your setup, most of the method below applies. Your CPU, RAM, and cards do not have to match ours.

---

## Tier 1 — Copy these (they transfer in full)

### The prefill patch
Three edits to `ggml/src/ggml-backend.cpp` (see [../patch/README.md](../patch/README.md)):
- **Spread the offloaded expert matmuls across all eligible GPUs**, keyed on the layer index. The default llama.cpp scheduler returns backend 0 for every offloaded op. On any multi-GPU offload setup, the scheduler sends the offload to one card. On Galactus, 731 of 1,186 GPU splits went to ROCm0.
- **Skip the expert-ids read from GPU to host at prefill batch sizes.** At large microbatches almost every expert is selected, so the read saves no bandwidth. It only adds a host-blocking `synchronize` that chains each layer's expert copy behind the previous layer's GEMM. Decode does not change; the selective path still runs at batch 1.

The method is architectural, not specific to Galactus. The size of the gain depends on your GPU count and your interconnect.

### Benchmark faults
These are properties of llama.cpp, not of Galactus. Each one cost us time:
- **`-p N` limits `n_ubatch` to N.** This was the largest error in the project. Every `op_offload` test before we found it had run at ub 512, not the ub 8192 we intended. Always set `-ub` directly. Do not trust a prefill number until you confirm the effective ubatch.
- **`llama-bench` separates `-ot` rules with semicolons. Commas create separate benchmark configurations** (the opposite of `llama-server`). A comma-separated rule set drops the `exps=CPU` catch-all from all but the first configuration. The GPU then tries to allocate the full expert tensor and runs out of memory. The failure is in [../raw-logs/newtest-ot-oom.txt](../raw-logs/newtest-ot-oom.txt).
- **`llama-bench` installs a null log callback.** `GGML_SCHED_DEBUG` output appears only with `-v`.
- **`llama-fit-params` turns off** if you pass any of `-ngl`, `-ts`, `-ot`, or `-ncmoe`.
- **`llama-cli` on a 1M-context model** takes the context length from the model unless you pass `-c`. It fills VRAM with the KV cache and runs out of memory. `llama-bench` hides this because it sizes the context per test.
- **Do not use `llama-cli` to measure speed.** `| tee` breaks its terminal display. `--log-file` clamps the log to error level and drops the timing lines. Capture with `script -q`, or measure through the JSON timings from `llama-server`.

### Diagnostic method
- **Run STREAM with the RFO correction** to find your true memory bandwidth (Scale ×1.5, Add/Triad ×4/3; Copy needs no correction if it compiles to non-temporal stores. Check: uncorrected Copy plus RFO must not exceed your theoretical limit.) See [../scripts/galactus-diag.sh](../scripts/galactus-diag.sh) and [../specs/galactus_triad.txt](../specs/galactus_triad.txt).
- **Run `GGML_SCHED_DEBUG=2 ... -v` and count the split histogram** (`grep '## SPLIT' | sort | uniq -c`). This shows whether the offload goes to one card. This check justified the patch. It also tells you whether the patch will help you.
- **Confirm the histogram first, then the throughput.** An equal histogram and a faster prefill are separate facts. Confirm that the mechanism changed before you trust the number.

### Speculative-decode settings
- **GLM-5.2**: `--spec-type draft-mtp --spec-draft-n-max 2`. The blk.78 NextN head loads from the existing Unsloth quant, with no re-download. Result: +31%.
- **DeepSeek-V4-Flash-0731**: `--spec-type draft-dspark --spec-draft-n-max 3`, p-min off, drafter = am17an's block-5 conversion in VRAM. Result: +45%.
- **`--spec-draft-p-min` reduced speed on technical prose at every value we tried.** Leave it off, unless your domain has a genuinely low acceptance rate. We did not test that case; it is a hypothesis only.

---

## Tier 2 — Measure again with your own numbers (the models transfer, the constants do not)

- **Decode two-term model:** `time_per_token ≈ C + (bytes_read_per_token ÷ your_bandwidth)`. `C` is a GPU-side constant (about 90 ms on Galactus). `bytes_read_per_token` is your active-expert size at your quant. On Galactus the model predicted 5.5 / 6.2 / 3.9 t/s against measured 5.53 / 6.01 / 3.87. Measure your bandwidth (STREAM) and your bytes per token. The form holds; the constants are yours.
- **Prefill ubatch ladder:** `t_ubatch ≈ (fixed streaming term) + (linear GEMM term × ub)`. On Galactus both terms came from the single card that the unpatched scheduler used. Your ladder will differ, but it should still fit two terms.
- **Prompt-length scaling:** a quadratic attention term plus the linear expert terms. Attention was 72% of a 32k prefill pass here. Your split depends on your attention implementation and your context depth.
- **Cost:** the $/GB and $/decode-token method in [results §7](results-and-takeaways.md) is a template. Your prices and your parts will differ.

---

## Tier 3 — Specific to Galactus (context, not instructions)

- The exact speeds (119.36 t/s prefill; 7.1 and 14.7 t/s decode).
- 152 GB/s platform bandwidth, 65.7 GB/s 4-stream H2D, and the split counts (731 → 285/300/294/292).
- The V620, EPYC 7713, and 8-channel DDR4-2933 cost, and the DDR5 comparison.
- Bugs seen in passing (the v2 pinned-buffer and ZenDNN crashes; the `llama-bench -d` KV-restore crash at 16,384 cells). These are specific to these builds. It is useful to know they exist.

---

## The honest limits of this data

- This is one machine, and mostly one prompt (a technical-prose ZFS explainer for the decode and speculation tests), with greedy decoding. We did not capture the acceptance rates for the Session 10 DSpark runs. An instrumentation failure caused this, and we recorded it as a finding.
- The numbers come from different llama.cpp builds across three weeks. The text names the build where the build matters.
- Where a figure is an estimate from a model rather than a measurement, the source documents say so. Trust the labels over any summary.
