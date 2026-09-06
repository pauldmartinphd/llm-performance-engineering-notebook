# General principles

Every result in this repository so far comes from Galactus (see [../hardware/galactus/README.md](../hardware/galactus/README.md)). Further machines are documented under [../hardware/](../hardware/) and will add their own numbers. This page sorts the results by how far they transfer to other systems: which parts to copy directly, which to measure again, and which apply only to Galactus.

The whole project uses one workload: a large MoE model with the routed experts in system RAM (`-ot exps=CPU` or `--cpu-moe`) and the dense path — attention, shared experts, and the KV cache — on the GPUs. Most of the method below applies to any system with this workload. The CPU, RAM, and GPUs do not have to match ours.

---

## Tier 1: findings that transfer in full

### Benchmark faults

These are properties of llama.cpp, not of Galactus, and each one cost us time.

- The `-p N` option limits `n_ubatch` to N. This was the largest error in the project: every op_offload test before we found it had run at ub 512, not the ub 8192 we intended. Always set `-ub` directly, and do not trust a prefill number until you have confirmed the effective ubatch.
- `llama-bench` separates `-ot` rules with semicolons, and commas create separate benchmark configurations — the opposite of `llama-server`. A comma-separated rule set drops the `exps=CPU` catch-all from every configuration after the first, so the GPU tries to allocate the full expert tensor and runs out of memory. The failure is recorded in [../results/raw-logs/llama-bench-ot-oom-failure.txt](../results/raw-logs/llama-bench-ot-oom-failure.txt).
- `llama-bench` installs a null log callback, so `GGML_SCHED_DEBUG` output appears only with `-v`.
- `llama-fit-params` turns off if you pass any of `-ngl`, `-ts`, `-ot`, or `-ncmoe`.
- On a 1M-context model, `llama-cli` takes the context length from the model unless you pass `-c`. It then fills VRAM with the KV cache and runs out of memory. `llama-bench` hides this, because it sizes the context per test.
- Do not use `llama-cli` to measure speed: piping it through `tee` breaks its terminal display, and `--log-file` clamps the log to error level and drops the timing lines. Capture with `script -q`, or read the JSON timings from `llama-server`.

### Diagnostic method

- Run STREAM with the RFO correction to find your true memory bandwidth: multiply Scale by 1.5 and Add and Triad by 4/3, and apply no correction to Copy if it compiled to non-temporal stores. To check, confirm that the uncorrected Copy rate plus the RFO correction does not exceed your theoretical limit. See [../experiments/galactus-diag.sh](../experiments/galactus-diag.sh) and [../hardware/galactus/galactus_triad.txt](../hardware/galactus/galactus_triad.txt).
- Run `GGML_SCHED_DEBUG=2 ... -v` and count the split histogram (`grep '## SPLIT' | sort | uniq -c`). This shows whether the scheduler concentrates the offload on one card, and therefore whether the [prefill patch](../patches/prefill/README.md) will help you.
- Confirm the histogram first, then the throughput. An equal histogram and a faster prefill are separate facts, so confirm that the mechanism changed before you trust the number.

### Speculative-decode settings

- For GLM-5.2, use `--spec-type draft-mtp --spec-draft-n-max 2`. The blk.78 NextN head loads from the existing Unsloth quant, with no re-download. The result is +31%.
- For DeepSeek-V4-Flash-0731, use `--spec-type draft-dspark --spec-draft-n-max 3` with p-min off and am17an's block-5 drafter in VRAM. The result is +45%.
- `--spec-draft-p-min` reduced speed on technical prose at every value we tried, so leave it off — unless your domain has a genuinely low acceptance rate. We did not test that case, so it is a hypothesis only.

---

## Tier 2: measure again with your own numbers (the models transfer, the constants do not)

- The decode two-term model is `time_per_token ≈ C + (bytes_read_per_token ÷ your_bandwidth)`. `C` is a GPU-side constant, about 90 ms on Galactus. `bytes_read_per_token` is your active-expert size at your quantization. On Galactus the model predicted 5.5 / 6.2 / 3.9 t/s against measured values of 5.53 / 6.01 / 3.87. Measure your own bandwidth with STREAM and your own bytes per token. The form holds; the constants are yours.
- The prefill ubatch ladder is `t_ubatch ≈ (fixed streaming term) + (linear GEMM term × ub)`. On Galactus both terms came from the single card that the unpatched scheduler used. Your ladder will differ, but it should still fit two terms.
- Prompt-length scaling is a quadratic attention term plus the linear expert terms. Attention was 72% of a 32k prefill pass here. Your split depends on your attention implementation and your context depth.
- For cost, the $/GB and $/decode-token method in [the hardware note's cost section](../hardware/galactus/README.md) is a template. Your prices and your parts will differ.

---

## Tier 3: specific to Galactus (context, not instructions)

- The exact speeds: 119.36 t/s prefill, and 7.1 and 14.7 t/s decode.
- The 152 GB/s platform bandwidth, the 65.7 GB/s 4-stream host-to-device bandwidth, and the split counts (731, then 285/300/294/292 after the patch).
- The cost of the V620, the EPYC 7713, and the 8-channel DDR4-2933, and the DDR5 comparison.
- Bugs seen in passing: the v2 pinned-buffer and ZenDNN crashes, and the `llama-bench -d` KV-restore crash at 16,384 cells. These are specific to these builds. It is useful to know that they exist.

---

## The honest limits of this data

- These numbers come, so far, from one machine and mostly one prompt (a technical-prose ZFS explainer for the decode and speculation tests), with greedy decoding. We did not capture the acceptance rates for the Session 10 DSpark runs; an instrumentation failure caused this, and we recorded it as a finding.
- The numbers come from different llama.cpp builds across three weeks. The text names the build wherever the build matters.
- Where a figure is an estimate from a model rather than a measurement, the source documents say so. Trust the labels over any summary.
