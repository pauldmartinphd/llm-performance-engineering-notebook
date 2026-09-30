# Findings and limits

This page covers the Galactus investigation (see [../hardware/galactus/README.md](../hardware/galactus/README.md)). The other machines are documented under [../hardware/](../hardware/). It brings together the benchmark pitfalls, explanatory models, and limits of the evidence. Although the method applies to other machines, these numerical results remain specific to Galactus.

The main workload is a large MoE model with the routed experts in system RAM (`-ot exps=CPU` or `--cpu-moe`) and the dense path — attention, shared experts, and the KV cache — on the GPUs. Resident-expert and CPU-only runs provide additional comparisons. Most of the method below applies to any system with this workload. The CPU, RAM, and GPUs do not have to match mine.

---

## Benchmark faults

These behaviors were observed in the llama.cpp builds used for the experiments. Check them when reproducing the work on a different build.

- The `-p N` option limits `n_ubatch` to N. This was the largest error in the project: every op_offload test before I found it had run at ub 512, not the ub 8192 I intended. Set `-ub` directly, keep `-p` large enough, and confirm the effective micro-batch size.
- `llama-bench` separates `-ot` rules with semicolons, and commas create separate benchmark configurations — the opposite of `llama-server`. A comma-separated rule set drops the `exps=CPU` catch-all from every configuration after the first, so the GPU tries to allocate the full expert tensor and runs out of memory. The failure is recorded in [../results/raw-logs/llama-bench-ot-oom-failure.txt](../results/raw-logs/llama-bench-ot-oom-failure.txt).
- `llama-bench` installs a null log callback, so `GGML_SCHED_DEBUG` output appears only with `-v`.
- `llama-fit-params` turns off if you pass any of `-ngl`, `-ts`, `-ot`, or `-ncmoe`.
- On a 1M-context model, `llama-cli` takes the context length from the model unless you pass `-c`. It then fills VRAM with the KV cache and runs out of memory. `llama-bench` hides this, because it sizes the context per test.
- Capture llama-cli through a pseudo-terminal or the server interface: piping it through `tee` breaks its terminal display, and `--log-file` clamps the log to error level and drops the timing lines. Capture with `script -q`, or read the JSON timings from `llama-server`.

## Diagnosing placement and bandwidth

- Run STREAM with the RFO correction to find your true memory bandwidth: multiply Scale by 1.5 and Add and Triad by 4/3, and apply no correction to Copy if it compiled to non-temporal stores. Check the generated stores before applying the correction; an adjusted result above the theoretical limit indicates a problem with the assumed traffic model. See [../experiments/galactus-diag.sh](../experiments/galactus-diag.sh) and [../hardware/galactus/galactus_triad.txt](../hardware/galactus/galactus_triad.txt).
- Run `GGML_SCHED_DEBUG=2 ... -v` and count the split histogram (`grep '## SPLIT' | sort | uniq -c`). This shows whether the scheduler concentrates the offload on one card, and whether the placement behavior addressed by the [prefill patch](../patches/prefill/README.md) is present. It does not establish a throughput gain.
- Check the histogram and throughput separately. Equal placement establishes that the scheduler changed; it does not establish that prefill became faster.

## Speculative decode

GLM-5.2 used `--spec-type draft-mtp --spec-draft-n-max 2`; its blk.78 NextN head loaded from the existing Unsloth quant without a re-download. The first successful measurement was 7.1 t/s (+31%). On the August common build, the repeated result was **6.6 ± 0.3 t/s**, about +25% over the stock baseline.

DeepSeek-V4-Flash-0731 used `--spec-type draft-dspark --spec-draft-n-max 3`, with p-min off and am17an's block-5 drafter in VRAM. The initial 14.7 t/s result was about +45%; the August common-build result was **14.1 ± 0.7 t/s**, about +36%. These gains come from separate speculative runs, not the prefill patch.

Confidence truncation with `--spec-draft-p-min` did not establish an improvement on the tested technical prose. One early row has conflicting reports, and Session 10 lacks acceptance-rate captures. Whether truncation helps a low-acceptance domain remains untested. See [speculative decoding](speculative-decoding.md) for the cost model and measurement limits.

---

## Models to test on another system

- The decode two-term model is `time_per_token ≈ C + (bytes_read_per_token ÷ your_bandwidth)`. `C` is a GPU-side constant, about 90 ms on Galactus. `bytes_read_per_token` is your active-expert size at your quantization. On Galactus the model predicted 5.5 / 6.2 / 3.9 t/s against measured values of 5.53 / 6.01 / 3.87. Measure your own bandwidth with STREAM and your own bytes per token. Treat the two-term form as an approximation to check against measurements.
- The prefill ubatch ladder is `t_ubatch ≈ (fixed streaming term) + (linear GEMM term × ub)`. On Galactus both terms came from the single card that the unpatched scheduler used. Test whether those two terms explain your measured ladder.
- Prompt-length scaling is a quadratic attention term plus the linear expert terms. Attention was 72% of a 32k prefill pass here. Your split depends on your attention implementation and your context depth.
- For cost, the $/GB and $/decode-token method in [the hardware note's cost section](../hardware/galactus/README.md) is a template. Your prices and your parts will differ.

---

## Hardware and build limits

The historical results—119.36 t/s patched prefill and 7.1 and 14.7 t/s speculative decode—belong to their recorded configurations. So do the measured 152 GB/s DRAM bandwidth, 65.7 GB/s four-stream host-to-device bandwidth, and split counts of 731 on ROCm0 before the patch and 285/300/294/292 afterward. Component costs and the DDR5 comparison reflect the purchase and pricing dates in the [hardware note](../hardware/galactus/README.md).

The notebook also records pinned-buffer and ZenDNN crashes during v2 testing, and a `llama-bench -d` KV-restore crash at 16,384 cells. Those reports concern the tested builds; they do not establish the behavior of later versions.

---

## Limits of the evidence

- These Galactus numbers come from one machine and mostly one prompt (a technical-prose ZFS explainer for the decode and speculation tests), with greedy decoding. An instrumentation failure prevented me from capturing acceptance rates for the Session 10 DSpark runs, as recorded in the notebook.
- The record spans April through August and includes different llama.cpp builds, model exports, memory populations, and benchmark settings. [Entry 12](../results/lab-notebook/12-common-baseline-2tb.md) provides a common August baseline; earlier results are historical comparisons.
- I label estimates from a model separately from measurements in the source documents.
