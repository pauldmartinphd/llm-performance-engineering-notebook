# Reproducing the measurements

These experiments use a large MoE model with routed experts in system RAM and the dense path on one or more GPUs. Repeating the method on another machine starts with its measured bandwidth, placement, and effective batch sizes. The commands below describe the recorded llama.cpp workflow; flags and backend behavior can differ by build.

For the initial March installation, the [earlier build record](results/lab-notebook/00-march-build-record.md) links the retained compiler, device-access, and server commands. It is a historical configuration on a different build and storage setup, rather than the recipe for the August common baseline. The [source-note index](results/reference-notes/README.md) records the evidence status of the later model-selection and residency plans.

## Prepare and record the configuration

Build llama.cpp for your GPU backend and record the commit, build options, and loaded backend libraries. The measurements here used ROCm. CUDA, Metal, and Vulkan share scheduler code, but this repository does not establish the patch's performance on those backends.

Choose an MoE GGUF whose routed experts must live in RAM, and record the exact export, quantization, shard names, and sizes. Compile STREAM (`stream.c`) for the machine. Record the memory population, channel count, configured transfer rate, CPU topology, GPU placement, and available VRAM. These details distinguish a repeat measurement from a comparison between different configurations.

## Measure memory bandwidth

Run STREAM across a thread-count sweep. For ordinary stores that incur read-for-ownership traffic, the corrections used here are Scale ×1.5 and Add/Triad ×4/3. Copy needs no correction when compiled to non-temporal stores. Check the generated store behavior before applying these factors on another system; an adjusted rate above the theoretical DRAM peak is a reason to investigate the assumption. The [diagnostic script](experiments/galactus-diag.sh) and [Galactus STREAM capture](hardware/galactus/galactus_triad.txt) provide the recorded procedure and output.

Use the result to estimate decode time:

```
time_per_token ≈ C + bytes_read_per_token / bandwidth
```

The bytes term is the active-expert footprint at the chosen quantization. `C` represents the remaining cost for the configuration; the GLM-5.2 investigation estimated about 90 ms on Galactus. Fit and check that term for your model and placement. A large gap between prediction and measurement leaves two possibilities to check: the setup and the model's assumptions.

## Establish prefill and decode baselines

Substitute the model path and one thread count per run:

```bash
llama-bench -m <model> -ngl 99 -ot "exps=CPU" -fa 1 \
  -t <threads> -b 8192 -ub 8192 -p 8192 -n 128 -r 2 -o md
```

Sweep thread counts rather than assuming that all logical CPUs will help. Galactus's GLM-5.2 decode peaked around 24–32 threads and slowed sharply when the sweep reached SMT siblings. Other models and prefill had different optima.

Set `-ub` explicitly and keep `-p` large enough for the intended micro-batch: in the recorded builds, `-p` clamps the effective `n_ubatch`. Asking for `-ub 8192` with `-p 512` still tests the smaller regime. Record the effective values from the run. The [August baseline](results/lab-notebook/12-common-baseline-2tb.md) used f16 KV; the April runs used q8_0 and smaller prompts.

## Inspect scheduler placement

```bash
GGML_SCHED_DEBUG=2 llama-bench -m <model> -ngl 99 -ot "exps=CPU" -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > sched.txt 2>&1
grep '## SPLIT' sched.txt | sed -E 's/.*: (GPU?[0-9]|CPU|ROCm[0-9]|CUDA[0-9]).*/\1/' | sort | uniq -c
```

Adapt the backend names and thread count for your machine. This small-batch run checks placement; it is not the large-batch throughput benchmark. `-v` is required because llama-bench otherwise suppresses the scheduler log.

Concentration on one GPU is a reason to inspect the [prefill patch](patches/prefill/README.md). On Galactus, balancing the split counts alone did not improve throughput: the routing-index synchronization also had to change. An equal histogram establishes placement, not a speedup.

## Compare the patched and stock builds

Follow the patch note's source anchors, build instructions, and two checks. Hold the model, flags, loading mode, and hardware state constant across stock and patched runs. Confirm the changed split distribution, measure large-batch prefill, and check decode for regressions. The recorded gain was 13.7% on GLM-5.2; the four-model comparison on the August build remains open.

The July commands use `-mmp 0` for pinned host loading. The [Session 10 interval notes](results/lab-notebook/10-dspark-deepseek-v4-flash.md) record its later replacement by `--load-mode none`; `-dio 1` maps to `--load-mode dio`. Use the flags supported by the build under test and record the buffer type actually allocated.

## Measure speculative decode separately

For the tested exports, GLM-5.2 used `--spec-type draft-mtp --spec-draft-n-max 2`. DeepSeek-V4-Flash used `--spec-type draft-dspark --spec-draft-n-max 3`, with the block-5 drafter in VRAM. The [model notes](results/README.md) give the files and conditions. These are starting points for a sweep, not established optima for other models or prompts.

Sweep `n-max` from 1 to 5 with the prompt, context, sampling settings, placement, and fitter state held constant. Use an explicit context size (`-c`): the tested 1M-context model otherwise attempted a KV allocation that exhausted VRAM. Record the baseline and repetitions, generated token count, acceptance rates, and timings. The Galactus tests used greedy decoding and mostly one technical-prose prompt; p-min truncation did not establish an improvement for that workload.

Use llama-server's JSON timings or capture llama-cli through a pseudo-terminal. On the Linux test host, `script -q <file> -c "<command>"` preserves terminal output; piping llama-cli or relying on its log file lost timing lines in the recorded builds. llama-bench did not support speculation in these experiments. Keep the tool difference explicit when comparing its baseline with a llama-cli result.

## Save the evidence

Keep raw output alongside the [CSV extracts](results/data/). Record date, experiment, configuration (including exact flags and build), metric, value, and source. Preserve individual repetitions and explain what a reported ± value represents. The speculative repeats in Entry 12 show about 9% timing variation, so a single run cannot reliably distinguish nearby settings.

Record failed runs, unsupported configurations, and null results as well as improvements. The [negative-results table](takeaways/refuted-hypotheses.md) is specific to the tested workload and machine; it helps prioritize new experiments without ruling out different behavior elsewhere.

## oMLX measurements on Apple Silicon

The workflow above describes Galactus/llama.cpp. [Magneto's oMLX record](results/qwen-3.8-27b.md) is separate. A Mac comparison depends on the model identifier and chassis as well as chip/GPU-core count, RAM, macOS and oMLX/MLX versions, model repository/revision, and recipe. The leaderboard's chip/RAM label does not identify chassis.

The recorded benchmark conditions include the corpus, context and generation lengths, full/quick warm-up, ANE-aligned prompt setting, cache state, and acceleration toggles. Single-request PP/TG and continuous-batching throughput are separate metrics. Thermal comparisons also depend on power mode, adapter, cooling conditions, and the timing of powermetrics relative to prefill and decode. A thermal snapshot does not establish the state of an entire sweep.

The initial recorded configuration used Lightning MTP on and ANE Prefill, SpecPrefill, DFlash, VLM MTP, and TurboQuant KV off. It provides a reference with MTP active; no MTP-off control was recorded. Since the model/MLX revisions, macOS build, full benchmark recipes and original logs were not retained, the evidence in [Entry 13](results/lab-notebook/13-magneto-omlx-qwen38.md) does not support exact reproduction of those runs. My settings at the end of the investigation are logged in the [model note](results/qwen-3.8-27b.md).
