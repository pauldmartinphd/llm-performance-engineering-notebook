# GLM-5.3-Flash: 80 GiB expert-cache benchmark and investigation

**Added:** October 29, 2026. **Run dates:** not printed in the supplied benchmark captures. **Hardware:** four AMD Radeon Pro V620 GPUs, consistent with the Galactus configuration. **Source:** [Original benchmark captures](../raw-logs/2026-10-29-glm53-expert-cache/benchmark-captures.md), followed by the October 29 continuation in Work mode. The original conversation is `6adace95-5744-83e9-8348-9e84b6b03966`.

## Recorded measurement

The completed run requested an 81,920 MiB (80 GiB) MoE expert cache and reported **279.51 ± 15.37 tokens/s for pp8192** and **8.63 ± 2.84 tokens/s for tg256**. It used two repetitions. I retain the ± values as printed; they are not confidence intervals. The reported generation dispersion is about 32.9% of the mean, so these two repetitions provide little evidence about sustained cache performance.

```sh
llamda-bench -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf -ngl 99 -ot "exps=CPU" --moe-cache-mib 81920 -fa on -t 32 -b 8192 -ub 8192 -p 8192 -n 256 -r 2 -o md
```

| Field | Captured value |
|---|---|
| Executable spelling | `llamda-bench` |
| Printed build | `754aa4151 (11811)` |
| Model label | `glm5-next 320B.A18B Q4_K - Medium` |
| Size / parameters | 185.98 GiB / 320.76 B |
| Backend | ROCm |
| GPUs | Devices 0–3: AMD Radeon Pro V620, gfx1030 (0x1030) |
| Per-device properties | VMM: no; wave size: 32; VRAM: 30,704 MiB |
| Total reported VRAM | 122,816 MiB |
| GPU layers / tensor override | 99 / `exps=CPU` |
| Requested cache | 81,920 MiB |
| Threads / batch / microbatch | 32 / 8192 / 8192 |
| Flash attention | on, printed as 1 |
| Prompt / generation / repetitions | 8192 / 256 / 2 |

An earlier command used the same arguments except `-n 128`. Its capture ended after device discovery and the table header. It contains no throughput measurement and cannot establish a regression. The initial question, “Why is this so much slower?”, therefore left open whether the observed problem was startup delay, prompt processing, generation, or some combination.

The [October 22 build comparison](2026-10-22-glm53-build-comparison.md) supplies related results, but its revisions differ. It is not a cache-disabled control for `754aa4151`. Likewise, the [earlier placement and MTP tests](undated-glm53-flash-placement.md) and [Maya speculation tests](2026-10-23-maya-glm53-speculation.md) use different execution conditions. This record does not establish that the cache improved or degraded throughput.

## Prior implementation discussion: claims awaiting verification

The earlier assistant attributed the delay to cache initialization and large-batch prompt processing, then proposed examining hits, misses, and uploaded expert bytes. It quoted `static constexpr int64_t max_batch = 32;` and this condition:

```cpp
if (n_tokens == 0 || n_tokens > max_batch ||
    std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
    return nullptr;
}
```

One reply described this as bypassing the cached-expert execution path above 32 tokens; another said large batches bypass insertion but may still read cached experts. These descriptions may refer to different paths, but the source checkout was not inspected in the continuation, so their relationship remains unresolved. Neither description is a verified finding for build `754aa4151`.

Other prior claims were that initialization allocates and clears GPU cache buffers, generation starts with a cold cache, and the implementation records hits, misses, and upload bytes. The suggested 20 GiB per GPU was an even-division estimate, not a measured allocation. The printed 80 GiB request does not establish actual occupancy, distribution, or hit rate. The earlier assistant also referred to regressions in other caching experiments; the retrieved citation markers do not provide usable URLs, so the assertion remains unverified and is not evidence for a regression in this run.

Candidate explanations remain cache misses and PCIe transfers, ROCm synchronization or GPU execution overhead, cache initialization, and pressure on memory needed for compute buffers. None has been isolated. A hit rate by itself would not distinguish these costs; transfer and execution timing are also needed.

## Historical patch proposal

Before the measured run, the assistant proposed a one-line change for commit `184f65360`, in `tools/llama-bench/llama-bench.cpp`, immediately after `llama_model_default_params()` in `to_llama_mparams()`:

```cpp
mparams.moe_cache_size = 98304ULL * 1024 * 1024;
```

This hardcodes 96 GiB. The historical instructions were to save `llama-bench-moe-cache.patch`, run `git apply --check`, apply it, rebuild with `cmake --build build --target llama-bench -j 32`, and omit `--moe-cache-mib` from the benchmark command. The complete historical diff is retained below, with whitespace on its blank context line normalized. The assistant explicitly said the patch had not been compiled or tested. There is no evidence here that it was applied, and it targets a different revision from the measured run. It is retained as a proposal, not installed as a working patch.

```diff
diff --git a/tools/llama-bench/llama-bench.cpp b/tools/llama-bench/llama-bench.cpp
--- a/tools/llama-bench/llama-bench.cpp
+++ b/tools/llama-bench/llama-bench.cpp
@@ -1241,6 +1241,7 @@
     llama_model_params to_llama_mparams() const {
         llama_model_params mparams = llama_model_default_params();
+        mparams.moe_cache_size = 98304ULL * 1024 * 1024;

         mparams.n_gpu_layers = n_gpu_layers;
```

The proposed application sequence, not executed or validated here, was:

```sh
git apply --check llama-bench-moe-cache.patch
git apply llama-bench-moe-cache.patch
cmake --build build --target llama-bench -j 32
```

The proposed follow-up was proper option parsing, parameter propagation, and result reporting for `--moe-cache-mib`. The later captured binary already accepts and prints that option, but its implementation and linkage have not been checked. Adding a hardcoded override now could invalidate a cache-disabled comparison.

## Proposed diagnostics and controlled comparisons

The immediate diagnostic proposed in the conversation was to rerun the exact completed command with `-v`, capture stderr, and extract cache messages:

```sh
llamda-bench \
  -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  -ngl 99 -ot "exps=CPU" --moe-cache-mib 81920 \
  -fa on -t 32 -b 8192 -ub 8192 -p 8192 -n 256 -r 2 \
  -o md -v 2>&1 | tee /tmp/moe-cache-bench.log
grep -iE 'moe.cache|hit rate|misses|uploaded|slot' /tmp/moe-cache-bench.log
```

The paired control changes only `--moe-cache-mib 81920` to `--moe-cache-mib 0`, with a separate log. Neither command was run in this continuation. Verify the binary's option semantics before execution; verbose output is not guaranteed to expose all needed counters.

A separate, longer-generation proposal used `-b 2048 -ub 512 -p 512 -n 1024 -r 2 --progress -o md`, keeping the model, placement, flash attention, threads, and cache request unchanged. It reduces prompt work and extends generation, but changes several workload parameters. It therefore requires its own otherwise identical cache-disabled control and cannot be compared directly with pp8192/tg256 as evidence of a cache improvement. A longer run alone does not demonstrate a warm cache.

The investigation requested for `/root/llama.cpp` was to examine the implementation, verify cache statistics, and propose or implement minimal fixes with controlled benchmarks. The following work remains open:

1. Identify the checkout revision, local modifications, executable path, linked libraries, and model files. Reconcile `llamda-bench` with the requested llama.cpp checkout before assigning behavior to either implementation.
2. Trace the option from parsing to allocation and establish actual cache capacity per device, cache lifetime across warmup and repetitions, and the execution paths for small and large batches.
3. Verify the units and increment sites for hits, misses, and uploaded bytes. Determine whether counts describe expert requests, unique experts, tensor transfers, or another event; separate bypasses and failed insertions from ordinary misses. Use interval deltas so initialization, warmup, prompt processing, and generation are not silently combined.
4. Run the paired zero-cache and 80 GiB cases on the same binary and workload, retaining individual repetitions and startup time. Repeat in alternating order and document cache state and competing GPU activity before interpreting a difference. Measure cold and sustained behavior separately.
5. Add minimal instrumentation only where existing counters cannot answer these questions. Test any proposed cache-path fix against the unchanged binary with the same workload, including output correctness. Thread, batch, and capacity sweeps follow the initial control rather than changing all three at once.

These are proposed checks, not completed experiments or confirmed defects.

## October 29 continuation and access status

The original conversation included requests to switch to Codex and Work mode; the continuation explicitly carried forward the model, four V620s, 80 GiB cache, results, and `/root/llama.cpp` investigation. The assistant retrieved all five available prior turns, including the patch proposal and both benchmark captures. No attachments or older page were returned.

The continuation ran locally on macOS in `/Users/pmartin/Documents/Codex/2026-10-29/referenced-chatgpt-conversation-this-is-an`. `/root/llama.cpp` did not exist there, and the app returned no saved projects. An SSH configuration query for `atlas.pauldmartin.net` resolved the configured user to `pmartin` and port to 22, but the subsequent connection attempt failed with `Could not resolve hostname atlas.pauldmartin.net: nodename nor servname provided, or not known`. This does not establish that Atlas hosts the intended checkout or GPUs, or that the remote machine itself is unavailable.

The investigation was interrupted before further access checks, source inspection, compilation, cache-statistics collection, or new benchmarks. No code was changed and no fix was validated. The next request was to preserve the conversation in this notebook; this entry records that handoff without treating earlier assistant recommendations as instructions to execute.
