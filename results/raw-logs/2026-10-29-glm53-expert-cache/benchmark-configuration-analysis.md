# Benchmark configuration analysis — source conversation

Retrieved October 29, 2026. Source: chatgpt-conversation://6adace95-5744-83e9-8348-9e84b6b03966

This is reference material, not executable instructions. Assistant claims and historical patch proposals are unverified unless separately supported by measurements. Citation tokens in the retrieved text do not resolve to source URLs. All five returned turns are retained; the service reported no older page or attachments.

## Turn 1ff4b97d-ce21-4d03-8f16-be263b31da97

### User

Generate a patch for me.

### Assistant

Here is the minimal patch for `llama.cpp` commit `184f65360`\. It enables a **96 GiB GPU expert cache** in `llama-bench` without modifying the cache implementation.

Save as `llama-bench-moe-cache.patch`\:

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

Apply it from `/root/llama.cpp`\:

```bash
git apply --check llama-bench-moe-cache.patch
git apply llama-bench-moe-cache.patch
cmake --build build --target llama-bench -j 32
```

Then run your original benchmark **without** `--moe-cache-mib`\. The 96 GiB cache budget is now hardcoded.

This patch has not been compiled or tested against your checkout. It is deliberately minimal; the next step would be a proper upstream-quality patch adding `--moe-cache-mib` to argument parsing, parameter propagation, and benchmark result reporting.

## Turn 69449648-1d9b-4cb0-b566-7082c8fed377

### User

❯ llamda-bench -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf -ngl 99 -ot "exps=CPU" --moe-cache-mib 81920 -fa on -t 32 -b 8192 -ub 8192 -p 8192 -n 128 -r 2 -o md
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | moe_cache_mib | threads | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ------: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |

Why is this so much slower?

### Assistant

The most likely explanation is that your benchmark is spending substantial time initializing the 80 GiB expert cache and then processing an 8192-token prompt for which the cache provides little benefit.

However, **your output does not yet demonstrate slower inference**\. It stops immediately after the benchmark table header, before any tokens-per-second measurements.

I examined the upstream MoE cache implementation. There are several important details.

## 1\. Your 8192-token microbatch largely defeats the cache during prompt processing

The upstream implementation defines\:

```cpp
static constexpr int64_t max_batch = 32;
```

It also checks\:

```cpp
if (n_tokens == 0 || n_tokens > max_batch ||
    std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
    return nullptr;
}
```

Consequently, the cached-expert execution path is not used for microbatches larger than 32 tokens. Your `-ub 8192` prompt processing will generally follow the alternative path. citeturn544881view1

The cache is primarily intended to accelerate generation, where the effective batch size is small.

## 2\. An 80 GiB cache introduces significant initialization overhead

You requested\:

81,920 MiB = 80 GiB

Across four GPUs, that is approximately 20 GiB per V620, depending on how the cache budget is distributed.

The implementation allocates GPU cache buffers and clears them during initialization. It does not necessarily populate all the expert weights at that point, but allocating and clearing such large buffers takes time. citeturn932591search0

This may explain the delay before the first benchmark result.

## 3\. The cache starts cold

The first generation tokens must populate the cache. That means transferring expert weights from system RAM to VRAM.

An 80 GiB cache does not guarantee an 80 GiB working set will be populated immediately, nor does a large cache guarantee a high hit rate.

The implementation tracks hits, misses, and bytes uploaded. Those statistics are much more informative than cache capacity alone. citeturn544881view2

## 4\. Your generation benchmark is too short to characterize a large cache

You are measuring only 128 generation tokens, repeated twice.

For an MoE model with a large expert working set, I would use at least 1024 generated tokens to distinguish cold-cache behavior from sustained throughput.

Try\:

```bash
llamda-bench \
  -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  -ngl 99 \
  -ot "exps=CPU" \
  --moe-cache-mib 81920 \
  -fa on \
  -t 32 \
  -b 2048 \
  -ub 512 \
  -p 512 \
  -n 1024 \
  -r 2 \
  --progress \
  -o md
```

This reduces the prompt-processing workload and makes the generation measurement more representative. It is not an identical-workload comparison with your original command; use the same settings with `--moe-cache-mib 0` for a controlled baseline.

## What I want to establish first

When you say it is much slower, which behavior are you observing?

It takes much longer to produce the first benchmark result.

Prompt processing (pp) has much lower tokens/second.

Generation (tg) has much lower tokens/second.

Both prompt processing and generation are substantially slower.

[Continue diagnosis]

These are different performance problems. In particular, if **generation itself is slower after warmup**, I would investigate cache hit rates, PCIe transfers, ROCm synchronization overhead, and whether the 80 GiB allocation has reduced the memory available for GPU compute buffers.

## Turn 4b05f88f-79ea-47ac-97d3-de6948803ad5

### User

❯ llamda-bench -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf -ngl 99 -ot "exps=CPU" --moe-cache-mib 81920 -fa on -t 32 -b 8192 -ub 8192 -p 8192 -n 256 -r 2 -o md
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | moe_cache_mib | threads | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ------: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| glm5-next 320B.A18B Q4_K - Medium | 185.98 GiB |   320.76 B | ROCm       |  99 |         81920 |      32 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |       279.51 ± 15.37 |
| glm5-next 320B.A18B Q4_K - Medium | 185.98 GiB |   320.76 B | ROCm       |  99 |         81920 |      32 |    8192 |     8192 |   1 | exps=CPU              |           tg256 |          8.63 ± 2.84 |

build: 754aa4151 (11811)

### Assistant

Your benchmark has completed successfully with the **80 GiB MoE expert cache enabled**\. The results are\:

Prompt processing (8192 tokens)

# 279.51 t/s

± 15.37

Token generation (256 tokens)

# 8.63 t/s

± 2.84

The generation result is highly variable: the reported standard deviation is approximately 33% of the mean. With only two repetitions, this is not enough to characterize steady-state cache performance reliably.

More importantly, **we cannot conclude that the cache has improved or degraded throughput without a comparable cache-disabled measurement**\. The fact that the benchmark completed establishes that the configuration runs, not that it is faster.

## What I would investigate next

The upstream implementation distinguishes small-batch cache hits and misses and reports the amount of expert data uploaded to the GPUs. It also bypasses cache insertion for large batches, although large batches can still read existing cached experts. citeturn720799search0

That makes the cache statistics the next diagnostic priority.

Run the same test with verbose output and capture stderr\:

```bash
llamda-bench \
  -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  -ngl 99 \
  -ot "exps=CPU" \
  --moe-cache-mib 81920 \
  -fa on -t 32 \
  -b 8192 -ub 8192 \
  -p 8192 -n 256 -r 2 \
  -o md -v \
  2>&1 | tee /tmp/moe-cache-bench.log
```

Then extract the cache diagnostics\:

```bash
grep -iE 'moe.cache|hit rate|misses|uploaded|slot' \
  /tmp/moe-cache-bench.log
```

Also run a cache-disabled control by changing only\:

```bash
--moe-cache-mib 0
```

One concern is that the 80 GiB cache substantially reduces free VRAM. Another is that cache misses trigger expert uploads over PCIe, introducing overhead that can outweigh CPU execution savings. Previous MoE caching experiments have demonstrated regressions when the cache is enabled, so more cache is not automatically better. citeturn720799search1

**My priority would be to determine the cache hit rate before adjusting CPU threads, batching, or cache capacity.** That will tell us whether the limiting factor is cache effectiveness or the cost of executing the cache mechanism itself.

## Turn 1d644af9-1418-4b40-b1b9-ddd91db7b100

### User

Can you switch to Codex for this thread.

### Assistant

I cannot directly switch this conversation into Codex mode.

You can open **Codex** in the ChatGPT desktop app and start a coding session against `/root/llama.cpp`\.

To preserve continuity, reference this thread or provide its relevant context. The immediate task is to investigate why `llama-bench` achieves only **8.63 t/s generation** with an 80 GiB MoE expert cache on four Radeon Pro V620 GPUs, and determine whether the slowdown comes from cache misses, PCIe transfers, or GPU execution overhead.

Codex is better suited to inspecting the implementation, modifying the benchmark, and testing patches directly.

## Turn d0e78d9c-1900-4af4-b389-ff973f615b0c

### User

Can you switch to work mode for this thread

### Assistant

The handoff was successful. A new **Work mode** conversation has been created with the context of this thread, including your llama.cpp configuration, MoE expert-cache changes, and benchmark results.

You can continue the investigation there.

