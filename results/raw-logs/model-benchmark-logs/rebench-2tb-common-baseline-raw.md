# Raw capture — 2 TB common-baseline rerun (2026-08-15/16)

Reconstructed from the session transcript; the tee originals live on Galactus in `/root/rebench-2tb/`. All runs: build `3653e6d6d (10326)`, stock scheduler (PP patch not applied), 2 TB population post-DIMM-replacement. The `threads` column is absent from tables where t=64, because llama-bench suppresses columns at their default value (64 = physical cores on this machine).

Note: the DeepSeek table below was recovered after a tee-filename mistake overwrote `/root/rebench-2tb/deepseek-v4-flash.md` with the Kimi run (repair commands were issued in-session; the closing build stamp of the DSV4 run was lost from the paste — same binary as the adjacent runs).

## GLM-5.2 (t=32)

```
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| glm-dsa 744B.A40B Q4_K - Medium | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |         95.99 ± 3.36 |
| glm-dsa 744B.A40B Q4_K - Medium | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |          5.30 ± 0.00 |
build: 3653e6d6d (10326)
```

## DeepSeek-V4-Flash-0731 (t=64)

```
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| deepseek4 ?B MXFP4 MoE         | 150.75 GiB |   284.33 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |        143.54 ± 1.64 |
| deepseek4 ?B MXFP4 MoE         | 150.75 GiB |   284.33 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |         10.34 ± 0.10 |
[build stamp lost to the tee overwrite; same binary as adjacent runs: 3653e6d6d (10326)]
```

## Kimi K2.6 (t=64) — file: `/models/Kimi-K2.6/UD-Q8_K_XL/` (native-INT4 MoE; arch reads `deepseek2 671B BF16` cosmetically)

```
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| deepseek2 671B BF16            | 553.71 GiB |  1026.41 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |         94.23 ± 4.45 |
| deepseek2 671B BF16            | 553.71 GiB |  1026.41 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |          5.79 ± 0.01 |
build: 3653e6d6d (10326)
```

## Qwen 3.5 397B-A17B (t=64) — file: Unsloth UD-Q6_K_XL (April ran bartowski Q6_K_L, 319.21 GiB)

```
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| qwen35moe 397B.A17B Q6_K       | 337.43 GiB |   396.35 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |       249.61 ± 19.78 |
| qwen35moe 397B.A17B Q6_K       | 337.43 GiB |   396.35 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |          9.37 ± 0.16 |
build: 3653e6d6d (10326)
```

## MiniMax M2.7 (t=64) — terminal measurement; model deleted after

```
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |          pp8192 |       418.83 ± 24.11 |
| minimax-m2 230B.A10B Q5_K - Medium | 157.23 GiB |   228.69 B | ROCm       |  99 |    8192 |     8192 |   1 | exps=CPU              |           tg128 |         15.18 ± 0.18 |
build: 3653e6d6d (10326)
```

## Qwen offload-class attempts (abandoned)

April 5-layer/card placement under the standing config: `llama_bench: error: failed to load model` (weight stage). 4-layer/card: `llama_bench: error: failed to create context with model` — weights fit, ub-8192 compute buffers + f16 KV did not. The two errors bracket the per-card VRAM budget; llama-bench's null log callback hides the allocation detail. Offload rerun abandoned by decision; see Entry 12 observations.

## Tier-2 speculative re-stamp (llama-cli, script(1) captures on Galactus)

Common flags: `-c 8192 -b 8192 -ub 8192 -n 256 --temp 0 -st -no-cnv --perf`, ZFS prompt (`Explain how ZFS copy-on-write snapshots work, including the uberblock, block pointers, and space maps.`). Build `b10326-3653e6d6d` per each run's banner. Greedy — token streams identical across reps of the same model. The new llama-cli prints only the compact perf line.

GLM-5.2, `-ngl 99 -ot exps=CPU -t 32 --spec-type draft-mtp` — capture `glm-mtp-n2.txt`, runs in order:

```
n-max 2:  [ Prompt: 2.5 t/s | Generation: 6.3 t/s ]
n-max 2:  [ Prompt: 1.9 t/s | Generation: 6.9 t/s ]
n-max 3:  [ Prompt: 2.5 t/s | Generation: 6.3 t/s ]
n-max 1:  [ Prompt: 2.5 t/s | Generation: 5.9 t/s ]
```

DeepSeek-V4-Flash-0731, `-ngl 99 --cpu-moe -t 64 --spec-type draft-dspark -md DeepseekV4-Flash-20260731-DSpark.gguf -ngld 99` — capture `dsv4-dspark-n3.txt`, runs in order (the `dflash requires ctx_other` line during memory fitting is cosmetic; every run proceeded):

```
n-max 3:  [ Prompt: 12.2 t/s | Generation: 14.7 t/s ]
n-max 3:  [ Prompt: 13.0 t/s | Generation: 13.4 t/s ]
n-max 2:  [ Prompt: 13.2 t/s | Generation: 14.3 t/s ]
n-max 4:  [ Prompt: 13.0 t/s | Generation: 13.6 t/s ]
n-max 8:  [ Prompt: 12.8 t/s | Generation: 12.5 t/s ]   (clamps to drafter block size 5)
n-max 1:  [ Prompt: 12.8 t/s | Generation: 12.9 t/s ]
```

The `Prompt:` figures are the tiny-prompt artifact (~20 tokens) and are not prefill measurements.

## Qwen MTP attempt (failed to arm)

```
Loading model... |0.33.138.901 E common_speculative_init_result: failed to create MTP context
0.33.138.912 E srv    load_model: failed to create MTP context
0.33.143.610 E srv  llama_server: exiting due to model loading error
```

Cause not isolated (export lacking NextN tensors vs MTP-context allocation failure); discriminator (`n_layer_nextn` load line) is in `/root/rebench-2tb/qwen-mtp-n2.txt` on Galactus. No further Qwen speculative work planned.
