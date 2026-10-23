# Galactus experiment archive

I added these records on October 22, 2026, from `tests.tar.gz`. They cover April thread/batch sweeps, July diagnostics and scheduler traces, and August rebenchmarks. Undated DSpark server logs name DeepSeek-V4-Flash-0731, not V4.1. These earlier tests do not measure the later combined V620/GLM-5.3/DeepSeek-V4.1 build.

I preserved all member contents, including failed runs, two empty files, terminal control characters, and the original scripts. Files over 1 MB use lossless gzip compression. The scripts have a `.txt` suffix because they are historical records, not current run instructions. I did not execute them. The manifest maps the descriptive filenames to original archive members and records both original and stored SHA-256 hashes.

[April experiments](../../lab-notebook/2026-04-07-to-17-thread-and-batch-tests.md) · [July diagnostics](../../lab-notebook/2026-07-14-diagnostic-reconstruction.md) · [Speculation and August records](../../lab-notebook/2026-08-16-speculation-and-rebench-records.md) · [All benchmark rows](../../data/galactus-archive-benchmarks.md)

## File inventory

| Original archive member | Preserved file | Original bytes |
|---|---|---:|
| `tests/bench-qwen35-397b-20260411-203129.log` | [april/bench-qwen35-397b-20260411-203129.log](april/bench-qwen35-397b-20260411-203129.log) | 21859 |
| `tests/bench_batchsweep_20260411_091944.log` | [april/bench_batchsweep_20260411_091944.log](april/bench_batchsweep_20260411_091944.log) | 8674 |
| `tests/bench_batchsweep_20260411_092044.log` | [april/bench_batchsweep_20260411_092044.log](april/bench_batchsweep_20260411_092044.log) | 3018 |
| `tests/bench_batchsweep_20260411_093230.log` | [april/bench_batchsweep_20260411_093230.log](april/bench_batchsweep_20260411_093230.log) | 14136 |
| `tests/diag-20260714-110813/MAIN.log` | [july/diag-20260714-110813/MAIN.log.gz](july/diag-20260714-110813/MAIN.log.gz) | 26699702 |
| `tests/diag-20260714-110813/monitors/cpumhz.txt` | [july/diag-20260714-110813/monitors/cpumhz.txt](july/diag-20260714-110813/monitors/cpumhz.txt) | 44498 |
| `tests/diag-20260714-110813/monitors/meminfo.txt` | [july/diag-20260714-110813/monitors/meminfo.txt](july/diag-20260714-110813/monitors/meminfo.txt) | 110250 |
| `tests/diag-20260714-110813/monitors/rocm-smi.txt` | [july/diag-20260714-110813/monitors/rocm-smi.txt](july/diag-20260714-110813/monitors/rocm-smi.txt) | 877909 |
| `tests/diag-20260714-110813/monitors/vmstat.txt` | [july/diag-20260714-110813/monitors/vmstat.txt](july/diag-20260714-110813/monitors/vmstat.txt) | 143645 |
| `tests/diag-v4-20260714-150115/MAIN.log` | [july/diag-v4-20260714-150115/MAIN.log.gz](july/diag-v4-20260714-150115/MAIN.log.gz) | 74833811 |
| `tests/diag-v4-20260714-150115/monitors/rocm-smi.txt` | [july/diag-v4-20260714-150115/monitors/rocm-smi.txt.gz](july/diag-v4-20260714-150115/monitors/rocm-smi.txt.gz) | 1319628 |
| `tests/diag-v4-20260714-150115/monitors/vmstat.txt` | [july/diag-v4-20260714-150115/monitors/vmstat.txt](july/diag-v4-20260714-150115/monitors/vmstat.txt) | 214392 |
| `tests/dspark-armed-test.log` | [dspark-v4/draft-armed-nmax5.log](dspark-v4/draft-armed-nmax5.log) | 3418 |
| `tests/dspark-baseline.log` | [dspark-v4/target-baseline-empty.log](dspark-v4/target-baseline-empty.log) | 0 |
| `tests/dspark-nmax1.log` | [dspark-v4/dspark-nmax1.log](dspark-v4/dspark-nmax1.log) | 3605 |
| `tests/dspark-nmax2.log` | [dspark-v4/dspark-nmax2.log](dspark-v4/dspark-nmax2.log) | 3605 |
| `tests/dspark-nmax5.log` | [dspark-v4/dspark-nmax5.log](dspark-v4/dspark-nmax5.log) | 3730 |
| `tests/galactus-prefill.sh` | [scripts/galactus-prefill.sh.txt](scripts/galactus-prefill.sh.txt) | 24215 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260407_102636.log` | [april/llama_bench_cpu_experts_thread_sweep_20260407_102636.log](april/llama_bench_cpu_experts_thread_sweep_20260407_102636.log) | 12867 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260408_130516.log` | [april/llama_bench_cpu_experts_thread_sweep_20260408_130516.log](april/llama_bench_cpu_experts_thread_sweep_20260408_130516.log) | 25734 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260409_121606.log` | [april/llama_bench_cpu_experts_thread_sweep_20260409_121606.log](april/llama_bench_cpu_experts_thread_sweep_20260409_121606.log) | 11985 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260410_183103.log` | [april/llama_bench_cpu_experts_thread_sweep_20260410_183103.log](april/llama_bench_cpu_experts_thread_sweep_20260410_183103.log) | 10293 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260417_091321.log` | [april/llama_bench_cpu_experts_thread_sweep_20260417_091321.log](april/llama_bench_cpu_experts_thread_sweep_20260417_091321.log) | 3594 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260417_091356.log` | [april/llama_bench_cpu_experts_thread_sweep_20260417_091356.log](april/llama_bench_cpu_experts_thread_sweep_20260417_091356.log) | 8841 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260417_103237.log` | [april/llama_bench_cpu_experts_thread_sweep_20260417_103237.log](april/llama_bench_cpu_experts_thread_sweep_20260417_103237.log) | 8829 |
| `tests/ot-placement-test.txt` | [july/glm52-expert-placement-model-allocation-failure.log](july/glm52-expert-placement-model-allocation-failure.log) | 123320 |
| `tests/ot3.txt` | [july/glm52-placement-ot3-load-failure.log](july/glm52-placement-ot3-load-failure.log) | 900 |
| `tests/ot6.txt` | [july/glm52-placement-ot6-context-failure.log](july/glm52-placement-ot6-context-failure.log) | 915 |
| `tests/ot7.txt` | [july/glm52-placement-ot7-pp2048.log](july/glm52-placement-ot7-pp2048.log) | 1050 |
| `tests/qwen-benchmark.log` | [april/qwen35-q6-cpu-experts-thread-sweep.log](april/qwen35-q6-cpu-experts-thread-sweep.log) | 12867 |
| `tests/qwen_batch_sweep.sh` | [scripts/qwen-batch-sweep.sh.txt](scripts/qwen-batch-sweep.sh.txt) | 1321 |
| `tests/rebench-2tb/build.txt` | [august/build.txt](august/build.txt) | 66 |
| `tests/rebench-2tb/deepseek-v4-flash.md` | [august/deepseek-named-file-with-1026b-model-output.log](august/deepseek-named-file-with-1026b-model-output.log) | 1143 |
| `tests/rebench-2tb/dsv4-dspark-n3.txt` | [august/deepseek-v4-dspark-draft-length-sweep.log](august/deepseek-v4-dspark-draft-length-sweep.log) | 29945 |
| `tests/rebench-2tb/glm-5.2.md` | [august/glm-5.2.md](august/glm-5.2.md) | 1145 |
| `tests/rebench-2tb/glm-base.txt` | [august/glm-base.txt](august/glm-base.txt) | 31085 |
| `tests/rebench-2tb/glm-mtp-n2.txt` | [august/glm52-mtp-draft-length-sweep.log](august/glm52-mtp-draft-length-sweep.log) | 14066 |
| `tests/rebench-2tb/minimax-m2.7.md` | [august/minimax-m2.7.md](august/minimax-m2.7.md) | 1151 |
| `tests/rebench-2tb/qwen-3.5-397b-offload.md` | [august/qwen-3.5-397b-offload.md](august/qwen-3.5-397b-offload.md) | 915 |
| `tests/rebench-2tb/qwen-3.5-397b.md` | [august/qwen-3.5-397b.md](august/qwen-3.5-397b.md) | 1143 |
| `tests/rebench-2tb/qwen-mtp-n2.txt` | [august/qwen-mtp-n2.txt](august/qwen-mtp-n2.txt) | 1772 |
| `tests/sched.txt` | [july/glm52-scheduler-empty.log](july/glm52-scheduler-empty.log) | 0 |
| `tests/sched_full.txt` | [july/glm52-scheduler-original.log.gz](july/glm52-scheduler-original.log.gz) | 4754444 |
| `tests/sched_patched.txt` | [july/glm52-scheduler-distributed.log.gz](july/glm52-scheduler-distributed.log.gz) | 4753492 |
| `tests/ts.txt` | [july/glm52-tensor-split-context-failure.log](july/glm52-tensor-split-context-failure.log) | 945 |
| `tests/ts2.txt` | [july/glm52-tensor-split-device-list-only.log](july/glm52-tensor-split-device-list-only.log) | 427 |

Use `gzip -dc path/to/log.gz` to read compressed logs. `SHA256SUMS` verifies stored files; `manifest.json` also verifies the decompressed originals. The two empty files establish no measurement.

Undated supporting records: [scheduler traces](../../lab-notebook/undated-glm52-scheduler-traces.md) and [V4 DSpark acceptance logs](../../lab-notebook/undated-deepseek-v4-dspark-acceptance.md).
