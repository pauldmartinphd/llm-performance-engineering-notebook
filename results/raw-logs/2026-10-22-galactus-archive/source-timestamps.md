# Original file modification times

I recovered these timestamps from the retained extraction of `tests.tar.gz` on October 23, 2026 and checked every file against its original SHA-256 hash in the manifest. The original archive is no longer at its supplied Downloads path, so this check uses the extracted originals rather than rereading the tar headers. The July diagnostic files end at times consistent with their embedded monitoring timestamps.

I use these modification dates to place captures that contain only elapsed time. They identify when a file was last written, not necessarily when a run began; a copied or edited capture can have a later modification time. Times below use America/New_York (UTC-04:00 for these dates). The manifest also preserves Unix timestamps and UTC values. Git checkout times and the notebook import dates are not experiment dates.

| Original member | Last modified (America/New_York) |
|---|---|
| `tests/bench-qwen35-397b-20260411-203129.log` | 2026-04-12T04:15:04-04:00 |
| `tests/bench_batchsweep_20260411_091944.log` | 2026-04-11T09:19:45-04:00 |
| `tests/bench_batchsweep_20260411_092044.log` | 2026-04-11T09:29:36-04:00 |
| `tests/bench_batchsweep_20260411_093230.log` | 2026-04-11T10:26:00-04:00 |
| `tests/diag-20260714-110813/MAIN.log` | 2026-07-14T14:25:10-04:00 |
| `tests/diag-20260714-110813/monitors/cpumhz.txt` | 2026-07-14T14:25:02-04:00 |
| `tests/diag-20260714-110813/monitors/meminfo.txt` | 2026-07-14T14:25:00-04:00 |
| `tests/diag-20260714-110813/monitors/rocm-smi.txt` | 2026-07-14T14:25:02-04:00 |
| `tests/diag-20260714-110813/monitors/vmstat.txt` | 2026-07-14T14:25:06-04:00 |
| `tests/diag-v4-20260714-150115/MAIN.log` | 2026-07-14T19:53:32-04:00 |
| `tests/diag-v4-20260714-150115/monitors/rocm-smi.txt` | 2026-07-14T19:54:06-04:00 |
| `tests/diag-v4-20260714-150115/monitors/vmstat.txt` | 2026-07-14T19:54:11-04:00 |
| `tests/dspark-armed-test.log` | 2026-08-07T19:55:26-04:00 |
| `tests/dspark-baseline.log` | 2026-08-07T19:43:10-04:00 |
| `tests/dspark-nmax1.log` | 2026-08-07T19:47:02-04:00 |
| `tests/dspark-nmax2.log` | 2026-08-07T19:47:46-04:00 |
| `tests/dspark-nmax5.log` | 2026-08-07T19:48:34-04:00 |
| `tests/galactus-prefill.sh` | 2026-07-14T15:01:08-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260407_102636.log` | 2026-04-07T12:31:54-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260408_130516.log` | 2026-04-08T17:22:20-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260409_121606.log` | 2026-04-09T13:46:54-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260410_183103.log` | 2026-04-10T19:02:18-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260417_091321.log` | 2026-04-17T09:13:24-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260417_091356.log` | 2026-04-17T09:35:52-04:00 |
| `tests/llama_bench_cpu_experts_thread_sweep_20260417_103237.log` | 2026-04-17T11:24:01-04:00 |
| `tests/ot-placement-test.txt` | 2026-07-14T18:13:25-04:00 |
| `tests/ot3.txt` | 2026-07-14T18:15:45-04:00 |
| `tests/ot6.txt` | 2026-07-14T18:18:59-04:00 |
| `tests/ot7.txt` | 2026-07-14T18:22:53-04:00 |
| `tests/qwen-benchmark.log` | 2026-04-07T20:47:17-04:00 |
| `tests/qwen_batch_sweep.sh` | 2026-04-11T12:24:32-04:00 |
| `tests/rebench-2tb/build.txt` | 2026-08-15T10:16:00-04:00 |
| `tests/rebench-2tb/deepseek-v4-flash.md` | 2026-08-15T21:58:56-04:00 |
| `tests/rebench-2tb/dsv4-dspark-n3.txt` | 2026-08-16T12:38:58-04:00 |
| `tests/rebench-2tb/glm-5.2.md` | 2026-08-15T13:07:13-04:00 |
| `tests/rebench-2tb/glm-base.txt` | 2026-08-16T10:33:24-04:00 |
| `tests/rebench-2tb/glm-mtp-n2.txt` | 2026-08-16T12:23:28-04:00 |
| `tests/rebench-2tb/minimax-m2.7.md` | 2026-08-16T09:54:48-04:00 |
| `tests/rebench-2tb/qwen-3.5-397b-offload.md` | 2026-08-16T09:44:29-04:00 |
| `tests/rebench-2tb/qwen-3.5-397b.md` | 2026-08-16T09:01:23-04:00 |
| `tests/rebench-2tb/qwen-mtp-n2.txt` | 2026-08-16T10:05:26-04:00 |
| `tests/sched.txt` | 2026-07-14T18:33:00-04:00 |
| `tests/sched_full.txt` | 2026-07-14T19:07:54-04:00 |
| `tests/sched_patched.txt` | 2026-07-14T19:58:26-04:00 |
| `tests/ts.txt` | 2026-07-14T18:30:53-04:00 |
| `tests/ts2.txt` | 2026-07-14T18:31:33-04:00 |
