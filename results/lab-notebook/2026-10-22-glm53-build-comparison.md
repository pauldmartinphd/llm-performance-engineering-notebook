# GLM-5.3-Flash: revision-identified builds and placement tests

**Record:** October 22, 2026, from the supplied filename; individual launches have no printed timestamps. **Machine:** Galactus, four V620s. [Source and checksums](../raw-logs/2026-10-23-maya-glm-and-builds/README.md).

I built llAMDa and llama.cpp for gfx1030 with HIP, GCC 14.2.0, and Zen 3 CPU flags (`-O3 -march=znver3 -mtune=znver3`, `GGML_NATIVE=OFF`). The configuration identifies HIP Clang 22.0.0. The build records name `ca3b1fc30` for llAMDa and `184f65360` for llama.cpp, and both installations used `/usr/local`.

The direct benchmark outputs distinguish those revisions, but the CLI captures expose a provenance problem: commands named both `llama-cli` and `llamda-cli` print **b11829-184f65360**. I cannot treat those names alone as proof that the CLI runs exercised different implementations. Shared installation paths or libraries are possible explanations; the capture does not identify the cause.

## Dedicated benchmarks

Both commands use the six-shard GLM-5.3-Flash UD-Q4_K_XL export, reported as 185.98 GiB and 320.76 B parameters, with 32 threads, `-ngl 99`, `-ot "exps=CPU"`, flash attention, and batch/microbatch 8192. Each requests two repetitions. The command supplies both `-n 128` and `-n 256`, and the output contains both generation tests.

| Executable and printed build | pp8192, tokens/s | tg128, tokens/s | tg256, tokens/s |
|---|---:|---:|---:|
| llama-bench, 184f65360 (11829) | 284.36 ± 16.33 | 10.06 ± 0.01 | 10.16 ± 0.04 |
| llamda-bench, ca3b1fc30 (11810) | 284.20 ± 16.42 | 10.07 ± 0.10 | 10.09 ± 0.02 |

These measurements show closely similar performance under this configuration. They do not isolate the scheduler patch: the builds differ, and there is no patch-off control. The ± values are reproduced as printed, not confidence intervals. No speculative settings appear in these benchmark commands.

```sh
llama-bench \
  -m /models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  -ngl 99 -ot "exps=CPU" -fa on -t 32 \
  -b 8192 -ub 8192 -p 8192 -n 128 -r 2 -o md -n 256
```

The second command is identical apart from the executable name, `llamda-bench`. The source also contains an earlier invalid attempt that passed CLI-style arguments to `llama-bench`; it failed on `-ngl all` and produced no timing.

## CLI placement and draft-length tests

The completed CLI runs use the ZFS explanation prompt, 32 threads, context 8192, flash attention, greedy decoding, a 256-token output limit, and `--spec-type draft-mtp`. I report generation only. “Forced RAM” means `-ngl all -cmoe`; “automatic” means both options were omitted. Effective tensor placement and draft acceptance are not captured.

| Source command line | Executable name | Requested placement | Draft n-max | Generation, tokens/s |
|---|---|---|---:|---:|
| 2417 | llama-cli | Forced RAM | 2 | 13.7 |
| 2478 | llamda-cli | Forced RAM | 2 | 12.3 |
| 2539 | llamda-cli | Forced RAM | 2 | 13.7 |
| 2612 | llamda-cli | Automatic | 2 | 17.8 |
| 2672 | llamda-cli | Automatic | 2 | 18.0 |
| 2732 | llama-cli | Automatic | 2 | 18.1 |
| 2792 | llama-cli | Automatic | 4 | 16.2 |
| 2839 | llama-cli | Automatic | 3 | 17.8 |
| 2887 | llama-cli | Automatic | 1 | 17.7 |

Every completed CLI row prints b11829-184f65360. At source line 2600, requesting `-ngl all` without `-cmoe` failed while allocating **40844.55 MiB on ROCm0**. Two further CLI command lines at 2947–2948 have no output and are not counted as completed runs.

Automatic placement produced the highest generation rates in this record, while increasing the requested draft length beyond two did not improve the observed rate. These are short runs with no acceptance counts or speculation-off control, so they do not establish MTP's independent contribution. The [next day's Maya A/B test](2026-10-23-maya-glm53-speculation.md) supplies that control within Maya, using a different prompt suite and context configuration.
