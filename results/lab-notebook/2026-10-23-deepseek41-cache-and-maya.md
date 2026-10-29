# DeepSeek-V4.1: expert-cache capacity and Maya DSpark results

**Recorded:** October 23, 2026. Maya prints a benchmark date of October 23 at 14:55; the llAMDa cache runs lack individual timestamps. **Machine:** Galactus, Proxmox LXC, EPYC 7713, 2 TiB RAM, four Radeon Pro V620 GPUs. **Source:** [retained conversation export](../raw-logs/2026-10-23-maya-comparison/README.md). The device output reports 30,704 MiB per GPU, 122,816 MiB in total, gfx1030, wave size 32, and no VMM.

I am comparing llAMDa.cpp with Project Maya v1.1.24 to understand how much performance depends on the inference engine and its placement decisions. The latest DeepSeek result preserves a command and printed output for one cache setting, although the executable’s source revision remains unknown. The supplied conversations preserve a broader cache sweep, a Maya benchmark, and a successful DSpark calibration. Only the 32 GiB cache row has its full llAMDa command and output; Maya has its own benchmark and calibration output. Their workloads differ, so I keep the engine results separate.

## DeepSeek cache sweep

The model is DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL. The printed benchmark describes it as `deepseek41 ?B MXFP4 MoE`, 473.46 GiB and 748.49 B parameters; the filename does not mean every tensor is stored at eight bits. Requested cache capacity is not a measurement of cache occupancy or a per-GPU allocation.

| Requested expert cache | pp8192, tokens/s | tg1024, tokens/s | Evidence retained |
|---|---:|---:|---|
| Disabled | 132.54 | 10.20 | Conversation summary only |
| 32 GiB (32768 MiB) | 132.66 ± 0.30 | 9.51 ± 0.07 | Command and printed benchmark table |
| 64 GiB | 132.82 | 11.54 | Conversation summary only |
| 80 GiB | — | — | OOM reported in summary; failure stage not captured |

The reported generation rates are not monotonic: 32 GiB is about 6.8% below the cache-disabled result, while 64 GiB is about 13.1% above it. The three reported prompt rates span only 0.28 tokens/s. This makes 64 GiB a useful provisional baseline for the next run, but does not identify an optimal capacity. Cache misses, replacement work, and transfers could explain the difference; without hit rates, placement records, or matched repeated captures, I cannot identify the cause.

The complete command available for the 32 GiB run is:

```sh
llamda-bench \
  -m /models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
  -ngl 99 \
  -sm layer \
  -ot 'engram_embd=CPU;exps=CPU' \
  --moe-cache-mib 32768 \
  -lm mlock -lzm off \
  -fa on \
  -t 32 \
  -b 8192 -ub 8192 \
  -p 8192 -n 1024 -r 2 \
  -o md
```

This requests CPU placement for Engram embeddings and routed experts, layer splitting, locked loading, lazy mode off, flash attention, and 32 threads. The ± values are reproduced from the benchmark, not confidence intervals. The output does not identify the executable's source revision, actual cache occupancy, or effective tensor placement. The command contains no draft-model configuration. Its separate pp8192 and tg1024 rows do not establish generation performance after an 8,192-token prompt, and they are not interchangeable with the older short-prompt CLI measurements or August tg128 baseline.

## LXC limits and Maya residency

The supplied container configuration shows an unprivileged Debian LXC with `cores: 128`, `memory: 2097152`, four render-device mappings, and `/dev/kfd` passed through. The configuration already assigned 2 TiB of memory; this record does not establish a RAM-capacity increase during the investigation.

The conversation identifies an earlier 8 MiB memlock limit and proposes `lxc.prlimit.memlock: unlimited` in the container configuration. After reconnecting, both `ulimit -Sl` and `ulimit -Hl` print `unlimited`. Those are direct checks of the login shell's limits, not a measurement of how many bytes the inference process actually locked. The original low-limit output and updated container configuration are not included.

The next proposed Maya configuration sets `DS4_ENGRAM_RAM=1`, `DS4_REQUIRE_ALL_RAM=1`, and `MAYA_DS4_Q8_BATCH=1`, with `--mmap-experts off`. The later discussion describes the benchmark as running after these residency changes. I have retained that sequence, but the final JSON and engine residency diagnostics are still needed to confirm every effective setting.

Maya v1.1.24's [release notes](https://github.com/mw00/project-maya/releases/tag/v1.1.24) document the locked-Engram and required-RAM options, along with the Q8 batching option. The discussion proposes several settings together, but their effective application is not fully captured. The improvement cannot be assigned to memlock alone. The release's new RDNA 4 attention path also does not establish acceleration on the gfx1030 V620s.

For llAMDa, `-lm mlock` in the command is not evidence that locking succeeded. A failed locking request would also not, by itself, prove that SSD reads limited inference. I need the loader warnings, process limits and locked-memory counters, and I/O measurements before attributing its performance to the container limit.

## Maya benchmark and DSpark calibration

The retained command `./maya.sh --backend hip --bench` selected `maya-ds4-hip.json`. Its output identifies **Project Maya v1.1.24**, all four V620s, **context 32768**, **2004 GB RAM as reported by the tool**, and the EPYC 7713. It loaded in **691 seconds**.

| Maya benchmark measurement | Result |
|---|---:|
| Prefill, 2048-token prompt | 135 tokens/s |
| Prefill, 8192-token prompt | 282 tokens/s |
| CSV parsing answer, 256 tokens | 20.4 tokens/s |
| Photosynthesis answer, 256 tokens | 20.4 tokens/s |
| HTML/canvas answer, 256 tokens | 20.6 tokens/s |
| Printed decode mean | 20.5 tokens/s |

The three answers used greedy decoding with thinking off. The surrounding conversation identifies this as the undrafted baseline; the benchmark output alone does not print the draft setting.

The attempted `--speed-calibration --no-start` command did **not** run a DSpark comparison: it returned “nothing to tune here.” The successful comparison instead called `maya.speed_calibration_ds4(path, cfg, "retest after full RAM residency")` from Maya's Python environment. That is an internal function used in this version, not a verified general-purpose CLI procedure.

| Calibration mode | Load time | Per-answer tokens/s | Sustained tokens/s | Draft acceptance |
|---|---:|---|---:|---:|
| Without drafting | 169 s | 20.9 / 21.3 / 21.2 | 21.2 | — |
| DSpark | 202 s | 30.8 / 27.2 / 30.1 | 29.3 | 69% |

Each mode first read a 512-token document and answered once as an uncounted warm-up, then generated three 384-token answers. Although the progress messages target 1024 tokens, the sustained summary explicitly reports **1152 tokens** per mode: 54 seconds without drafting and 39 seconds with DSpark, rounded as printed. I retain the tool's sustained rates rather than recompute them from rounded seconds.

DSpark improved sustained generation by **38%** within this calibration, and the tool reports selecting it for the configuration. This is the clearest measured comparison in the record. The initial 3.1 tokens/s without drafting and 1.3 with drafting appear only in conversation summaries. They provide context for the retest, but not a controlled before/after measurement of the memory changes. Likewise, the shorter calibration load times cannot be attributed to a particular change without controlling the cache state.

The older discussion cites a 64 GiB llAMDa result of **132.11 / 11.56 tokens/s**, while the later cache summary gives **132.82 / 11.54**. I retain both as distinct summary records, rather than silently substitute one for the other. Neither includes its printed output in the supplied material. Maya's three-prompt benchmark and its DSpark calibration also differ from llAMDa's separate pp8192/tg1024 tests. The rates make Maya worth pursuing, but do not establish an engine-only speedup. The conversation describes the intended common weights as UD-Q8_K_XL; the actual Maya model path and effective configuration still need capture.
