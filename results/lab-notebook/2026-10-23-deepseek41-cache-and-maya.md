# DeepSeek-V4.1: expert-cache capacity and Maya DSpark results

**Recorded:** October 23, 2026. Maya prints a benchmark date of October 23 at 14:55; the llAMDa cache runs lack individual timestamps. **Machine:** Galactus, Proxmox LXC, EPYC 7713, 2 TiB RAM, four Radeon Pro V620 GPUs. **Source:** [retained benchmark and configuration evidence](../raw-logs/2026-10-23-maya-comparison/README.md). The device output reports 30,704 MiB per GPU, 122,816 MiB in total, gfx1030, wave size 32, and no VMM.

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



## Configuration recovered from shell history on October 28

I recovered the setup and configuration-edit commands from my exported `zsh_history`. The timestamps below are the history's Unix timestamps converted to America/New_York (EDT, UTC−04:00). History records commands entered, not their exit status or output; I therefore distinguish these commands from the benchmark and calibration output retained above. I have included the relevant settings and sequence here without publishing unrelated shell history.

At **13:51:33 on October 23** (1792777893), the recorded setup command selected:

```sh
./maya.sh \
  --backend hip \
  --setup \
  --model DeepSeek-V4.1-Flash-1017 \
  --ds4-quant gguf-q8 \
  --model-dir /models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL \
  --gpus 0,1,2,3 \
  --context 32768 \
  --no-vision \
  --port 8099 \
  --no-start
```

The history records cloning the v1.1.24 tag immediately before this setup. The setup entry also begins with `cd ~/project-maya`, while neighboring entries use `/opt/project-maya`; the history alone cannot establish whether that directory change succeeded. The later configuration edits explicitly begin in `/opt/project-maya`.

At **14:38:56** (1792780736), a Python command read `maya-ds4-hip.json`, saved its prior text to `maya-ds4-hip.json.bak`, set `DS4_ENGRAM_RAM=1` and `DS4_REQUIRE_ALL_RAM=1`, and set or appended `--mmap-experts off`. After memlock checks and a recorded shutdown, another command at **14:42:04** (1792780924) repeated those edits and added `MAYA_DS4_Q8_BATCH=1`. That second command used the same backup filename, so the history does not establish that `.json.bak` still contains the original pre-edit configuration.

| Setting | Recovered value | Evidence and remaining limit |
|---|---|---|
| Backend and GPUs | HIP; 0,1,2,3 | Setup command; four V620s also confirmed by benchmark output |
| Model | DeepSeek-V4.1-Flash-1017, UD-Q8_K_XL | Explicit setup model directory and `gguf-q8`; shard hashes not captured |
| Context | 32768 | Setup command and benchmark output |
| Vision | Off | Explicit `--no-vision` in setup command |
| Engram residency | `DS4_ENGRAM_RAM=1` | Recorded JSON edit; successful locking still requires runtime evidence |
| CPU expert residency | `DS4_REQUIRE_ALL_RAM=1`, `--mmap-experts off` | Recorded JSON edit; actual allocation still requires runtime evidence |
| Q8 batching | `MAYA_DS4_Q8_BATCH=1` | Recorded JSON edit immediately before benchmark invocation |
| Initial port | 8099 | Setup command; later changed to 8081 |
| CPU threads, TP groups, layer split, VRAM reserve | Not recovered as effective values | No final JSON or corresponding output in the supplied history |

The benchmark command was entered at **14:42:25** (1792780945), after the second edit. Its retained output prints 14:55 after loading and running the tests. These timestamps are consistent with a command start followed by a later benchmark report; they are not two separate measured runs.

At **15:00:28** (1792782028), the history records:

```sh
cp -a maya-ds4-hip.json maya-ds4-hip.json.before-dspark-retest
```

Three seconds later, the recorded Python invocation calls `maya.speed_calibration_ds4(path, cfg, "retest after full RAM residency")`. The separate retained output confirms that this calibration completed at 21.2 tokens/s without drafting and 29.3 tokens/s with DSpark, with 69% draft acceptance. If `maya-ds4-hip.json.before-dspark-retest` still exists, it is the strongest identified candidate for recovering the exact pre-calibration arguments. Its present existence and contents have not been checked.

### Changes after the measured baseline

At **16:54:40 on October 23** (1792788880), the history records renaming `maya-ds4-hip.json` to `maya-ds4.1-flash-hip.json`. At **16:54:54** (1792788894), a JSON edit sets its port to **8081**. At **16:59:46** (1792789186), another edit changes `--max-context` to **1048576**.

The 1M-token context therefore belongs to a later configuration change. It is not the context associated with the measured 32K benchmark and subsequent DSpark retest. These commands do not establish that the larger context caused the later stall.

The history also records a detached checkout command for **v1.1.34 on October 27 at 22:44:13** (1793155453), followed by setup commands. At **22:57:35** (1793156255), it records `cp -a maya-*.json config-backup/`. That directory is another potential source of earlier tuned configurations, but the copy occurred after the first recorded setup invocation on the new version.

Finally, the history contains checkout attempts for **v1.1.35**, including one after fetching tags on **October 28 at 12:24:41** (1793204681), followed by `./setup.sh --backend hip`. The live checkout must therefore be inspected rather than assumed to remain at v1.1.34. No command output in this history confirms the checkout or build outcome.

### Rebuilding the recorded configuration

A v1.1.34 rebuild can explicitly restore the recovered 32K, four-GPU, text-only, RAM-residency settings. It cannot yet be called an exact reproduction of the October 23 configuration because the effective CPU-thread count, TP groups, layer split, and other generated arguments remain unknown. The [v1.1.34 setup source](https://github.com/mw00/project-maya/blob/b04264526e93a3c7d3c051cb9cec97af86ae8a37/maya.py) defaults to TP=2 with four GPUs; that default is not evidence of October 23's effective TP setting.

The verified v1.1.34 tag resolves to `b04264526e93a3c7d3c051cb9cec97af86ae8a37`. Its [settings documentation](https://github.com/mw00/project-maya/blob/b04264526e93a3c7d3c051cb9cec97af86ae8a37/docs/SETTINGS.md) documents the three recovered environment settings and the relationship between `DS4_REQUIRE_ALL_RAM=1` and `--mmap-experts off`.

For the next run, I will preserve the old checkout, raw results, tuned configurations, and existing model pack before rebuilding. Model shards and DSpark files remain in place. I will record the Maya and llama.cpp revisions, package versions, build log, ROCm version, GPU inventory, shell and process memlock limits, generated JSON, and model file identities. Model sizes and modification times provide an inventory; content hashes are needed to verify identical weights. Benchmark output and calibration output remain separate, with configuration snapshots before and after calibration.



**October 29 follow-up:** The backup JSON was subsequently found and printed. TP=2 and a 3072 MiB reserve are now documented, and v1.1.30 was restored. The forward-looking v1.1.34 rebuild plan above is superseded by [the October 28 recovery and tuning record](2026-10-28-maya-prefill-and-regression.md).


## October 24 continuation: live context, cooling, and Hermes

**Added:** October 29, 2026. **Sources:** [Speculative Node Setup](https://chatgpt.com/c/6adb5c97-ef6c-83e9-b985-9d01d55530f8), the October 29 continuation supplied with this notebook update, and the existing records linked below. The retrieved conversation contains the final setup exchange, but no older-page cursor; the live-context figures below are therefore retained as user-reported observations rather than reconstructed benchmark output. This section records the October 23–24 deployment and does not supersede the [October 29 production configuration](2026-10-29-maya-production-64k.md).

### Performance and context in use

The platform remains the EPYC 7713 with **2 TB RAM and four Radeon PRO V620s**. The [GLM entry](2026-10-23-maya-glm53-speculation.md) retains Project Maya **v1.1.24**, **GLM-5.3-Flash UD-Q4_K_XL**, the speculation A/B comparison, the speculative-head sweep, and the `MAYA_GLM_MTP_KEEP` sweep. The best observed default was **31.6 tokens/s generation and 242 tokens/s PP8192**, with two head GPUs and KEEP=2. Those defaults won the recorded trials, but repeated interleaved runs remain necessary to distinguish the nearby KEEP results reliably.

For **DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL**, the full-RAM-residency configuration and DSpark retest above produced **29.3 tokens/s calibrated versus 21.2 undrafted**, with **69% draft acceptance**. The configuration requests Engram and CPU-expert residency through `DS4_ENGRAM_RAM=1`, `DS4_REQUIRE_ALL_RAM=1`, and `--mmap-experts off`; shell-history edits establish the requested settings, while the [October 24 dashboard](../deepseek-v4.1-flash.md#october-24-dashboard-from-ongoing-use) provides separate runtime evidence of 11,773 experts in pinned RAM, 3,587 in VRAM, and zero SSD-only experts. This supports RAM/VRAM expert residency for that session, without establishing every Engram allocation or proving that memory locking alone caused the calibration gain.

| Observation | Value | Evidence and limit |
|---|---|---|
| Configured context after calibration | **1,048,576 tokens (1M / 1024K)** | October 23 configuration-edit command and October 24 dashboard; the calibration itself used a 32K configuration |
| Context during later live use | **Approximately 69–82K tokens** | User-reported in the October 29 continuation; no corresponding request log retained here |
| Generation during that live use | **Approximately 27–30 tokens/s** | User-reported operating range, not a controlled long-context benchmark |
| Separate retained dashboard | **28.0 tokens/s aggregate**, 90 requests, 86,075 output tokens | October 24 screenshot; its last request used 2.2K context, so it does not verify the reported 69–82K interval |

The configured ceiling, live context in use, and benchmark prompt length are different quantities. Neither the live-use report nor the dashboard establishes throughput or answer quality at a full million tokens.

### Deployment and Hermes on magneto

The intended route is **Hermes on `magneto` → authenticated Open WebUI API → Maya on the inference container's localhost**. The conversation records Open WebUI's upstream connection as `http://localhost:8081/v1`, with Chat Completions enabled and no upstream authentication. It describes Maya as bound to localhost on port 8081; that is the recorded deployment arrangement, not a fresh listener inspection in this update.

| Component | Recorded name or endpoint | Status |
|---|---|---|
| Open WebUI | `http://10.0.0.51:8080` | Existing frontend; user reports API keys enabled |
| Hermes API base | `http://10.0.0.51:8080/api` | Setup value; answer **no** to appending `/v1` |
| Model discovery | `GET /api/models` | Authenticated check still to confirm |
| Chat inference through Open WebUI | `POST /api/chat/completions` | Normal response and tool calling still to validate from Hermes |
| DeepSeek configuration | `maya-ds4.1-flash-hip.json`, port **8081** | Rename and port edit appear in shell history; dashboard separately confirms port 8081 |
| GLM configuration | `maya-glm5.3-flash-hip.json`, proposed port **8082** | Requested deployment arrangement; the original run used a temporary **8100** benchmark port, and the retained source does not confirm the final listener on 8082 |

On `magneto`, run `hermes model`, select the custom OpenAI-compatible endpoint, and use the API base above. Supply the Open WebUI account's API key privately. Use the exact model ID returned by `/api/models`; `deepseek-v4.1-flash` is the proposed ID, pending discovery. API-key enablement is recorded, but a successful authenticated model listing, completed Hermes configuration, and end-to-end tool call have not yet been demonstrated in the available exchange.

The model-list check uses zsh's hidden-input syntax. This procedure holds the key in a temporary shell variable and clears it afterward; the notebook contains no key value.

```zsh
read -rs "WEBUI_KEY?Open WebUI API key: "
echo
curl -fsS 'http://10.0.0.51:8080/api/models' \
  -H "Authorization: Bearer $WEBUI_KEY" \
  | python3 -m json.tool
unset WEBUI_KEY
```

After configuring Hermes, test a normal response and a small tool-calling request. Offering Hermes itself as an additional Open WebUI model was discussed as a separate arrangement; no completed deployment of that route is established here.

### GPU and RDIMM cooling

The [October 1 thermal record](../../hardware/galactus/README.md#thermal-testing-and-fan-control) establishes that the GPU duct also cools the adjacent RDIMMs. FAN2 drives the two high-pressure 80 mm fans feeding the four-GPU duct. At **70%**, the recorded speed was about **5,100 RPM**, with the hottest DIMM around **70–74 °C** during inference. At 100%, the hottest DIMM reached about 63 °C after ten minutes; this was an observation during cooling, not a demonstrated equilibrium. The earlier low-airflow incident reached approximately 84–85 °C and was followed by ECC/RAS trouble, without captured error counts or a diagnosis of permanent damage.

The **70–80% fan comparison remains an open test**. The existing proposed curve includes 80% duty at a 70 °C DDR4C reading, but that control point is not a measured 80% trial. A matched sustained inference workload should record GPU temperature, RDIMM temperatures by channel, fan RPM, ambient conditions, and throughput at each duty setting, with enough time to assess stabilization. The available record does not establish an 80% thermal result, a GPU throttling threshold, or a throughput gain from increasing the fans.

### Remaining validation

- [ ] Confirm the authenticated model list and exact model ID, then complete a normal Hermes response and tool call through Open WebUI.
- [ ] Capture the effective configuration and listener state for the historical naming/port arrangement if it is still needed; keep it distinct from the newer production configuration.
- [ ] Retain request timings, actual prompt/context counts, and quality checks for the reported 69–82K live-use range; a full 1M-context test remains unperformed in this record.
- [ ] Run a controlled 70% versus 80% fan comparison with GPU and RDIMM telemetry.
- [ ] Repeat the close GLM MTP_KEEP cases with interleaved baselines before treating the observed default as a stable optimum.


## Residency diagnostics and comparison proposals retained from the source discussion

**Consolidated October 29:** The [extracted evidence](../raw-logs/2026-10-23-maya-comparison/benchmark-and-residency-evidence.md) retains the supplied LXC configuration, unlimited shell limits, full 32 GiB cache command/table, Maya benchmark, calibration response, and internal DSpark invocation. The removed dialogue also proposed the following checks; they remain proposals unless an earlier section records their results.

The supplied CT 100 configuration includes an 8T root filesystem, 512 MiB swap allowance, autostart, nesting/keyctl/fuse, render devices 128–131 with gid 992, card devices 0–4 with gid 44, and `/dev/kfd`. The proposed memlock change was `lxc.prlimit.memlock: unlimited` in `/etc/pve/lxc/100.conf`, followed by `pct shutdown 100` and `pct start 100`. Such a restart interrupts Open WebUI and other container services. Only the later shell limits were printed: the proposed `prlimit --pid 1 --memlock` check has no captured result. If PID 1 and a login shell differ, their service/login limits would need separate inspection. Making the container privileged was not part of the proposed fix.

The discussion estimated the Engram tables at approximately 203 GB and described strict residency as failing initialization rather than silently falling back to SSD. Those are source claims, not an allocation measurement in the retained calibration output. The JSON-edit procedure saved `.json.bak`, set the three environment values shown above, and replaced or appended `--mmap-experts off`. `mlock` requests do not raise process limits, and unlocked file pages can still remain in the page cache. The earlier claims that this llAMDa build warns and continues on locking failure and that its `-lm mlock` path is non-mmap require checking the actual revision; neither claim establishes SSD traffic.

The proposed diagnostic collection was `./maya.sh --backend hip --report`, followed by inspection of `maya-ds4.log` for `Engram RAM ONLY`, experts copied to RAM, VRAM, PCIe, disk, `STAT`, decode, and prefill lines. The intended measurements were GPU expert-cache hit rate, CPU expert work, transfers, and disk reads per token. These would help distinguish placement from HIP execution and multi-GPU scheduling. The 691-second initial load and 169/202-second calibration loads remain separate observations because initialization and cache state were not controlled.

The successful retest used DSpark, not the DFlash2 name mentioned in the initial question. A suggestion to rerun the ordinary benchmark after calibration would verify that normal serving retained DSpark; no such additional result is in these exports. A hardware-calibration suggestion aimed to explore CPU threads and tensor parallelism, but does not itself establish successful HIP calibration.

The proposed llAMDa retest used the same DeepSeek shard path as the measured 32 GiB command, `-ngl 99`, `-ot 'engram_embd=CPU;exps=CPU'`, `--moe-cache-mib 65536`, `-lm mlock -lzm off`, `-fa on`, `-t 32`, `-b 8192 -ub 8192`, `-p 8192 -n 1024 -r 2 -o md`. Unlike the retained 32 GiB command, it did not explicitly include `-sm layer`. The earlier reference was 132.11/11.56 tokens/s; the later summary's 132.82/11.54 remains distinct. Filesystem cache state and simultaneous Maya activity would need control before attributing a change to memlock.

A second proposal changed threads from 32 to 64 after extracting Maya's `gpu`, `args`, and `env` fields from `maya-ds4-hip.json`. It aimed for the same four GPUs, weights, 32K context, resident CPU experts/Engram, and initially no speculation, but neither Maya's worker count nor equivalent context handling was established by that suggestion. A 64 GiB dynamic expert cache is not equivalent to Maya's proposed popularity-based preload/adaptive residency and CPU lane, nor does a 3 GiB-per-GPU reserve translate directly into that cache capacity. Separate llama-bench PP/TG tests also differ from generation following Maya's actual prompts. A 48–60 GiB cache sweep was suggested as a possible follow-up, with hit rates, PCIe traffic, and GPU execution measured before assigning a cause.

The original Flash llAMDa proposal used the six-shard GLM path, CPU expert override, 64 GiB cache, locked loading/lazy mode off, 32 threads, and PP8192/TG1024 with batch and microbatch 8192 and two repetitions. The flagship proposal retained that workload but required checking its actual tensor structure before choosing placement. Neither proposal supplies a new result here. Later native-quant and flagship suggestions are retained in the [GLM setup record](2026-10-23-maya-glm53-speculation.md#setup-choices-and-proposals-retained-from-the-source-discussion).
