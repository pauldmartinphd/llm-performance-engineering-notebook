# Maya v1.1.54: prefill sweep and HIP peer all-reduce failure

[All lab records](README.md) · [Related investigation: chapter 5](../../investigations/05-tuning-long-prompts.md)

## Purpose and reasoning

Test whether new prefill, kernel, and transport switches improve DeepSeek on four V620s, then isolate the HIP peer-probe limitation with more reserve memory.

## Experimental design

I ran ten variants on v1.1.54, with two measured requests at each 8K/32K/64K length after warm-up. I retained two separate 256-token decode cases per variant. I checked transport activation in the logs. I then ran a focused HIP peer case requesting 3072 MiB reserve. All measurements and source details from this sweep are collected in this one entry.

## Results

The completed sweep contains 60 prefill rows and 20 decode cases. Default prefill means were 324.8, 459.4, and 479.4 tokens/s. Automatic chunk selection was slower; neither requested peer path activated. The focused retest completed no startup-probe samples and never reached a ready engine.

## Conclusions

No tested switch established a useful improvement over the retained prefill choice. Completed host fallback is not successful peer transport, and the later startup failure has no throughput result. Two distinct decode cases per variant do not establish a repeatable decode gain. The complete CSV is the result set; partial terminal output is not another experiment.

## Detailed record

**Experiments:** November 5, 2026. **Sources:** the supplied *Fast Card Prefill Acceleration* conversation export and `maya_benchmark_1154_fixed.py`. Conversation timestamps and result-directory names establish the run date; the host timezone is not independently captured. [Retained evidence and hashes](../raw-logs/2026-11-05-maya-v1154/README.md).

I tested ten Maya v1.1.54 configurations at 8K, 32K, and 64K prompt lengths, with two repetitions per length. The default measured **324.8, 459.4, and 479.4 tokens/s**. None of the switches established a useful prefill improvement in this sweep. Automatic chunk selection reduced throughput at all three lengths, while both requested peer transports retained host transport. A separate test requesting 3072 MiB of reserve reached the HIP transport probe but failed before the engine became ready, so it produced no throughput result.

This continues the [October 28 prefill investigation](2026-10-28-maya-prefill-and-regression.md), [October 29 production confirmation](2026-10-29-maya-production-64k.md), and [October 30 v1.1.41 tests](2026-10-30-maya-v1141-benchmark.md). No new inference run or production change accompanied this notebook update.

## Workload and provenance

The conversation identifies the system as Galactus with four Radeon Pro V620s, HIP, and DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL. The later pasted engine log independently identifies four `gfx1030` devices and 32768-token prompt chunks. TP2 and an 800 MiB ring are established configuration context; the initial conversation did not include the effective JSON, a model-pack hash, a binary hash, or the resolved release commit. The later [collection](../raw-logs/2026-11-10-maya-collection/README.md) now retains the per-variant configs and update logs; it does not supply a model-pack hash. The transport summary identifies groups GPU0/GPU1 and GPU2/GPU3.

The completed command used the production configuration as its input:

```sh
cd /opt/project-maya
.benchmark-venv/bin/python maya_benchmark_1154_fixed.py \
  /opt/project-maya \
  /opt/project-maya/maya-ds4-production.json \
  2>&1 | tee /opt/maya-benchmark-run.log
```

It reported `/opt/maya-bench-1154-20261105-142331`. The [supplied runner](../../experiments/maya_benchmark_1154_fixed.py) fetches tag `v1.1.54`, switches the supplied checkout to its detached commit, refreshes the HIP engine, and launches a separate engine process for each variant. It does not create a separate checkout, despite an earlier description in the conversation. It copies configuration data and removes the tested environment overrides from each child before applying that variant. Thus “default” means defaults for the tested switches with the remaining supplied configuration retained, not an independently verified stock configuration.

The runner builds prompts from the checked-out README, documentation, and benchmark-result Markdown. It tokenizes this corpus, rotates the token sequence by prompt length and repetition, and repeats/truncates it to exactly 8192, 32768, or 65536 tokens. Each process performs a short decode warm-up, two 256-token decode cases, and an 8K prefill warm-up before the six measured prefill requests. Prefill requests generate one token; the recorded rate is the requested token count divided by the engine's reported prompt time. Rotated leading tokens are intended to avoid prefix reuse, but these printed records contain no cache-hit counter. I therefore do not label the sweep independently verified zero-cache prefill.

The complete run contains 60 prefill measurements and 20 decode cases. The original [run CSV](../raw-logs/2026-11-10-maya-collection/extracted/maya-bench-1154-20261105-142331/results.csv) retains them all. The [conversation-derived CSV](../data/maya-v1154-prefill-observations.csv) contains only the 47 prefill observations printed in the earlier export; it is a partial transcription, not the full run. The [printed-summary CSV](../data/maya-v1154-prefill-summary.csv) retains all 30 prefill means. Repeated pasted output is counted once.

## Completed prefill comparison

These are the original printed means, in tokens/s, with two repetitions per cell. Retaining their one-decimal precision avoids inventing precision for variants whose individual records are missing.

| Variant | Override relative to tested defaults | 8K | 32K | 64K |
|---|---|---:|---:|---:|
| `default` | None | 324.8 | 459.4 | 479.4 |
| `tp_even` | `DS4_TP_SHARES=off` | 324.1 | 458.8 | 478.8 |
| `p2p_generic` | `DS4_AR_P2P=1`, `DS4_AR_P2P_LOG=1` | 324.1 | 458.6 | 479.4 |
| `p2p_hip_auto` | `MAYA_DS4_HIP_PEER_AR=auto` | 319.8 | 459.2 | 478.8 |
| `flow_dma` | `DS4_FLOW_COPY_DMA=1` | 324.4 | 458.7 | 478.9 |
| `flow_ai` | `DS4_AI_RUN=1` | 315.9 | 460.9 | 480.9 |
| `flow_chunk` | `DS4_CHUNK_AUTO=1` | 286.3 | 420.8 | 458.1 |
| `flow_yoco` | `DS4_YOCO_FLOW=1` | 324.4 | 458.9 | 478.7 |
| `flow_combined` | All four preceding flow flags | 286.5 | 425.0 | 460.1 |
| `expert_mmq` | `MAYA_DS4_EXPERT_MMQ=1` | 324.6 | 459.3 | 479.9 |

Automatic chunk selection was approximately 11.9%, 8.4%, and 4.4% slower than default at 8K, 32K, and 64K, calculated from the printed means. Combined flow was approximately 11.8%, 7.5%, and 4.0% slower. The default 32768-token chunk remains the supported choice from these measurements. The sweep does not isolate the internal chunk decisions or establish that every requested flow path activated.

`flow_ai` was approximately 0.3% faster at 32K and 64K, but its 8K repetitions were 324.74 and 307.15 tokens/s. Two repetitions in a fixed variant order cannot establish a reliable improvement of this size. MMQ also remained close to default. Its dispatch was not captured, so this is neither proof that MMQ ran nor evidence against the potential of RDNA 2 integer dot-product kernels. No quality evaluation was supplied for MMQ or the variants the runner labels approximate.

The 479.4-token/s 64K default is close to earlier long-prompt results around 480 tokens/s. It is not a matched release comparison: the October HTTP requests contained 65,568 actual tokens, used saved chat payloads, and generated 128 tokens, while this runner uses 65,536 raw corpus tokens and one output token. There is no paired v1.1.49/v1.1.54 run in this evidence.

## The completed P2P cases used host transport

The generic P2P summary repeatedly reports both groups staying host-mapped; one group is quoted here:

```text
DS4_AR_P2P: the group CUDA0 BAR1 0 MiB CUDA1 BAR1 0 MiB stays host-mapped (BAR1 under 1 GiB, or no peer access)
```

This is Maya's reported transport decision. It does not independently establish the physical BAR aperture, the BIOS configuration, or whether HIP peer access is impossible on this hardware. The near-baseline rate is a host-transport measurement with P2P requested, not a successful P2P benchmark.

The HIP-specific case reports a different obstacle:

```text
peer prefill all-reduce unavailable on GPU0/GPU1 (GPU0 has 1026 MiB free, needs 1984 (auto: + the probe's input row)); retaining host transport
peer prefill all-reduce unavailable on GPU2/GPU3 (GPU2 has 1006 MiB free, needs 1984 (auto: + the probe's input row)); retaining host transport
```

The available workspace was below the probe's stated requirement. This explains the host fallback in the completed sweep, but does not establish that additional reserve alone would make peer transport work.

## Focused 3072 MiB reserve retest

The conversation then proposed a copy of the production JSON named `maya-ds4-p2p-test.json`, changing only the engine argument `--reserve-mib` to `3072`, and a derived `maya_benchmark_p2p_test.py` restricted to `p2p_hip_auto`. The pasted output confirms a one-variant run under `/opt/maya-bench-1154-20261105-201458`; the initial export did not include either derived file. Both are now retained in the November 10 collection, including the config requesting 3072 MiB reserve.

The process exited before `READY`. Its log ends with:

```text
PEER_AR_AUTO gpu=0,1 floats=167772160 samples=0 host_ms=0.000000 ar2_ms=0.000000 peer_ms=0.000000 selected=host elapsed_ms=10295.2 complete=0
maya-ds4: startup transport probe did not drain; reload required
```

The probe describes 167,772,160 floats, equivalent to 640 MiB at four bytes per float, and zero completed samples after about 10.3 seconds. The zero timing fields are not zero-cost transfer measurements. Although the status includes `selected=host`, initialization did not complete and this was not a successful host fallback. No prefill or decode throughput was measured in this retest.

Model loading had progressed further: the log reports two complete Engram tables with 202.758 GB locked, all 12,715 CPU-lane experts selected with no disk fallback, and 239.1 GB of experts copied into RAM in 221.5 seconds. It separately reports 239.9 GB pinned in 19.8 seconds. These are distinct engine counters, preserved as printed. The failure occurred after that work, during transport probing. A HIP synchronization problem, a transfer stall, or a kernel failure remains an explanation to investigate, not an established root cause. The subsequent Python `BrokenPipeError` and nested-tmux warning do not supply a different cause for the engine's logged failure.

## Setup failures and production correction

Before the completed sweep, one launch collided with an existing second-resolution output directory, and another failed importing `regex` through `strata_tokenizer`. The latter occurred after the script's tag checkout step. The later successful command uses `.benchmark-venv/bin/python`. The conversation suggested a `tempfile.mkdtemp` patch for directory collisions, but the supplied script still uses second-resolution directory names; I have retained it without silently applying that historical suggestion.

The runner preserves the input JSON but changes the supplied Maya checkout, rebuilds the engine, runs experiments, and leaves production stopped. It has blocking reads without startup/request deadlines and captures no numerical-quality test. It also saves configuration copies that may contain an API key. These properties matter for reproducing the experiment; the file is retained as a historical artifact, not presented as a revised unattended runner. Its `--max-context >= 65536` check also does not reserve additional room for the generated token. The optional `DS4_LENT_RAM_GB=8` case is conditional on explicit borrowing configuration and is absent from this ten-variant result; server-level prompt memoization is outside the engine protocol.

After the failed probe, the conversation incorrectly suggested `./maya.sh --serve` and unnecessary explicit transport-disable overrides. The user's terminal rejected `--serve`. The final correction was the ordinary production launch:

```sh
cd /opt/project-maya
./maya.sh --config maya-ds4-production.json
```

The experimental overrides were scoped to benchmark child processes; no permanent `MAYA_DS4_HIP_PEER_AR=off` or `DS4_AR_P2P=0` requirement follows from the test. The source does not include a subsequent successful production-start log. The retest log also reports default RDNA 2 stream priority; the old `DS4_STREAM_PRIO=0` workaround is not a new requirement from this investigation.

## Related observations and untested directions

The discussion began with a separate PC containing one R9700 and two V620s and the question of buying a second R9700. No heterogeneous-card comparison was run. The four-V620 sweep does not answer that purchasing question or demonstrate dedicated fast-card prefill scheduling.

A quoted September 24 W4A8 report motivated the kernel investigation. It reports a different vLLM workload on four V620s at TP4/EP4 and 120 W caps: 908→1160 tokens/s at about 3.3K, 957→1217 at 14K, and 962→1232 at 26K, with decode at 62.8–62.9 tokens/s. It describes 64×128×16 tiles, packed INT8 dot products, 2.0–2.45× kernel speedups, and validation/NLL results. These are quoted prior claims, not newly collected Maya results or an independently verified local run. Their exact source revision and raw artifacts were not supplied. The conversation's release-benefit percentages and Q8/Q4 equivalence discussion likewise do not establish measurements for the actual V4.1 model pack.

Two screenshot attachments are absent from the export. The assistant's reading reports an interactive request with 48,794 total prompt tokens, 47,087 cached tokens, 1,707 new tokens in 10,020.1 ms (170.4 tokens/s), and 4,602 generated tokens at 31.2 tokens/s with 71.9% draft acceptance. It also reports 3,102 experts in VRAM, 12,258 in RAM, and an 82.3% VRAM hit rate. These remain screenshot-summary-only observations, not recovered timing JSON. The reported usage total of 27,517 conflicts with the 53,396 sum of prompt and completion counts. Neither the accounting nor the cache metric can be resolved without the original response or image. This cached-suffix workload is not comparable to fresh 32K/64K prefill.

The original result directories and configs have now been collected. The remaining experimental work is to instrument or isolate the HIP peer probe and verify kernel dispatch before attributing performance to MMQ or flow flags. Any future larger-reserve comparison needs a matched host-transport control at that reserve, because changing reserve also changes expert residency. Quality checks and repeated decode comparisons are needed before adopting approximate settings or trading residency for faster prefill. None of that follow-up was completed by the supplied evidence.


## Decode results from the same sweep

Each variant has two distinct 256-token decode cases after a warm-up; these are different workloads, not repetitions of one prompt. Rates are tokens/s. The arithmetic mean here is derived from the two raw observations, not Maya's separate three-answer built-in benchmark.

| Variant | Case 1 | Case 2 | Derived mean |
|---|---|---|---|
| default | 31.31 | 27.08 | 29.20 |
| tp_even | 30.86 | 27.11 | 28.98 |
| p2p_generic | 31.24 | 26.19 | 28.71 |
| p2p_hip_auto | 27.66 | 26.89 | 27.27 |
| flow_dma | 31.78 | 26.97 | 29.38 |
| flow_ai | 31.77 | 26.36 | 29.06 |
| flow_chunk | 32.16 | 27.62 | 29.89 |
| flow_yoco | 31.09 | 26.25 | 28.67 |
| flow_combined | 32.14 | 27.36 | 29.75 |
| expert_mmq | 32.06 | 26.19 | 29.12 |


The results do not establish a reliable decode gain from any switch. Case content, engine warm-up and fixed run order matter, and two cases per variant do not measure run-to-run variance. The benchmark's primary finding remains that the default 32K chunk reaches about 479.4 tokens/s at 64K while automatic chunk selection is slower.

### Source files

All ten config copies, engine logs, the experiment plan, skipped-case notes, summary and update logs are retained under [the completed run](../raw-logs/2026-11-10-maya-collection/extracted/maya-bench-1154-20261105-142331/). They complement the supplied script, rather than replacing the earlier conversation transcription. Three preceding directories retain the failed setup attempts. The separate `201458` directory retains the focused peer retest config and log, so the requested 3072 MiB reserve is now directly inspectable.

The original [P2P runner](../raw-logs/2026-11-10-maya-collection/extracted/maya_benchmark_p2p_test.py) is also recovered. The retest CSV has a header and no completed measurements; its startup probe ended without a ready engine. The earlier distinction between successful host fallback in the main sweep and failure before readiness in the retest still holds.

The decode cases are part of the same ten-variant sweep. They do not add a numerical-quality test or a successful HIP peer-transport result.

### Sources and recording dates

**Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md). Historical commands describe the experiments; this documentation update ran no inference tests.
