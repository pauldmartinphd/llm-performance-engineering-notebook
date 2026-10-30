# Maya v1.1.41: short benchmark and uncached 32K/64K production tests

**Recorded:** October 30, 2026 (America/New_York). **Source:** my pasted terminal output in the update and benchmark discussion. The benchmark prints `2026-10-30 11:34` without a timezone. No benchmark was executed by the notebook editor, and no live server settings were changed.

I ran the built-in benchmark after updating from v1.1.36, then made two passes replaying the original saved 32K and 64K requests against the production server. All tests completed. All four long requests reported zero cached tokens and generated 128 tokens. The 64K prefill measured 469.6 and 485.6 tokens/s; the second pass is approximately 0.6% below the October 29 result of 488.6. The two-pass median is 477.6 tokens/s, approximately 2.3% below that single historical result. The historical stream-priority workaround is no longer in use, and the measurements do not isolate a release regression.

## Command and results

```sh
./maya.sh --bench --config maya-ds4-production.json
```

| Measurement | v1.1.41 result |
|---|---:|
| Model load | 207 seconds |
| Decode: CSV parsing function, 256 tokens | 31.0 tokens/s |
| Decode: photosynthesis explanation, 256 tokens | 28.6 tokens/s |
| Decode: bouncing-ball HTML page, 256 tokens | 36.3 tokens/s |
| Mean decode, greedy with thinking off | 32.0 tokens/s |
| Prefill, 2,048-token prompt | 194 tokens/s |
| Prefill, 8,192-token prompt | 334 tokens/s |

The report identifies four AMD RADEON PRO V620 Azure 30 GB GPUs, devices [0, 1, 2, 3], 2004 GB RAM, and an AMD EPYC 7713 64-Core Processor. It names `maya-ds4-production.json`, reports context 1048576, and says it wrote `/opt/project-maya/maya-bench.txt`. The original file has not been collected for this entry; the supplied output is the evidence.

The banner lists the models supported by Maya rather than positively identifying the loaded model and quantization. The established production model is DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL, but this output alone does not independently verify its weights.

## Comparison with the working baseline

The appropriate preceding record is the [October 29 v1.1.36 production confirmation](2026-10-29-maya-production-64k.md), together with the [prefill sweep and regression investigation](2026-10-28-maya-prefill-and-regression.md).

| Workload | v1.1.36 | v1.1.41 |
|---|---:|---:|
| Built-in mean decode | 31.1 tokens/s | 32.0 tokens/s |
| Built-in prefill, 2,048 | 195 tokens/s | 194 tokens/s |
| Built-in prefill, 8,192 | 334 tokens/s | 334 tokens/s |
| Separate uncached request, 65,568 prompt tokens | 488.6 prefill; 24.4 decode tokens/s | Pass 1: 469.6 prefill / 36.0 decode; pass 2: 485.6 / 39.2 tokens/s |

The short results are numerically close: mean decode is approximately 2.9% higher, 2K prefill approximately 0.5% lower, and 8K prefill unchanged at the printed precision. One benchmark per version, with the current config contents uncollected, cannot establish a release speedup or regression.

The initial discussion instead compared today's 334 tokens/s at 8K with an older 271.7 result and suggested an approximately 23% improvement. That comparison is superseded here. The older row belongs to the v1.1.30 sweep with 16K chunks and an 800 MiB ring; its 32K and 64K rates were 432.3 and 464.2. I subsequently selected **32K chunks, an 800 MiB ring, TP2, and automatic workspaces**, with approximately 480 tokens/s at 64K. The October 29 production request then measured 488.6 tokens/s with zero cache hits.

The October 29 v1.1.36 record used `maya-ds4-production-no-stream-prio.json`, with `DS4_STREAM_PRIO=0` and API port 8081. Disabling stream priority was a historical bug workaround. I explicitly clarified on October 30 that we do not use that workaround in the current production configuration; it must not be carried forward as a benchmark requirement or tuning recommendation. Today's command uses `maya-ds4-production.json`. The supplied output does not print the current stream-priority value, and removing the old workaround does not by itself establish a particular environment value or the upstream fix mechanism. It also does not print chunk, ring, TP, or workspace settings.

## Recovering the long-prompt comparison

The built-in run measured only 2K and 8K prompts. The configured 1M context is a capacity limit, not a request to benchmark that length. The earlier long tests remain documented and were conducted separately.

The first sweep used `/opt/maya-prefill-sweep.py`, retained as [experiments/maya-prefill-sweep.py](../../experiments/maya-prefill-sweep.py), and wrote to `/opt/maya-test-results/20261028-200032`. It measured two repetitions at each of 8K, 32K, and 64K across chunk/ring settings. The later TP/workspace sweep used [maya-prefill-tp-sweep.py](../../experiments/maya-prefill-tp-sweep.py) and wrote to `/opt/maya-test-results/20261028-223138`. That second sweep was intentionally stopped after 15 completed cases.

The October 29 64K confirmation selected this exact saved request:

```text
/opt/maya-test-results/20261028-223138/requests/r1-65536.json
```

It posted the request to `http://127.0.0.1:8081/v1/chat/completions` with a 1200-second timeout, using `.venv/bin/python` and the production config's API key when present. It printed `timings.prompt_n`, `timings.cache_n`, `timings.prompt_per_second`, and `timings.predicted_per_second`. The result was 65,568 prompt tokens, zero cache hits, 488.6 tokens/s prefill, and 24.4 tokens/s decode.

## Completed October 30 long-prompt tests

The first replay attempt stopped at the file-existence check because `/opt/maya-test-results/20261028-223138/requests/r1-32768.json` was missing. No inference request was sent by that attempt. The supplied recovery command downloaded both original request JSON files from the notebook's extracted archive and checked their Git blob hashes before writing them back to the original request directory. The recorded hashes are `77e60039ecb80393ebfbb092fbc31c9e7aa95e61` for 32K and `ad624e4658a6acf85493a5a9dcbb13c73ccfbde5` for 64K.

I then ran the replay client from `/opt/project-maya` using `.venv/bin/python`. It loaded `maya-ds4-production.json` for the port and API key, and posted the saved requests unchanged to `http://127.0.0.1:8081/v1/chat/completions` with a 1200-second timeout. This client uses the running server; reading the config for connection details does not independently verify the server's effective engine settings. It changes no tuning settings and does not restore the old stream-priority workaround.

| Request | Actual prompt tokens | Cached tokens | Output tokens | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| `r1-32768.json` | 32,800 | 0 | 128 | 436.4 | 26.4 |
| `r1-65536.json` | 65,568 | 0 | 128 | 469.6 | 36.0 |

The supplied output reports these full responses saved on Galactus:

```text
/opt/maya-test-results/20261030-114549-production-recheck/response-r1-32768.json
/opt/maya-test-results/20261030-114549-production-recheck/response-r1-65536.json
```

The shell prompt identifies the checkout as `#v1.1.41`. The response files themselves have not been collected into the notebook; this entry retains the pasted result and the client's save locations. The client copied the request payloads into the same output directory before posting them.

For the first pass, relative to the October 29 single 64K request, prefill is 19.0 tokens/s lower (3.9%) and decode is 11.6 tokens/s higher (47.5%). The preceding record does not print its actual generated-token count, while this run reports 128, so the decode difference cannot establish a matched release improvement. Relative to the selected v1.1.30 first-sweep medians of 438.8 at 32K and 480.0 at 64K, the first-pass rates are approximately 0.5% and 2.2% lower.

Calculated prefill durations are approximately 75.2 seconds at 32K and 139.6 seconds at 64K. The previous 64K rate implies approximately 134.2 seconds, giving about 5.4 additional seconds of prefill today. These are calculations from rounded throughput and prompt counts, not independently measured request latency. The terminal's roughly 3m45s elapsed time covers both requests and client overhead, rather than either request's time to first token.

### Second pass

I repeated the same client and supplied a second result directory, `/opt/maya-test-results/20261030-115003-production-recheck`. Both requests again report zero cached tokens and 128 output tokens. The second capture does not establish whether the server was restarted between passes, so I do not describe these as independent fresh-start trials or assign the increase to warm-up.

| Request | Actual prompt tokens | Cached tokens | Output tokens | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| `r1-32768.json` | 32,800 | 0 | 128 | 464.2 | 37.3 |
| `r1-65536.json` | 65,568 | 0 | 128 | 485.6 | 39.2 |

The client reports saving `response-r1-32768.json` and `response-r1-65536.json` in that second directory. These full response files have not been collected into the notebook.

| Workload | Prefill range, tokens/s | Two-pass prefill median | Decode range, tokens/s | Two-pass decode median |
|---|---:|---:|---:|---:|
| 32,800 prompt tokens | 436.4–464.2 | 450.3 | 26.4–37.3 | 31.85 |
| 65,568 prompt tokens | 469.6–485.6 | 477.6 | 36.0–39.2 | 37.6 |

The second-pass prefill is approximately 6.4% higher at 32K and 3.4% higher at 64K than the first pass. Against the October 29 single 64K result of 488.6, the second pass is 3.0 tokens/s lower (0.6%) and the two-pass median is 11.0 tokens/s lower (2.3%). Against the selected v1.1.30 sweep medians, the current two-pass prefill medians are approximately 2.6% higher at 32K and 0.5% lower at 64K. These summaries preserve both passes rather than selecting only the faster results.

The working conclusion is that the updated installation completed all four uncached long-prompt requests with prefill near the selected historical range. The second pass substantially narrows the apparent 64K difference from the October 29 measurement. Two samples at each length and one historical production sample do not establish a release regression or a stable decode improvement; effective configuration evidence and more repetitions would be needed to isolate those effects. The [collected archive](../raw-logs/2026-10-29-maya-collection/README.md) preserves the earlier scripts, request files, settings, and results.

## Update and build provenance

The preceding discussion concerned moving a detached v1.1.36 checkout to a newer release. A detached HEAD at a release tag is valid. The suggested tag-based update was `git fetch origin --tags` followed by `git switch --detach v1.1.41`, after preserving any local changes. Checking out `main` would instead track development beyond a release.

The discussion proposed `./setup.sh` and then `./setup.sh --rebuild --no-start`. No update-command transcript or build log was supplied, so this entry does not claim which command was executed, that every engine was rebuilt, or that all earlier claims about setup flags and model preservation were verified. The observed outcome is that the v1.1.41 banner appeared and the pasted short benchmark completed.
