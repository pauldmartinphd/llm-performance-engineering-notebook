# Maya v1.1.36: production configuration and uncached 64K confirmation

**Recorded:** October 29, 2026 (America/New_York). **Machine:** Galactus. **Source:** my pasted benchmark output in [Analyze GPU temperatures](https://chatgpt.com/c/6ae2e072-8b64-83ea-b98f-89d8e13af59f), followed by my explicit confirmation of the working production configuration when continuing that discussion. This entry records an existing run; no benchmark was repeated and no server configuration was changed to prepare it.

The production configuration completed a **65,568-token prefill at 488.6 tokens/s with zero cache hits**, followed by decode at **24.4 tokens/s**. This extends the earlier successful short-prompt benchmark with stream priority disabled to the retained 64K request. I am carrying this configuration forward as the working baseline, without reopening the completed tuning sweeps or assuming that resuming the investigation requires reconfiguration.

**October 30 follow-up:** The configuration below records the October 29 workaround run. I clarified that `DS4_STREAM_PRIO=0` was a bug workaround and is not used in the current production configuration. The [v1.1.41 benchmark and two uncached 32K/64K passes](2026-10-30-maya-v1141-benchmark.md) use `maya-ds4-production.json`; the historical workaround must not be carried forward as a tuning recommendation.

## Working production configuration

| Component | Confirmed setting |
|---|---|
| Host | Galactus, four AMD Radeon Pro V620 GPUs |
| Runtime | Project Maya v1.1.36, ROCm 7.2 |
| Model | DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL |
| Working directory | `/opt/project-maya` |
| Configuration | `maya-ds4-production-no-stream-prio.json` |
| Tensor parallelism | TP2 |
| Prefill chunk | 32,768 tokens |
| Ring | 800 MiB |
| Stream priority | `DS4_STREAM_PRIO=0` |
| API port | 8081 |

The [recovery and tuning record](2026-10-28-maya-prefill-and-regression.md) documents the selection of TP2, the chunk and ring sizes, and automatic workspaces. It also records my reproduction of a deadlock with `DS4_STREAM_PRIO=1`. Disabling priority is the working workaround; the exact synchronization mechanism and introducing commit remain unresolved. A completed request establishes success for that workload, not indefinite production stability.

The [earlier collection](../raw-logs/2026-10-29-maya-collection/README.md) retains a copy of the no-stream-priority config and identifies the collected v1.1.36 checkout and binary. That collection was taken at 14:15 Eastern; this conversation-derived result is separate evidence and is not represented as a new raw response recovered from that archive.

## Request and captured result

The client ran from `/opt/project-maya` with `.venv/bin/python`, loaded the production JSON for the API key when present, and posted the retained request to `http://127.0.0.1:8081/v1/chat/completions` with a 1200-second timeout. It selected the most recently modified matching `/opt/maya-test-results/*/requests/r1-65536.json`; the printed selection below identifies the request actually used. The preceding instructions described a 128-token decode request. The pasted output does not print the actual generated-token count, so I retain 24.4 tokens/s without claiming that all 128 tokens were generated.

The user-pasted output is preserved here:

```text
Using prompt: /opt/maya-test-results/20261028-223138/requests/r1-65536.json
Running 64K prefill benchmark on port 8081...

Prompt tokens: 65568
Cached tokens: 0
Prefill:       488.6 tok/s
Decode:        24.4 tok/s

Previous TP2 baseline: ~480 tok/s PP at 64K
```

The client printed `timings.prompt_n`, `timings.cache_n` with a default of zero when absent, `timings.prompt_per_second`, and `timings.predicted_per_second`. I explicitly confirmed the run as having no cache hits in the continuation. The pasted output alone cannot distinguish an explicit zero cache field from the client's fallback; the full response JSON was not supplied with this result.

At the reported rate, 65,568 / 488.6 gives approximately **134.2 seconds**, or **2 minutes 14 seconds**, of prefill. This is a calculation from rounded throughput, not independently measured wall-clock latency. It excludes decode and other request overhead. The earlier 2–4 minute expectation and five-/ten-minute troubleshooting thresholds were advice in the conversation, not additional timings or measured stall boundaries.

## Comparison with the earlier baseline

The previous selected chunk/ring result was approximately **480 tokens/s** at 64K on v1.1.30. The new rate is **8.6 tokens/s, or about 1.8%, higher**. The first sweep's 480.0 is a two-request median; the partial TP/workspace sweep's auto-workspace baseline was 479.80. The current 488.6 is one reported request. These observations support carrying forward approximately the same prefill performance with the v1.1.36 workaround, but do not establish a statistically meaningful release improvement or quantify normal run-to-run variation.

The earlier v1.1.36 built-in benchmark reported **31.1 tokens/s mean decode**, **195 tokens/s prefill at 2048**, and **334 at 8192**. Those short-prompt measurements and the present 24.4 tokens/s decode after a 65,568-token prompt are different workloads. I do not treat their difference as an isolated decode regression.

## Historical progression and its evidence limits

The conversation's answer to “What did I start with originally?” summarized the following progression. The figures are useful historical context, but not a controlled speedup series.

| Stage | Prefill tokens/s | Decode tokens/s | Interpretation |
|---|---:|---:|---|
| Original “llama.cpp baseline,” as summarized in the conversation | 132.5 | 11.6 | Rounded retrospective summary; not established as one paired run |
| Early Maya v1.1.24 | 282 at 8192 | 20.5 | Separate built-in benchmark retained in the October 23 record |
| Optimized Maya v1.1.30 | ~480 at 64K | ~30–36 in the conversation summary | Prefill supported by the sweep; decode is contextual, not a matched 64K control |
| Current Maya v1.1.36 | 488.6 at 65,568 | 24.4 | User-pasted result above, zero cache hits confirmed |

The [October 23 evidence](2026-10-23-deepseek41-cache-and-maya.md) is more precise than the retrospective “llama.cpp” label: it identifies llAMDa cache-sweep summary values of 132.54 prefill / 10.20 decode with cache disabled and 132.82 / 11.54 with 64 GiB, alongside a printed 32 GiB capture of 132.66 ± 0.30 / 9.51 ± 0.07. Thus, 132.5 and 11.6 should not be promoted to a newly verified baseline pair.

Dividing 488.6 by the historical rounded 132.5 gives **3.69×**, or approximately **269% higher prefill throughput**. This is an indicative comparison across workloads, prompt lengths, and execution configurations. It cannot assign the gain to the engine alone, RAM residency, TP2, or any individual change. The earlier discussion attributed the larger gains to moving to Maya, retaining experts in RAM, and tuning TP2 prefill; those changes are part of the history, but their separate causal contributions were not isolated by this run.

No GPU temperature readings were included in the retrieved conversation excerpt despite its title. This entry therefore makes no new thermal or throttling claim. The existing constraint against quality-reducing prompt skipping remains unchanged.
