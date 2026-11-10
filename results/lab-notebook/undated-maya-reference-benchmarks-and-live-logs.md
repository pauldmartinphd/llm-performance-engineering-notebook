# Maya archive: upstream reference tests and unassigned live logs

[All lab records](README.md) · [Related investigation: chapter 4](../../investigations/04-from-the-fork-to-maya.md)

**Source dates: October 5–6 and November 1–2, 2026; live log dates unverified.**

The supplied archive mixes Galactus artifacts with benchmark notes from other machines. You retain all of them, but their hardware and authorship do not justify treating them as your local measurements. The three README captures identify their own hosts and protocols; their dates are source-record dates, not newly inferred Galactus test days.

## External V100 reference captures

| Source | Host and workload | Reported result |
|---|---|---|
| `2026-10-05-glm-single-gpu/README.md` | V100, Xeon E5-2690 v4, only 30 GB system RAM, GLM-Flash UD-IQ1_S, simulated VRAM budgets | At 32 GB, three requests 5.8 / 6.7 / 7.4 tokens/s; at 12 GB, 2.3 / 2.9 / 4.5; disk reads are a major limit |
| `2026-10-06-glm-spec/README.md` | Two V100-32GB and a separate one-V100 host, GLM-Flash UD-IQ1_S | Batched prefill, speculative pipeline and correctness fixes; reports roughly 36–40 tokens/s, then 41.4 after later fixes and about 45.7 over 1000 tokens |
| `2026-11-02-4gpu-4x128k/README.md` | Three V100-16GB plus RTX 3060, dual Xeon, 80 GB RAM, Maya-S v2 IQ2_XXS | Four conversations at 20.1–21.5 tokens/s each, 80–84 aggregate; several batching/multi-MTP alternatives slower |

The [complete original notes](../raw-logs/2026-11-10-maya-collection/extracted/) preserve every table, correctness finding, negative result and configuration switch. These reference notes explain why expert residency, legal splits and pipeline balance were worth investigating. The disk-constrained V100 results do not show an SSD bottleneck on Galactus, whose RAM is much larger. The four-conversation test does not show concurrent two-model performance on your machine.

## Loose production and benchmark files

`maya-glm-production.log` identifies four GPUs and 131,072 context, unlike the final two-GPU, 1M-configured service. It records a 26,920-token answer at about 29.5 tokens/s, 89.0% expert-cache hit rate, 81.0% draft acceptance for that answer, and 99.75% KV streaming block-read hits. The final cumulative speculative statistic is 77.0%; it is a different aggregation. These counters describe the captured session. The log lacks a wall-clock start date, so you do not assign it to the November 9 two-GPU measurements.

`maya-ds4-hip.log`, `maya-ud-q4_k_xl-hip.log`, `maya-bench.log`, `maya-bench.txt` and `maya-benchmark-run.log` are also retained in full. Shared filenames may contain the latest run rather than every run that used that name. You use dated directories, effective layouts and conversation captures to associate results, instead of attaching a loose log to a test solely because its filename matches.

The root `maya.py`, `setup.py`, `chat.py` and benchmark scripts are source snapshots supplied with the results. Their bytes and hashes are retained; they do not confirm that every historical test executed those exact bytes. The dated metadata's script and binary hashes take precedence for that purpose.

## Sources

**Added:** November 10, 2026 (America/New_York). **Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md).
