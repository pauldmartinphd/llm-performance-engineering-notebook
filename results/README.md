# Results by model and machine

This page helps locate a result. The [lab notebook](lab-notebook/README.md) keeps the full experiments; the [Experiments](../investigations/README.md) explain why the tests were run. The [Findings](../takeaways/general-principles.md) give the short conclusions.

## Models in the November 10 Galactus setup

Both models use Project Maya, with a separate pair of V620s for each. Their tests measured individual-model performance. The rates below come from different request types and are not a controlled model ranking.

| Model | Selected layout | Useful measurements | Full account |
|---|---|---|---|
| [DeepSeek-V4.1-Flash-1017](deepseek-v4.1-flash.md) | GPUs 0–1; UD-Q8_K_XL; DSpark; FP16-output prefill | Two-GPU built-in generation mean 29.0 tokens/s; later optimized prefill medians 389.55 / 501.95 / 589.70 at 8K / 16K / 32K targets | [Two-GPU test](lab-notebook/2026-11-09-deepseek-tp2-and-tp4.md), [prefill test](lab-notebook/2026-11-09-deepseek-fp16-prefill.md) |
| [GLM-5.3-Flash](glm-5.3-flash.md) | GPUs 2–3; UD-Q4_K_XL; MTP; INT8 streaming KV | FP16-KV reserve test reached 21.5 tokens/s built-in mean generation; separate INT8 test showed higher mean generation and about 292 tokens/s long prefill | [Reserve test](lab-notebook/2026-11-09-glm-flash-reserve-and-speculation.md), [KV test](lab-notebook/2026-11-09-glm-flash-kv-cache.md) |

The [deployment record](lab-notebook/2026-11-10-maya-dual-model-production.md) gives the reported operating state and retained settings. A configured one-million-token ceiling is not a measured million-token result. The FP16-output and INT8-KV tests also lack a local answer-quality comparison.

The [full GLM-5.3 experiment](lab-notebook/2026-11-09-full-glm53-placement.md) is separate from Flash. Its corrected four-GPU split completed at 6.7 tokens/s mean generation and 47 tokens/s at 8K prefill. We returned to Flash for the intended daily use.

## Earlier Galactus common baseline

These five models were measured on August 15–16 with stock llama.cpp build `3653e6d6d`, 2 TB RAM, CPU-resident routed experts, an 8192-token prompt, 128 generated tokens, and two requested repetitions. GLM used 32 CPU threads; the others used 64. This is a historical common benchmark, not the current Maya setup.

| Model | Prefill, tokens/s | Generation, tokens/s | Separate speculative generation |
|---|---:|---:|---|
| [MiniMax M2.7](minimax-m2.7.md) | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| [Qwen 3.5 397B](qwen-3.5-397b.md) | 249.61 ± 19.78 | 9.37 ± 0.16 | MTP did not initialize |
| [DeepSeek-V4-Flash-0731](deepseek-v4-flash.md) | 143.54 ± 1.64 | 10.34 ± 0.10 | 14.1 ± 0.7 with DSpark |
| [GLM-5.2](glm-5.2.md) | 95.99 ± 3.36 | 5.30 ± 0.00 | 6.6 ± 0.3 with MTP |
| [Kimi K2.6](kimi-k2.5.md) | 94.23 ± 4.45 | 5.79 ± 0.01 | — |

The baseline spreads are printed llama-bench values. The speculative figures use a different prompt and llama-cli, with two repetitions; their spreads are not confidence intervals. The [common-baseline entry](lab-notebook/2026-08-15-to-16-common-baseline-2tb.md) records all conditions. Each model page also preserves its earlier tests.

## Other machines

| Machine and model | Retained result | What it supports |
|---|---|---|
| [Magneto: Qwen3.8-27B with oMLX](qwen-3.8-27b.md) | Initial 4K baseline: 156.1 tokens/s prefill, 16.4 generation; later screenshot: 171.0 / 16.9 | Local M2 Max observations; ANE, MTP, thermal state, and recipe differences need to be kept explicit |
| [SilverSurfer: Qwen3.8-Flash-Next with Gufo](qwen-3.8-flash-next.md) | 318 generated tokens at 29.85 tokens/s on battery, after a 63-token prompt | Direct timing JSON for one successful request; longer Jan rates remain attachment-summary evidence |

The [other-machines chapter](../investigations/07-other-local-machines.md) explains these investigations. Different models and engines prevent a hardware ranking from this table.

## Where to inspect the detail

The [chronological index](lab-notebook/README.md) includes failures, plans, and recovered evidence. [Measurement methods](methodology.md) define the metrics and controls. [Data tables](data/) preserve individual observations, while [raw captures](raw-logs/) and [source notes](reference-notes/README.md) preserve their origin.

For known limitations, see the [unsuccessful approaches](../takeaways/refuted-hypotheses.md). For the experimental fork's outcome, see the [llamda retrospective](lab-notebook/2026-10-29-llamda-experimental-fork-retrospective.md).
