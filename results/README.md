# Results

These are the per-model performance notes. Every note states the machine it was measured on; all results to date come from Galactus. Each note distills the raw benchmarks into the main numbers, the best configuration found, and the conclusions. For the machine itself, see [../hardware/galactus/README.md](../hardware/galactus/README.md). For the method behind the numbers, see [methodology.md](methodology.md). For what transfers to other systems, see [../takeaways/](../takeaways/).

## The normalized baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

This table uses a common command structure and one build over August 15–16. Thread counts and model files are recorded in Entry 12; the speculative column uses separate llama-cli runs. [Entry 12](lab-notebook/12-common-baseline-2tb.md) has the full conditions ledger. The rows are sorted by baseline decode.

| Model | pp8192 (t/s) | tg128 (t/s) | Speculative decode, same build |
|---|---|---|---|
| [MiniMax M2.7](minimax-m2.7.md) † | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| [DeepSeek-V4-Flash-0731](deepseek-v4-flash.md) | 143.54 ± 1.64 | 10.34 ± 0.10 | **14.1 ± 0.7** (DSpark n=3) |
| [Qwen 3.5 397B](qwen-3.5-397b.md) | 249.61 ± 19.78 | 9.37 ± 0.16 | — (MTP does not arm) |
| [Kimi K2.6](kimi-k2.5.md) | 94.23 ± 4.45 | 5.79 ± 0.01 | — |
| [GLM-5.2](glm-5.2.md) (t=32) | 95.99 ± 3.36 | 5.30 ± 0.00 | **6.6 ± 0.3** (MTP n=2) |

† This is the final MiniMax measurement; the model was removed afterward.

Baseline ± values are reproduced from llama-bench. The speculative ± values summarize two llama-cli repetitions, not confidence intervals; those runs used a ZFS prompt, greedy decoding, and 256 generated tokens rather than the baseline’s tg128 test. Entry 12 records the cross-tool comparison and about 9% speculative timing variation.

The model notes retain earlier results with their builds and configurations. They include the July patched prefill (GLM-5.2 at 119.36), the April resident-offload rows (MiniMax at 17.37 t/s, Qwen at about 11.7 t/s), and the April K2.5 numbers.

Memory bandwidth and the active-expert footprint explain part of decode time; the GPU-side cost also differs by model. Prefill is fastest for MiniMax (A10B), followed by Qwen (A17B), V4-Flash, GLM (A40B), and Kimi (A32B). This is not a simple ordering by active parameter count: quantization and architecture also differ. MTP and DSpark improve decode on the two models tested successfully, but their timings vary more than the ordinary decode benchmarks.

## The full record

| Path | Content |
|---|---|
| [methodology.md](methodology.md) | The measurement method these results were produced with. |
| [lab-notebook/00-overview.md](lab-notebook/00-overview.md) | The chronological record: Sessions 1–10 and Entries 11–12, every command and result in order. |
| [raw-logs/](raw-logs/) | The primary captures: the diagnostic run, the MTP A/B test, the per-model benchmark logs, and the failure logs. |
| [data/](data/) | CSV extracts, with every benchmark row in machine-readable form. |
