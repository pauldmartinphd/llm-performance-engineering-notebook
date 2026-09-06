# Results

These are the per-model performance notes. Every note states the machine it was measured on; all results to date come from Galactus. Each note distills the raw benchmarks into the main numbers, the best configuration found, and the conclusions. For the machine itself, see [../hardware/galactus/README.md](../hardware/galactus/README.md). For the method behind the numbers, see [methodology.md](methodology.md). For what transfers to other systems, see [../takeaways/](../takeaways/).

## The normalized baseline (2026-08-15/16, build 3653e6d6d, stock scheduler)

This table uses one command, one build, and one day, with one class of optimization per number. [Entry 12](lab-notebook/12-common-baseline-2tb.md) has the full conditions ledger. The rows are sorted by baseline decode.

| Model | pp8192 (t/s) | tg128 (t/s) | Speculative decode, same build |
|---|---|---|---|
| [MiniMax M2.7](minimax-m2.7.md) † | 418.83 ± 24.11 | 15.18 ± 0.18 | — |
| [DeepSeek-V4-Flash-0731](deepseek-v4-flash.md) | 143.54 ± 1.64 | 10.34 ± 0.10 | **14.1 ± 0.7** (DSpark n=3) |
| [Qwen 3.5 397B](qwen-3.5-397b.md) | 249.61 ± 19.78 | 9.37 ± 0.16 | — (MTP does not arm) |
| [Kimi K2.5/K2.6](kimi-k2.5.md) | 94.23 ± 4.45 | 5.79 ± 0.01 | — |
| [GLM-5.2](glm-5.2.md) (t=32) | 95.99 ± 3.36 | 5.30 ± 0.00 | **6.6 ± 0.3** (MTP n=2) |

† This is the final measurement. We removed MiniMax M2.7 from the machine after this row.

The best results under other classes and builds live in the per-model notes as history, each labeled with its class and build. They include the July patched prefill (GLM-5.2 at 119.36), the April resident-offload rows (MiniMax at 17.37, Qwen at about 11.7), and the April K2.5 numbers.

Two patterns explain the table. First, DRAM bandwidth bounds decode, so a model with a smaller active-expert footprint decodes faster; prefill orders by active size (A10B > A17B > V4-Flash > A40B ≈ A32B). Second, the largest decode gains come from speculative decoding (MTP and DSpark) on the models that support it, and the gain scales with the amortizable GPU-side share of the token budget.

## The full record

| Path | Content |
|---|---|
| [methodology.md](methodology.md) | The measurement method these results were produced with. |
| [lab-notebook/](lab-notebook/) | The chronological record: Sessions 1–10 and Entries 11–12, every command and result in order. |
| [raw-logs/](raw-logs/) | The primary captures: the diagnostic run, the MTP A/B test, the per-model benchmark logs, and the failure logs. |
| [data/](data/) | CSV extracts, with every benchmark row in machine-readable form. |
