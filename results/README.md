# Results

Per-model performance notes; every note states the machine it was measured on (all results to date: Galactus). Each note distills the raw benchmarks into headline numbers, the best configuration found, and the takeaways. The machine itself: [../hardware/galactus/README.md](../hardware/galactus/README.md). The method behind the numbers: [methodology.md](methodology.md). What carries to other systems: [../takeaways/](../takeaways/).

Sorted by best measured decode:

| Model | Params / quant | Best decode | Best prefill | What got it there |
|---|---|---|---|---|
| [MiniMax M2.7](minimax-m2.7.md) | 229 B / Q5_K | **17.37 t/s** | 154.93 t/s | resident-expert offload; smallest active set |
| [DeepSeek-V4-Flash-0731](deepseek-v4-flash.md) | Q8_K_XL, MXFP4 | **14.7 t/s** | — | DSpark speculative decode, n=3 |
| [Qwen 3.5 397B](qwen-3.5-397b.md) | 396 B / Q6_K | ~11.7 t/s | ~110 t/s | resident offload at DDR4-2933 |
| [GLM-5.2](glm-5.2.md) | 753 B / Q4_K_XL | 7.1 t/s | **119.36 t/s** | the deep investigation: MTP + scheduler patch |
| [Kimi K2.5](kimi-k2.5.md) | 1.03 T / Q4_K_XL | 6.76 t/s | 44 t/s | largest model; bandwidth-bound |

How to read these: decode is bounded by DRAM bandwidth, so a smaller active-expert footprint decodes faster; prefill responds to thread count and to resident-expert offload; the largest decode gains came from speculative decoding (MTP, DSpark) on the models that support it.

## The full record

| Path | Content |
|---|---|
| [methodology.md](methodology.md) | The measurement method these results were produced with. |
| [lab-notebook/](lab-notebook/) | The chronological record: Sessions 1–10, every command and result in order. |
| [raw-logs/](raw-logs/) | Primary captures: the diagnostic run, the MTP A/B, per-model benchmark logs, failure logs. |
| [data/](data/) | CSV extracts — every benchmark row, machine-readable. |
