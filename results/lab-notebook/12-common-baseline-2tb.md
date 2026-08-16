# Entry 12 — the 2 TB common baseline (stock build)

**Date:** August 15–16, 2026. **Machine:** Galactus, LXC `openwebui`.
**Object:** re-measure all five models with one command set, one build, on one day. The April and July headline rows were produced under non-identical conditions (different builds, flags, ubatch sizes, KV types, and RAM populations), so they could not be compared against each other. This entry replaces them with a normalized set. It is also the closing snapshot for the retired models, and the stock reference against which the prefill-patch A/B will be measured.

## Conditions

Identical for every row:

- **Build:** `3653e6d6d (10326)`, ROCm. **Stock scheduler — the prefill patch is not applied.** The patched pass is a separate future measurement.
- **Memory:** the 2 TB population, after the failing-DIMM replacement ([Entry 11](11-stream-rebaseline-2tb.md); 148–151 GB/s corrected).
- **Command:** `llama-bench -m <file> -ngl 99 -ot "exps=CPU" -fa on -t <T> -b 8192 -ub 8192 -p 8192 -n 128 -r 2 -o md` — all routed experts in RAM, dense path on the four V620s, f16 KV.
- Raw tables: [rebench-2tb-common-baseline-raw.md](../raw-logs/model-benchmark-logs/rebench-2tb-common-baseline-raw.md). CSV: [rebench-2tb-baseline.csv](../data/rebench-2tb-baseline.csv).

Per-model files:

| Model | File | Size | t |
|---|---|---|---|
| GLM-5.2 | Unsloth UD-Q4_K_XL | 435.19 GiB | 32 |
| DeepSeek-V4-Flash-0731 | Unsloth UD-Q8_K_XL (MXFP4 experts) | 150.75 GiB | 64 |
| Kimi K2.6 | Unsloth UD-Q8_K_XL (native-INT4 MoE) | 553.71 GiB | 64 |
| Qwen 3.5 397B-A17B | Unsloth UD-Q6_K_XL | 337.43 GiB | 64 |
| MiniMax M2.7 | Unsloth UD-Q5_K_M | 157.23 GiB | 64 |

Two file-identity notes for the conditions ledger: the Kimi row is **K2.6** (the April rows were K2.5; the user treats them as equivalent, and this is recorded as an assumption, not a measurement); the Qwen file changed from April's bartowski Q6_K_L (319.21 GiB) to Unsloth's UD-Q6_K_XL (+5.7%). MiniMax is the only model with exact file identity to April.

## Results — baseline class

| Model | pp8192 (t/s) | tg128 (t/s) |
|---|---|---|
| MiniMax M2.7 | **418.83 ± 24.11** | **15.18 ± 0.18** |
| Qwen 3.5 397B | 249.61 ± 19.78 | 9.37 ± 0.16 |
| DeepSeek-V4-Flash-0731 | 143.54 ± 1.64 | 10.34 ± 0.10 |
| GLM-5.2 | 95.99 ± 3.36 | 5.30 ± 0.00 |
| Kimi K2.6 | 94.23 ± 4.45 | 5.79 ± 0.01 |

The optimization classes are kept strictly separate from here on: **baseline** (this table), **offload** (resident experts in VRAM), **MTP** and **DSpark** (speculative decode), and combinations — one label per number. The prefill patch is a fifth dimension, absent from everything in this entry.

## Against the historical record (class-matched only)

- **GLM-5.2:** pp 95.99 confirms the stock build (the patched July figure was 119.36; the unpatched July figure 104.97). The −8.6% against July-unpatched is not separable retroactively — build churn is the dominant candidate, the ~2% slower DIMM population contributes at most a couple of points. Decode 5.30 vs the C+S prediction at 148.2 GB/s (5.47): −3%, inside the variance envelope; equivalently C ≈ 96 ms on this build if S holds.
- **DeepSeek-V4-Flash:** tg 10.34 vs July's llama-bench 7.16 — **+44% from three weeks of upstream churn alone**, the single strongest argument for re-baselining. pp 143.54 is the first recorded DSV4 prefill at the standing config. C+S: measured +3% over the 10.03 prediction.
- **Kimi:** first fully-conditioned Kimi row. The April 6.76/44 (K2.5) is not class-comparable (unrecorded April conditions). Decode 172.7 ms/token decomposes as S ≈ 101 ms (≈15 GB/token of INT4 routed experts) + C ≈ 72 ms — in family with DSV4, below GLM. K2.6 out-decodes GLM despite being the larger model: similar streamed bytes, smaller GPU-side constant.
- **Qwen:** decode 9.37 vs April baseline-class 9.56 (t=64) = −2.0% — the moving parts (larger file, −2% bandwidth, build churn) cancel almost exactly. pp 249.61 is 2.85× the April baseline-class peak. Implied C ≈ 20–26 ms, the lowest measured.
- **MiniMax:** pp 418.83 is the machine record — 4.1× its April baseline-class 101.71 and 2.7× even the April *offload*-class 154.93. tg 15.18 is **+2.8% over the April baseline-class 14.76 on the identical file** — the only model to beat its April number, and a clean isolation of build gains (despite the −2% population). Still below the April offload-class 17.37, as the class separation predicts.

## Observations

- **Prefill orders by active-expert size:** A10B 419 > A17B 250 > DSV4 (~13B MXFP4) 143 > A40B 96 ≈ A32B 94. DSV4 sits below the trend line; MXFP4 dequant or arch overhead are candidates, not investigated.
- **Prefill repetition spread grows as active size shrinks:** MiniMax ±5.8% and Qwen ±7.9% against DSV4's ±1.1%. Cause not identified; noted for anyone quoting the big numbers.
- **Decode is stable everywhere** (tg stddev ≤ 0.18 across all five) — bandwidth-bound behavior, as always.
- **Large-ubatch prefill and resident-expert offload compete for VRAM.** The April Qwen 5-layer/card placement failed to load under the standing config, and 4-layer/card loaded weights but failed at context creation — the ub-8192 compute graphs plus f16 KV now consume what April's p512-sized graphs left for resident experts. April's offload placements were only feasible because `-p 512` kept the compute buffers small. Offload reruns were abandoned on this finding.
- **Qwen MTP does not arm on this export.** llama.cpp on this build supports qwen35moe MTP (the hybrid Qwen3.5 MTP context), but the run failed at `common_speculative_init_result: failed to create MTP context`. Cause not isolated (NextN tensors absent from the Unsloth export vs allocation failure); no further Qwen speculative work planned.

## Program decisions recorded in this entry

- **MiniMax M2.7 is retired.** The row above is its final measurement on this machine; the model was deleted after the run. Qwen 3.5 397B was considered for retirement and **kept**.
- Kept set under active testing: **GLM-5.2, DeepSeek-V4-Flash-0731, Kimi K2.6, Qwen 3.5 397B.** The prefill-patch A/B will cover these four; the retired architectures can no longer join it.
- The headline tables now carry **only normalized rows, class-labeled**. All April and July figures remain in the per-model notes and this notebook as history.

## Speculative re-stamp (same build)

llama-cli under script(1); ZFS prompt, `-n 256`, temp 0 — greedy, so token streams are identical between reps and rep spread is pure timing noise. Comparators are the Tier-1 tg128 rows (cross-tool; the cli-vs-bench offset was ≤3% in Session 10).

**GLM-5.2 MTP** (t=32; capture `glm-mtp-n2.txt`, run order n2, n2, n3, n1):

| n-max | t/s | reps |
|---|---|---|
| 1 | 5.9 | 1 |
| 2 | 6.3 / 6.9 → **6.6 ± 0.3** | 2 |
| 3 | 6.3 | 1 |

+25% ± 6 over the 5.30 baseline (July: +31%, 5.4 → 7.1). n=2 remains the optimum, with more separation from n=1 than July showed (there n=1 sat nearly at the peak). **Production: MTP n=2 — 6.6 ± 0.3 t/s on this build.**

**DeepSeek-V4-Flash DSpark** (t=64, p-min off; capture `dsv4-dspark-n3.txt`, run order n3, n3, n2, n4, n8, n1):

| n-max | t/s | reps |
|---|---|---|
| 1 | 12.9 | 1 |
| 2 | 14.3 | 1 |
| 3 | 14.7 / 13.4 → **14.1 ± 0.7** | 2 |
| 4 | 13.6 | 1 |
| 8 (clamps to block size 5) | 12.5 | 1 |

+36% ± 7 over the 10.34 baseline (July: +45% over 9.8–10.3). **The entire July depth curve reproduces across three weeks of build churn:** 12.9 / 14.3 / 14.1±0.7 / 13.6 / 12.5 today against 12.8 / 14.3–14.5 / 14.6–14.8 / — / 11.4 (n=5) then — same rise, same n=2–3 plateau (rep 1 at n=3 landed exactly on July's 14.7), same verify-tax decay. The n=8 request clamps to the drafter's block size 5. **Production: DSpark n=3 — 14.1 ± 0.7 t/s on this build; n=2 is equivalent within today's noise.**

**Speculative runs are an order of magnitude noisier than plain decode on this build:** rep spread ~9% on both models (6.3/6.9 and 14.7/13.4) against ≤0.2% for llama-bench tg128, with identical token streams. Cause not investigated; treat any single spec rep as carrying that error bar. Tooling notes: this llama-cli now routes through the server core (banner, compact `[ Prompt | Generation ]` perf line, `srv` log prefixes) even under `-st -no-cnv`; the dflash `ctx_other` error during memory fitting remains cosmetic, as established in Session 10.

## Open items after this entry

- Prefill-patch A/B across the kept four (stock 95.99 / 143.54 / 94.23 / 249.61 are the references).
- Speculative-run timing variance (~9% rep spread at identical token streams) — cause unknown.
- Upstream PR for the prefill patch.
- Kimi K2.6 vs K2.7-Code disposition (same arch and size; K2.7-Code is the coding specialization and drops non-thinking mode; K2.6 measured here).
- Kimi K3 evaluation (unchanged).
- The Session-10 rerun protocol is superseded by this entry's spec re-stamp and is closed.
