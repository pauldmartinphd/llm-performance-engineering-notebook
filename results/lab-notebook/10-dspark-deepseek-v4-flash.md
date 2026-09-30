# Lab Notebook Addendum — Session 10 and Interval Notes

[Notebook index](00-overview.md) · [Model summaries](../README.md)

*Later results: [Entry 11](11-stream-rebaseline-2tb.md) completed the memory rebaseline, and [Entry 12](12-common-baseline-2tb.md) repeated speculative decode on a common build. The table below remains provisional, with the p-min conflict unresolved. The description of 14.7 t/s as the fastest result on any model overlooks the April MiniMax resident-offload result of 17.37 t/s.*

*Continuation of the Galactus Lab Notebook (Sessions 1–9, July 13–21, 2026). Compiled August 8, 2026 from the working dialogue; benchmark figures in Session 10 were hand-collected from the terminal by Paul — the log-capture instrumentation failed (recorded below as its own finding). Times of day were not recorded, so the entries follow the order of the conversation, as in Session 9.*

---

## Interval notes — July 24 to August 7, 2026 (approximate dates)

- **RAM upgrade completed.** The 8 × 256 GB DDR4-2933 3DS RDIMMs (ServerPartDeals, reportedly Micron) went in; Galactus now has 2 TB, run at rated 2933 (not overclocked). *Open item:* the STREAM re-baseline on the new DIMMs has not been run; the 152 GB/s platform figure is presumed but unverified for the new population.
- **llama.cpp flag migration.** `-mmp`/`--mmap` deprecated in favor of `--load-mode <none|mmap|mlock|mmap+mlock|dio>` (default `mmap`). The old `-mmp 0` in the best-prefill command maps to `--load-mode none`; `-dio 1` maps to `--load-mode dio`. Verified in source (`include/llama.h`, llama-bench usage text).
- **GLM-5.2 MTP measured (~August 1, date approximate).** Upstream llama.cpp gained MTP support for `glm-dsa` (blk.78 NextN head now loads from the existing Unsloth UD-Q4_K_XL — no re-download). A/B on llama-cli, `-ot exps=CPU -fa on -t 32 -c 8192`, ZFS-explainer prompt, `--temp 0 -n 256`:

  | Config | tg (t/s) |
  |---|---|
  | baseline | 5.4 |
  | `--spec-type draft-mtp --spec-draft-n-max 1` | 6.8 |
  | `--spec-draft-n-max 2` | **7.1** |
  | `--spec-draft-n-max 3` | 6.9 |

  **Decision:** n=2 kept for GLM-5.2. First decode improvement of the entire project (+31%); crosses the 7–10 t/s reading-speed bar. The shallow-depth peak confirms the verify-batch expert-read cost predicted in Session 5.
  **Dead end (lesson):** first attempt omitted `-c`; llama-cli takes the context length from the model — 1,048,576 for GLM-5.2 — filled the cards with KV and failed allocating a 4,362.07 MiB compute buffer on ROCm0. llama-bench never exposed this because it sizes context per test. Explicit `-c` is mandatory with llama-cli on 1M-context models.
- **PR #25784 merged (~August 2):** "DeepseekV4 MTP + DSpark" (am17an). Adds DeepSeek-V4 backbone support to the `dflash` draft architecture and MTP for deepseek4. Notable from the PR: DeepSeek did not ship the MTP head in the 0731 checkpoint — DSpark is the intended speculative path; headline DGX-Spark numbers (16.5 → ~30 t/s) are all-in-VRAM and did not transfer to the hybrid CPU-MoE rigs in the thread, most of which reported gains only at draft depth 1, or regressions.

---

## Session 10 — August 8, 2026 — DSpark speculative decoding on DeepSeek-V4-Flash-0731

**Object:** test DSpark (DeepSeek's official block-diffusion drafter, llama.cpp `--spec-type draft-dspark`) on DeepSeek-V4-Flash-0731, Unsloth UD-Q8_K_XL (162 GB, 5 shards, MXFP4 routed experts), with routed experts in system RAM and everything else — attention, shared experts, drafter — in VRAM.

**System state:** llama.cpp built same morning from master (includes PR #25784); 2 TB DDR4; 4 × Radeon Pro V620.

### Entry 1 — Placement semantics settled

The intended split is `--cpu-moe`. The source confirms that its regex `\.ffn_(up|down|gate|gate_up)_(ch|)exps` catches routed experts only — `ffn_*_shexp` (shared experts) stay on GPU under `-ngl 99`, which is both the intended placement for performance (shared experts fire every token) and the only placement that fits (routed experts ≈ 135+ GiB of the 162 GB model vs ~120 GiB total VRAM).

### Entry 2 — Drafter identification

The 0731 release has no loadable MTP head; DeepSeek's companion drafter is DSpark (released ~June 27, advertised at +60–85% over MTP-1). The drafter is a separate ~10.9 GB model passed via `-md`; it is not bundled with the Unsloth quant.

**Dead end:** first candidate, `alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF` (10.9 GB, MXFP4+Q8_0, extracted from the official 0731 checkpoint, trained block size 5), fails to load:

```
llama_model_load: error loading model: unknown model architecture: 'deepseek_v4_flash_dspark_draft'
```

The file carries a nonstandard architecture string (its card explicitly disclaims llama.cpp compatibility); llama.cpp's registered drafter architecture is `dflash`. Removed the file and both huggingface caches (no files left behind), replaced with the PR author's official conversion:

```bash
huggingface-cli download am17an/DeepseekV4-Flash-20260731-DSpark \
  DeepseekV4-Flash-20260731-DSpark.gguf \
  --local-dir /models/DeepSeek-V4-Flash/DSpark
```

Check with gguf_dump before loading:

```
general.architecture = 'dflash'
dflash.block_size = 5
```

### Entry 3 — Log-capture failures

Three log-capture schemes failed in sequence; all benchmark figures below were read from the terminal by hand.

- `| tee` breaks the llama-cli interactive UI (TTY required).
- `--log-file` alone captures nothing useful: `tools/cli/cli.cpp` clamps the CLI's default verbosity to error level, and the acceptance/timing lines are info level (`SLT_INF`/`SRV_INF`).
- `-lv 3 --log-file` still failed to capture the needed lines in practice (mechanism unresolved — the CLI's embedded-server log routing apparently does not reach the file sink as the source suggests).

**Decision (for future sessions):** capture TTY sessions with `script -q <file> -c "<command>"`, or benchmark speculation through llama-server, whose `/completion` response returns timings as JSON. The llama-cli UI is for interaction, not measurement.

**Correction (diagnosis retracted):** identical (near-empty) logs from the first sweep were misread as "drafter never armed," and an `E ... dflash requires ctx_other to be set (this warning is normal during memory fitting)` line was twice misjudged — first as fatal, then as the cause of the apparent failure. Hand-read runs showed the drafter arming throughout; the ctx_other line is cosmetic (fitter probe pass), and `--fit off` has no effect on results (14.3 with, 14.5 without, at n=2 — within run variance). With `-ngl 99 --cpu-moe -c 8192` explicit, the fitter has nothing to decide.

### Entry 4 — Test configuration

```bash
llama-cli -m /models/DeepSeek-V4-Flash/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
  -ngl 99 --cpu-moe -fa on -t 64 -c 8192 -b 8192 -ub 8192 \
  -p "Explain how ZFS copy-on-write snapshots work, including the uberblock, block pointers, and space maps." \
  -n 256 --temp 0 -st -no-cnv --perf \
  --spec-type draft-dspark \
  -md /models/DeepSeek-V4-Flash/DSpark/DeepseekV4-Flash-20260731-DSpark.gguf \
  -ngld 99 --spec-draft-n-max <N> [--spec-draft-p-min <P>]
```

Each configuration used a single technical-prose prompt with greedy decoding and one measurement unless noted. Since the log-capture failures above left draft acceptance rates unrecorded, the analysis below cannot use them to separate draft quality from verification cost.

### Entry 5 — Results

The baseline without speculation was measured twice during the session: **10.3** and **9.8 t/s** — taken as 10.1 ± 0.3. Compared with July: the same model baselined at 7.16 t/s on build 9942; upstream DSv4 work (fused kernels), the 0731 checkpoint, and current config delivered ~+40% before speculation.

| Configuration | tg (t/s) |
|---|---|
| baseline | 9.8 / 10.3 |
| n=1 | 12.8 |
| n=2 | 14.3 (--fit off) / 14.5 |
| n=2, p-min 0.3 | 14.4 |
| n=2, p-min 0.5 | 13.9 |
| **n=3** | **14.6 / 14.8** |
| n=3, p-min 0.3 | 14.7 (see conflict note) |
| n=3, p-min 0.8 | 12.6 |
| n=5 | 11.4 |
| n=5, p-min 0.5 | 13.3 |

**Data conflict, unresolved:** an earlier report in the session gave n=3 / p-min 0.3 as 13.4; a later run of the nominally identical configuration gave 14.7. These are irreconcilable at observed run variance (±0.2); one report is presumed mislabeled (one prior mislabeling was caught and corrected during the session). The later p-min sweep (0.3 → 14.7, 0.5-at-n=2 → 13.9, 0.8 → 12.6), being internally consistent and monotone, is taken as authoritative.

**Rerun:** `experiments/session-10-rerun.sh` regenerates this table under identical conditions — fixed order, two reps per configuration, baseline at start and end as a drift check, `--fit` at default throughout, TTY captured with `script(1)`. The table above is provisional until it runs; it also settles the conflict row.

### Entry 6 — Analysis

- **Optimum: fixed draft depth 2–3, plateau at ~14.5–14.8 t/s.** The production setting remained `--spec-draft-n-max 3`, p-min off: **14.7 ± 0.2 t/s, ≈ +45% over baseline.** The fastest decode Galactus has produced on any model.
- **Depth curve** (12.8 / 14.5 / 14.7 / 11.4 at n = 1/2/3/5) follows the verify-cost curve predicted in Session 5: on a top-k-of-many MoE, drafted tokens activate nearly disjoint expert sets, so each verified position pays close to full streaming cost; positions 4–5 cost more expert-read bytes than their acceptance yields.
- **Confidence truncation (p-min) steadily reduces throughput on this domain.** It did not provide a useful tuning range: 0.3 ≈ no-op (the head rarely fires below 0.3 in the first three positions), 0.5 costs ~0.5–1.3 t/s, 0.8 costs ~2.1. Truncated configurations converge to ~13.3–13.4 regardless of configured n-max — once confidence truncation controls the draft, configured depth stops mattering. Interpretation: the drafter's confidence head is too pessimistic on technical prose. Hypothesis for later: p-min may help on genuinely low-acceptance domains (creative text); untested.
- **Cost decomposition (estimate — acceptance rates not captured):** the curve is consistent with ~70 ms/token amortizable cost (GPU dense path, sync) + ~29 ms/token CPU expert streaming. Cross-check: 29 ms × 152 GB/s ≈ 4.4 GB/token, matching 13B active with MXFP4 routed experts. Implied speculation ceiling ≈ 35 t/s (streaming-only); perfect-acceptance value of block-5 drafts at n=3 ≈ 22 t/s. The 14.7 → 22 gap is drafter quality, rather than configuration.
- **Contrast with the PR #25784 field reports:** hybrid CPU-MoE rigs there mostly saw gains at n=1 only, or outright regressions; Galactus gets +45% at n=3 without workarounds. Two structural advantages: MXFP4 routed experts make the per-token verification cost small, and 8-channel DDR4 absorbs verify batches better than consumer boards. This is, as far as the thread shows, the first working Radeon Pro V620 data point, and the first hybrid rig to land DSpark's advertised range.
- **Contrast with GLM-5.2 MTP** (+31% at n=2): V4-Flash speculates better because its streaming share of the token budget is smaller (~29 of ~100 ms vs GLM's ~91 of ~180 ms), leaving more cost that speculation can amortize.

### State of knowledge at end of Session 10

- Production decode configurations, both via speculation, both established since August 1: GLM-5.2 at **7.1 t/s** (draft-mtp, n=2); DeepSeek-V4-Flash-0731 at **14.7 ± 0.2 t/s** (draft-dspark, n=3, p-min off, drafter am17an block-5 in VRAM, `--fit off` unnecessary).
- V4-Flash decode progression: 7.16 (July, build 9942) → ~10.1 (upstream + 0731, no speculation) → 14.7 (DSpark). Roughly 2× since the investigation began, entirely from software.
- Speculation ceiling on V4-Flash ≈ 35 t/s (estimate); further gains depend on drafter training quality, domain, or concurrency rather than flags.
- Recording limitation: llama-cli cannot be relied on for benchmark capture; use `script(1)` or llama-server JSON timings. Acceptance rates for every run in this session are unrecorded.
- Open items: STREAM re-baseline on the 2 TB DIMM population; domain envelope (code / creative) at n=3; concurrency scaling via llama-server; PR #25784 data-point post (V620 works, optimum n=2–3 not n=1, conf-head pessimistic on prose); Kimi K3 evaluation pending (fits in 2 TB; est. ~2.4–3 t/s decode, untested).
