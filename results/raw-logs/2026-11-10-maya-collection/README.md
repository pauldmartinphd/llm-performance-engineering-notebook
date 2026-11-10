# Maya evidence collected November 10, 2026

The supplied [archive](maya-test-results.tar.gz) contains **343 regular files, 10,916,520 uncompressed bytes**. All are retained under [extracted/](extracted/). Log filenames end in `.log.txt` here so their complete contents remain versioned under the notebook's existing log-ignore policy; the archive preserves original names and bytes. [inventory.csv](inventory.csv) maps every original path to its retained path, size and SHA256. [SHA256SUMS](SHA256SUMS) covers the archive and retained evidence.

The two complete conversation exports remain outside this public repository, following its existing policy. [Update terminal captures](update-terminal-captures.md) and [theory/test terminal captures](theory-terminal-captures.md) preserve the user-supplied experimental output with original turn numbers. Selected commands, source claims and results are discussed in the dated entries. Instructions in the conversations were treated as historical material, not permission to operate Galactus or alter Maya. No inference run accompanied this import.

## Source ledger

| Supplied source | SHA256 | Evidence retained |
|---|---|---|
| `ChatGPT-Update Project Maya-20261110-0858.md` | `2746cf29757dfc23bacd8025e52156754a936ab54a6b144903f6977bd6aa3f4a` | Selected terminal captures; dated analysis; full export remains private |
| `ChatGPT-Calculate Theoretical TPS Limits-20261110-0858.md` | `3994ddfc2438dcb3c03ca2eeb542551914dfbe18dc1891dbf39206d21608c44a` | Selected terminal captures; dated analysis; full export remains private |

## Coverage and gaps

| Artifact family | Retained scope | Notebook treatment |
|---|---|---|
| October 5–6 V100 and November 1–2 mixed-NVIDIA README notes | Three original README files | External reference hosts; not Galactus measurements |
| October 28 requests | Two original long requests | Reproduction payloads; not another inferred experiment |
| October 30 production rechecks | Two directories, four responses and four payloads | Closes original-response gaps in the existing October 30 entry |
| November 2 production recheck | Two responses and two requests | Dated production replay |
| November 2 GPU/cache matrix | Nine completed variants, 54 measurements, one failed GLM start | 18 uncached and 36 cached observations; no single-GPU GLM zero-cache result |
| November 5 sweep | Ten variants, 80 rows (60 prefill, 20 decode) | Completes previously partial evidence; no new test |
| November 5 setup and peer retest | Three setup directories plus failed `201458` probe | Preserves configuration and startup failure; no retest throughput |
| November 9 `175610` | One completed TP2 start, six measurements | v1.1.54 baseline; planned second start absent |
| November 9 `181250` | Preparation corpus/requests only | No measurements counted |
| November 9 `181257` | Two starts, 12 layout-mismatch rows | v1.1.57 requested TP4 actually TP2; excluded as TP4 evidence |
| November 9 `184435` | Two starts, 12 valid timing rows | FP16-output prefill; logs identify nonfinite checks and early-row skipping |
| Full GLM and Flash short tests | Configs, shared latest report/log, transcript captures | Eight complete printed short reports plus full-model partial failure |
| GLM chunk and KV long sweeps | Scripts, configs and printed transcript output | 24 transcribed observations; named result directories absent |
| Production-named profiles | Requested names and final user status | Experimental equivalents retained; final filenames and joint process readback absent |
| Loose logs and Python sources | Complete supplied bytes | Dates and executing revisions not invented |

## Reading sequence

- [Maya v1.1.49: production replay and the two-pair utilization question](../../lab-notebook/2026-11-02-maya-v1149-production-recheck.md).
- [Maya v1.1.49: one versus four GPUs and reduced expert-cache budgets](../../lab-notebook/2026-11-02-maya-gpu-cache-matrix.md).
- [Maya releases, memory budgets, and alternative deployment plans](../../lab-notebook/2026-11-02-maya-release-and-capacity-planning.md).
- [Maya v1.1.54: recovered decode measurements and complete sweep evidence](../../lab-notebook/2026-11-05-maya-v1154-recovered-decode.md).
- [Throughput bounds and alternative Qwen deployments](../../lab-notebook/2026-11-09-throughput-models-and-alternative-hardware.md).
- [DeepSeek on two V620s: baseline and requested TP4 fallback](../../lab-notebook/2026-11-09-deepseek-tp2-and-tp4.md).
- [DeepSeek RDNA 2 FP16-output prefill: measured speedup and fidelity limits](../../lab-notebook/2026-11-09-deepseek-fp16-prefill.md).
- [Full GLM-5.3 Q4: allocation failures, legal layer boundaries, and a slow working split](../../lab-notebook/2026-11-09-full-glm53-placement.md).
- [GLM-5.3-Flash on two V620s: MTP, reserve reduction, and cancellation](../../lab-notebook/2026-11-09-glm-flash-reserve-and-speculation.md).
- [GLM-Flash prefill: requested versus effective chunks and the failed first harness](../../lab-notebook/2026-11-09-glm-flash-prefill-chunks.md).
- [GLM-Flash FP16 versus INT8 KV: repeated prefill and decode observations](../../lab-notebook/2026-11-09-to-10-glm-flash-kv-cache.md).
- [Galactus: GLM-Flash INT8 KV and DeepSeek on separate GPU pairs](../../lab-notebook/2026-11-10-maya-dual-model-production.md).
- [Maya archive: upstream reference tests and unassigned live logs](../../lab-notebook/undated-maya-reference-benchmarks-and-live-logs.md).

## Provenance details

Directory names, UTC metadata, benchmark report clocks and conversation timestamps sometimes differ. Entries preserve those differences. Model inventories record paths and sizes, not weight hashes. A configuration's installer timestamp can predate later edits, so it is not a complete configuration history. Shared root logs may have been overwritten by later tests.

The JSON configurations were checked for populated credential fields before inclusion; none were found. Full conversation exports and unavailable screenshot bytes are not embedded in the archive. The archive's external source notes retain their original attribution and are evidence snapshots, not newly authored benchmark claims. Any interpretation of current upstream behavior would require a separate source review; this import records the supplied historical material.
