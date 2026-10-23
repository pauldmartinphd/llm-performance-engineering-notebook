# Chronological lab notebook

I organize these records by when I ran the experiments, not when I recovered the logs or wrote the entries. Each recovered record keeps its addition date in the metadata. When a capture lacks a run date, I mark it as undated and place it beside the related investigation without assigning it an invented date.

The original July transcripts still refer to sessions in their historical sequence. Those references describe the original conversation, not a numbering system for new notebook entries. Overlapping diagnostic runs and discussions remain separate records.

[Background, hardware, and source ledger](overview.md) · [Previous filename map](filename-map.md) · [Model summaries](../README.md) · [Original archive inventory](../raw-logs/2026-10-22-galactus-archive/README.md)

## March and April: building Galactus and testing the models

- **March 23:** [Initial Galactus build](2026-03-23-galactus-build.md).
- **April 7–17:** [Thread counts, resident experts, and Qwen batch/context tests](2026-04-07-to-17-thread-and-batch-tests.md). I recovered these captures on October 22; the Qwen capture without a run date stays explicitly undated within the comparison.

## July: GLM-5.2 prefill and the scheduler patch

- **July 13, evening:** [Background and plan](2026-07-13-background-and-plan.md).
- **July 14, 07:00:** [Baselines and memory bandwidth](2026-07-14-0700-baselines-and-memory-bandwidth.md).
- **July 14, 10:22:** [Crash forensics and diagnostic relaunch](2026-07-14-1022-crash-forensics-and-relaunch.md).
- **July 14, 11:08–14:25:** [The v3 diagnostic run](2026-07-14-1108-v3-diagnostic-run.md).
- **July 14, from 12:13:** [Interpreting v3 and planning larger microbatches](2026-07-14-1213-interpreting-v3.md). This discussion overlaps the diagnostic run.
- **July 14:** [Full v3/v4 logs and hardware-monitor reconstruction](2026-07-14-diagnostic-reconstruction.md), added October 22.
- **July 14, from 15:29:** [Microbatch ladder and hardware economics](2026-07-14-1529-ubatch-ladder-and-economics.md).
- **July 14, from 18:13:** [Multi-card placement](2026-07-14-1813-multi-card-placement.md).
- **July 14–16:** [H2D results and the interrupted patch investigation](2026-07-14-to-16-h2d-and-patch-investigation.md).
- **July 21:** [Building and benchmarking the prefill patch](2026-07-21-prefill-patch.md).
- **Related, run date unknown:** [Original and distributed scheduler traces](undated-glm52-scheduler-traces.md). Their filenames do not establish a day; I keep them beside the patch investigation.

## August: DSpark and the 2 TB rebaseline

- **Compiled August 8; individual run dates unknown:** [DeepSeek-V4 DSpark investigation and interval notes](undated-deepseek-v4-dspark-investigation.md).
- **Related, run date unknown:** [V4-Flash-0731 DSpark server acceptance logs](undated-deepseek-v4-dspark-acceptance.md). These are separate from the dated August 16 CLI captures.
- **August 15:** [STREAM on the 2 TB memory population](2026-08-15-stream-rebaseline-2tb.md).
- **August 15–16:** [Five-model common baseline](2026-08-15-to-16-common-baseline-2tb.md).
- **August 16:** [Speculation transcripts and adjacent rebenchmark files](2026-08-16-speculation-and-rebench-records.md), recovered October 22. The CLI transcripts carry dates; the adjacent benchmark files do not independently identify their run times.

## September: Magneto

- **Compiled September 30; some run dates unknown:** [Qwen3.8-27B and oMLX on the M2 Max](undated-magneto-omlx-qwen38.md).

## October: the AMD fork and Flash models

- **Earlier V620 integration, run dates unknown:** [GLM-5.3-Flash placement, cache, and MTP tests](undated-glm53-flash-placement.md) and [DeepSeek-V4.1 target inference and DSpark allocation failure](undated-deepseek41-dspark.md). I recorded these on October 21 and clarified the combined patch set on October 22; those are documentation dates, not inferred experiment dates.
- **October 21:** [AMD fork integration](2026-10-21-llamda-amd-fork.md).
- **October 21:** [Bandwidth, placement, and decode analysis](2026-10-21-bandwidth-and-placement.md).
