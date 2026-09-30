# Dated source notes and deployment proposals

These notes were retained from my article folder on September 30, 2026. They add early build details, model-selection context, and design proposals to the measured notebook. The copies preserve the supplied bytes; the [manifest](source-manifest.json) records their original filenames and SHA-256 values. Several sources had already been edited on September 30, so their original subject dates do not imply untouched exports from those dates.

The notes are authored records, not raw machine output. A local observation, a vendor specification, an external benchmark, and a proposed role remain different kinds of evidence even when they appear in the same document. Historical commands and open items describe the source's state; they are not instructions to execute now.

| Source | Date and status | Use in this notebook |
|---|---|---|
| [Galactus build notebook](2026-03-23-galactus-build.md) | March 23 subject date; September 30 edited copy | Container/device setup, compiler choice, DIMM replacement, placement patterns, unsuccessful trials, and the complete production command. [March entry](../lab-notebook/00-march-build-record.md) separates those conditions from later benchmarks. |
| [Open-weights survey](2026-07-31-open-weights-survey.md) | July 31 shortlist; September 30 source checks | OCR, speech, and translation candidates. External scores and runtime support guide selection; the note contains no completed local deployment comparison. |
| [Practical local frontier stack](2026-09-13-local-frontier-stack.md) | September 13 comparison; September 30 edited copy | A proposed role table, quantization distinctions, and a Jan/HIP/Vulkan observation. The overlapping public essay uses the material on benchmark equivalence; this copy preserves the dated candidates. |
| [September frontier-model roundup](2026-09-13-frontier-model-roundup.md) | September 13 snapshot; September 30 edited copy | Architecture and cost-accounting questions. Its recorded index and endpoint values are historical notes, with live links that do not reproduce an archived September result. |
| [Local AI infrastructure proposal](undated-local-ai-infrastructure-proposal.md) | Original date unspecified; edited September 30 | Proposed two-node residency and scheduling. The record establishes neither simultaneous fit nor a validated hierarchy of model quality. |

## What can be carried into a deployment decision

The practical-stack note proposes a compact default, an efficient agent, a serious default, a maximum local escalation model, and an alternative lineage. Its named candidates are dated choices, rather than results of a local routing evaluation. The useful part is the distinction between roles and the evidence each role requires: quality on the intended tasks, intervention needed, latency, and actual runtime support.

The Max+ 395 observation is more concrete. The note records roughly 3 tokens/s for Qwen27 through Jan's selected HIP path and a much faster Vulkan path, without a numerical Vulkan rate or a complete matched recipe. Jan was installed as a Flatpak. This makes the runtime/backend path a reasonable suspect, but it does not identify sandbox permissions as the cause. It remains an observation for that machine, separate from the oMLX measurements on Magneto.

The architectural notes distinguish total storage from active work. Quantization width, additional lookup tables, MTP weights, cache state, and runtime buffers all affect residency; nominal active parameter count cannot account for them. Their vendor links provide specifications to consult for a selected revision. This import does not newly validate every release claim or endorse the historical rankings as current.

The two-node proposal similarly separates residency from execution capacity. Keeping several models loaded may avoid a reload, but those models still share memory bandwidth and accelerator resources when they run. Its system-RAM ledger is:

```text
sum(host weight allocations) + sum(host cache allocations)
    + runtime/staging + operating-system reserve < usable host RAM
```

Each GPU needs a separate ledger for its weights, caches, and buffers. A GPU copy does not remove a retained host copy from the host budget. Until the selected files, context allocations, and concurrent workloads are accounted for, keeping every proposed model available without swapping remains an objective.

Model agreement also does not supply independent confirmation. A primary source, checked calculation, or reproduced result can support a claim that several models happen to repeat incorrectly. The proposed workflow remains useful as a division of responsibilities, provided evidence collection and review retain that distinction.

## How these notes relate to the results

The [August common baseline](../lab-notebook/12-common-baseline-2tb.md) remains the latest normalized Galactus model comparison. [Entry 13](../lab-notebook/13-magneto-omlx-qwen38.md) remains the latest measurement entry, with the final Magneto settings recorded in its model summary. None of the imported surveys supplies a later common baseline, a measured multi-model residency configuration, or a controlled replacement test.

The build record supports reproducing an earlier configuration. The surveys support understanding why a candidate was considered. The architecture proposal supports examining a design before implementation. Retaining all three kinds of material makes the notebook more complete without turning a shortlist or plan into a measured result.
