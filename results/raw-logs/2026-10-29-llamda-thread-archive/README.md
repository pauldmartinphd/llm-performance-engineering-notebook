# llamda investigation artifacts retained October 29, 2026

These files preserve the local experimental work and October 24 candidate ledger before the original conversation is deleted. They are source artifacts, not hardware benchmark results. The [retrospective](../../lab-notebook/2026-10-29-llamda-experimental-fork-retrospective.md) records the findings and evidence limits.

- `01-provisional-cpu-batching-and-staging.patch`: fceebbb3f, based on amd-main e3f526dd7. Provisional MXFP4 CPU batching and pinned transfer staging.
- `02-pipeline-integration.patch`: 3061e2b96, on top of the provisional changes; adapted PR 29963 with original author credit.
- `03-candidate-ledger.patch`: c9431a8c2, documentation only.
- `04-isolated-pipeline-alternative.patch`: 04587fd2f, applied directly to e3f526dd7. This is an ALTERNATIVE to patches 01-03, not the next patch in that series.
- `performance-patch-queue.md`, `expert-execution.md`, and `CREDITS-AMD.md`: snapshots from c9431a8c2. Their relative repository links refer to the source fork layout, not this archive. Their PR status statements describe October 24, not current upstream.
- `LICENSE`: source fork MIT notice, retained with the patches.

These patches require the source baseline; they are not a complete checkout. The experimental changes remained local and unpushed at the last recorded step. CPU tests passed, but no Galactus A/B validates this patch series. Original model experiments remain in the separately linked raw-log collections; this archive does not reconstruct missing terminal output.

The temporary local build/test logs were no longer available when this archive was assembled. The build and test outcomes in the retrospective come from the recorded checks in the original conversation; this archive does not contain their complete terminal transcripts. Those checks describe macOS CPU validation, not HIP compilation or inference performance.
