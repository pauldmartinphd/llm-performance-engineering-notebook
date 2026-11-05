# SilverSurfer Gufo source record

Added November 5, 2026, from the privately retained export `ChatGPT-Run Gufo on Aeon-20261105-1608.md`, titled *Run Gufo on Aeon*. The notebook publishes the experimental information and selected technical evidence, not the conversation transcript. The experiment is dated November 2 by the user messages; container logs print November 3 with an apparent five-hour offset. Setup and follow-up span November 1–3.

| Information | Preserved location |
|---|---|
| Runtime inventory, loading failures, memory diagnostics, successful launch, model list | [Selected terminal evidence](terminal-captures.md) |
| Original final streaming timing event for the battery run | [Battery completion JSON](battery-completion.json) |
| Seven longer-request prefill rows, summary-only | [Jan summary CSV](../../data/silversurfer-gufo-jan-summary.csv) |
| Interpretation, configuration, comparisons, integration outcomes, and limitations | [Notebook entry](../../lab-notebook/2026-11-02-silversurfer-gufo.md) |

The battery completion was parsed from the user's November 2, 21:38 message, which explicitly identified battery operation. The export names `Pasted text(20261103-030733).txt` but does not include the attachment's contents. Later prefill, cache-restore, queueing, and memory observations therefore retain summary-only status. Publisher benchmarks and proposed Maya/Open WebUI settings are not local test results.

## Private source hash

SHA-256 of the original export: `e1c05cc440d6b08824f10e7f687e77a4523b492f2df4b304192fa3b6b6e931a1`.

This hash identifies the private source, not the selected evidence files. The full transcript is intentionally excluded from the published repository. Terminal evidence omits dialogue, password prompts, and unrelated boot arguments; the notebook retains the relevant kernel and allocation settings.
