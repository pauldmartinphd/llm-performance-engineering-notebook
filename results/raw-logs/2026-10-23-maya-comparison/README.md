# DeepSeek cache and Maya evidence

The October 23 conversation exports were consolidated on October 29, 2026. The current notebook keeps their experimental information without the AI dialogue. Original exports remain recoverable from Git history before this consolidation; their hashes below identify the source versions and are not checksums of the new extract.

| Information | Preserved location |
|---|---|
| 32 GiB command, device inventory, and benchmark table | [Terminal evidence](benchmark-and-residency-evidence.md) |
| LXC configuration, unlimited soft/hard shell limits | [Configuration evidence](benchmark-and-residency-evidence.md) |
| Maya benchmark, CLI calibration response, internal DSpark invocation and output | [Calibration evidence](benchmark-and-residency-evidence.md) |
| Summary-only cache sweep and early Maya rates; residency edits and shell history | [DeepSeek notebook](../../lab-notebook/2026-10-23-deepseek41-cache-and-maya.md) |
| Diagnostics, unperformed retests, 32/64-thread proposals, memory and comparison limits | [Residency and comparison proposals](../../lab-notebook/2026-10-23-deepseek41-cache-and-maya.md#residency-diagnostics-and-comparison-proposals-retained-from-the-source-discussion) |
| Native quantization sizes/quality claims, Flash/flagship plan, superseded setup choices | [GLM setup record](../../lab-notebook/2026-10-23-maya-glm53-speculation.md#setup-choices-and-proposals-retained-from-the-source-discussion) |

The Maya benchmark has a printed date; individual cache-sweep run dates remain unknown. The earlier 8 MiB limit and initial Maya rates are summary-only. Proposed commands remain distinct from executed commands. Hostname, MAC and private IP placeholders in the LXC excerpt are inherited from the original sanitized export.

## Historical source hashes

| Removed export | SHA-256 |
|---|---|
| `source-conversation.txt` | `da1f0fc4b371372872cced78e138ce92d0c374c047cf1c727facd29ab5239ec9` |
| `maya-benchmark-conversation.txt` | `e813391e95d54c0a2f7287b620fd90a22bce947c9a0c5c3dd0055caf1b2d9ea2` |
| `lxc-memlock-conversation.txt` | `1f712484f49beaddd124d29b5fa791255c9a856ebe9ccbce5567b5ce252ff0e9` |
