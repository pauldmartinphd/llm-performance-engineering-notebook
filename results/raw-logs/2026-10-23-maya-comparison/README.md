# DeepSeek cache and Maya comparison sources

Added October 23, 2026 from three user-supplied conversation exports. These contain both terminal output and assistant discussion; proposed commands are not evidence of execution.

| Source | Evidence | Preservation |
|---|---|---|
| [Cache benchmark conversation](source-conversation.txt) | Full 32 GiB llAMDa command/table; summary-only 0/64/80 GiB results | Byte-for-byte original |
| [Maya benchmark and calibration conversation](maya-benchmark-conversation.txt) | Maya v1.1.24 benchmark dated October 23 at 14:55, and DSpark off/on calibration output | Byte-for-byte original |
| [LXC and memlock conversation](lxc-memlock-conversation.txt) | Container configuration, shell limits returning unlimited, and proposed Maya residency edits | Network hostname, MAC and private IP addresses replaced with labeled placeholders; remaining text preserved |

The Maya benchmark has a printed date. The cache sweep's individual run dates are not established. The earlier 8 MiB limit and initial Maya rates are discussed but their original diagnostic captures are absent. The final Maya JSON and residency diagnostics are not included.

SHA-256 of retained files:

- `lxc-memlock-conversation.txt`: `1f712484f49beaddd124d29b5fa791255c9a856ebe9ccbce5567b5ce252ff0e9`
- `maya-benchmark-conversation.txt`: `e813391e95d54c0a2f7287b620fd90a22bce947c9a0c5c3dd0055caf1b2d9ea2`
- `source-conversation.txt`: `da1f0fc4b371372872cced78e138ce92d0c374c047cf1c727facd29ab5239ec9`

[Analysis](../../lab-notebook/2026-10-23-deepseek41-cache-and-maya.md).
