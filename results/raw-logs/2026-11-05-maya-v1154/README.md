# November 5 Maya v1.1.54 source record

Added November 6, 2026. [Notebook entry](../../lab-notebook/2026-11-05-maya-v1154-prefill-and-peer-ar.md).

- [Conversation export](conversation.md): supplied as `ChatGPT-Fast Card Prefill Acceleration-20261106-1545.md`. It contains original user terminal captures, assistant interpretation, proposed commands, and later corrections. Historical commands are evidence, not instructions to run. Two signed screenshot download URLs were removed; the images were not supplied. Other text and line numbering are preserved.
- [Benchmark runner](../../../experiments/maya_benchmark_1154_fixed.py): the supplied 225-line file, retained byte-for-byte. It is not an independently collected copy from the benchmark host. The conversation proposes subsequent edits, so exact identity with the executed file is unverified.
- [Manifest](manifest.json): SHA-256 hashes of both supplied inputs and the archived conversation.
- [Individual observations](../../data/maya-v1154-prefill-observations.csv): 47 unique, complete prefill dictionaries extracted only from user terminal captures. Repeated pastes are deduplicated; `source_lines` records all occurrences. The truncated HIP 8K row is not parsed, but its complete later repetition is retained.
- [Printed summary](../../data/maya-v1154-prefill-summary.csv): all 30 printed mean rows, each reporting two repetitions. Values retain the original one-decimal precision. Missing individual repetitions are not reconstructed from rounded means. Source line numbers refer to `conversation.md`.

The main run reports `/opt/maya-bench-1154-20261105-142331`. The focused retest reports `/opt/maya-bench-1154-20261105-201458`. Neither directory, the generated `results.csv`, effective configs, binary hashes, full engine logs, nor screenshot attachments were supplied. The conversation excerpts are the retained evidence; the extracted CSVs are not the original server CSV.
