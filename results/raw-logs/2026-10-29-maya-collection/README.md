# Maya artifacts collected October 29, 2026

The supplied archive was created on Galactus at 14:15 Eastern and imported into the notebook on October 29. The gzip/tar payload was read and safely extracted: 521 members, no symlinks or path traversal, and no missing files among the paths in `checkout-files.list`. This validates the archive against its collection manifest, not every file that may have existed on the server. Build directories, Git internals, model weights, and virtual environments were excluded by the collection command.

The original [archive](maya-logs-20261029-141501.tar.gz) is retained unchanged. [SHA256SUMS](SHA256SUMS) covers the archive and every extracted file. The complete [extracted collection](extracted/) includes configs, benchmark reports, engine/server logs, API responses, sweep metadata, and both scripts as run. Large `.log` files remain subject to the repository's existing Git ignore policy; the original archive preserves their bytes. Nothing has been published upstream.

## Completeness and provenance

- `20261028-200032`: 10 completed cases, 60 rows, all `ok`, and the original `summary.csv`.
- `20261028-223138`: intentionally stopped by Paul after the trend was clear. Fifteen cases completed; case 16 was interrupted during warm-up. Ninety timed rows were saved: 54 `ok`, 36 `layout-mismatch-exclude`. Nine planned cases never started. There is no final harness `summary.csv` or closing baseline. [partial-tp-summary.csv](partial-tp-summary.csv) is derived here from only the 54 `ok` rows; it is not an original harness output.
- All six completed TP=4 requests actually used TP=2. Their 36 rows remain in the raw results but cannot establish TP=4 performance.
- Both server scripts match their recorded `script_sha256` in sweep metadata and the archive's `scripts-sha256.txt`. The separately uploaded scripts match `experiments/` and differ from the server copies only by one trailing blank line. Preserve the server copies for exact reproduction.
- Both sweeps used v1.1.30 commit `8621e33159f426da2d376be4ec012b0ce8fe2948`, binary SHA256 `48fcd11d9701a836069db0b633b7f42ef175ec1363bee06d29b2f65c37e10697`, and corpus SHA256 `2478f6f55df95f210def6fb9db16f5d61d144e2ab41d39f35042ddbeff33b571`.
- The metadata's attempted `third_party/llama.cpp` revision query returned the Maya commit. This does not verify an independent llama.cpp revision: Git may have found the parent repository. Do not use that field as a dependency identity.
- The separately supplied shell history ends with the earlier update commands. It does not add a transcript of the overnight sweep or the October 29 regression test. Its recovery evidence is incorporated in the October 23 note; the full unrelated history is not republished here.

## Current checkout identities

At collection, `project-maya-production` was v1.1.30 (`8621e33159f426da2d376be4ec012b0ce8fe2948`); `project-maya` was v1.1.36 (`f031a23b9682551e744e29d0e9f8d527df49f282`). The tracked diffs are empty. The v1.1.36 binary hash is `5a4009f53960b535a0518bcc618d9d6e34f8db23999a273f18189da07fa54774`.

Config files inside the production checkout still reference `/opt/project-maya/build-hip/maya-ds4` and `/opt/project-maya` as the working directory. Their directory location does not guarantee which binary they launch. The collection makes no changes to these files or the server.

The v1.1.36 `maya-bench.txt` records a complete benchmark with `maya-ds4-production-no-stream-prio.json`: decode mean 31.1 tokens/s, prefill 195 at 2048 and 334 at 8192. The matching config sets `DS4_STREAM_PRIO=0`. Paul separately confirmed on October 29 that `DS4_STREAM_PRIO=1` causes a deadlock. This is a user-reproduced bug with a captured successful workaround run; a paired priority-on failure capture has not been positively identified in this bundle.

See [the notebook analysis](../../lab-notebook/2026-10-28-maya-prefill-and-regression.md). Keep the original archive until any cleanup and later transfer have been verified. The archive is a point-in-time snapshot; logs written after 14:15 are not covered.
