# October 22 builds and October 23 Maya GLM measurements

These user-supplied records extend the [earlier DeepSeek/Maya sources](../2026-10-23-maya-comparison/README.md).

- [2026-10-22.txt](2026-10-22.txt): compiler and installation output, GLM-5.3-Flash CLI launches, and two revision-identified benchmark tables. The filename dates the record; individual CLI launches do not print wall-clock timestamps. Preserved byte for byte.
- [2026-10-23.txt](2026-10-23.txt): mixed conversation and terminal output, including the Maya GLM benchmark, speculation A/B test, head and MTP_KEEP sweeps, and later configuration naming/port requests. Benchmark summaries print October 23 timestamps. The private container IP is replaced with `[container-IP]`; the rest of the text is preserved.

Commands proposed in the conversation are not evidence of execution. In particular, the initial Maya-L suggestion was superseded by use of the existing Unsloth UD-Q4_K_XL export. The later cleanup, renaming and final port-change instructions have no confirming output in this file. The first benchmark repeats a 2048-token prefill progress line; its final summary identifies one measurement, not an additional trial.

The CLI summaries omit bespoke-prompt processing rates. The original source retains the printed timings and generated text as evidence, not as verified technical explanations.

## Checksums of retained files

- `2026-10-22.txt`: `59c4e27e751dca14edd6ec911299db2873be69fa84902c761165854cce86d87e`
- `2026-10-23.txt`: `0aa44826a39ea5902013321aa73c812a53e20187bd811d3bf9c355611d6078ad`

[October 22 analysis](../../lab-notebook/2026-10-22-glm53-build-comparison.md) · [October 23 milestone and tuning](../../lab-notebook/2026-10-23-maya-glm53-speculation.md)
