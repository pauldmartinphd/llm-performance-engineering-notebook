# Qwen3.8-Flash-Next on SilverSurfer

These are Gufo measurements on SilverSurfer. One on-battery completion has direct timing evidence; the later Jan prefill figures have a weaker source. They are not a matched Gufo-versus-Maya comparison.

[Related investigation: chapter 7](../investigations/07-other-local-machines.md) · [All model results](README.md)

On November 2, 2026, you ran the UD-Q4_K_XL model with its shared Q8_0 MTP predictor under Gufo 0.28.1 on SilverSurfer's Ryzen AI Max+ PRO 395 and 128 GB unified memory. The configured context was 262,144 tokens, with one session and up to seven draft tokens.

| Observation | Result | Evidence |
|---|---|---|
| Battery API completion | 318 generated tokens at 29.854842 tokens/s | Original timing JSON |
| Short uncached prompt | 63 tokens at 112.322210 tokens/s | Original timing JSON; not sustained prefill |
| MTP acceptance | 191 of 273 proposed tokens, 69.96% | Derived from original counters |
| Longer Jan requests | 539–740 tokens/s prefill; total context up to 54,781 | Assistant summary of missing attachment |
| Load time / GPU usage at completion | 25,240 ms / 99,994 MiB | Original startup log; not peak memory |

The launch used thinking on, temperature 1, top-k 20, and top-p 0.95. There is no retained AC-power repetition, MTP-off baseline, full-context test, or matched Maya comparison. The longer Jan measurements involve cached prefixes and newly processed tokens; they must not be treated as cold prefill at the listed total context lengths.

See the [full notebook entry](lab-notebook/2026-11-02-silversurfer-gufo.md), [hardware record](../hardware/silversurfer/README.md), and [source archive](raw-logs/2026-11-02-silversurfer-gufo/README.md) for setup, failures, exact commands, all seven Jan rows, and source details.
