# GLM-Flash FP16 versus INT8 KV: repeated prefill and decode observations

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

**Date:** November 9, 2026.

You compared FP16 and INT8 attention caches while keeping the Q4 weights and 8K chunks.

## Setup

FP16 ran before INT8, with two repeats at each 8K/32K/64K target. Both used GPUs 2–3, MTP, 1024 MB reserve, and streaming KV with 32K resident positions. The saved terminal output has twelve rows; the named result directory is missing.

## Results

INT8 average generation was about 12.1%, 12.8%, and 11.2% higher at the three targets. Prefill changed by about +3.4%, −1.4%, and −0.9%. The longer INT8 prompts processed near 292 tokens/s. Individual generation samples varied substantially.


You compared GLM-5.3-Flash with FP16 and INT8 attention KV caches after selecting 8K effective prefill chunks. Both configurations kept the same Q4 model weights, two GPUs, 1M configured context, streaming KV with 32K resident positions, 1024 MB reserve, native MTP and 4096 MB prefill budget. This changes cached attention state, not the GGUF weight quantization.

## Timing and source details

The run directory is `glm-kv-20261109-225228`. The archive contains the [runner](../raw-logs/2026-11-10-maya-collection/extracted/maya-glm-kv-sweep.py) and [INT8 config](../raw-logs/2026-11-10-maya-collection/extracted/maya-glm-5.3-flash-q4-2gpu-1m-int8kv.json), but not the named result directory, its response JSONs, or engine logs.

The runner starts the FP16 case and then INT8, with two repetitions each. It prints effective chunk 8192 for both and `ok` for all twelve observations. Its cache-status rule permits up to 1% prefix reuse; exact cache counts cannot be reconstructed from these terminal lines. Raw observations are transcribed in the [combined CSV](../data/maya-glm-november-long-observations.csv).

## Results

Rates are tokens/s; means use the two repetitions at each prompt length. Reported chat counts are 8076 / 32652 / 65420. The FP16 control here is newly measured, not reused from the previous chunk experiment.

| Target | FP16 PP pair | INT8 PP pair | PP mean change | FP16 TG pair | INT8 TG pair | TG mean change |
|---|---|---|---:|---|---|---:|
| 8K | 224.9 / 224.2 | 232.6 / 231.9 | +3.4% | 16.5 / 21.6 | 18.2 / 24.5 | +12.1% |
| 32K | 295.5 / 296.5 | 291.2 / 292.7 | −1.4% | 16.1 / 23.1 | 17.9 / 26.3 | +12.8% |
| 64K | 294.4 / 294.4 | 291.3 / 292.0 | −0.9% | 21.7 / 22.0 | 24.9 / 23.7 | +11.2% |

INT8 shows higher average decode in all three pairs, with modest 8K prefill improvement and a small long-prefill cost. The within-setting decode variation is substantial: FP16 at 32K ranges from 16.1 to 23.1, while INT8 ranges from 17.9 to 26.3. Fixed run order, two samples and short continuations limit the precision of the apparent 11–13% gain.

The previous FP16 chunk sweep had faster decode in some cells than this FP16 control. You retain both experiments rather than selecting the earlier, faster FP16 samples or mixing controls. Their difference reinforces the need to distinguish an observed configuration benefit from a stable causal speedup.

## Quality and context limits

No local KLD, perplexity, retrieval test, long reasoning evaluation or task-accuracy result was supplied. Upstream assertions about INT8 quality and the conversation's Q4-versus-Q6 discussion are not local validation of this KV change. Cache quantization can affect outputs even when weight quantization stays fixed. These tests support the timing decision but do not show quality equivalence.

Likewise, 1,048,576 is the configured capacity, not a tested prompt length. The longest supplied target is 65,536 and the reported chat count is 65,420. The two streaming formats were not compared at the capacity limit.

## Production selection

On November 10 you requested two production-named GLM profiles on port 8082, one FP16 and one INT8. They are alternatives sharing GPUs 2 and 3, so they are not launched together. Your later task statement confirms INT8 as the hosted choice alongside DeepSeek on the other pair. The [deployment entry](2026-11-10-maya-dual-model-production.md) preserves the FP16 fallback and the limits of the captured cohosting evidence.

## Sources

**Sources:** the supplied November 10 Maya archive and conversation exports; [source ledger and retained evidence](../raw-logs/2026-11-10-maya-collection/README.md).

## Conclusion

You chose INT8 and kept an FP16 fallback. Two samples in a fixed order do not show a universal speedup. No local quality comparison or million-token test accompanied the choice.
