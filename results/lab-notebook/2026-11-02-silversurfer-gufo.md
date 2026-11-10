# November 1–3: Gufo on SilverSurfer, memory allocation, and Jan

[All lab records](README.md) · [Related investigation: chapter 7](../../investigations/07-other-local-machines.md)

## Purpose and reasoning

Bring Qwen3.8-Flash-Next up on SilverSurfer and test whether the shared-memory laptop can serve it usefully on battery. Diagnose file access and memory allocation as separate obstacles.

## Experimental design

I launched Gufo 0.28.1 in the recorded Podman setup, with Q4 weights, the MTP predictor, one session, and a 262144-token configured ceiling. I checked memory readbacks through the failed and successful starts, then retained the direct API timing event. I kept the later Jan attachment-summary figures separate from the direct result.

## Results

The first attempts failed at file access and then allocation. A later launch loaded in about 25 seconds. The retained on-battery event generated 318 tokens at 29.85 tokens/s after a 63-token prompt. Later reported Jan prefill rates of 539–740 tokens/s lack their original attachment.

## Conclusions

The laptop successfully hosted the tested model and request beyond the initial firmware-reserved GPU slice. Several memory settings changed and the final BIOS value was not recorded, so one change cannot be credited alone. The short request did not test the context ceiling or establish a matched comparison with Maya.

## Detailed record

**Experiments:** November 2, 2026, with setup on November 1 and follow-up on November 3. **Added:** November 5. **Machine:** SilverSurfer, HP ZBook Ultra G1a, Ryzen AI Max+ PRO 395, 128 GB unified memory, openSUSE Aeon. **Source:** [source ledger and selected technical evidence](../raw-logs/2026-11-02-silversurfer-gufo/README.md).

I brought Gufo 0.28.1 up on SilverSurfer with Qwen3.8-Flash-Next UD-Q4_K_XL and its shared Q8_0 MTP predictor. The first retained completion generated **318 tokens at 29.85 tokens/s on battery**. The later conversation summary records **539–740 tokens/s for newly processed prompt tokens** as Jan's conversation grew to 54,781 tokens. The first result has its original timing JSON; the longer-request results survive only in the assistant's analysis of an attachment whose contents are absent from the export. I keep those evidence levels separate throughout this entry.

The exported messages place the successful run on November 2 in the evening. Container logs print November 3 around 02:35, consistent with a five-hour offset from America/New_York, though the log timezone is not explicitly stated. The selected terminal evidence preserves both clocks.

## Runtime and model

Podman reported version 6.1.2 and initially selected `runc`. The successful launch explicitly selected `crun`, preserved UID/GID 1000 and supplementary groups, passed `/dev/kfd` and `/dev/dri`, and set unlimited memlock. Gufo identified the GPU as **gfx1151**. The kernel path in `/proc/cmdline` identified `7.3.0-1-default`.

The working launch was:

```bash
podman run --rm -it \
  --name gufo \
  --runtime=crun \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ulimit memlock=-1 \
  --security-opt label=disable \
  -p 127.0.0.1:8080:8080 \
  -v "$HOME/Models/Qwen3.8-Flash-Next:/models:ro" \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
    --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --speculative mtp \
    --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    --sessions 1 --context 262144
```

This is the historical command, including disabled SELinux container labeling; it is not a new host configuration change. The container tag was `latest`, but the banner pins the observed runtime to 0.28.1. No image digest or Gufo source revision was captured. The proposed Hugging Face download pinned `unsloth/Qwen3.8-Flash-Next-GGUF` to `38bb39ee97821de2c9009abb7e93950eec396e66`; the export has no completed download verification or model hashes, so that remains intended provenance rather than verified file identity.

The server reported one session, 262,144 configured context tokens, MTP with a draft limit of seven, and disk cache off. Sampling defaults were thinking on, temperature 1, top-k 20, top-p 0.95, min-p 0, seed -1, repeat penalty 1, and zero frequency/presence penalties. The model-list response advertised `Qwen3.8 Flash Next`, with spaces, and text input only. Sending the literal placeholder `MODEL_ID` failed with `model_not_found`; using the advertised ID resolved that mismatch.

## Loading failures and unified memory

The first launch failed to open the GGUF with `Permission denied`. Adding `--security-opt label=disable` let loading advance, consistent with a labeling restriction, but no audit trace establishes the exact policy denial. The next failure was `std::bad_alloc`, which required separate investigation.

| Capture | Linux memory | VRAM reported | GTT reported | Outcome |
|---|---|---:|---:|---|
| Initial diagnostics | 94 GiB total, 83 GiB available | 34,359,738,368 bytes (32 GiB) | 50,493,997,056 bytes (47.03 GiB) | Loading failed with `std::bad_alloc` |
| After kernel settings, still 32 GiB VRAM | 94 GiB total, 84 GiB available | 34,359,738,368 bytes | 100,987,990,016 bytes (94.05 GiB) | The preceding retry still failed |
| Successful load | 119,129 MiB available before load; 24,986 MiB afterward | No final sysfs readback | No final sysfs readback | Loaded in 25,240 ms |

Initially `amdgpu.gttsize` read `-1` and `ttm.pages_limit` read `12327636`. The later boot command line contained `ttm.pages_limit=33554432 ttm.page_pool_size=33554432 amdgpu.gttsize=131072`; both TTM values were confirmed in sysfs. Reading `amdgpu.gttsize` without elevated privileges failed, so its effective value was not confirmed in that capture.

Increasing those limits alone did not resolve the failure while the 32 GiB firmware reservation remained. The next recommendation was to reduce that reservation to 512 MiB, or 1–2 GiB if necessary. The subsequent launch succeeded with substantially more host memory available. This sequence supports the memory-allocation explanation, but the final BIOS value was not recorded and no failing-allocation trace isolates the cause. I cannot claim that a particular BIOS setting was read back or that the kernel change alone fixed loading.

At load completion Gufo reported `gpu_device_used_mib=99994`, `gpu_device_total_mib=131072`, and `rss_mib=2704`. The GPU usage is about **97.65 GiB**, with **24.40 GiB** host memory available. These are runtime counters on a unified-memory machine, not independent physical pools to add together, and the load snapshot is not a measured session peak. The configured component-cache capacity was 14,246,539,264 bytes, with 128 snapshot entries, RAM cache on, and no disk backing.

The earlier claim that SilverSurfer could only serve models fitting the firmware-reserved GPU slice was too restrictive. This run demonstrates GPU access beyond that slice through the shared-memory allocation path; it does not establish unlimited usable capacity or successful operation at the full context ceiling.

## Direct API result on battery

The user explicitly identified this run as being on battery. The preceding suggested prompt was “Explain the difference between a process and a thread.” The actual successful request body was not pasted, but its final streaming timing event was.

| Field | Retained value |
|---|---:|
| Prompt tokens | 63 |
| Prompt time | 560.886398 ms |
| Prompt throughput | 112.322210 tokens/s |
| Generated tokens | 318 |
| Generation time | 10,651.538481 ms |
| Generation throughput | 29.854842 tokens/s |
| Cached tokens | 0 |
| Draft rounds | 127 |
| Draft tokens proposed / accepted | 273 / 191 |
| Derived draft acceptance | 69.96% |
| Cache snapshot time | 8.067906 ms |

The completion ended with `finish_reason=stop`. The 63-token prompt is too short to characterize sustained prefill; its 112.32 tokens/s must not be compared directly with a long-prompt benchmark. There was no retained AC-power repetition, no MTP-off control, and no measurement of power draw. The 262,144-token setting describes configured capacity, not the amount of context exercised here.

## Longer Jan conversation: summary-only results

The November 2, 22:07 message references `Pasted text(20261103-030733).txt`. The export contains the attachment name but not the original file. The following rows reproduce the assistant's extracted table, not independently recovered request logs:

| Total context tokens | Newly processed tokens | Prefill tokens/s |
|---:|---:|---:|
| 5,563 | 4,905 | 539 |
| 11,657 | 6,094 | 740 |
| 23,385 | 9,725 | 726 |
| 34,878 | 8,742 | 729 |
| 48,337 | 9,191 | 718 |
| 51,362 | 3,025 | 562 |
| 54,781 | 3,419 | 710 |

The same analysis quotes a request with `prompt_tokens=48337`, `prefill_tokens=9191`, `cached_tokens=39146`, `cache=memory`, and `cache_restore_ms=21.6`. The token counts reconcile: 39,146 reused plus 9,191 new equals 48,337. A later response describes a 23,385-token prefix restored in about 12 ms, generation at roughly 30–45 tokens/s, and some reported first-token timings near 14 ms. Without the attachment I cannot verify those timing definitions or treat them as end-to-end latency measurements.

The analysis also reports repeated 477-token requests alongside the main conversation, occasional seconds of queue delay, a cache ledger growing from about 2.5 to 10.6 GB, and available host memory falling toward 10 GiB. These observations remain summary-only; the background requests were not conclusively attributed to Jan, and battery operation was explicitly confirmed only for the earlier short API test.

On November 3 I reported that Qwen through Gufo felt enormously faster than DeepSeek V4.1 even though prompt processing was not much better. That is a useful user observation, not a matched engine benchmark. Different models, hardware, prompts, reasoning lengths, cache reuse, and queueing can all affect time to a useful answer. The notebook's existing Galactus Maya measurements remain valid for their own workloads; this session does not establish a Gufo/Maya speedup ratio.

## Jan, Galactus, and prompt reuse

The local Jan provider was configured around `http://127.0.0.1:8080/v1`, provider name `Gufo`, and model ID `Qwen3.8 Flash Next`. The server logged `auth=off`; the suggested `gufo` key was only a non-secret placeholder for a required client field. Host port publication was restricted to loopback.

For Galactus, I supplied the server address and corrected the assistant's assumed port 3000 to **8080**, giving Jan the base URL **`http://<GALACTUS_HOST>:8080/api`** (host address omitted from this public note) with the existing Open WebUI API key. No actual key is in the retained record. Jan displayed its custom-provider capability warning. I subsequently reported that Jan's tools were not being used through the proxy; after advice to enable the model's Tools capability and compare structured tool calls directly and through the proxy, I reported “Fixed.” The exact change and API test outputs were not captured, so the outcome is confirmed but its root cause is not.

The caching discussion proposed `MAYA_DS4_SLOT_KEEP=1` and a separate test of `MAYA_DS4_KEEP_THINKING=1`, along with stable system prompts and tool definitions, native function calling, and less automatic citation, file-context, and memory injection in Open WebUI. These are historical investigation proposals, not settings verified or applied in this session. Retaining previous reasoning changes subsequent model inputs and should be evaluated separately from retaining computed state. The proposed direct-Maya provider had an unknown port and no captured test result.

Vision setup for Gufo was proposed using `mmproj-BF16.gguf`, but no projector-enabled launch or image test was recorded. The captured model list advertised text only. Audio/video deployments and the later DeepSeek modality discussion likewise supplied no local test results; assistant capability assertions do not establish functionality on either machine.

## Published comparisons discussed, not local measurements

The conversation compared Gufo's published prefill figures of approximately 1,801 tokens/s at an empty prefix and 1,707 / 1,673 / 1,603 / 1,557 at 16K / 32K / 64K / 128K cached-prefix depths. It also cited generation around 28.29 without MTP, 35.7–35.8 for mixed text with MTP, and 116–117 for repetitive text. Those figures came from the assistant's account of publisher benchmarks, with greedy decoding and thinking disabled, unlike this run's defaults. They are recorded here as research context, not independently verified SilverSurfer results.

The Hipfire discussion cited 0.4.4 prefill at 2,073 tokens/s for 8K and 2,473 for a 16,355-token cold request with 6.61 s to first token; an older 0.4.1 result was 1,777 at 32K. It also described NPU-assisted prefill and an optimized 512-token append to a 256K context. No Hipfire installation or benchmark on SilverSurfer was captured. Gufo's incremental prefill at a cached depth and Hipfire's cold-request measurements cannot establish a head-to-head winner.

## Remaining measurements

Recover the missing Jan log before promoting its summary rows to primary measurements. Record the final firmware reservation, effective memory limits, image digest, model hashes, power profile, and AC/battery state. Then measure cold prompts and cached appends separately at fixed lengths, recording new tokens, reused tokens, prefill time, first-token latency, time to a useful answer, and peak memory. A controlled MTP on/off comparison and output-quality checks are still needed. STREAM bandwidth, VMM behavior, and llama.cpp split-mode behavior remain unmeasured on SilverSurfer.
