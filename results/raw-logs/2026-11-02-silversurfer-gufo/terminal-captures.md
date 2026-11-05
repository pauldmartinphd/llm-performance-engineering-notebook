# SilverSurfer setup and runtime evidence

Selected technical excerpts from the November 1–2 user-pasted terminal output. Dialogue, password prompts, and unrelated boot arguments (including filesystem identifiers) are omitted. The boot-line omissions are marked; other command/output lines retain their source text. Times are preserved as exported. See the [source ledger](README.md) for provenance.

## 11/1/2026, 9:11:30 PM

```text
pmartin@silversurfer:~> podman --version
podman version 6.1.2
pmartin@silversurfer:~> podman info --format '{{.Host.OCIRuntime.Name}}'
runc
pmartin@silversurfer:~> 
pmartin@silversurfer:~> ls -l /dev/kfd /dev/dri/renderD*
crw-rw----+ 1 root render 226, 128 Oct 31 01:37 /dev/dri/renderD128
crw-rw----+ 1 root render 237,   0 Oct 31 01:37 /dev/kfd
pmartin@silversurfer:~> id
uid=1000(pmartin) gid=1000(pmartin) groups=1000(pmartin),495(wheel) context=unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023
pmartin@silversurfer:~>
```

## 11/2/2026, 9:17:05 PM

```text
pmartin@silversurfer:~> podman run --rm -it \
>   --name gufo \
>   --runtime=crun \
>   --userns=keep-id:uid=1000,gid=1000 \
>   --device /dev/kfd \
>   --device /dev/dri \
>   --group-add keep-groups \
>   --ulimit memlock=-1 \
>   -p 127.0.0.1:8080:8080 \
>   -v "$HOME/Models/Qwen3.8-Flash-Next:/models:ro" \
>   ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
>   gufo serve --host 0.0.0.0 --port 8080 llm \
>     --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
>     --speculative mtp \
>     --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
>     --sessions 1 \
>     --context 262144
  ,___,   gufo 0.28.1
  (◉,◉)   Fast LLM & DiT inference · gfx1151
2026-11-03 02:16:47 [INFO] [gpu] event=queue_budget kind=llm observed=0 budget=8 expected_max=3 cap=2
2026-11-03 02:16:47 [INFO] [loader] event=load_started kind=text artifact=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf rss_mib=136 host_available_mib=84396
Error loading model '/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf': Failed to open GGUF: Failed to open or map file: /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf: cannot open: Permission denied
2026-11-03 02:16:47 [ERROR] [loader] event=load_failed kind=text rss_mib=137 host_available_mib=84396
pmartin@silversurfer:~>
```

## 11/2/2026, 9:18:36 PM

```text
pmartin@silversurfer:~> podman run --rm -it \
>   --name gufo \
>   --runtime=crun \
>   --userns=keep-id:uid=1000,gid=1000 \
>   --device /dev/kfd \
>   --device /dev/dri \
>   --group-add keep-groups \
>   --ulimit memlock=-1 \
>   --security-opt label=disable \
>   -p 127.0.0.1:8080:8080 \
>   -v "$HOME/Models/Qwen3.8-Flash-Next:/models:ro" \
>   ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
>   gufo serve --host 0.0.0.0 --port 8080 llm \
>     --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
>     --speculative mtp \
>     --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
>     --sessions 1 \
>     --context 262144
  ,___,   gufo 0.28.1
  (◉,◉)   Fast LLM & DiT inference · gfx1151
2026-11-03 02:17:28 [INFO] [gpu] event=queue_budget kind=llm observed=0 budget=8 expected_max=3 cap=2
2026-11-03 02:17:28 [INFO] [loader] event=load_started kind=text artifact=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf rss_mib=136 host_available_mib=84414
Error loading model '/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf': std::bad_alloc
2026-11-03 02:18:01 [ERROR] [loader] event=load_failed kind=text rss_mib=275 host_available_mib=85753
pmartin@silversurfer:~>
```

## 11/2/2026, 9:19:48 PM

```text
pmartin@silversurfer:~> free -h
               total        used        free      shared  buff/cache   available
Mem:            94Gi        10Gi        82Gi        93Mi       5.7Gi        83Gi
Swap:           94Gi       350Mi        93Gi
pmartin@silversurfer:~> 
pmartin@silversurfer:~> cat /proc/cmdline
[boot command line excerpt; kernel 7.3.0-1-default; unrelated arguments omitted]
[no explicit TTM/GTT sizing arguments]
pmartin@silversurfer:~> 
pmartin@silversurfer:~> cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null
34359738368
pmartin@silversurfer:~> cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null
50493997056
pmartin@silversurfer:~>
```

## 11/2/2026, 9:21:18 PM

```text
pmartin@silversurfer:~> sudo cat /sys/module/amdgpu/parameters/gttsize
-1
pmartin@silversurfer:~> cat /sys/module/ttm/parameters/pages_limit 2>/dev/null
12327636
pmartin@silversurfer:~>
```

## 11/2/2026, 9:30:46 PM

```text
pmartin@silversurfer:~> podman run --rm -it   --name gufo   --runtime=crun   --userns=keep-id:uid=1000,gid=1000   --device /dev/kfd   --device /dev/dri   --group-add keep-groups   --ulimit memlock=-1   --security-opt label=disable   -p 127.0.0.1:8080:8080   -v "$HOME/Models/Qwen3.8-Flash-Next:/models:ro"   ghcr.io/gufo-org/toolboxes/gufo-runtime:latest   gufo serve --host 0.0.0.0 --port 8080 llm     --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf     --speculative mtp     --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf     --sessions 1     --context 262144
  ,___,   gufo 0.28.1
  (◉,◉)   Fast LLM & DiT inference · gfx1151
2026-11-03 02:30:01 [INFO] [gpu] event=queue_budget kind=llm observed=0 budget=8 expected_max=3 cap=2
2026-11-03 02:30:01 [INFO] [loader] event=load_started kind=text artifact=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf rss_mib=135 host_available_mib=89192
Error loading model '/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf': std::bad_alloc
2026-11-03 02:30:28 [ERROR] [loader] event=load_failed kind=text rss_mib=280 host_available_mib=90073
```

## 11/2/2026, 9:31:22 PM

```text
pmartin@silversurfer:~> cat /proc/cmdline
[boot command line excerpt; kernel 7.3.0-1-default; unrelated arguments omitted]
ttm.pages_limit=33554432 ttm.page_pool_size=33554432 amdgpu.gttsize=131072
pmartin@silversurfer:~> 
pmartin@silversurfer:~> echo "GPU memory limits:"
GPU memory limits:
pmartin@silversurfer:~> cat /sys/module/amdgpu/parameters/gttsize
cat: /sys/module/amdgpu/parameters/gttsize: Permission denied
pmartin@silversurfer:~> cat /sys/module/ttm/parameters/pages_limit
33554432
pmartin@silversurfer:~> cat /sys/module/ttm/parameters/page_pool_size
33554432
pmartin@silversurfer:~> 
pmartin@silversurfer:~> echo "Reported GPU memory:"
Reported GPU memory:
pmartin@silversurfer:~> for f in /sys/class/drm/card*/device/mem_info_{vram,gtt}_total; do
>     echo "$f: $(cat "$f")"
> done
/sys/class/drm/card1/device/mem_info_vram_total: 34359738368
/sys/class/drm/card1/device/mem_info_gtt_total: 100987990016
pmartin@silversurfer:~> 
pmartin@silversurfer:~> echo "System memory:"
System memory:
pmartin@silversurfer:~> free -h
               total        used        free      shared  buff/cache   available
Mem:            94Gi       9.8Gi        85Gi        95Mi       2.5Gi        84Gi
Swap:           94Gi        79Mi        93Gi
pmartin@silversurfer:~>
```

## 11/2/2026, 9:36:17 PM

```text
pmartin@silversurfer:~> podman run --rm -it   --name gufo   --runtime=crun   --userns=keep-id:uid=1000,gid=1000   --device /dev/kfd   --device /dev/dri   --group-add keep-groups   --ulimit memlock=-1   --security-opt label=disable   -p 127.0.0.1:8080:8080   -v "$HOME/Models/Qwen3.8-Flash-Next:/models:ro"   ghcr.io/gufo-org/toolboxes/gufo-runtime:latest   gufo serve --host 0.0.0.0 --port 8080 llm     --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf     --speculative mtp     --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf     --sessions 1     --context 262144
  ,___,   gufo 0.28.1
  (◉,◉)   Fast LLM & DiT inference · gfx1151
2026-11-03 02:34:52 [INFO] [gpu] event=queue_budget kind=llm observed=0 budget=8 expected_max=3 cap=2
2026-11-03 02:34:52 [INFO] [loader] event=load_started kind=text artifact=/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf rss_mib=135 host_available_mib=119129
2026-11-03 02:35:18 [INFO] [cache] schema=component-cache-v1 event=backing_prepared committed_bytes=2259419136 long_term_pinned_bytes=0
2026-11-03 02:35:18 [INFO] [cache] schema=component-cache-v1 event=configured sessions=1 snapshot_entries=128 capacity_bytes=14246539264 staging_capacity_bytes=0 automatic_bytes=14246539264 max_bytes=24198111232 disk_capacity_bytes=0 slot_pinned_bytes=0 ram_cache=on disk_backing_bytes=0
2026-11-03 02:35:18 [INFO] [loader] event=structured_output_prepared vocabulary_tokens=248320 prepare_ms=29.7 rss_delta_kib=31100 rss_mib=2734 host_available_mib=24986
2026-11-03 02:35:18 [INFO] [sampling] event=defaults thinking=on temperature=1 top_k=20 top_p=0.95 min_p=0 min_keep=0 seed=-1 repeat_penalty=1 repeat_last_n=64 frequency_penalty=0 presence_penalty=0
2026-11-03 02:35:18 [INFO] [loader] event=load_completed kind=text elapsed_ms=25240 monotonic_us=97247185 model=Qwen3.8 Flash Next sessions=1 context_tokens=262144 speculative=mtp draft_limit=7 disk_cache=off rss_mib=2704 host_available_mib=24986 gpu_device_used_mib=99994 gpu_device_total_mib=131072
2026-11-03 02:35:18 [INFO] [server] event=listening monotonic_us=97247636 address=http://0.0.0.0:8080 auth=off max_connections=16 max_body_bytes=8388608
```

## 11/2/2026, 9:37:04 PM

```text
pmartin@silversurfer:~> curl -N http://127.0.0.1:8080/v1/chat/completions \
>   -H "Content-Type: application/json" \
>   -d '{
>     "model": "MODEL_ID",
>     "messages": [
>       {"role": "user", "content": "Explain the difference between a process and a thread."}
>     ],
>     "max_tokens": 512,
>     "stream": true
>   }'
{"error":{"message":"model 'MODEL_ID' is not served by this process","type":"invalid_request_error","code":"model_not_found"}}pmartin@silversurfer:~> 
pmartin@silversurfer:~>
```

## 11/2/2026, 9:37:26 PM

```text
{
    "object": "list",
    "data": [
        {
            "id": "Qwen3.8 Flash Next",
            "object": "model",
            "created": 1793673438,
            "owned_by": "gufo",
            "context_length": 262144,
            "architecture": {
                "input_modalities": [
                    "text"
                ]
            }
        }
    ]
}
```
