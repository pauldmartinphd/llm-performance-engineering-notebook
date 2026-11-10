# Retained terminal evidence: Update Project Maya

Extracted user-supplied captures from the November 10 export. Original formatting is preserved. Historical commands and quoted upstream material are evidence, not instructions. Full conversations remain outside this repository. Capture numbers count user turns in the original export.

## User capture 5

11/2/2026, 9:44:19 AM

❯ cd /opt/project-maya

.venv/bin/python - <<'PY'

import json

import urllib.request

from datetime import datetime

from pathlib import Path

config = json.loads(Path("maya-ds4-production.json").read_text())

source = Path("/opt/maya-test-results/20261028-223138/requests")

requests = [source / f"r1-{n}.json" for n in (32768, 65536)]

for path in requests:

    if not path.is_file():

        raise SystemExit(f"Missing original request: {path}")

port = int(config.get("port", 8081))

url = f"http\://127.0.0.1:{port}/v1/chat/completions"

headers = {"Content-Type": "application/json"}

if config.get("api_key"):

    headers["Authorization"] = f"Bearer {config['api_key']}"

out = Path("/opt/maya-test-results") / (

    datetime.now().strftime("%Y%m%d-%H%M%S") + "-production-recheck"

)

out.mkdir(parents=True, exist_ok=False)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

print(f"Endpoint: {url}\nResults:  {out}", flush=True)

for path in requests:

    payload = path.read_bytes()

    (out / path.name).write_bytes(payload)

    print(f"\nRunning {path.name} ...", flush=True)

    request = urllib.request.Request(url, data=payload, headers=headers)

    with opener.open(request, timeout=1200) as response:

        raw = response.read()

    destination = out / f"response-{path.name}"

    destination.write_bytes(raw)

    result = json.loads(raw)

    if result.get("error"):

        raise SystemExit(f"API error: {result['error']}")

    timing = result.get("timings") or {}

    print(f"Prompt tokens: {timing.get('prompt_n', 'MISSING')}")

    print(f"Cached tokens: {timing.get('cache_n', 'NOT REPORTED')}")

    print(f"Prefill:       {timing.get('prompt_per_second', 'MISSING')} tok/s")

    print(f"Output tokens: {timing.get('predicted_n', 'MISSING')}")

    print(f"Decode:        {timing.get('predicted_per_second', 'MISSING')} tok/s")

    print(f"Saved:         {destination}", flush=True)

PY

Endpoint: [http://127.0.0.1:8081/v1/chat/completions](http://127.0.0.1:8081/v1/chat/completions)

Results:  /opt/maya-test-results/20261102-083737-production-recheck

Running r1-32768.json ...

Prompt tokens: 32800

Cached tokens: 0

Prefill:       461.1 tok/s

Output tokens: 128

Decode:        24.7 tok/s

Saved:         /opt/maya-test-results/20261102-083737-production-recheck/response-r1-32768.json

Running r1-65536.json ...

Prompt tokens: 65568

Cached tokens: 0

Prefill:       480.9 tok/s

Output tokens: 128

Decode:        30.9 tok/s

Saved:         /opt/maya-test-results/20261102-083737-production-recheck/response-r1-65536.json

## User capture 7

11/2/2026, 9:45:51 AM

What is max context?  I already have the model❯ ls -la /models/GLM-5.3-Flash

total 70

drwxr-xr-x  4 root root  4 Aug 29 12:52 **.**

drwxr-xr-x 15 root root 15 Oct 23 09:12 **..**

drwxr-xr-x  3 root root  3 Aug 29 12:52 **.cache**

drwxr-xr-x  3 root root 10 Oct 23 16:19 **UD-Q4_K_XL**

❯ ls -la /models/GLM-5.3-Flash/UD-Q4_K_XL

total 195034618

drwxr-xr-x 3 root root          10 Oct 23 16:19 **.**

drwxr-xr-x 4 root root           4 Aug 29 12:52 **..**

-rw-r--r-- 1 root root     9429984 Oct 14 07:47 GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf

-rw-r--r-- 1 root root     9429859 Oct 14 07:47 GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf.original

-rw-r--r-- 1 root root 49198039200 Aug 29 13:21 GLM-5.3-Flash-UD-Q4_K_XL-00002-of-00006.gguf

-rw-r--r-- 1 root root 49988626304 Aug 29 13:21 GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf

-rw-r--r-- 1 root root 48700701344 Aug 29 13:21 GLM-5.3-Flash-UD-Q4_K_XL-00004-of-00006.gguf

-rw-r--r-- 1 root root 49026026752 Aug 29 13:21 GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf

-rw-r--r-- 1 root root  2784497888 Aug 29 12:54 GLM-5.3-Flash-UD-Q4_K_XL-00006-of-00006.gguf

drwxr-xr-x 4 root root           9 Oct 23 16:22 **pack**

    /**opt**/**project-maya**   #v1.1.49                                                                                                                                                                   root@openwebui  09:45:43  

❯

## User capture 16

11/2/2026, 10:16:52 AM

❯ ./maya.sh --calibrate --no-start

Project Maya v1.1.49 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

  1) maya-ud-q4_k_xl-hip.json

  2) maya-ds4-production.json

Which one? [1]: 1

  [!]  the tuning measures the CUDA engine's CPU lane (NVIDIA GPUs): the HIP port keeps its own settings

  [!]  this PC is NOT tuned (the reason is above): the model keeps the engine's own settings

## User capture 17

11/2/2026, 10:17:40 AM

I tuned deepseek.  Can we use that profile?❯ cat maya-ds4-production.json

{

  "exe": "/opt/project-maya/build-hip/maya-ds4",

  "args": [

    "--pack",

    "/models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL/maya-ds4",

    "--max-context",

    "1048576",

    "--tp",

    "2",

    "--reserve-mib",

    "3072",

    "--mmap-experts",

    "off",

    "--dspark",

    "on",

    "--prefill-chunk",

    "32768",

    "--threads",

    "on"

  ],

  "cwd": "/opt/project-maya",

  "tokenizer": "/models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL/maya-ds4/tokenizer",

  "backend": "hip",

  "model_name": "deepseek-v4.1-flash",

  "gpu": [

    0,

    1,

    2,

    3

  ],

  "sampling": {

    "temperature": 1.0,

    "top_p": 1.0

  },

  "reasoning_effort": "high",

  "lib_dirs": [

    "/opt/rocm-7.2.0/lib"

  ],

  "port": 8081,

  "env": {

    "DS4_ENGRAM_RAM": "1",

    "DS4_REQUIRE_ALL_RAM": "1",

    "MAYA_DS4_Q8_BATCH": "1",

    "DS4_CHUNK": "32768",

    "DS4_RING_MB": "800"

  },

  "log": "/opt/project-maya/maya-ds4-hip.log",

  "installer": {

    "family": "deepseek",

    "models_dir": "/opt/Maya-data/models",

    "quant": "DeepSeek-V4.1-Flash-1017",

    "ds4_quant": "gguf-q8",

    "model_dir": "/models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL",

    "written": "2026-10-28 17:05",

    "speed_calibration": {

      "date": "2026-10-28 17:05",

      "version": "1.1.30",

      "skipped": "chosen",

      "kept": "dspark"

    }

  }

}

## User capture 18

11/2/2026, 10:18:27 AM

But I want to make an optimized GLM as well.  Can we draft something off of the deepseek defaults

❯ cat maya-ud-q4_k_xl-hip.json

{

 "exe": "/opt/project-maya/build-hip/maya",

 "args": [

  "--glm-pack",

  "/models/GLM-5.3-Flash/UD-Q4_K_XL/pack",

  "--max-context",

  "131072",

  "--expert-cache",

  "auto",

  "--prefill",

  "auto",

  "--kv-streaming",

  "on",

  "--expert-profile",

  "/opt/project-maya/data/expert-profile-glm.bin"

 ],

 "cwd": "/opt/project-maya",

 "tokenizer": "/models/GLM-5.3-Flash/UD-Q4_K_XL/pack/tokenizer",

 "model_name": "glm-5.3-flash",

 "gpu": [

  0,

  1,

  2,

  3

 ],

 "sampling": {

  "temperature": 1.0,

  "top_p": 0.95

 },

 "reasoning_effort": "max",

 "lib_dirs": [

  "/opt/rocm-7.2.0/lib"

 ],

 "port": 8081,

 "backend": "hip",

 "env": {

  "MAYA_GLM_RESERVE_MB": "3072",

  "MAYA_GLM_RAM_HEADROOM_GB": "16",

  "MAYA_GLM_PREFILL_SUB": "1024",

  "MAYA_GLM_PREFILL_MB": "4096"

 },

 "vision": {

  "exe": "/opt/project-maya/build-vision/bin/maya-vision",

  "mmproj": "/models/GLM-5.3-Flash/UD-Q4_K_XL/vision/mmproj-GLM-5.3-Flash-F16.gguf",

  "model": "/models/GLM-5.3-Flash/UD-Q4_K_XL/vision/GLM-5.3-Flash-vocab.gguf",

  "gpu": true

 },

 "log": "/opt/project-maya/maya-ud-q4_k_xl-hip.log",

 "installer": {

  "models_dir": "/opt/Maya-data/models",

  "quant": "UD-Q4_K_XL",

  "gguf": "/models/GLM-5.3-Flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf",

  "gguf_dir": "/models/GLM-5.3-Flash/UD-Q4_K_XL",

  "draft": "mtp",

  "written": "2026-11-02 09:49"

 }

}**#**

## User capture 31

11/2/2026, 3:16:32 PM

Ended up with a crash but I thnk the results are good ehough to extrapolate:

❯ cat 20261102-124651-maya-matrix/results.csv

variant,model,gpus,mode,context,repeat,prompt_n,cache_n,prefill_tps,predicted_n,decode_tps,status,error

deepseek-4gpu-full,deepseek,4,full,32768,1,32800,0,460.9,128,25.0,ok,

deepseek-4gpu-full,deepseek,4,full,32768,2,1,32799,15.6,128,32.6,cached-not-comparable,

deepseek-4gpu-full,deepseek,4,full,32768,3,1,32799,16.3,128,32.3,cached-not-comparable,

deepseek-4gpu-full,deepseek,4,full,65536,1,65568,0,483.4,128,40.4,ok,

deepseek-4gpu-full,deepseek,4,full,65536,2,1,65567,15.2,128,39.9,cached-not-comparable,

deepseek-4gpu-full,deepseek,4,full,65536,3,1,65567,15.2,128,32.8,cached-not-comparable,

deepseek-4gpu-reduced,deepseek,4,reduced,32768,1,32800,0,463.1,128,24.2,ok,

deepseek-4gpu-reduced,deepseek,4,reduced,32768,2,1,32799,14.2,128,25.1,cached-not-comparable,

deepseek-4gpu-reduced,deepseek,4,reduced,32768,3,1,32799,14.5,128,26.8,cached-not-comparable,

deepseek-4gpu-reduced,deepseek,4,reduced,65536,1,65568,0,481.7,128,32.6,ok,

deepseek-4gpu-reduced,deepseek,4,reduced,65536,2,1,65567,13.4,128,33.2,cached-not-comparable,

deepseek-4gpu-reduced,deepseek,4,reduced,65536,3,1,65567,13.6,128,32.7,cached-not-comparable,

deepseek-4gpu-zero,deepseek,4,zero,32768,1,32800,0,461.5,128,31.2,ok,

deepseek-4gpu-zero,deepseek,4,zero,32768,2,1,32799,15.9,128,29.3,cached-not-comparable,

deepseek-4gpu-zero,deepseek,4,zero,32768,3,1,32799,16.1,128,30.6,cached-not-comparable,

deepseek-4gpu-zero,deepseek,4,zero,65536,1,65568,0,483.9,128,34.9,ok,

deepseek-4gpu-zero,deepseek,4,zero,65536,2,1,65567,14.8,128,37.4,cached-not-comparable,

deepseek-4gpu-zero,deepseek,4,zero,65536,3,1,65567,15.0,128,37.5,cached-not-comparable,

deepseek-1gpu-full,deepseek,1,full,32768,1,32800,0,297.7,128,16.5,ok,

deepseek-1gpu-full,deepseek,1,full,32768,2,1,32799,11.8,128,17.8,cached-not-comparable,

deepseek-1gpu-full,deepseek,1,full,32768,3,1,32799,11.6,128,17.9,cached-not-comparable,

deepseek-1gpu-full,deepseek,1,full,65536,1,65568,0,316.2,128,20.3,ok,

deepseek-1gpu-full,deepseek,1,full,65536,2,1,65567,11.2,128,20.0,cached-not-comparable,

deepseek-1gpu-full,deepseek,1,full,65536,3,1,65567,11.4,128,20.1,cached-not-comparable,

deepseek-1gpu-reduced,deepseek,1,reduced,32768,1,32800,0,297.7,128,17.0,ok,

deepseek-1gpu-reduced,deepseek,1,reduced,32768,2,1,32799,11.8,128,17.3,cached-not-comparable,

deepseek-1gpu-reduced,deepseek,1,reduced,32768,3,1,32799,11.8,128,17.3,cached-not-comparable,

deepseek-1gpu-reduced,deepseek,1,reduced,65536,1,65568,0,316.4,128,19.7,ok,

deepseek-1gpu-reduced,deepseek,1,reduced,65536,2,1,65567,11.3,128,20.5,cached-not-comparable,

deepseek-1gpu-reduced,deepseek,1,reduced,65536,3,1,65567,11.3,128,20.5,cached-not-comparable,

deepseek-1gpu-zero,deepseek,1,zero,32768,1,32800,0,297.7,128,17.1,ok,

deepseek-1gpu-zero,deepseek,1,zero,32768,2,1,32799,11.6,128,17.1,cached-not-comparable,

deepseek-1gpu-zero,deepseek,1,zero,32768,3,1,32799,11.8,128,17.1,cached-not-comparable,

deepseek-1gpu-zero,deepseek,1,zero,65536,1,65568,0,316.3,128,19.5,ok,

deepseek-1gpu-zero,deepseek,1,zero,65536,2,1,65567,11.2,128,19.2,cached-not-comparable,

deepseek-1gpu-zero,deepseek,1,zero,65536,3,1,65567,11.3,128,21.4,cached-not-comparable,

glm-4gpu-full,glm,4,full,32768,1,32017,0,348.7,128,28.9,ok,

glm-4gpu-full,glm,4,full,32768,2,1,32016,19.6,128,32.1,cached-not-comparable,

glm-4gpu-full,glm,4,full,32768,3,1,32016,19.6,128,32.9,cached-not-comparable,

glm-4gpu-full,glm,4,full,65536,1,63733,0,351.2,128,31.4,ok,

glm-4gpu-full,glm,4,full,65536,2,1,63732,19.3,128,33.3,cached-not-comparable,

glm-4gpu-full,glm,4,full,65536,3,1,63732,19.0,128,32.4,cached-not-comparable,

glm-4gpu-reduced,glm,4,reduced,32768,1,32017,0,355.2,128,25.2,ok,

glm-4gpu-reduced,glm,4,reduced,32768,2,1,32016,18.1,128,31.2,cached-not-comparable,

glm-4gpu-reduced,glm,4,reduced,32768,3,1,32016,18.2,128,31.5,cached-not-comparable,

glm-4gpu-reduced,glm,4,reduced,65536,1,63733,0,356.6,128,27.0,ok,

glm-4gpu-reduced,glm,4,reduced,65536,2,1,63732,18.1,128,30.5,cached-not-comparable,

glm-4gpu-reduced,glm,4,reduced,65536,3,1,63732,18.8,128,29.9,cached-not-comparable,

glm-1gpu-full,glm,1,full,32768,1,32017,0,175.2,128,10.2,ok,

glm-1gpu-full,glm,1,full,32768,2,1,32016,9.7,128,10.7,cached-not-comparable,

glm-1gpu-full,glm,1,full,32768,3,1,32016,7.5,128,10.4,cached-not-comparable,

glm-1gpu-full,glm,1,full,65536,1,63733,0,169.6,128,14.0,ok,

glm-1gpu-full,glm,1,full,65536,2,1,63732,10.7,128,14.2,cached-not-comparable,

glm-1gpu-full,glm,1,full,65536,3,1,63732,11.5,128,13.8,cached-not-comparable,

## User capture 36

11/2/2026, 3:55:10 PM

1. [Releases](https://github.com/mw00/project-maya/releases)
2. [v1.1.50](https://github.com/mw00/project-maya/releases/tag/v1.1.50)

# v1.1.50

[**Latest**](https://github.com/mw00/project-maya/releases/latest)

[@mw00](https://avatars.githubusercontent.com/u/14845178?s=40\&v=4) [**mw00**](https://github.com/mw00) released this 4 hours ago

[Tag v1.1.50](https://github.com/mw00/project-maya/tree/v1.1.50)

[Commit ](https://github.com/mw00/project-maya/commit/10e69a0eb11541004abb24f9fa47147e9002326f)[`10e69a0`](https://github.com/mw00/project-maya/commit/10e69a0eb11541004abb24f9fa47147e9002326f)

Models you can download, swap and delete from the Models page or the API, a Turbo button, a thinking budget you can set, GLM Maya-XS, faster long prompts on DeepSeek, and a round of community pull requests.

- **Models page:** the catalog lists every Maya model with its size. Download one with a progress bar, load it, unload the running one, or delete one (a file that another model's folder links to stays). Nothing needs a command line.
- **Model API:** `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` and `/api/delete` list, load, download and delete models from any app or script. A request for a model that is not loaded swaps to it, and `keep_alive` unloads it when idle. `/api/ps` shows "loading" while the weights load.
- **Turbo:** one button beside the model's parameters sets the fastest settings for the model (greedy sampling, no thinking, 32K context on GLM and MiniMax, expert skipping, the fastest drafting). It says "Turbo ON" with a flame while it is on, and switching it off restores your settings exactly. It trades some quality for speed; on the smallest quants (GLM and DeepSeek Maya-XS) it keeps every expert, so their answers stay sound.
- **Thinking budget:** a field in the settings sets how many tokens the model may think before it answers, and applies at once. The default stays 32768; GLM Maya-XS defaults to 8192.
- **GLM-5.3-Flash Maya-XS:** a new, smaller GLM quant (about 73 GB) for PCs with 64 GB of RAM and a 24 GB card at 128K context. The setup offers it as `glm-maya-xs`, with the int8 KV cache, a thinking budget of 8192 and sampling of temperature 0.7, top-p 0.95, min-p 0.05.
- **DeepSeek long prompts:** the prompt chunk now follows how many experts live in RAM. Where RAM holds most of them, a long prompt reads up to about 19% faster; where the VRAM holds them, nothing changes.
- **DeepSeek thinking budget 0:** the reasoning closes again, so answers and tool calls no longer land in `reasoning_content` ([#786](https://github.com/mw00/project-maya/issues/786)).
- **Maya-S n-gram tables:** Maya-S reuses Maya-XS's tables instead of downloading a second pair ([#776](https://github.com/mw00/project-maya/issues/776)).
- **Log timestamps:** every `[maya]` log line starts with the time, in the terminal and in the log file ([#783](https://github.com/mw00/project-maya/issues/783)).
- **Gentoo:** the HIP check finds versioned LLVM folders ([#787](https://github.com/mw00/project-maya/pull/787), thanks [**@Wilsonfabozi**](https://github.com/Wilsonfabozi)).
- **Community pull requests:**
  - `/tokenize` and `/detokenize` endpoints ([#764](https://github.com/mw00/project-maya/pull/764), thanks [**@cresslank**](https://github.com/cresslank)).
  - A chat request's `response_format` reaches the model ([#763](https://github.com/mw00/project-maya/pull/763), thanks [**@cresslank**](https://github.com/cresslank)).
  - `ignore_eos`, and `logit_bias` on DeepSeek ([#772](https://github.com/mw00/project-maya/pull/772), thanks [**@cresslank**](https://github.com/cresslank)).
  - GLM prompts: the CPU lane takes a share where PCIe is not much slower than the CPU, with `MAYA_GLM_PREFILL_CPU_GATE` ([#752](https://github.com/mw00/project-maya/pull/752), thanks [**@hugou74130**](https://github.com/hugou74130)).
  - DeepSeek on AMD keeps the router's weights in BF16 ([#745](https://github.com/mw00/project-maya/pull/745), thanks [**@mrsirg97-rgb**](https://github.com/mrsirg97-rgb)).
  - DeepSeek flow prompt chunks: an opt-in DMA/host all-reduce ([#736](https://github.com/mw00/project-maya/pull/736), thanks [**@hajekt2**](https://github.com/hajekt2)).
  - Metal: no free-block exhaustion on long-context prefill ([#742](https://github.com/mw00/project-maya/pull/742), thanks [**@appleguy**](https://github.com/appleguy)).
  - Opt-in, off by default: peer-to-peer all-reduce over ReBAR for DeepSeek on 2+ GPUs, up to 29% faster prompts on the PCs it was measured on (`DS4_AR_P2P=1`, [#762](https://github.com/mw00/project-maya/pull/762), thanks [**@486DX4-100**](https://github.com/486DX4-100)). It was a little slower on V100s, so Maya does not turn it on.
  - Speed table and AMD docs ([#756](https://github.com/mw00/project-maya/pull/756) thanks [**@Cayaf**](https://github.com/Cayaf), [#749](https://github.com/mw00/project-maya/pull/749) thanks [**@Diudid**](https://github.com/Diudid)).

Maya is free; if it helps you, you can sponsor it: [https://github.com/sponsors/mw00|](https://github.com/sponsors/mw00|)

Interesting options here

## User capture 40

11/5/2026, 8:54:59 PM

# v1.1.55

[**Latest**](https://github.com/mw00/project-maya/releases/latest)

[@mw00](https://avatars.githubusercontent.com/u/14845178?s=40\&v=4) [**mw00**](https://github.com/mw00) released this 8 hours ago

[Tag v1.1.55](https://github.com/mw00/project-maya/tree/v1.1.55)

[Commit ](https://github.com/mw00/project-maya/commit/c4effee67cb95fb4303720919b075fddf04b4e69)[`c4effee`](https://github.com/mw00/project-maya/commit/c4effee67cb95fb4303720919b075fddf04b4e69)

Codex CLI works with Maya, web-chat answers always run to their end and survive a restart, and a few speed settings now on by default.

- **Codex CLI ([#726](https://github.com/mw00/project-maya/issues/726)):** Maya's Responses API (`/v1/responses`) now accepts what Codex CLI sends: its grouped tools (`namespace`, shown to the model as `namespace::name` and returned with their namespace), built-in tools Maya cannot run (`web_search`, `local_shell`, ...) left out instead of failing the request, `reasoning.summary`, and `include: ["reasoning.encrypted_content"]` - the reasoning comes back as an opaque blob that Codex sends with the next turn. Tested end to end with Codex CLI 0.160.0 on GLM-5.3-Flash, DeepSeek-V4.1-Flash and Qwen-3.8-Flash-Next (a shell command run and a file written correctly on each).
- **Web chat: an answer runs to its end unless you press Stop.** Before, the server stopped an answer that no page had read for 5 minutes - a phone whose screen locked during a long thinking answer lost it. Now it always finishes, and the page takes the whole answer up whenever it is opened again; the page also reconnects once a waking phone's Wi-Fi is back. Finished answers are kept for a day (was 15 minutes) and saved in `answers/` next to the config, so a server restart keeps them too; one that a restart cut off comes back with what it had written and a note saying so.
- **Speed settings now on by default** (measured with this release against v1.1.54 on the same PC, quant and prompts):
  - **DeepSeek, prompt reading (prefill):** the indexer's buffers are sized by the prompt instead of the whole context (`DS4_AI_RUN`), and on NVIDIA a two-GPU group's prompt sums run in one pipelined launch, bit for bit the same (`MAYA_DS4_AR2=1`). On 3x V100 with the EXL3 2.9 bpw quant and `--tp 2`: **+1.3%** at 2K tokens, **+3%** at 8K, **+4%** at 32K. Decode is unchanged.
  - **DeepSeek, loading:** the model's files are read in turn instead of one after another (`MAYA_DS4_LOAD_INTERLEAVE`): the EXL3 2.9 bpw quant loads **31% faster** on a Gorgon Halo; the same on 3x V100.
  - **Every model, long agent chats:** the server prepares a re-sent conversation by increment (`MAYA_PROMPT_MEMO`, the same token ids): a follow-up turn after a 115K-token history takes about **6% less time** (measured on GLM-5.3-Flash, 3x V100).
  - **GLM, memory:** the attention's cache in 8 bits by default (`MAYA_GLM_KV_INT8`, as the setup already set it for new installs): half the cache's VRAM (one V100 at 128K context: 1.78 -> 1.13 GB, room for 84 more experts). Checked against the full model: the same quality as 16 bits (within its own run-to-run spread) and no looping. `--kv fp16` keeps 16 bits.
  - `0` turns each one off, in the config's `"env"` block or with `--env`.
- **Docs:** `MAYA_GLM_PREFILL_FUSED` and `MAYA_GLM_MV_RDNA` are listed as on for RDNA4 (gfx12) too, as the code has them.

Supported by [**@vikcious**](https://github.com/vikcious).

Maya is free; if it helps you, you can sponsor it: [https://github.com/sponsors/mw00](https://github.com/sponsors/mw00)

## User capture 43

11/10/2026, 8:35:13 AM

❯ git fetch origin --tags

❯ git remote -v

origin  [https://github.com/mw00/project-maya.git](https://github.com/mw00/project-maya.git) (fetch)

origin  [https://github.com/mw00/project-maya.git](https://github.com/mw00/project-maya.git) (push)

❯ git config --get remote.origin.tagOpt

❯ git rev-parse --is-shallow-repository

false

❯ git fetch --unshallow --tags origin

fatal: --unshallow on a complete repository does not make sense

❯ git fetch --tags origin

    /**opt**/**project-maya**   #v1.1.57                                                                                                                  unsloth-updater   root@openwebui  08:35:01  

❯
