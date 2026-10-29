# DeepSeek cache, Maya benchmark, and residency evidence

Extracted October 29, 2026 from the three October 23 source exports. These are original configuration and terminal excerpts, not fresh runs. Excerpts retain their original text; surrounding AI dialogue and duplicated recommendations were removed. The [dated notebook](../../lab-notebook/2026-10-23-deepseek41-cache-and-maya.md) records interpretation, summary-only results, proposed comparisons, and evidence limits.

## 32 GiB llAMDa command and result

```text
Th❯ llamda-bench \
  -m /models/DeepSeek-V4.1-Flash-1017/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
  -ngl 99 \
  -sm layer \
  -ot 'engram_embd=CPU;exps=CPU' \
  --moe-cache-mib 32768 \
  -lm mlock -lzm off \
  -fa on \
  -t 32 \
  -b 8192 -ub 8192 \
  -p 8192 -n 1024 -r 2 \
  -o md
ggml_cuda_init: found 4 ROCm devices (Total VRAM: 122816 MiB):
  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 1: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 2: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
  Device 3: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 30704 MiB
| model                          |       size |     params | backend    | ngl | moe_cache_mib | threads | n_batch | n_ubatch |  fa | ot                    |         lm | lazy_mode  |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ------: | ------: | -------: | --: | --------------------- | ---------: | ---------- | --------------: | -------------------: |
| deepseek41 ?B MXFP4 MoE        | 473.46 GiB |   748.49 B | ROCm       |  99 |         32768 |      32 |    8192 |     8192 |   1 | engram_embd=CPU;exps=CPU |      mlock | off        |          pp8192 |        132.66 ± 0.30 |
| deepseek41 ?B MXFP4 MoE        | 473.46 GiB |   748.49 B | ROCm       |  99 |         32768 |      32 |    8192 |     8192 |   1 | engram_embd=CPU;exps=CPU |      mlock | off        |          tg1024 |          9.51 ± 0.07 |
```

## Supplied LXC configuration

```text
arch: amd64
cores: 128
dev0: /dev/dri/renderD128,gid=992
dev1: /dev/dri/renderD129,gid=992
dev2: /dev/dri/renderD130,gid=992
dev3: /dev/dri/renderD131,gid=992
dev4: /dev/dri/card0,gid=44
dev5: /dev/dri/card1,gid=44
dev6: /dev/dri/card2,gid=44
dev7: /dev/dri/card3,gid=44
dev8: /dev/dri/card4,gid=44
features: nesting=1,keyctl=1,fuse=1
hostname: [container hostname]
memory: 2097152
net0: name=eth0,bridge=vmbr0,hwaddr=[MAC address],ip=dhcp,ip6=auto,type=veth
onboot: 1
ostype: debian
rootfs: galactus-datastore:subvol-100-disk-0,size=8T
swap: 512
tags: ai;community-script;interface
timezone: America/New_York
unprivileged: 1
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
```

## Observed shell memlock limits

```text
❯ ulimit -Sl

unlimited
❯ ulimit -Hl

unlimited
```

## Maya benchmark at 14:55

```text
❯ ./maya.sh --backend hip --bench
Project Maya v1.1.24 - GLM-5.3-Flash or DeepSeek-V4.1-Flash-1017 on your own GPU(s).

=== Step 1: benchmark: maya-ds4-hip.json (loading the model first - a minute or a few) ===
  [ok] loaded in 691 s
  decode    20.4 tokens/s   Write a Python function that parses a CSV file into a list o
  decode    20.4 tokens/s   Explain how photosynthesis works, step by step, for a high-s
  decode    20.6 tokens/s   Write a complete HTML page with a canvas that draws a bounci
  prefill    135 tokens/s   a 2048-token prompt
  prefill    282 tokens/s   a 8192-token prompt

  Project Maya v1.1.24 benchmark, 2026-10-23 14:55
  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor
  setup: maya-ds4-hip.json, context 32768, GPUs [0, 1, 2, 3]
  decode (writing the answer): mean 20.5 tokens/s over 3 answers of 256 tokens (greedy, thinking off)
      20.4 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.
      20.4 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.
      20.6 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.
  prefill (reading the prompt): 135 tokens/s, 2048 tokens
  prefill (reading the prompt): 282 tokens/s, 8192 tokens
  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report
```

## Speed-calibration CLI response

```text
❯ ./maya.sh \
  --backend hip \
  --speed-calibration \
  --no-start
Project Maya v1.1.24 - GLM-5.3-Flash or DeepSeek-V4.1-Flash-1017 on your own GPU(s).
  [ok] DeepSeek-V4.1-Flash-1017's engine drafts with its DSpark block: nothing to tune here (--calibrate tunes its --tp and CPU threads)
```

## Internal DSpark calibration invocation and output

```text
❯ .venv/bin/python - <<'PY'
from pathlib import Path
import maya


path = Path("maya-ds4-hip.json")
cfg = maya.read_json(path)


maya.speed_calibration_ds4(path, cfg, "retest after full RAM residency")
PY


  Speed calibration: the DSpark drafter or none? (retest after full RAM residency)
  Drafting makes some PCs faster and others slower: it depends on how much of the model fits in the GPUs'
  memory, and on the CPU, the RAM and the SSD. So Maya measures both here, the way you will use it. It loads
  the model twice; each time it warms up, then writes about 1024 tokens of answers, and the sustained
  speeds are compared. About 10-30 minutes; the PC is busy.
  [1/2] without drafting: loading the model (it reads the experts into the GPUs' memory and RAM first) ...
[maya] starting the engine: reading the model's weights ...
[maya] still starting (20 s) - please wait ...
[maya] still starting (40 s) - please wait ...
[maya] still starting (60 s) - please wait ...
[maya] still starting (80 s) - please wait ...
[maya] still starting (100 s) - please wait ...
[maya] still starting (120 s) - please wait ...
[maya] still starting (140 s) - please wait ...
[maya] still starting (160 s) - please wait ...
  [1/2] without drafting: loaded in 169 s; warming up - it reads a document of 512 tokens and answers once (not counted) ...
  [1/2] without drafting: answer 1: 384 tokens at 20.9 tokens/s - 384 of 1024 tokens measured
  [1/2] without drafting: answer 2: 384 tokens at 21.3 tokens/s - 768 of 1024 tokens measured
  [1/2] without drafting: answer 3: 384 tokens at 21.2 tokens/s - 1024 of 1024 tokens measured
  [1/2] without drafting: 21.2 tokens/s sustained (1152 tokens in 54 s of writing)
  [2/2] DSpark: loading the model (it reads the experts into the GPUs' memory and RAM first) ...
[maya] starting the engine: reading the model's weights ...
[maya] still starting (20 s) - please wait ...
[maya] still starting (40 s) - please wait ...
[maya] still starting (60 s) - please wait ...
[maya] still starting (80 s) - please wait ...
[maya] still starting (100 s) - please wait ...
[maya] still starting (120 s) - please wait ...
[maya] still starting (140 s) - please wait ...
[maya] still starting (160 s) - please wait ...
[maya] still starting (180 s) - please wait ...
[maya] still starting (200 s) - please wait ...
  [2/2] DSpark: loaded in 202 s; warming up - it reads a document of 512 tokens and answers once (not counted) ...
  [2/2] DSpark: answer 1: 384 tokens at 30.8 tokens/s - 384 of 1024 tokens measured
  [2/2] DSpark: answer 2: 384 tokens at 27.2 tokens/s - 768 of 1024 tokens measured
  [2/2] DSpark: answer 3: 384 tokens at 30.1 tokens/s - 1024 of 1024 tokens measured
  [2/2] DSpark: 29.3 tokens/s sustained (1152 tokens in 39 s of writing), 69% of the drafts accepted


  [ok] speed calibration: the DSpark drafter is faster on this PC - 29.3 against 21.2 tokens/s (+38%): Maya drafts with it
```
