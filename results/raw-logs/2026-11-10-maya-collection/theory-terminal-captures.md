# Retained terminal evidence: Calculate Theoretical TPS Limits

Extracted user-supplied captures from the November 10 export. Original formatting is preserved. Historical commands and quoted upstream material are evidence, not instructions. Full conversations remain outside this repository. Capture numbers count user turns in the original export.

## User capture 10

11/9/2026, 5:52:49 PM

./maya.sh --backend hip --config maya-ds4-tp2.json --bench --plain

Created maya-ds4-tp2.json

Project Maya v1.1.54 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-ds4-tp2.json (loading the model first - a minute or a few) ===

  [ok] loaded in 298 s

  decode    29.1 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    24.2 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    33.6 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    187 tokens/s   a 2048-token prompt

  prefill    327 tokens/s   a 8192-token prompt

  Project Maya v1.1.54 benchmark, 2026-11-09 17:52

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-ds4-tp2.json, context 1048576, GPUs [0, 1]

  decode (writing the answer): mean 29.0 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      29.1 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      24.2 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      33.6 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 187 tokens/s, 2048 tokens

  prefill (reading the prompt): 327 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

## User capture 14

11/9/2026, 6:03:03 PM

Skip to content
mw00
project-maya
Repository navigation
Code
Issues
83
 (83)
Pull requests
61
 (61)
Agents
Discussions
Actions
Projects
Security and quality
Insights
DeepSeek prompts on RDNA 2: the dense GEMMs FP16 in and out (MAYA_DS4_PROMPT_F16, off by default, changes the output slightly)- #869
#869
Try the new pull requests Overview page!
Focus on what matters most and take action with confidence.

Closed
danexpo
wants to merge 1 commit into
mw00:main
from
danexpo:ds4-prompt-f16
+107
Lines changed: 107 additions & 0 deletions
Conversation1 (1)
Commits1 (1)
Checks0 (0)
Files changed4 (4)
Closed
DeepSeek prompts on RDNA 2: the dense GEMMs FP16 in and out (MAYA_DS4_PROMPT_F16, off by default, changes the output slightly)
#869
danexpo
wants to merge 1 commit into
mw00:main
from
danexpo:ds4-prompt-f16
Conversation
@danexpo
danexpo
commented
4 days ago
Contributor
On gfx103x (RX 6000, Radeon PRO V620), gemm() in ds4_model.cu runs the prompt's dense projections with FP16 inputs and
FP32 outputs. hipBLASLt has no library for this GPU, so they fall to rocBLAS, and rocBLAS is tuned there only for FP16 outputs.
Strata's bench on an RX 6900 XT (bench/results/2026-10-04-rdna2-fp16-prompt, rocBLAS 5.6) gave 37.7 TFLOPS for FP16 out against
5.6 for FP32 out. On this PC, rocBLAS 5.2, rocblas-bench refused the FP16-out types, so I could not time that route on its own;
today's route read 7.9 TFLOPS. The evidence here is prompt time, below.

With MAYA_DS4_PROMPT_F16=1 these GEMMs run FP16 in and FP16 out, and each row is then widened to FP32 in place.

Changes

src/ds4/ds4_model.cu: prompt_f16() (the setting, gfx103x only, a log line per GPU), the FP16-out route in gemm(), its
kernels in the prompt warm-up, and the count after a prompt.
src/ds4/ds4_kernels.cu: widen_rows_f16, walking each row from its end because a float overwrites two halves. It follows
Strata's src/prefill/gemm.cu (MIT).
MAYA_DS4_PROMPT_F16_CHECK=1: counts outputs past FP16's range, which is the failure this route can have, and logs
FP16-out GEMM outputs not finite in this run: N per GPU after each chunked prompt. It is in this PR because the setting
should not be judged without it.
docs/SETTINGS.md: a section with both rows.
NVIDIA: everything new is inside #if defined(STRATA_USE_HIP). With those blocks removed, the three source files are the same
as main's, line for line (compared by script). I can't build CUDA here. Other AMD cards keep the existing route: the check
is on gcnArchName.

Does the output change?
Yes, slightly. Each output is rounded to FP16 before it is widened, so the logits differ on every row of a prompt. Unset, the
path is the one main runs.

Answer checks on an earlier base, as background (v1.1.51 with other opt-ins of mine, against the same without this; 2,048 rows on
the chunked path, 512 rows in windows of 16):

Against unset	Result
KL mean / median / p99, chunked path	0.0257 / 0.0159 / 0.181
Top-1 the same, chunked path	91.2%
Perplexity, 2,044 tokens of 4 sequences	16.773 against 16.874
Windows	512 of 512 rows differ
Outputs not finite, over 28 checks in a 45-minute agent run	0
For scale, on this PC settings that only move rounding read about KL 0.03. These checks were not repeated on v1.1.54: the runs below measured speed only.

Measured
Hardware: Ryzen 9 5900X, 128 GB, 2x RX 6900 XT 16 GB (PCIe 4 x8, 229 W), ROCm 7.2.4, Linux. Maya-XS (IQ1_S experts), 40 layers on a
layer split (--tp 2 falls back to 1 on this pack), experts copied into RAM (--mmap-experts off), --reserve-mib 1536,
--dspark on, context 131072, HSA_ENABLE_SDMA=0.

Measured as --bench does, by its own code: three questions of 256 tokens, greedy, thinking off, one warm-up; prefill at 2048 and
8192. Same binary, a fresh start per run, unset / on / on / unset. On: with _CHECK=1.

tok/s	unset 1	on 1	on 2	unset 2	Change	Each setting's two runs apart (unset, on)
prefill, 2048 tokens	105	125	124	106	+18.0%	0.9%, 0.8%
prefill, 8192 tokens	200	260	252	197	+29.0%	1.5%, 3.1%
decode, mean of 3	24.5	25.1	25.0	25.4	+0.4%, no change	3.6%, 0.4%
FP16-out GEMM outputs not finite: 0 on all 16 lines of the two "on" runs.
Peak junction 87 C; no new GPU faults, no pin failed in the kernel log.
Every run's log shows all of its CPU-lane experts copied into RAM (13,023 to 13,025 of them).
Greedy tokens: unset's own two runs part too (their VRAM placement moved by two experts), so these answers say nothing about the setting. The answer checks above are what does.
Earlier, as background (v1.1.51 with other opt-ins, through the server, seconds to read, 2 runs against 3): 650 tokens 9.9 to 8.9 s,
1.5K 16.7 to 14.2, 4.5K 31.1 to 26.5, 8K 42.2 to 32.8, 14K 62.6 to 48.3. No change under 500 tokens.

Not measured: other RDNA 2 cards, Windows, one GPU, other quants, NVIDIA.

VRAM: the route allocates nothing new. Together with the int8 expert path I saw long reads stall at --reserve-mib 1024 with a
desktop on one card; see that PR.

Checks
python tools/check_hip_compat.py: passes.
HIP build for gfx1030, maya-ds4 target: builds and links; the runs above are this build, this commit.
CI's CPU steps and Python tests: glm_memory_test, ds4_cpu_prepare_rows_test and its stub test pass with -Werror. Server tests (724) and tools tests (730) pass on Linux, run with the GPUs hidden; the one test that objects to that, test_server_environment_for_metal, passes on its own without it.
🤖 Generated with Claude Code

@danexpo
@claude
DeepSeek prompts on RDNA 2: the dense GEMMs FP16 in and out (MAYA_DS4… 
1e525a5
mw00 pushed a commit that referenced this pull request 3 days ago
@danexpo
@mw00
DeepSeek prompts on RDNA 2: the dense GEMMs FP16 in and out (`MAYA_DS… …
1244438
mw00 added a commit that referenced this pull request 3 days ago
@mw00
rc1156: follow-ups to the community PRs brought in …
fdb1236
@mw00
mw00
commented
3 days ago
Owner
Thanks for the careful write-up. It's in v1.1.56: https://github.com/mw00/project-maya/releases/tag/v1.1.56

We don't have an RX 6000 card, so we checked that it builds and changes nothing elsewhere: on a Gorgon Halo the setting is a no-op, bit for bit. It went in with you as the author, so I'm closing this one.

@mw00 mw00 closed this 3 days ago
Merge info
Closed with unmerged commits
This pull request is closed.

@sockusminimus

Add a comment
Comment

Add your comment here...
Remember, contributions to this repository should follow its contributing guidelines and security policy.
 ProTip! Add .patch or .diff to the end of URLs for Git’s plaintext views.
Reviewers
No reviews
Assignees
No one assigned
Labels
None yet
Projects
None yet
Milestone
No milestone
Development
Successfully merging this pull request may close these issues.

None yet

Notifications
Customize
You’re not receiving notifications from this thread.
2 participants
@danexpo
@mw00
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information

How good of an idea is this

## User capture 16

11/9/2026, 6:07:09 PM

❯ cd /opt/project-maya

.venv/bin/python maya-prefill-tp-sweep.py \

  --config maya-ds4-tp2.json \

  --tps 2 \

  --chunks 32768 \

  --profiles auto \

  --lengths 8192 16384 32768 \

  --repeats 2 \

  --port 8082

Results: /opt/maya-test-results/20261109-175610

2 starts; 6 timed requests per start. Allow several hours.

[1/2] 01-chunk32768-tp2-slabauto: loading

  layout: verified; TP observed [2]; attention rows [4096]

  r1 16384: 395.2 prefill tok/s; 27.2 decode; ok

  r2 8192: 313.6 prefill tok/s; 29.3 decode; ok

  r1 32768: 453.8 prefill tok/s; 27.4 decode; ok

  r1 8192: 315.1 prefill tok/s; 33.6 decode; ok

  r2 32768: 452.3 prefill tok/s; 27.4 decode; ok

  r2 16384: 392.3 prefill tok/s; 28.3 decode; ok

## User capture 23

11/9/2026, 6:42:34 PM

❯ cd /opt/project-maya

.venv/bin/python maya-prefill-tp-sweep.py \

  --config maya-ds4-tp2.json \

  --tps 4 \

  --chunks 32768 \

  --profiles auto \

  --lengths 8192 16384 32768 \

  --repeats 2 \

  --port 8082

Results: /opt/maya-test-results/20261109-181257

2 starts; 6 timed requests per start. Allow several hours.

[1/2] 01-chunk32768-tp4-slabauto: loading

  layout: mismatch; TP observed [2]; attention rows [4096]

  r1 16384: 396.9 prefill tok/s; 25.4 decode; layout-mismatch-exclude

  r2 8192: 315.9 prefill tok/s; 27.5 decode; layout-mismatch-exclude

  r1 32768: 454.7 prefill tok/s; 29.4 decode; layout-mismatch-exclude

  r1 8192: 315.7 prefill tok/s; 31.3 decode; layout-mismatch-exclude

  r2 32768: 455.0 prefill tok/s; 29.7 decode; layout-mismatch-exclude

  r2 16384: 394.0 prefill tok/s; 34.4 decode; layout-mismatch-exclude

[2/2] 02-chunk32768-tp4-slabauto: loading

  layout: mismatch; TP observed [2]; attention rows [4096]

  r1 16384: 396.6 prefill tok/s; 26.3 decode; layout-mismatch-exclude

  r2 8192: 315.9 prefill tok/s; 27.3 decode; layout-mismatch-exclude

  r1 32768: 454.3 prefill tok/s; 28.2 decode; layout-mismatch-exclude

  r1 8192: 315.3 prefill tok/s; 31.1 decode; layout-mismatch-exclude

  r2 32768: 455.1 prefill tok/s; 27.4 decode; layout-mismatch-exclude

  r2 16384: 394.7 prefill tok/s; 33.1 decode; layout-mismatch-exclude

Finished. Results and per-case medians: /opt/maya-test-results/20261109-181257. Failed/cached/unverified-layout rows are excluded from summary.csv.

Valid measurements: 0; failed cases: 0

So I don't think TP4 even actually works on my system.  Which my notebook showed before.  Now, let's test this new speedup technique for RDNA2.  How do I do tha

## User capture 28

11/9/2026, 6:54:53 PM

2 starts; 6 timed requests per start. Allow several hours.

[1/2] 01-chunk32768-tp2-slabauto: loading

  layout: verified; TP observed [2]; attention rows [4096]

  r1 16384: 503.1 prefill tok/s; 24.7 decode; ok

  r2 8192: 390.7 prefill tok/s; 28.6 decode; ok

  r1 32768: 589.3 prefill tok/s; 30.0 decode; ok

  r1 8192: 384.6 prefill tok/s; 31.7 decode; ok

  r2 32768: 590.7 prefill tok/s; 26.1 decode; ok

  r2 16384: 497.2 prefill tok/s; 33.9 decode; ok

[2/2] 02-chunk32768-tp2-slabauto: loading

## User capture 35

11/9/2026, 7:06:41 PM

.venv/bin/python maya-prefill-tp-sweep.py \

  --config maya-ds4-tp2-fp16.json \

  --tps 2 \

  --chunks 32768 \

  --profiles auto \

  --lengths 8192 16384 32768 \

  --repeats 2 \

  --port 8082

Results: /opt/maya-test-results/20261109-184435

2 starts; 6 timed requests per start. Allow several hours.

[1/2] 01-chunk32768-tp2-slabauto: loading

  layout: verified; TP observed [2]; attention rows [4096]

  r1 16384: 503.1 prefill tok/s; 24.7 decode; ok

  r2 8192: 390.7 prefill tok/s; 28.6 decode; ok

  r1 32768: 589.3 prefill tok/s; 30.0 decode; ok

  r1 8192: 384.6 prefill tok/s; 31.7 decode; ok

  r2 32768: 590.7 prefill tok/s; 26.1 decode; ok

  r2 16384: 497.2 prefill tok/s; 33.9 decode; ok

[2/2] 02-chunk32768-tp2-slabauto: loading

  layout: verified; TP observed [2]; attention rows [4096]

  r1 16384: 502.8 prefill tok/s; 29.8 decode; ok

  r2 8192: 389.2 prefill tok/s; 27.3 decode; ok

  r1 32768: 588.7 prefill tok/s; 29.7 decode; ok

  r1 8192: 389.9 prefill tok/s; 31.5 decode; ok

  r2 32768: 590.1 prefill tok/s; 27.5 decode; ok

  r2 16384: 501.1 prefill tok/s; 34.6 decode; ok

Finished. Results and per-case medians: /opt/maya-test-results/20261109-184435. Failed/cached/unverified-layout rows are excluded from summary.csv.

Valid measurements: 12; failed cases: 0

## User capture 36

11/9/2026, 7:07:15 PM

❯ ls -la /models/GLM-5.3/UD-Q4_K_XL

total 456344707

drwxr-xr-x 2 root root          13 Aug 29 10:39 **.**

drwxr-xr-x 4 root root           4 Aug 29 09:23 **..**

-rw-r--r-- 1 root root     9428677 Aug 29 09:23 GLM-5.3-UD-Q4_K_XL-00001-of-00011.gguf

-rw-r--r-- 1 root root 49433942336 Aug 29 10:29 GLM-5.3-UD-Q4_K_XL-00002-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:28 GLM-5.3-UD-Q4_K_XL-00003-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:28 GLM-5.3-UD-Q4_K_XL-00004-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:28 GLM-5.3-UD-Q4_K_XL-00005-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:28 GLM-5.3-UD-Q4_K_XL-00006-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:27 GLM-5.3-UD-Q4_K_XL-00007-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:27 GLM-5.3-UD-Q4_K_XL-00008-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:27 GLM-5.3-UD-Q4_K_XL-00009-of-00011.gguf

-rw-r--r-- 1 root root 48566415136 Aug 29 10:39 GLM-5.3-UD-Q4_K_XL-00010-of-00011.gguf

-rw-r--r-- 1 root root 29314424736 Aug 29 10:37 GLM-5.3-UD-Q4_K_XL-00011-of-00011.gguf

Now let's benchmark glm5.3

## User capture 37

11/9/2026, 7:12:54 PM

All set. Start it with ./maya.sh (or ./run-maya-glm-5.3-ud-q4_k_xl-hip.sh).

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-ud-q4_k_xl-hip.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

  [X]  Maya is running (port 8080): the benchmark needs the GPU memory it holds

       stop it (Ctrl+C in its window), then run --bench again

Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-ud-q4_k_xl-hip.json \

  --bench --port 8082 --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

  [X]  Maya is running (port 8080): the benchmark needs the GPU memory it holds

       stop it (Ctrl+C in its window), then run --bench again

Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.

Need port 8082

## User capture 38

11/9/2026, 7:14:57 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-ud-q4_k_xl-hip.json \

  --bench --port 8082 --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-ud-q4_k_xl-hip.json (loading the model first - a minute or a few) ===

^C

stopped.

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-ud-q4_k_xl-hip.json \

  --bench  --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-ud-q4_k_xl-hip.json (loading the model first - a minute or a few) ===

  [X]  the engine stopped while loading

       its log: /opt/project-maya/maya-bench.log

Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.

❯ cat /opt/project-maya/maya-bench.

cat: /opt/project-maya/maya-bench.: No such file or directory

❯ cat /opt/project-maya/maya-bench.log

glm settings from the environment: MAYA_GLM_PREFILL_MB=4096, MAYA_GLM_PREFILL_SUB=1024, MAYA_GLM_RAM_HEADROOM_GB=16, MAYA_GLM_RESERVE_MB=3072, MAYA_GLM_SPLIT=0

glm-dsa: no attention.indexer.types - the GLM-5.2 schedule (llama.cpp's rule, context_length 1048576)

glm-dsa: no attention.indexer.types - the GLM-5.2 schedule (llama.cpp's rule, context_length 1048576)

glm-dsa: WARNING 57 shared-indexer layers carry indexer tensors (3,4,5,7,8,9,11,12,...) - ignored, they reuse the previous full layer's top-k

maya generate: pack: out of memory

## User capture 39

11/9/2026, 7:15:57 PM

\=== Physical memory ===

               total        used        free      shared  buff/cache   available

Mem:           2.0Ti       3.0Gi       1.9Ti       204Ki        18Gi       2.0Ti

Swap:             0B          0B          0B

**MemTotal**:       2100828888 kB

**MemAvailable**:   2097693334 kB

**SwapTotal**:             0 kB

**CommitLimit**:    1050414444 kB

**Committed_AS**:   36182444 kB

\=== Process limits ===

-t: cpu time (seconds)              unlimited

-f: file size (blocks)              unlimited

-d: data seg size (kbytes)          unlimited

-s: stack size (kbytes)             8192

-c: core file size (blocks)         0

-m: resident set size (kbytes)      unlimited

-u: processes                       7678535

-n: file descriptors                1024

-l: locked-in-memory size (kbytes)  unlimited

-v: address space (kbytes)          unlimited

-x: file locks                      unlimited

-i: pending signals                 7678535

-q: bytes in POSIX msg queues       819200

-e: max nice                        0

-r: max rt priority                 0

-N 15: rt cpu time (microseconds)   unlimited

\=== Linux overcommit ===

vm.overcommit_memory = 0

vm.overcommit_ratio = 50

vm.max_map_count = 1048576

\=== Cgroup memory limit ===

/sys/fs/cgroup/memory.max

max

/sys/fs/cgroup/memory.current

23115063296

/sys/fs/cgroup/memory.events

low 0

high 0

max 0

oom 0

oom_kill 0

oom_group_kill 0

sock_throttled 0

\=== Kernel OOM messages ===

    /**opt**/**project-maya**   #v1.1.57 ?5                                                               unsloth-updater   root@openwebui  19:15:35  

❯ 

[0] 0:zsh\*                                                                                                         "root@openwebui:/opt/p" 19:15 09-Nov-26

## User capture 45

11/9/2026, 7:30:07 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-q4-4gpu-128k.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-q4-4gpu-128k.json (loading the model first - a minute or a few) ===

  [ok] loaded in 441 s

  decode     4.3 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode     3.6 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

Looks like llama.cpp was faster for some reason

## User capture 47

11/9/2026, 7:33:59 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-q4-4gpu-128k.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-q4-4gpu-128k.json (loading the model first - a minute or a few) ===

  [ok] loaded in 441 s

  decode     4.3 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode     3.6 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode     4.3 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  [X]  the engine stopped during the benchmark

       its log: /opt/project-maya/maya-bench.log

Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.

## User capture 49

11/9/2026, 7:35:47 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-q4-4gpu-balanced.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-q4-4gpu-balanced.json (loading the model first - a minute or a few) ===

  [X]  the engine stopped while loading

       its log: /opt/project-maya/maya-bench.log

Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.

## User capture 50

11/9/2026, 7:36:25 PM

grep -Ei \

  'error|failed|cannot|out of memory|split|CUDA[0-3]|dense weights|state/KV|prefill|pool' \

  maya-bench.log | tail -100

glm settings from the environment: MAYA_GLM_KV_INT8=1, MAYA_GLM_PREFILL_MB=4096, MAYA_GLM_PREFILL_SUB=1024, MAYA_GLM_RAM_HEADROOM_GB=16, MAYA_GLM_RESERVE_MB=3072, MAYA_GLM_SPLIT=20,40,60, MAYA_GLM_SPLIT_LOG=1

maya generate: glm split: layer 20 shares the previous layer's indexer top-k, so a part cannot start there; the nearest full-indexer layers are 18 or 22 (glm-dsa boundaries: 1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74)

❯ tail -30 maya-bench.log

glm settings from the environment: MAYA_GLM_KV_INT8=1, MAYA_GLM_PREFILL_MB=4096, MAYA_GLM_PREFILL_SUB=1024, MAYA_GLM_RAM_HEADROOM_GB=16, MAYA_GLM_RESERVE_MB=3072, MAYA_GLM_SPLIT=20,40,60, MAYA_GLM_SPLIT_LOG=1

glm-dsa: no attention.indexer.types - the GLM-5.2 schedule (llama.cpp's rule, context_length 1048576)

maya generate: glm split: layer 20 shares the previous layer's indexer top-k, so a part cannot start there; the nearest full-indexer layers are 18 or 22 (glm-dsa boundaries: 1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74)

## User capture 51

11/9/2026, 8:51:27 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-q4-4gpu-balanced.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-q4-4gpu-balanced.json (loading the model first - a minute or a few) ===

  [ok] loaded in 416 s

  decode     6.6 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode     5.5 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode     8.1 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill     60 tokens/s   a 2048-token prompt

  prefill     47 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 19:51

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-q4-4gpu-balanced.json, context 131072, GPUs [0, 1, 2, 3]

  decode (writing the answer): mean 6.7 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

       6.6 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

       5.5 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

       8.1 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 60 tokens/s, 2048 tokens

  prefill (reading the prompt): 47 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

## User capture 58

11/9/2026, 9:02:43 PM

./maya.sh --backend hip --setup --rebuild --no-start

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

  Model? 1) GLM-5.3-Flash

\=== Step 1: checking this PC ===

  [ok] using GPU 0 (AMD RADEON PRO V620 Azure, 30 GB, gfx1030); HIP, one GPU, text only

  [ok] ROCm: /opt/rocm-7.2.0; CPU: AMD EPYC 7713 64-Core Processor (AVX2)

  [ok] RAM: 2004 GB, 2008 GB available now

  [!]  gfx1030 (RDNA 2) is experimental: it builds and has the portable kernels, but Maya has not been validated on such a card yet - please report how it goes in an issue ([https://github.com/mw00/project-maya/issues](https://github.com/mw00/project-maya/issues))

\=== Step 2: your choices ===

  Which GLM-5.3-Flash? 10) GLM-5.3 (744B, text only): GGUF files you already have

  The setup's whole output: /opt/project-maya/maya-setup.log

  [X]  GLM-5.3 is set up from its GGUF files

       ./maya.sh --setup --gguf-dir \<the folder of its .gguf files>

Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.

    /**opt**/**project-maya**   #v1.1.57 ?5                                                                                                       39s   unsloth-updater   root@openwebui  21:02:33  

❯

## User capture 61

11/9/2026, 9:12:13 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-flash-q4-2gpu-1m.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-flash-q4-2gpu-1m.json (loading the model first - a minute or a few) ===

  [ok] loaded in 98 s

  decode    16.2 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    16.2 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    23.6 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    177 tokens/s   a 2048-token prompt

  prefill    234 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 21:11

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-flash-q4-2gpu-1m.json, context 1048576, GPUs [2, 3]

  decode (writing the answer): mean 18.7 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      16.2 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      16.2 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      23.6 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 177 tokens/s, 2048 tokens

  prefill (reading the prompt): 234 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

Are you using my mtp drafter?

## User capture 62

11/9/2026, 9:12:51 PM

❯ cd /opt/project-maya

grep -Ei 'glm mtp:|glm spec:|glm spec tune:|drafts accepted|NO_MTP|NO_SPEC' \

  maya-bench.log | tail -60

glm settings from the environment: MAYA_GLM_KV_INT8=0, MAYA_GLM_NO_MTP=0, MAYA_GLM_NO_SPEC=0, MAYA_GLM_PREFILL_MB=4096, MAYA_GLM_PREFILL_SUB=1024, MAYA_GLM_RAM_HEADROOM_GB=16, MAYA_GLM_RESERVE_MB=3072, MAYA_GLM_SPLIT=auto, MAYA_GLM_SPLIT_LOG=1

glm mtp: CUDA1 the NextN block (blk.45) loaded: 210 MB dense, experts q4_K/q5_K

glm spec: CUDA0 the head's recurrent-state backup (77 MB) before the expert pool

glm spec: 53 of 62 drafts accepted (85.5%)

glm spec: 232 of 254 drafts accepted (91.3%)

glm spec: 205 of 254 drafts accepted (80.7%)

glm spec: 246 of 254 drafts accepted (96.9%)

glm spec: 824 speculative positions, 89.3% accepted

## User capture 67

11/9/2026, 9:16:19 PM

❯ cd /opt/project-maya

cp maya-glm-5.3-flash-q4-2gpu-1m.json \

   maya-glm-5.3-flash-q4-2gpu-1m-precal.json

./maya.sh --backend hip \

  --config maya-glm-5.3-flash-q4-2gpu-1m.json \

  --calibrate --no-start --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

  [!]  the tuning measures the CUDA engine's CPU lane (NVIDIA GPUs): the HIP port keeps its own settings

  [!]  this PC is NOT tuned (the reason is above): the model keeps the engine's own settings

## User capture 72

11/9/2026, 9:21:23 PM

=== Step 1: benchmark: maya-glm-5.3-flash-q4-2gpu-1m-reserve1g.json (loading the model first - a minute or a few) ===

grep -n -A 2 -B 2 'reserve' maya-ds4-tp2-fp16-production.json

  [ok] loaded in 96 s

  decode    19.3 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    19.4 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    25.7 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    179 tokens/s   a 2048-token prompt

  prefill    234 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 21:21

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-flash-q4-2gpu-1m-reserve1g.json, context 1048576, GPUs [2, 3]

  decode (writing the answer): mean 21.5 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      19.3 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      19.4 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      25.7 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 179 tokens/s, 2048 tokens

  prefill (reading the prompt): 234 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

## User capture 75

11/9/2026, 9:26:28 PM

./maya.sh --backend hip \

  --config maya-glm-5.3-flash-q4-2gpu-1m-spec-cancel.json \

  --bench --plain

maya-glm-5.3-flash-q4-2gpu-1m-spec-cancel.json

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-flash-q4-2gpu-1m-spec-cancel.json (loading the model first - a minute or a few) ===

  [ok] loaded in 95 s

  decode    16.8 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    17.8 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    21.3 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    177 tokens/s   a 2048-token prompt

  prefill    234 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 21:25

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-flash-q4-2gpu-1m-spec-cancel.json, context 1048576, GPUs [2, 3]

  decode (writing the answer): mean 18.7 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      16.8 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      17.8 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      21.3 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 177 tokens/s, 2048 tokens

  prefill (reading the prompt): 234 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

## User capture 78

11/9/2026, 9:33:38 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-flash-q4-2gpu-1m-prefill16k.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-flash-q4-2gpu-1m-prefill16k.json (loading the model first - a minute or a few) ===

  [ok] loaded in 95 s

  decode    19.4 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    19.4 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    27.1 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    179 tokens/s   a 2048-token prompt

  prefill    235 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 21:32

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-flash-q4-2gpu-1m-prefill16k.json, context 1048576, GPUs [2, 3]

  decode (writing the answer): mean 22.0 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      19.4 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      19.4 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      27.1 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 179 tokens/s, 2048 tokens

  prefill (reading the prompt): 235 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

❯ grep -Ei \

  'glm prefill:.*chunks|cannot lend|token by token|out of memory' \

  maya-bench.log | tail -30

glm prefill: CUDA0 chunks of 8192 tokens (--prefill 16384; mixer sub-batches of 1024), borrowing 3791 MB of the expert pool while a prompt runs (activations 3468, weight scratch 96, expert staging 211, prestage 0: 0 experts); disk landing ring 96 experts (1687 MB pinned)

glm prefill: CUDA1 chunks of 8192 tokens (--prefill 16384; mixer sub-batches of 1024), borrowing 3767 MB of the expert pool while a prompt runs (activations 3468, weight scratch 96, expert staging 187, prestage 0: 0 experts); disk landing ring 96 experts (1494 MB pinned)

## User capture 79

11/9/2026, 9:37:50 PM

❯ ./maya.sh --backend hip \

  --config maya-glm-5.3-flash-q4-2gpu-1m-prefill32k.json \

  --bench --plain

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-flash-q4-2gpu-1m-prefill32k.json (loading the model first - a minute or a few) ===

  [ok] loaded in 95 s

  decode    17.5 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    17.0 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    22.2 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    175 tokens/s   a 2048-token prompt

  prefill    234 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 21:37

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-flash-q4-2gpu-1m-prefill32k.json, context 1048576, GPUs [2, 3]

  decode (writing the answer): mean 18.9 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      17.5 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      17.0 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      22.2 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 175 tokens/s, 2048 tokens

  prefill (reading the prompt): 234 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

❯ grep -Ei \

  'glm prefill:.*chunks|cannot lend|token by token|out of memory' \

  maya-bench.log | tail -30

glm prefill: CUDA0 chunks of 8192 tokens (--prefill 32768; mixer sub-batches of 1024), borrowing 3791 MB of the expert pool while a prompt runs (activations 3468, weight scratch 96, expert staging 211, prestage 0: 0 experts); disk landing ring 96 experts (1687 MB pinned)

glm prefill: CUDA1 chunks of 8192 tokens (--prefill 32768; mixer sub-batches of 1024), borrowing 3767 MB of the expert pool while a prompt runs (activations 3468, weight scratch 96, expert staging 187, prestage 0: 0 experts); disk landing ring 96 experts (1494 MB pinned)

## User capture 80

11/9/2026, 9:43:45 PM

./maya.sh --backend hip \

  --config maya-glm-5.3-flash-q4-2gpu-1m-prefill32k-8gb.json \

  --bench --plain

maya-glm-5.3-flash-q4-2gpu-1m-prefill32k-8gb.json

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

=== Step 1: benchmark: maya-glm-5.3-flash-q4-2gpu-1m-prefill32k-8gb.json (loading the model first - a minute or a few) ===

  [ok] loaded in 100 s

  decode    17.6 tokens/s   Write a Python function that parses a CSV file into a list o (256 tokens)

  decode    17.7 tokens/s   Explain how photosynthesis works, step by step, for a high-s (256 tokens)

  decode    25.7 tokens/s   Write a complete HTML page with a canvas that draws a bounci (256 tokens)

  prefill    179 tokens/s   a 2048-token prompt

  prefill    235 tokens/s   a 8192-token prompt

  Project Maya v1.1.57 benchmark, 2026-11-09 21:42

  GPUs: AMD RADEON PRO V620 Azure 30 GB, AMD RADEON PRO V620 Azure 30 GB; RAM 2004 GB; CPU AMD EPYC 7713 64-Core Processor

  setup: maya-glm-5.3-flash-q4-2gpu-1m-prefill32k-8gb.json, context 1048576, GPUs [2, 3]

  decode (writing the answer): mean 20.4 tokens/s over 3 answers of up to 256 tokens (greedy, thinking off)

      17.6 tokens/s  Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.

      17.7 tokens/s  Explain how photosynthesis works, step by step, for a high-school student.

      25.7 tokens/s  Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.

  prefill (reading the prompt): 179 tokens/s, 2048 tokens

  prefill (reading the prompt): 235 tokens/s, 8192 tokens

  [ok] written: /opt/project-maya/maya-bench.txt - attach it (with ./maya.sh --report) to a speed report

❯ grep -Ei \

  'glm prefill:.*chunks|cannot lend|token by token|out of memory' \

  maya-bench.log | tail -30

glm prefill: CUDA0 chunks of 32768 tokens (--prefill 32768; mixer sub-batches of 1024), borrowing 6676 MB of the expert pool while a prompt runs (activations 6353, weight scratch 96, expert staging 211, prestage 0: 0 experts); disk landing ring 96 experts (1687 MB pinned)

glm prefill: CUDA1 chunks of 16384 tokens (--prefill 32768; mixer sub-batches of 1024), borrowing 4830 MB of the expert pool while a prompt runs (activations 4532, weight scratch 96, expert staging 187, prestage 0: 0 experts); disk landing ring 96 experts (1494 MB pinned)

glm prefill: CUDA0 chunks of 16384 tokens (the split's smallest)

## User capture 86

11/9/2026, 9:49:27 PM

❯ cd /opt/project-maya

.venv/bin/python -c 'import tokenizers' ||

    .venv/bin/python -m pip install tokenizers

❯ .venv/bin/python bench-glm-long.py

Traceback (most recent call last):

  File "/opt/project-maya/bench-glm-long.py", line 32, in \<module>

    tok = Tokenizer.from_file(str(tokenizer_files[0]))

**Exception**: data did not match any variant of untagged enum ModelUntagged at line 2 column 16

❯ .venv/bin/python -c 'import tokenizers' ||

    .venv/bin/python -m pip install tokenizers

❯ .venv/bin/python bench-glm-long.py

Traceback (most recent call last):

  File "/opt/project-maya/bench-glm-long.py", line 32, in \<module>

    tok = Tokenizer.from_file(str(tokenizer_files[0]))

**Exception**: data did not match any variant of untagged enum ModelUntagged at line 2 column 16

## User capture 87

11/9/2026, 9:51:58 PM

❯ .venv/bin/python bench-glm-long.py

\=== 8k-chunks ===

Traceback (most recent call last):

  File "/opt/project-maya/bench-glm-long.py", line 171, in \<module>

    wait_read&#x79;**(proc)**

    \~\~\~\~\~\~\~\~\~\~**^^^^^^**

  File "/opt/project-maya/bench-glm-long.py", line 74, in wait_ready

    raise RuntimeError(f"Maya exited with code {proc.returncode}")

**RuntimeError**: Maya exited with code 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):

  File "/opt/project-maya/bench-glm-long.py", line 199, in \<module>

    os.killp&#x67;**(proc.pid, signal.SIGTERM)**

    \~\~\~\~\~\~\~\~\~**^^^^^^^^^^^^^^^^^^^^^^^^^^**

**ProcessLookupError**: [Errno 3] No such process

    /**opt**/**project-maya**   #v1.1.57 ?7                                                                                                       10s   unsloth-updater   root@openwebui  21:51:50  

❯

## User capture 88

11/9/2026, 9:52:17 PM

Project Maya v1.1.57 - GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017 or MiniMax-M3.1-Flash on your own GPU(s).

  ----------------------------------------------------------------------------------------------------

  Starting GLM-5.3 · UD-Q4_K_XL (maya-glm-5.3-flash-q4-2gpu-1m-reserve1g.json).

  Dashboard:        [http://127.0.0.1:8082/](http://127.0.0.1:8082/)

  API (OpenAI):     [http://127.0.0.1:8082/v1](http://127.0.0.1:8082/v1)        (any model name; the API key: any)

  API (Anthropic):  [http://127.0.0.1:8082/v1/messages](http://127.0.0.1:8082/v1/messages)

  Loading takes a few minutes: the engine pins up to all but about 6 GB of the free RAM for its expert tier

  (MAYA_GLM_RAM_HEADROOM_GB changes the 6) and warms its caches - the first answers are the slowest.

  Ctrl+C (or closing this terminal) stops it. Engine log: /opt/project-maya/maya-ud-q4_k_xl-hip.log

  ----------------------------------------------------------------------------------------------------

21:51:41 [maya] this server's lines go to /opt/project-maya/maya-ud-q4_k_xl-hip.log too

21:51:41 [maya] sampling defaults from the config: temperature=1.0, top_p=0.95

21:51:41 [maya] images: the vision encoder needs 2.75 GB on this GPU at up to 4096 image tokens

loading the model (the first start takes a minute or two) ...

21:51:41 [maya] layer split across GPUs [2, 3] (auto)

21:51:41 [maya] starting the engine: reading the model's weights ...

Traceback (most recent call last):

  File "/opt/project-maya/serve/server.py", line 7307, in \<module>

    sys.exit(main())

             \~\~\~\~^^

  File "/opt/project-maya/serve/server.py", line 7107, in main

    engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=engine_env)

  File "/opt/project-maya/serve/server.py", line 911, in \_\_init\_\_

    raise RuntimeError("the engine exited before it was ready" + (f" (see {log})" if log else ""))

RuntimeError: the engine exited before it was ready (see /opt/project-maya/maya-ud-q4_k_xl-hip.log)

## User capture 91

11/9/2026, 10:51:26 PM

❯ cd /opt/project-maya

.venv/bin/python maya-glm-prefill-sweep.py

Results: /opt/maya-test-results/glm-prefill-20261109-220014

=== 8k-chunks ===

Effective chunk: 8192

r0   8192 target (8076 chat tokens): 225.3 prefill tok/s; 19.2 decode tok/s; ok

r0  32768 target (32652 chat tokens): 297.7 prefill tok/s; 19.5 decode tok/s; ok

r0  65536 target (65420 chat tokens): 293.1 prefill tok/s; 23.5 decode tok/s; ok

r1   8192 target (8076 chat tokens): 224.9 prefill tok/s; 25.6 decode tok/s; ok

r1  32768 target (32652 chat tokens): 297.3 prefill tok/s; 24.0 decode tok/s; ok

r1  65536 target (65420 chat tokens): 294.9 prefill tok/s; 23.9 decode tok/s; ok

=== 16k-chunks ===

Effective chunk: 16384

r0   8192 target (8076 chat tokens): 225.4 prefill tok/s; 18.8 decode tok/s; ok

r0  32768 target (32652 chat tokens): 233.8 prefill tok/s; 20.4 decode tok/s; ok

r0  65536 target (65420 chat tokens): 231.9 prefill tok/s; 22.5 decode tok/s; ok

r1   8192 target (8076 chat tokens): 225.9 prefill tok/s; 22.1 decode tok/s; ok

r1  32768 target (32652 chat tokens): 234.0 prefill tok/s; 24.0 decode tok/s; ok

r1  65536 target (65420 chat tokens): 231.8 prefill tok/s; 24.0 decode tok/s; ok

Completed. Results: /opt/maya-test-results/glm-prefill-20261109-220014/results.csv

## User capture 93

11/10/2026, 8:13:48 AM

.venv/bin/python maya-glm-kv-sweep.py

Created: maya-glm-5.3-flash-q4-2gpu-1m-int8kv.json

Created: maya-glm-kv-sweep.py

Results: /opt/maya-test-results/glm-kv-20261109-225228

=== 8k-chunks ===

Effective chunk: 8192

r0   8192 target (8076 chat tokens): 224.9 prefill tok/s; 16.5 decode tok/s; ok

r0  32768 target (32652 chat tokens): 295.5 prefill tok/s; 16.1 decode tok/s; ok

r0  65536 target (65420 chat tokens): 294.4 prefill tok/s; 21.7 decode tok/s; ok

r1   8192 target (8076 chat tokens): 224.2 prefill tok/s; 21.6 decode tok/s; ok

r1  32768 target (32652 chat tokens): 296.5 prefill tok/s; 23.1 decode tok/s; ok

r1  65536 target (65420 chat tokens): 294.4 prefill tok/s; 22.0 decode tok/s; ok

=== int8-kv ===

Effective chunk: 8192

r0   8192 target (8076 chat tokens): 232.6 prefill tok/s; 18.2 decode tok/s; ok

r0  32768 target (32652 chat tokens): 291.2 prefill tok/s; 17.9 decode tok/s; ok

r0  65536 target (65420 chat tokens): 291.3 prefill tok/s; 24.9 decode tok/s; ok

r1   8192 target (8076 chat tokens): 231.9 prefill tok/s; 24.5 decode tok/s; ok

r1  32768 target (32652 chat tokens): 292.7 prefill tok/s; 26.3 decode tok/s; ok

r1  65536 target (65420 chat tokens): 292.0 prefill tok/s; 23.7 decode tok/s; ok

Completed. Results: /opt/maya-test-results/glm-kv-20261109-225228/results.csv
