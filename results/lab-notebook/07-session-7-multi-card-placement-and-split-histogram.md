## Session 7 — Monday, July 14, 2026 (evening) — Multi-card placement experiments and the split histogram

### 18:13 — newtest.txt named; first read of an incomplete log

Paul pointed at the log by name ("Okay, its newtest.txt") before the file itself arrived. What was reachable showed the run had not finished — the log stopped mid-load at `blk.78` skip messages, with no result row, no `load_tensors:` buffer summary, no error, no exit marker. Two findings from the partial capture:

1. The `-ot` rules were accepted verbatim:

```
tensor blk.3.ffn_gate_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm1
tensor blk.3.ffn_down_exps.weight (2112 MiB q5_K) buffer type overridden to ROCm1
... blk.4, 5, 6 → ROCm1
```

Placement mechanism confirmed: `-ot` puts expert tensors exactly where pointed. "That was the precondition for the patch to mean anything, and it holds." Missing, however: the blk.7–10 → ROCm2 lines never appear.

2. The load was slow and unintentionally pinned:

```
done_getting_tensors: tensor 'token_embd.weight' cannot be used with preferred buffer type ROCm_Host, using CPU instead
```

`ROCm_Host` is the pinned buffer — a ~14-minute pinned load rather than the seconds-long mmap load intended. Proposed re-run:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.(3|4|5|6)\.ffn_.*_exps\.=ROCm1,blk\.(7|8|9|10)\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 \
  -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot2.txt
```

```bash
grep -E 'overridden to ROCm2' /root/ot2.txt | head        # did blk.7-10 land on ROCm2?
grep -E '^\| glm-dsa' /root/ot2.txt                        # the number
```

**Observations**

- Baseline to beat: `ub 4096 → 84.62` (all experts on CPU).
- **Prediction (branches):** above 84.62 → multi-card expert compute is real and the patch premise confirmed; unchanged → resident experts do not help prefill, do not patch; OOM → ROCm1/ROCm2 too full — informative in itself.
- Caveat restated: this tests resident experts (necessary but not sufficient for the patch, which routes streamed experts).

### 18:14 — newtest.txt attached: the OOM and the broken regex

The file arrived. **Artifact — `newtest.txt` (saved 18:13:59 ET):** the failed `-ot` llama-bench log. Its header reads `llama-bench: benchmark 1/3: starting` — the comma-separated `-ot` rules had been parsed as three separate benchmark configurations, so configuration 1 carried only the ROCm1 rule and lost its `exps=CPU` catch-all; the load then attempted to allocate 157272.74 MiB on ROCm1 and died out of memory. (The full anatomy of that parse emerged over the next several entries.)

```
allocating 157272.74 MiB on device 1: cudaMalloc failed: out of memory
failed to allocate ROCm1 buffer of size 164912416512
```

157 GiB onto ROCm1's 30 GiB card. The first diagnosis targeted the regex: `blk\.(3|4|5|6)` is unanchored — `blk.3` is a substring of `blk.34`, and `regex_search` matches anywhere — so the rule swept roughly 40 layers (blk.3/4/5/6 plus blk.30–39, blk.40–49, blk.53, blk.63, and so on) onto ROCm1, attempted 157 GiB, and died before reaching the ROCm2 rule (hence zero ROCm2 matches, and why the earlier capture showed only blk.3–6). Fixed with a literal-dot anchor:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.[3-6]\.ffn_.*_exps\.=ROCm1,blk\.[7-9]\.ffn_.*_exps\.=ROCm2,blk\.10\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 \
  -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot3.txt
```

```bash
grep -c 'overridden to ROCm1' /root/ot3.txt    # want exactly 12  (4 layers × 3 tensors)
grep -c 'overridden to ROCm2' /root/ot3.txt    # want exactly 12
grep -E '^\| glm-dsa' /root/ot3.txt            # the result
```

**Observations**

- **Dead end:** the OOM exposed a bug in the test design, not the machine; it does not prove multi-card resident experts fail — the experiment never ran.
- **Correction (self):** "I've now handed you two broken `-ot` regexes in a row" — the regex_search-matches-anywhere warning had been written into Paul's own diagnostic script three versions earlier.
- Expected footprint: 4 layers per card ≈ 22 GiB of experts each plus ~5 GiB attention share and streaming scratch, inside 30 GiB.

### 18:16 — ot3: "failed to load model," a six-line log

Paul ran ot3 verbatim. Output: the four-device init block (Total VRAM 122816 MiB; 4× AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM 30704 MiB each), then:

```
llama_bench: error: failed to load model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

with an empty result table. The real error line was requested:

```bash
grep -E 'out of memory|failed to allocate|overridden to ROCm|error loading' /root/ot3.txt | head -20
```

```bash
grep -E 'cudaMalloc|out of memory|failed to allocate ROCm|buffer of size|overridden to' /root/ot3.txt
```

and staged a two-layer-per-card fallback that "cannot OOM":

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.3\.ffn_.*_exps\.=ROCm1,blk\.4\.ffn_.*_exps\.=ROCm1,blk\.5\.ffn_.*_exps\.=ROCm2,blk\.6\.ffn_.*_exps\.=ROCm2,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot4.txt
```

```bash
grep -c 'overridden to ROCm1' /root/ot4.txt   # must be exactly 6
grep -c 'overridden to ROCm2' /root/ot4.txt   # must be exactly 6
grep -E '^\| glm-dsa' /root/ot4.txt
```

**Observations**

- **Dead end:** second consecutive load failure, error stage unknown pending the tee'd log; placement math checked out this time (`[7-9]` = 3 layers + `10` = 1 → 4 layers per card).
- **Prediction (branches):** `allocating ~157000 MiB` again → the regex still over-matching; a smaller number (~25–30 GiB) → placement correct but 8 resident layers genuinely do not fit.

### 18:17 — The six-line log reinterpreted: parse failure, not allocation

`cat /root/ot3.txt` showed the entire log was the init block, the error line, and the empty table header — no placement lines, no allocation messages. Course corrected: this was not the 157 GiB over-match — the run died earlier, at argument parsing; the `-ot` string never took effect. Isolation test with a single rule:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.3\.ffn_.*_exps\.=ROCm1,exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot5.txt
```

```bash
grep -cE 'overridden to ROCm1' /root/ot5.txt   # want 3
grep -E '^\| glm-dsa|error|out of memory' /root/ot5.txt
```

```bash
llama-bench --list-devices
```

**Observations**

- **Correction:** the failure stage moved from allocation to parsing; the four-clause `-ot` string was suspected of hitting a parse limit or malformed-rule rejection.
- Open question flagged: does `-ot ...=ROCm1` even parse on this build (newtest.txt had shown "overridden to ROCm1," so it parsed before); stray characters from retyping also flagged.

### 18:18 — Paul forces the lookup; the comma/semicolon inversion found in the parser

Paul: "Well can you go look this up?  I think you're supposed to use | | | or , , ,". The llama-bench parser, read directly, gave the root cause — `-ot` uses two delimiter levels:

- comma `,` separates benchmark configurations — each comma-group becomes a separate run (line 901: `strcspn(value, ",")`);
- semicolon `;` separates rules within one configuration (line 930: `strcspn(override_group, ";")`).

The earlier comma string had therefore been parsed as three separate benchmark runs, each with one rule. Run 1 put blk.3–6 (unanchored, ~40 layers) on ROCm1 with nothing on CPU — the other experts defaulted to VRAM — hence the 157 GiB OOM recorded in newtest.txt. "Commas made three broken configs instead of one correct config." This is the inverse of `llama-server`/`llama-cli`, where `-ot` rules are comma-separated; that exact inversion had been noted in the v3 script briefing and the command then written with commas anyway — the same mistake twice. Device names come from `ggml_backend_buft_name(buft)` (line 897); a wrong name produces `error: unrecognized buffer type` plus a valid list (line 947).

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.[3-6]\.ffn_.*_exps\.=ROCm1;blk\.[7-9]\.ffn_.*_exps\.=ROCm2;blk\.10\.ffn_.*_exps\.=ROCm2;exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 4096 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot6.txt
```

```bash
grep -c 'overridden to ROCm1' /root/ot6.txt   # 12  (blk.3,4,5,6 × 3 tensors)
grep -c 'overridden to ROCm2' /root/ot6.txt   # 12  (blk.7,8,9,10 × 3 tensors)
grep -E '^\| glm-dsa' /root/ot6.txt
```

**Observations**

- **Correction (root cause, from source):** the comma/semicolon inversion, not a parse limit.
- Expected to load without OOM (~22 GiB experts per card plus attention and scratch, inside 30 GiB). Baseline to beat: 84.62.

### 18:19 — ot6: "failed to create context"

Paul ran ot6 verbatim; the same four-device init block, then:

```
llama_bench: error: failed to create context with model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

The changed error read as progress — "failed to create context" happens after the model loads; the semicolons fixed parsing, placement worked, and the run died building the compute context. Requested detail:

```bash
grep -E 'overridden to ROCm|error|failed|out of memory|cudaMalloc|buffer' /root/ot6.txt | head -40
```

**Prediction:** with expert layers resident on ROCm1/ROCm2 and op_offload on (`-nopo 0`) at `-ub 4096`, there is no room for the compute buffer — the weights fit; the working memory for a 4096-token batch does not. Fix step 1, drop the ubatch:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 \
  -ot "blk\.[3-6]\.ffn_.*_exps\.=ROCm1;blk\.[7-9]\.ffn_.*_exps\.=ROCm2;blk\.10\.ffn_.*_exps\.=ROCm2;exps=CPU" \
  -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 2048 -ub 2048 -p 2048 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ot7.txt
```

Baseline at ub 2048: 62.64. Fix step 2 if 2048 still fails — halve the resident layers:

```
-ot "blk\.3\.ffn_.*_exps\.=ROCm1;blk\.4\.ffn_.*_exps\.=ROCm1;blk\.5\.ffn_.*_exps\.=ROCm2;blk\.6\.ffn_.*_exps\.=ROCm2;exps=CPU"
```

**Observations**

- Insight flagged as central: resident experts and op_offload scratch compete for the same VRAM — the same ceiling would apply to the patched multi-card path. The `-ts 0,1,1,1` idea (ROCm0 kept empty as scratch) reiterated.

### 18:20 — The grep returns only the error line; two structural hypotheses

Paul's grep of ot6.txt returned exactly one line — the error itself; no "overridden to ROCm" lines at all, though in ot3.txt placement lines had appeared during load. The unfiltered view was requested:

```bash
wc -l /root/ot6.txt
sed -n '6,40p' /root/ot6.txt
```

and a narrowing grep:

```bash
grep -iE 'overridden|split|assigned to device|compute buffer|graph|context|reserve|cudaMalloc|out of memory|GGML_ASSERT|abort' /root/ot6.txt
```

**Observations**

- **Hypothesis (branch 1):** placement lines present → compute-buffer OOM as predicted; `-ub 2048` fixes it.
- **Hypothesis (branch 2):** no placement lines → a structural failure: `-ot` puts layer 3–10 experts on ROCm1/2 while those layers' attention stays on ROCm0 (the default `-ngl` split) — the graph builder may reject a layer spread across devices. Noted as informative for the patch, which separates weight location from run location in the same way.

### 18:24 — ot7 at ub 2048 runs: pp2048 = 51.74 — the negative result

The experiment finally executed (build 657e01125 (10001)):

```
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |  fa | ot                    |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | --------------------- | --------------: | -------------------: |
| glm-dsa ?B Q4_K - Medium       | 435.19 GiB |   753.86 B | ROCm       |  99 |      32 |     2048 |   1 | blk\.[3-6]\.ffn_.*_exps\.=ROCm1;blk\.[7-9]\.ffn_.*_exps\.=ROCm2;blk\.10\.ffn_.*_exps\.=ROCm2;exps=CPU |          pp2048 |         51.74 ± 0.00 |
```

pp2048 = 51.74 with 8 expert layers resident on ROCm1/ROCm2, against the all-CPU baseline of 62.64 at ub 2048: resident experts on multiple cards made prefill worse, by 17%. The `-ub 4096` context failure was thereby confirmed as compute-buffer OOM — the cards are tight.

**Observations**

- **Hypothesis (later retracted):** the 8 resident layers force activations to hop ROCm0→ROCm1, compute, and hop back, per layer per token, through the host bridge (`Hops: 2`, no XGMI) — "that is exactly what my patch would trigger, at scale."
- **Decision (later reversed):** "Do not build the patch."
- Standing best reaffirmed: ub 8192, all experts CPU, op_offload on → 104.97 t/s; "the single-card concentration that I thought was a bug to fix is, on this hardware, close to optimal... The 105 t/s you already have is, most likely, the ceiling." `-ts 0,1,1,1` still noted as a different mechanism worth one run.

### 18:25 — Paul pushes back; the P2P retraction

Paul: "Can you look into anything I can do here.  I have four cards.  Prefill should be faster and parallelizable.  I also think we should look into that p2p routing maybe?" The source, read again, forced a retraction of the earlier explanation: op_offload doesn't do GPU-to-GPU copies at all. In op_offload the weight is on the host, so the transfer is host→device; `GGML_CUDA_P2P` gates GPU↔GPU only (lines 334-348) and is irrelevant to the offload path. What the `-ot` experiment actually did: op_offload fires only for CPU-resident weights (`ggml_backend_buffer_is_host(src->buffer)`, line 919); the resident experts on ROCm1 are not host, so those 8 layers ran as ordinary resident-GPU matmuls while their attention stayed on ROCm0 — the hidden state crossed ROCm0→ROCm1 and back via `cudaMemcpyDeviceToDevice` (lines 754-756), which does need peer access, not enabled. The experiment created GPU↔GPU traffic through the resident-weight path, not the offload path. It tested the wrong mechanism.

Proposed tests, in order — first a 30-second P2P capability probe:

```bash
cat > /tmp/p2p.c <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
int main(void){int n;hipGetDeviceCount(&n);
for(int i=0;i<n;i++)for(int j=0;j<n;j++){if(i==j)continue;
int c=0;hipDeviceCanAccessPeer(&c,i,j);
printf("GPU%d->GPU%d: %s\n",i,j,c?"YES":"no");}return 0;}
EOF
hipcc /tmp/p2p.c -o /tmp/p2p && /tmp/p2p
```

then whether `GGML_CUDA_P2P=1` changes the existing offload number (source says it should not; confirm):

```bash
GGML_CUDA_P2P=1 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 -o md
```

then the still-undone `-ts 0,1,1,1` run. Both follow-ups were issued with tee targets:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 -o md 2>&1 | tee /root/ts.txt
```

```bash
GGML_CUDA_P2P=1 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 -o md
```

**Observations**

- **Correction:** "the `-ot` result does not actually condemn the patch... the experiment was not a valid test of the patch, and I called it decisive when it wasn't." The patch is untested, not disproven.
- Re-verified on build 657e01125: the offload path is host→device only; the copy falls through to the blocking path (lines 514-517) because `cpy_tensor_async` returns false for a CPU source. The bottleneck is serialization, not peer bandwidth.
- Self-assessment on the record: three wrong or oversold mechanism claims this session. The measured truth: prefill is host→device-bound, serialized, one card — 22 GB/s against ~89 GB/s of installed lane bandwidth.

### 18:26 — hipcc fails on a .c file; workarounds

The probe would not compile:

```
failed to execute:/opt/rocm-7.2.0/lib/llvm/bin/clang++  -O3 --driver-mode=g++ -O3 --hip-link  -x c /tmp/p2p.c -o "/tmp/p2p"
```

(establishing ROCm 7.2.0 on the box). Diagnosis: a hipcc quirk — `-x c` fights `--driver-mode=g++`. Fixes offered:

```bash
cp /tmp/p2p.c /tmp/p2p.hip
hipcc /tmp/p2p.hip -o /tmp/p2p && /tmp/p2p
```

```bash
/opt/rocm-7.2.0/lib/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx1030 \
  /tmp/p2p.c -o /tmp/p2p \
  -L/opt/rocm-7.2.0/lib -lamdhip64 && /tmp/p2p
```

No-compile alternatives:

```bash
rocm-smi --showtopo
```

```bash
rocm-bandwidth-test -t   # topology + peer access matrix
```

Recalled hardware fact: the earlier `--showtopo` showed `Link Type: PCIE, Hops: 2` for the pairs; the single needed bit is whether any GPU pair reports peer access possible.

**Observations**

- **Dead end (tooling):** hipcc mis-detects `.c` files; the `.hip` extension or an explicit `-x hip --offload-arch=gfx1030 -lamdhip64` invocation works around it. ROCm version on the box established: 7.2.0.

### 18:27 — P2P capability: all twelve directions YES

Paul reported "2 warnings generated when compiling for host," then:

```
GPU0->GPU1: YES
GPU0->GPU2: YES
GPU0->GPU3: YES
GPU1->GPU0: YES
GPU1->GPU2: YES
GPU1->GPU3: YES
GPU2->GPU0: YES
GPU2->GPU1: YES
GPU2->GPU3: YES
GPU3->GPU0: YES
GPU3->GPU1: YES
GPU3->GPU2: YES
```

**Observations**

- **Refuted:** every pair: YES. That kills the "cards can't talk to each other" claim outright. The patch is not dead — untested. It was also now unknown why the `-ot` experiment lost, the P2P explanation being disproven.
- Capability is not bandwidth: four cards on separate root complexes through the EPYC IOD fabric could peer anywhere from 3 GB/s to 25 GB/s. Bandwidth probes queued:

```bash
rocm-bandwidth-test 2>/dev/null | head -60 || echo "not installed"
```

```bash
cat > /tmp/pbw.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#define N (256*1024*1024)
int main(){
  hipSetDevice(0); hipDeviceEnablePeerAccess(1,0);
  void *a,*b; hipSetDevice(0); hipMalloc(&a,N);
  hipSetDevice(1); hipMalloc(&b,N);
  hipSetDevice(0);
  for(int w=0;w<3;w++) hipMemcpyPeer(b,1,a,0,N);
  hipDeviceSynchronize();
  hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s);
  for(int i=0;i<20;i++) hipMemcpyPeer(b,1,a,0,N);
  hipEventRecord(e); hipEventSynchronize(e);
  float ms; hipEventElapsedTime(&ms,s,e);
  printf("GPU0->GPU1 peer: %.1f GB/s\n", 20.0*N/1e9/(ms/1000.0));
  return 0;
}
EOF
hipcc /tmp/pbw.hip -o /tmp/pbw && /tmp/pbw
```

- **Decision thresholds:** ≥15 GB/s → peer traffic cheap, tensor-parallel expert compute viable, "path to ~300–400 t/s prefill is real"; 3–8 GB/s → careful per-scheme testing; near host-memory rate → little advantage. The `-ts` run (ts.txt) and `GGML_CUDA_P2P=1` control (p2ptest.txt) were re-issued verbatim in parallel.

### 18:29 — Single-pair peer bandwidth: 16.2 GB/s; ts.txt missing

Paul: "GPU0->GPU1 peer: 16.2 GB/s" — above the ≥15 threshold, "not the 3-8 GB/s host-relay penalty I feared." Four-card expert compute back on the table on measured evidence. Two unknowns remained: aggregate peer bandwidth with all four cards peering simultaneously through the shared IOD (could scale to ~64 GB/s or contend down to ~20), and how much peer traffic each scheme generates. Patch math sketch: spread 438 GB of expert streaming across four host→device links; peer cost = activation hops — at 16 GB/s, a 100 MB activation per layer ≈ ~6 ms, across 75 layers ≈ ~450 ms of peer traffic per token if every layer hops. The queued results were requested:

```bash
grep -E '^\| glm-dsa' /root/ts.txt
```

```bash
grep -E '^\| glm-dsa' /root/p2ptest.txt
```

(~105 expected for the P2P control, since offload is host→device.) The response:

```
root@openwebui:~# grep -E '^\| glm-dsa' /root/ts.txt
grep: /root/ts.txt: No such file or directory
```

The queued commands had not run. A check was proposed and the commands re-issued:

```bash
ls -la /root/ts.txt /root/p2ptest.txt 2>&1
```

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ts.txt
```

```bash
grep -E '^\| glm-dsa|error|out of memory|failed' /root/ts.txt
```

plus the four-pair aggregate peer probe (concurrent pairs {0→1},{1→0},{2→3},{3→2}, async, 256 MiB, 20 iterations):

```bash
cat > /tmp/pbw4.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#define N (256*1024*1024)
int main(){
  int pairs[4][2]={{0,1},{1,0},{2,3},{3,2}};
  void *src[4],*dst[4]; hipStream_t st[4];
  for(int k=0;k<4;k++){int s=pairs[k][0],d=pairs[k][1];
    hipSetDevice(s); hipDeviceEnablePeerAccess(d,0); hipMalloc(&src[k],N);
    hipStreamCreate(&st[k]);
    hipSetDevice(d); hipMalloc(&dst[k],N);}
  for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]);
    for(int w=0;w<3;w++) hipMemcpyPeerAsync(dst[k],pairs[k][1],src[k],pairs[k][0],N,st[k]);}
  for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]); hipStreamSynchronize(st[k]);}
  hipSetDevice(0); hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  hipEventRecord(s,0);
  for(int i=0;i<20;i++) for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]);
    hipMemcpyPeerAsync(dst[k],pairs[k][1],src[k],pairs[k][0],N,st[k]);}
  for(int k=0;k<4;k++){hipSetDevice(pairs[k][0]); hipStreamSynchronize(st[k]);}
  hipSetDevice(0); hipEventRecord(e,0); hipEventSynchronize(e);
  float ms; hipEventElapsedTime(&ms,s,e);
  printf("4 concurrent pairs: %.1f GB/s aggregate\n", 4.0*20.0*N/1e9/(ms/1000.0));
  return 0;
}
EOF
hipcc /tmp/pbw4.hip -o /tmp/pbw4 && /tmp/pbw4
```

**Observations**

- **Prediction (`-ts` branches):** above 105 → ROCm0 was VRAM-contended and headroom exists; flat at 105 → the blocking serialized host→device copy is the wall; OOM → drop to ub 4096. Baseline: 104.97.
- **Prediction (pbw4 thresholds):** near 4×16 = 64 GB/s → the fabric scales and the patch has headroom; contending to ~20 → four cards streaming would choke each other.

### 18:31 — -ts 0,1,1,1 fails to create context: self-defeating by design

Paul ran the ts command verbatim; the same init block, then:

```
llama_bench: error: failed to create context with model '/models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf'
```

The detail was requested:

```bash
grep -iE 'out of memory|cudaMalloc|failed to allocate|buffer of size|compute buffer|reserve' /root/ts.txt
```

and identified the design flaw in its own experiment: `-ts 0,1,1,1` puts zero model layers on ROCm0, packing all 78 attention layers onto ROCm1/2/3 (26 each) — but op_offload still sends every streamed expert to ROCm0 (the `return b` = backend 0 behaviour). ROCm1–3 cannot fit 26 attention layers each plus the ub 8192 compute buffer, so context creation fails. "`-ts 0,1,1,1` is self-defeating: it frees the wrong card." A fallback was staged to at least get a number:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -ts 0,1,1,1 -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 4096 -ub 4096 -p 4096 -n 0 -r 1 \
  -o md 2>&1 | tee /root/ts2.txt
```

**Observations**

- **Correction (own experiment design):** the third flag-level failure in a row.
- **Decision:** all three experiments (`-ot`, `-ts`, the retry) hit the same wall — "op_offload's ROCm0 hardcoding means no `-ts`/`-ot` juggling can spread the streaming load. Only code can." Flag-level experiments declared exhausted. Remaining: (a) the pbw4 aggregate number, the deciding fact; (b) the patch itself, only if justified, tested functionally at low ubatch first.

### 18:31 — Aggregate peer bandwidth: 49.4 GB/s

```
22 warnings generated when compiling for host.
4 concurrent pairs: 49.4 GB/s aggregate
```

49.4 GB/s aggregate against 16.2 single-pair: the fabric scales at ~76% efficiency across concurrent transfers and does not collapse to ~20. Measured-facts scoreboard at this point: peer access YES in all 12 directions; single-pair peer 16.2 GB/s; four-pair aggregate 49.4 GB/s; current prefill = one card, one link, 22 GB/s host→device, serialized.

Before writing the patch, the mechanism was verified — does `src[1]` (the activation) have a resolved buffer at graph-assignment time? The source answered no: lines 1047-1050 walk the nodes calling `backend_id_from_cur` without pre-assigning sources — the src-assignment block at lines 1052-1066 is `#if 0` dead code. `src[1]` therefore has `tensor_backend_id == -1` when the heuristic runs; the original activation-routing patch would read `asrc->buffer` as null and fall through to `return b` = ROCm0. A silent no-op. The scheduler's own design comment (line 1074): "ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known" — pass 2 (line 1078+) expands GPU assignments to adjacent nodes, but the offload heuristic at line 919 fires in pass 1 and hardcodes backend 0 before pass 2 runs.

A no-code diagnostic was ordered first:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 2>&1 | grep -E 'ffn_moe|split|ROCm[0-9]|off' | head -60 > /root/sched.txt
wc -l /root/sched.txt
head -60 /root/sched.txt
```

**Observations**

- **Prediction (later corrected):** not a clean 4× to 415, but "250-350 t/s is plausible, versus 105 today."
- **Confirmed (against own patch):** "it confirms my patch is broken" — the drafted activation-routing patch cannot work as written.
- **Decision:** the correct patch is narrower — either defer (return -1, let pass-2 expansion place the op with its layer; risk of landing on CPU) or round-robin the target backend (cruder, predictable).

### 18:35 — sched.txt is empty; the debug format discovered

The diagnostic produced `0 /root/sched.txt`. Zero lines meant grep mismatch or uncaptured output, not "no splits." Unfiltered capture proposed:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_full.txt 2>&1
wc -l /root/sched_full.txt
grep -icE 'split|SPLIT|backend|ROCm|CPU' /root/sched_full.txt
```

```bash
grep -inE 'split #|## SPLIT|Split|node.*backend|=.*ROCm|=.*CPU|1\.off|1\.wgt' /root/sched_full.txt | head -40
```

Source reading established the format: `## SPLIT #N: <backend>` plus per-node lines `node #N (op): name [BACKEND cause]` with cause tags `1.off` / `1.wgt` — and that the dump goes through `GGML_LOG_DEBUG`, filtered unless log verbosity is debug. Fix — add `-v`:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_full.txt 2>&1
echo "total lines:"; wc -l /root/sched_full.txt
echo "split headers:"; grep -c '## SPLIT' /root/sched_full.txt
echo "which backends own the splits:"; grep '## SPLIT' /root/sched_full.txt | grep -oE 'ROCm[0-9]|CPU' | sort | uniq -c
echo "where the offloaded (1.off) ops landed:"; grep '1.off' /root/sched_full.txt | grep -oE '\[(ROCm[0-9]|CPU)' | sort | uniq -c
```

with `GGML_SCHED_DEBUG=2 GGML_LOG_LEVEL=4 llama-bench ...` as the fallback if `-v` alone did not surface it.

**Observations**

- **Hypothesis:** zero lines because the debug log is suppressed, not absent.

### 18:44 — Without -v: 10 lines; llama-bench's null log callback found

Paul ran the no-`-v` version: 10 lines total, 3 grep matches — just device-init lines. Reading the tool's source settled it: llama-bench installs a null log callback (`llama_null_log_callback`, line 2143) that discards all ggml logging unless `-v` is set; even the "slow copy" warning that fires on the blocking offload path goes through `GGML_LOG_DEBUG` (line 490) and is swallowed. "Not a flag problem — the tool suppresses it by design." The needed run, verbatim:

```bash
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_full.txt 2>&1

wc -l /root/sched_full.txt
grep -c '## SPLIT' /root/sched_full.txt
grep '1.off' /root/sched_full.txt | grep -oE '\[ROCm[0-9]|\[CPU' | sort | uniq -c
```

**Observations**

- Caveat noted: at `-ub 512` the batch is 512, above the 32-token `GGML_OP_OFFLOAD_MIN_BATCH`, so offload should fire; if the `1.off` grep is empty, bump to `-p 2048 -ub 2048` and re-grep.

### 19:08 — With -v: 31,795 lines, 1,494 splits; the 1.off grep comes back empty

```
31795 /root/sched_full.txt
1494
```

The dump landed; the third command (the `1.off` grep) printed nothing — the pattern was wrong, not offload absent (the bracket format in source is `[%5.5s %8.8s]`, so the cause tag may render padded or truncated). Follow-up queries:

```bash
# what do the node lines actually look like? show a MoE expert node
grep -m5 'ffn_moe\|_exps' /root/sched_full.txt

# what cause tags exist, and how often?
grep -oE '1\.[a-z]+' /root/sched_full.txt | sort | uniq -c

# which backends own the 1494 splits?
grep '## SPLIT' /root/sched_full.txt | grep -oE 'ROCm[0-9]|CPU' | sort | uniq -c
```

**Observations**

- **Correction:** the empty `1.off` grep reflects a wrong pattern (the bracket format in source is `[%5.5s %8.8s]`, so the cause tag may render padded or truncated), not an absence of offload. The third command is the runtime answer: if ~1,400 of the 1,494 splits are ROCm0, the diagnosis holds; if evenly spread, the theory is wrong and the 105 ceiling has a different cause.

### 19:09 — Partial grep results: name fragments, and the ROCm_Host insight

After a stray keystroke ("v"), Paul pasted partial results. The first grep showed tensor lines:

```
tensor blk.3.ffn_gate_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm_Host
create_tensor: loading tensor blk.3.ffn_gate_exps.weight
tensor blk.3.ffn_down_exps.weight (2112 MiB q5_K) buffer type overridden to ROCm_Host
create_tensor: loading tensor blk.3.ffn_down_exps.weight
tensor blk.3.ffn_up_exps.weight (1728 MiB q4_K) buffer type overridden to ROCm_Host
```

The cause-tag grep (run twice, same output):

```
    576 1.attn
     56 1.exp
    564 1.ffn
      1 1.gguf
     40 1.indexer
```

Reinterpreted: the `1.xxx` tokens are tensor name fragments (`blk.1.attn…`, `blk.1.ffn…`), not scheduler cause tags. Key observation: `buffer type overridden to ROCm_Host` — pinned, page-locked host RAM; with `-ot exps=CPU` the experts live in the pinned host buffer, on no GPU. Direct queries issued for the actual question:

```bash
# the actual question: of 1494 splits, which backend runs each?
grep '## SPLIT' /root/sched_full.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c

# and the node-level backend column (5th field in "[BACKEND cause]")
grep -oE '\[(ROCm[0-9]|CPU) ' /root/sched_full.txt | sort | uniq -c
```

**Observations**

- Clean measured facts restated while leveling with Paul: prefill 104.97 t/s at ub 8192 (2.6× over start); the offload copy blocking and serialized (source, lines 514-517); peer bandwidth healthy (16 GB/s single, 49 GB/s aggregate); every VRAM-layout flag trick hits the same wall. "Squeezing past 105 requires a code change to the scheduler."
- **Prediction (branches):** splits ~1,400 on ROCm0 → write the patch knowing it is real; distributed → the bottleneck is the blocking copy itself, and the fix is the harder `cpy_tensor_async` change.

### 19:10 — The split distribution: ROCm0 owns 731 of 1,494

```
root@openwebui:~# grep '## SPLIT' /root/sched_full.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
    308 CPU
    731 ROCm0
    133 ROCm1
    175 ROCm2
    147 ROCm3
root@openwebui:~# grep -oE '\[(ROCm[0-9]|CPU) ' /root/sched_full.txt | sort | uniq -c
  18811 [ROCm0 
  14188 [ROCm1 
  18922 [ROCm2 
  15874 [ROCm3
```

"The diagnosis holds, and now it's measured." ROCm0 owns 731 splits — 49% of all 1,494, 62% of the 1,186 GPU splits — versus ~133/175/147 for the other three cards. The node-level counts (18811/14188/18922/15874) are a red herring: they count resident attention nodes, distributed normally by the layer split. The splits expose op_offload: 731 splits on ROCm0 = 731 host→device transfers serialized onto one card's link = the measured 22 GB/s single-link bottleneck. Everything-agrees checklist: splits ROCm0 62% ✓; copy path blocking/serialized (source) ✓; streaming 22 GB/s = one link ✓; peer fabric 49 GB/s aggregate, healthy ✓; flag tricks cannot move it, the pin is in code ✓.

**Decision:** the patch is justified by measurement. The original activation-routing diff being a no-op, the correct narrow change is round-robin. Patch v1 — file `ggml/src/ggml-backend.cpp`, find around line 919:

```c
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
```

Replace with:

```c
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                // Round-robin the offload target across all eligible GPU backends
                // instead of always taking the first. With a large CPU-resident
                // MoE this stops every offloaded expert matmul from serializing
                // onto backend 0's single PCIe link.
                int n_elig = 0;
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        n_elig++;
                    }
                }
                if (n_elig > 0) {
                    int pick = sched->off_rr % n_elig;
                    int seen = 0;
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            if (seen == pick) {
                                sched->off_rr++;
                                SET_CAUSE(tensor, "1.off");
                                return b;
                            }
                            seen++;
                        }
                    }
                }
            }
```

Second edit — add to `struct ggml_backend_sched {`:

```c
    int off_rr; // round-robin cursor for op_offload target selection
```

Zero-initialized if the struct is calloc'd (verify: `grep -n 'calloc\|ggml_backend_sched_new' ggml/src/ggml-backend.cpp`; if not, add `sched->off_rr = 0;` in init). Build and verify placement moved:

```bash
cd /root/llama.cpp
cmake --build build -j64 && cmake --install build

# re-run the exact split-distribution check that just confirmed the pin
GGML_SCHED_DEBUG=2 llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 1 -nopo 0 -fa 1 -v \
  -t 32 -b 512 -ub 512 -p 512 -n 0 -r 1 > /root/sched_patched.txt 2>&1
grep '## SPLIT' /root/sched_patched.txt | sed -E 's/.*: (ROCm[0-9]|CPU).*/\1/' | sort | uniq -c
```

Then the number:

```bash
llama-bench \
  -m /models/GLM-5.2/UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf \
  -ngl 99 -ot "exps=CPU" -mmp 0 -nopo 0 -fa 1 \
  -t 32 -b 8192 -ub 8192 -p 8192 -n 0 -r 2 -o md
```

**Observations**

- **Prediction (histogram):** before, ROCm0 731 and others ~150; if the patch works, all four ~290 each. If ROCm0 is still 731, the patch did not take effect — do not measure throughput.
- **Prediction (throughput, later corrected):** the fitted model says a path to ~250–350 against the 104.97 baseline. Caveats: round-robin ignores locality (peer hops affordable at 49 GB/s aggregate); the copy is still blocking (async within a card is a separate, harder `cpy_tensor_async` patch); decode never triggers offload at batch 1 — 6.01 t/s untouched.

### 19:23 — A counter-analysis arrives

Paul pasted a detailed technical review from a separate analysis session (referred to below as the pasted review). Its claims, condensed:

- The patch is aimed at the right place, will compile, and should equalize the split histogram — but will not deliver 15 s → 4–5 s streaming, because that assumes four links transfer in parallel and the confirmed blocking copy path prevents it. `cpy_tensor_async` in the CUDA/HIP backend returns false unless both backends are GPU, so every expert-weight transfer goes through the synchronous fallback in `ggml_backend_sched_compute_splits`, blocking the single scheduler thread. The loop is sequential: copies to ROCm1 and ROCm2 cannot overlap regardless of destination. Round-robin alternates the destination link; it does not aggregate them. Streaming stays ~22 GB/s and ~15 s total.
- What the patch does buy: the fallback synchronizes the destination backend before copying — today each weight copy waits for ROCm0 to finish the previous expert GEMM. With targets spread, the destination is idle, so copy l+1 overlaps GEMM l. The GEMMs remain a sequential chain across layers (layer l+1 depends on layer l through attention), so four cards do not parallelize compute for a single ubatch either.
- Bound from the fitted decomposition: 8192 tokens at 104.97 t/s ≈ 78 s; perfectly hiding a 15 s streaming term yields ~63 s ≈ ~130 t/s. "Worth having, but not 250–350."
- The dramatic version requires concurrent copies (async host→device with a pinned source, `memcpyAsync` on each backend's own stream, an event before compute launch, or a copy-thread pool). Before writing that: a ten-minute standalone test — `hipMemcpyAsync` from pinned host to all four cards on four streams from one thread. If the root complex or RAM bandwidth caps aggregate H2D near 22–40 GB/s rather than ~88, no scheduler change can reach the target.
- On the patch itself: the struct is calloc'd in `ggml_backend_sched_new` in every known version (keep the grep step); make `off_rr` unsigned. Per-op round-robin splits gate/up/down of one layer across three cards — two extra multi-hundred-MB peer hops per layer at ub 8192 and roughly double the split count. Prefer keying on the layer: `sscanf(src->name, "blk.%d.", &il); pick = il % n_elig`, falling back to the cursor if parsing fails — same size, keeps each layer's three expert matmuls local, stable across ubatches, needs no struct field.
- Expect each of ROCm1–3 to newly allocate roughly one layer's expert staging in its compute buffer; check VRAM headroom on first run. Compare total split count before/after, not just shares. Decode untouched (`offload_op` requires batch ≥ 32). Caveat on its own claims: the `cpy_tensor_async` rejection and calloc details date from mid-2025 — confirm the fallback still performs the destination sync before the copy in this checkout, "that stall is where this patch's entire gain comes from."

**Observations**

- **Hypothesis (pasted review):** the realistic bound for round-robin is ~130 t/s, not 250–350 — the single blocking host thread prevents parallel transfer; the patch's entire gain is overlap of copy l+1 with GEMM l.
- **Decision seed (pasted review):** key the assignment on the layer index rather than a per-op cursor.

### 19:23 — Verification, adoption of layer-keying, and the h2d4 gate

The counter-analysis was verified against the current tree: lines 516-518 — `ggml_backend_tensor_copy_async` hits the blocking fallback for host→device (the async path at lines 508-509 needs both ends GPU); the split loop (~line 1583) does `ggml_backend_synchronize(split_backend)` before the copy when there is no event — each expert copy waits for that card to finish its previous GEMM; copy and compute fully serialize on ROCm0.

A new source finding altered the magnitude: a comment at line ~1587 — "when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used" — llama.cpp already sparse-copies offloaded experts (only the ~8-per-token actually routed, gated by the `used_ids` bitset). The "438 GB per ubatch" was the dense figure; the real transfer is far smaller, and the measured 22 GB/s is moving only active experts. This does not change the ~130 bound (derived from the measured 78 s wall clock, which already reflects sparse copying), but the streaming term is smaller than 15 s, so the overlap win is more modest — "possibly closer to ~120 than 130." The layer-keyed revision was adopted:

```c
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                int n_elig = 0;
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) n_elig++;
                }
                if (n_elig > 0) {
                    int il = -1, key;
                    if (tensor->src[0] && sscanf(tensor->src[0]->name, "blk.%d.", &il) == 1 && il >= 0) {
                        key = il % n_elig;                 // co-locate a layer's gate/up/down
                    } else {
                        key = (int)(sched->off_rr++ % (unsigned)n_elig);   // fallback
                    }
                    int seen = 0;
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            if (seen == key) { SET_CAUSE(tensor, "1.off"); return b; }
                            seen++;
                        }
                    }
                }
            }
```

(`unsigned off_rr` per the overflow note; the `blk.%d.` sscanf is on `src[0]`, the weight, whose name carries the layer.) Everything was gated on the aggregate host→device probe — the pbw4 probe measured GPU↔GPU, not four concurrent host→device streams:

```bash
cat > /tmp/h2d4.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#define N (256*1024*1024)
int main(){
  void *h; hipHostMalloc(&h,N);            // pinned source
  void *d[4]; hipStream_t s[4];
  for(int k=0;k<4;k++){hipSetDevice(k);hipMalloc(&d[k],N);hipStreamCreate(&s[k]);}
  for(int k=0;k<4;k++){hipSetDevice(k);hipMemcpyAsync(d[k],h,N,hipMemcpyHostToDevice,s[k]);}
  for(int k=0;k<4;k++){hipSetDevice(k);hipStreamSynchronize(s[k]);}
  hipSetDevice(0);hipEvent_t a,b;hipEventCreate(&a);hipEventCreate(&b);
  hipEventRecord(a,0);
  for(int i=0;i<20;i++)for(int k=0;k<4;k++){hipSetDevice(k);hipMemcpyAsync(d[k],h,N,hipMemcpyHostToDevice,s[k]);}
  for(int k=0;k<4;k++){hipSetDevice(k);hipStreamSynchronize(s[k]);}
  hipSetDevice(0);hipEventRecord(b,0);hipEventSynchronize(b);
  float ms;hipEventElapsedTime(&ms,a,b);
  printf("4 concurrent H2D: %.1f GB/s aggregate\n",4.0*20.0*N/1e9/(ms/1000.0));
  return 0;}
EOF
hipcc /tmp/h2d4.hip -o /tmp/h2d4 && /tmp/h2d4
```

**Observations**

- **Correction (of the earlier prediction):** the earlier claim — streaming 15 s → 4-5 s across four links → ~300 t/s — was wrong; it assumes parallel transfer, which the single blocking host thread prevents. The pasted review's bound stands: ~63 s → ~130 t/s. Spread across cards is not parallel across cards.
- **Decision:** layer-keying adopted as strictly better — co-locates a layer's gate/up/down, halves peer traffic, stable across ubatches.
- **Decision (thresholds):** "This number decides the whole endeavor. ~80+ → the async-copy patch is worth writing and the ceiling is high. ~40 → round-robin's ~130 is near the wall and the harder patch buys little. ~22 → the bottleneck was never the scheduler, it's your H2D fabric, and nothing in software moves it."
- The session ends with the h2d4 probe not yet run.

### State of knowledge at end of Session 7

- The zero-code mechanism tests all failed to move the needle or to run: three `-ot` command failures (unanchored regex; comma-vs-semicolon delimiter inversion, documented in the `newtest.txt` artifact's 157272.74 MiB ROCm1 OOM; compute-buffer OOM at ub 4096), then the one that ran — resident experts on ROCm1/ROCm2 at ub 2048 — measured pp2048 = 51.74 vs the 62.64 baseline, 17% worse.
- The 51.74 negative result was initially read as condemning the patch, then retracted: the experiment exercised the resident-weight peer-copy path, not op_offload's host→device path. The patch is untested, not disproven.
- Peer fabric measured: capability YES in all 12 directions; single-pair 16.2 GB/s; four-pair aggregate 49.4 GB/s (~76% scaling). Current offload streaming: ~22 GB/s, one link, blocking, against ~89 GB/s of installed lane bandwidth.
- `-ts 0,1,1,1` is self-defeating (op_offload hardcodes ROCm0 as target; freeing ROCm0 overloads ROCm1–3 with 26 attention layers each and the context fails to build). Flag-level experiments are exhausted; only code can spread the streaming load.
- The runtime split histogram (31,795-line GGML_SCHED_DEBUG dump, obtainable only with `-v` because llama-bench installs a null log callback): CPU 308 / ROCm0 731 / ROCm1 133 / ROCm2 175 / ROCm3 147 — ROCm0 owns 62% of the 1,186 GPU splits. The `return b` pin is confirmed at runtime, no longer inferred.
- The original activation-routing patch is a proven no-op (`src[1]` unassigned at heuristic time; the src-assignment block is `#if 0` dead code). Round-robin patch v1 was drafted, then revised to layer-keyed v2 (`sscanf(src->name, "blk.%d.")`) after the pasted counter-analysis.
- Expected gain corrected downward: not 250–350 t/s but ~130 (possibly ~120), because the single blocking host thread prevents parallel transfer; llama.cpp already sparse-copies only used experts.
- The h2d4 four-stream pinned host→device probe is queued as the decider (~80+ / ~40 / ~22 GB/s thresholds). Standing numbers: prefill 104.97 t/s, decode 6.01 t/s.

---

