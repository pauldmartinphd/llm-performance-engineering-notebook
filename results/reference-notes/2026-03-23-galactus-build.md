# Galactus LLM Inference Server Build — Lab Notebook

## Date: 2026-03-23

*Edited: September 30, 2026*

## Objective

This notebook records the initial build of `galactus`, a dedicated inference server based on an EPYC 7713 and four Radeon Pro V620s. The objective was to serve large MoE models, initially Kimi K2.5 and Qwen 3.5 397B, through Open WebUI using llama.cpp and ROCm under Proxmox.

The useful configuration emerged from two changes: placing selected expert layers explicitly on the GPUs and replacing the mixed-capacity DIMM population. Together, these brought Qwen generation into the 12.4–12.7 t/s range. The record below preserves the March configuration and results; a flag that failed in this build is not necessarily a permanent limitation of the backend.

---

## Hardware

- **CPU:** AMD EPYC 7713 (64C/128T, Milan, Zen 3)
- **RAM:** 1 TB DDR4-2933 ECC (8× 128 GB matched DIMMs, 1DPC, 8-channel)
  - Previous population: 4× 128 GB + 4× 64 GB; this configuration measured much lower bandwidth (see Memory section)
- **GPU:** 4× AMD Radeon Pro V620 (gfx1030, RDNA 2, 32 GB VRAM each). The original inventory recorded 122 GB total; the nominal capacities sum to 128 GB, so the 122 GB figure needs its reporting units or usable-memory basis confirmed. Each card has its own memory allocation limit.
- **Storage:** BTRFS on NVMe
- **Platform:** Proxmox VE 9.1.1, kernel 6.17.2-1-pve

## Software Stack

- Unprivileged LXC container (ID 100, Debian 13, hostname `openwebui.pauldmartin.net`)
- ROCm 7.2.0 (`hip-runtime-amd`, `hip-dev`, `rocm-hip-sdk`)
- llama.cpp build b8477-ec2b787eb, compiled with `-DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1030"` using ROCm's clang at `/opt/rocm-7.2.0/llvm/bin/clang++`
- Open WebUI (uv-based install) on port 8080
- llama-server on port 8081, connected to Open WebUI as OpenAI-compatible endpoint at `http://localhost:8081/v1`

---

## Container Setup

I provisioned an unprivileged container with community-scripts/ProxmoxVE `openwebui.sh`, using the advanced install to expose 128 logical CPUs, a 770 GB memory ceiling, and 2 TB of disk. This is a single-purpose inference server, but these LXC limits still act as ceilings rather than reservations; assigning the limit does not itself allocate all of that memory.

### GPU Passthrough

The community script exposed `/dev/dri/renderD128-131` but omitted `/dev/kfd`. ROCm needed access to both in this installation, so the remaining setup was:

- [x] Added `/dev/kfd` bind mount to container config: `lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file`
- [x] Permissions fix: `/dev/kfd` appears as `nobody:nogroup` inside unprivileged container due to UID/GID offset. Fixed with `chmod 666 /dev/kfd` on the host.
- [x] Persistent udev rule on host: `KERNEL=="kfd", MODE="0666"` in `/etc/udev/rules.d/99-kfd.rules`

The permissions workaround granted all host users access to the device. It records how this installation was made to work; a group-based mapping would provide a narrower access boundary.

### ROCm Build Issues

The build failed with Debian's system clang 19.1.7 because the ROCm 7.2 device bitcode, produced with [LLVM 22](https://rocm.docs.amd.com/en/docs-7.2.0/compatibility/compatibility-matrix.html), contained attribute kinds that the older reader did not recognize. Pointing CMake at ROCm's own clang resolved that mismatch:

```
cmake -S . -B build \
    -DGGML_HIP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.0/llvm/bin/clang++ \
    -DAMDGPU_TARGETS="gfx1030"
```

After the build, I installed it with `cmake --install build --prefix /usr/local` and ran `ldconfig` to register the shared libraries.

### Hostname Rename Incident

Renaming the Proxmox host from `borg` to `galactus` left container 100's configuration under `/etc/pve/nodes/borg/lxc/` while the active node used the new name. I copied the configuration into the new node directory. During the migration, pmxcfs operations gave conflicting indications about whether the destination file existed; renaming the container configuration to 101, copying it, and renaming it back allowed the migration to complete. The symptoms establish a migration problem, but do not identify its filesystem mechanism.

---

## Models

### Kimi K2.5 (UD-Q4_K_XL)

- Loader architecture: `deepseek2`; approximately 1.03T parameters in the recorded export, 384 routed experts, 8 selected per token, and ~32B active parameters. Moonshot's [model card](https://huggingface.co/moonshotai/Kimi-K2.5) rounds the total to 1T.
- Quantization: Unsloth Dynamic Q4_K_XL — experts at Q4_K, attention/norms at Q6_K/Q8_0
- Size on disk: 579 GB (13 GGUF shards), 4.85 BPW
- The source model uses native INT4 quantization. This makes a four-bit export a sensible candidate, but a GGUF Q4 format does not automatically preserve the original quantizer, scales, or outputs. Native low-bit training is not evidence that this particular conversion is lossless.

### Qwen 3.5 397B-A17B (Q6_K_L and UD-Q4_K_XL)

- Architecture: 512 routed experts, 10 selected per token, and 17B nominally active parameters, according to Qwen's [released configuration](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json).
- Q6_K_L: ~325 GB on CPU in the recorded placement; this is the export label used in the March tests
- UD-Q4_K_XL: ~214 GB total (~140 GB on CPU with expert offload), dynamic quantization

Qwen ran substantially faster than Kimi in these trials. Its smaller activated path and the amount of expert traffic are plausible contributors, but the models also differ in attention, quantization, and placement. The benchmark does not isolate one architectural cause, and Qwen does not have fewer routed experts.

---

## Performance Optimization Record

### Kimi K2.5 Results

These trials used the mixed-capacity DIMMs, whose STREAM result was 30 GB/s. STREAM is a memory benchmark, not a direct measurement of the quantized expert kernel's sustained bandwidth.

| Config | Threads | GPU Layers | Expert Placement | TPS (gen) |
|--------|---------|------------|-----------------|-----------|
| `--cpu-moe -ngl 99` | 64 (container limited) | All non-expert | All experts CPU | 1.15 |
| `--cpu-moe -ngl 99` | 128 | All non-expert | All experts CPU | 3.4 |
| `--cpu-moe -ngl 99` | 64 (128 available) | All non-expert | All experts CPU | 4.0 |
| `-ngl 8 -ts 1,1,1,1` | 64 | 8 full layers | Full layers on GPU | 2.0 |
| `--cpu-moe -ngl 99 -ts 1,1,1,1` | 64 | All non-expert | All experts CPU | 2.7 |

The best recorded Kimi result used 64 threads with all 128 logical CPUs available to the container. It exceeded the 128-thread result, which is consistent with SMT adding contention in this workload; thread count alone does not establish physical-core affinity. The much slower 64-thread result under the earlier container limit also shows why those two configurations should not be treated as equivalent.

Offloading eight complete layers left attention for 53 layers on the CPU and produced only 2.0 t/s. Explicit tensor splitting with CPU experts produced 2.7 t/s rather than 4.0. Additional movement or synchronization could explain the latter result, but profiling would be needed to identify the cost.

### Qwen 3.5 397B — Optimization Progression

#### Phase 1: Baseline (Q6_K_L, mismatched DIMMs, 30 GB/s)

| Config | TPS (gen) | Notes |
|--------|-----------|-------|
| `--cpu-moe -ngl 99 -t 64` | 4.4 | Baseline |
| `--cpu-moe -ngl 99 -t 60` | 4.4 | Same displayed rate at both thread counts |
| Manual `-ot` 16 layers on GPU | 5.5 | Explicit distribution of experts across GPUs |
| + `--no-repack` | 5.5 | No gain visible at the reported precision |

#### Phase 2: UD-Q4_K_XL (mismatched DIMMs, 30 GB/s)

| Config | TPS (gen) | Notes |
|--------|-----------|-------|
| Manual `-ot` 24 layers on GPU | 8.5 | Q4 experts smaller, more fit on GPU |

#### Phase 3: Matched DIMMs (141 GB/s STREAM Copy)

| Config | TPS (gen) | Notes |
|--------|-----------|-------|
| Manual `-ot` 24 layers on GPU | 12.4–12.7 | STREAM Copy increased 4.7×; generation rose less |
| Testing 30+ layers on GPU | In progress | ROCm0 OOM at 8 layers |

### What the trials establish

Manual `-ot` placement was the largest recorded software improvement: moving 16 expert layers onto the GPUs raised Q6 generation from 4.4 to 5.5 t/s. The Q4 export then allowed 24 expert layers to fit, with 8.5 t/s before the DIMM replacement. That comparison combines a smaller quantization with a different placement, so it does not measure the quantization effect in isolation.

I retained `-t 60`, `--prio 3`, and `--no-repack` in the production command. Sixty threads matched the displayed 64-thread result, but this record does not contain a separate controlled speedup for the priority setting. Both rows surrounding `--no-repack` display 5.5 t/s; they do not substantiate the earlier claim of a marginal gain or identify repacking overhead on RDNA 2.

The remaining trials are useful as a record of this build:

- `-fa off` / `-fa on`, `--poll 100`, and `-ub 256` produced no recorded improvement in these tests.
- `--no-op-offload` did not improve throughput and caused OOM in some configurations.
- THP produced no measurable change, and `--mlock` did not help after the pages were cached.
- `-ts 1,1,1,1` with `--cpu-moe`, 128 threads, and the `-ngl 8` full-layer placement were slower in the comparisons above.
- `--cache-type-k q8_0` segfaulted on this ROCm/RDNA 2 setup.
- A 9B speculative draft did not improve throughput. An expensive verification path is one possible explanation; acceptance and verification timings were not recorded here.

Several approaches were unusable at the time of this notebook. In the tested multi-GPU ROCm configuration, `-ncmoe` placed the GPU-resident experts on the last device rather than distributing them as intended. The attempted ik_llama.cpp build failed on ROCm 7.2/RDNA 2 with HIP shim, bfloat16, and warp synchronization mask issues. The Eagle3 path examined was not merged into the build; the notes identified a TensorRT-LLM/Blackwell deployment route, but did not establish a usable ROCm path. These are March implementation observations, not claims about every later release or every implementation of those techniques.

---

## The `-ot` Expert Distribution Technique

llama.cpp's `--override-tensor` flag allowed me to assign selected layers' expert tensors to specific GPUs while keeping the remaining expert bank on the CPU. This supplied the placement control that the tested `-ncmoe` and `--cpu-moe` paths did not provide.

Pattern structure:

```
-ot "blk\.(5[4-9])\.ffn_.*_exps=ROCm0,blk\.(4[8-9]|5[0-3])\.ffn_.*_exps=ROCm1,blk\.(4[2-7])\.ffn_.*_exps=ROCm2,blk\.(3[6-9]|4[0-1])\.ffn_.*_exps=ROCm3,exps=CPU"
```

The specific patterns precede the fallback `exps=CPU`; in this configuration, that ordering is necessary for the intended matches. The expression assigns six layers to each GPU, for 24 in total.

The recorded sizing estimate was roughly 3.5 GB per Qwen Q4 expert layer, with about 22 GB per V620 available after non-expert weights and KV cache. Those are configuration-specific budgets. ROCm0 also carries substantial compute-buffer allocation, so equal numbers of expert layers do not imply equal remaining headroom. The attempt to put eight layers there failed with OOM; future placements must be checked against the allocation on each card rather than aggregate VRAM.

---

## Memory Bandwidth Discovery

The mixed population, 4× 128 GB + 4× 64 GB, measured only 30 GB/s in the large STREAM test. Replacing it with 8× 128 GB matched DIMMs produced:

- STREAM Copy: 141 GB/s, compared with 30 GB/s, a 4.7× increase;
- STREAM Triad: 107 GB/s;
- nominal eight-channel DDR4-2933 bandwidth: approximately 187 GB/s, with Copy reaching about 75% of that figure.

Generation rose from 8.5 to 12.4 t/s on the same recorded software configuration. This is strong evidence that the DIMM population change addressed an important performance constraint, although token generation includes work that STREAM does not measure and therefore did not improve by the same factor.

The measurements do not establish the original interleaving explanation. A 30 GB/s result does not show that most addresses operated in single-channel mode, and the capacity calculation of 512 GB (64 GB × 8 channels) does not by itself reveal the controller's actual address mapping. Channel population, firmware configuration, DIMM compatibility, and the test's placement would need to be inspected to distinguish those mechanisms. The practical result is clear; the precise reason for the mixed population's poor bandwidth remains unverified.

---

## Production Configuration Recorded in March

```bash
llama-server \
    -m /root/models/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/Qwen3.5-397B-A17B-UD-Q4_K_XL-00001-of-00006.gguf \
    -ngl 99 \
    --fit off \
    --prio 3 \
    -t 60 \
    --no-repack \
    -ot "blk\.(5[4-9])\.ffn_.*_exps=ROCm0,blk\.(4[8-9]|5[0-3])\.ffn_.*_exps=ROCm1,blk\.(4[2-7])\.ffn_.*_exps=ROCm2,blk\.(3[6-9]|4[0-1])\.ffn_.*_exps=ROCm3,exps=CPU" \
    --temp 0.7 \
    --min-p 0.01 \
    --ctx-size 16384 \
    --port 8081 \
    --host 0.0.0.0 \
    --special \
    --jinja \
    --alias "Qwen3.5-397B" \
    --no-warmup
```

The recorded performance for this production configuration was **12.4–12.7 t/s generation and 17.4 t/s prompt processing**, with a 16K context allocation. These figures describe this workload and build; the context capacity alone does not specify how many prompt tokens were processed in the test.

---

## Open Items in the March Record

- [ ] Push more expert layers onto GPU — currently 24/60, targeting 30+. ROCm0 VRAM is the constraint (gets disproportionate compute buffer allocation).
- [ ] Test larger context sizes (32K, 64K, 128K), measuring the resulting KV and workspace allocations against the expert VRAM budget
- [ ] Revisit `-ncmoe` when llama.cpp fixes multi-GPU distribution
- [ ] Evaluate Qwen 3.5 122B-A10B as a faster alternative for less demanding tasks
- [ ] Set up systemd service for llama-server auto-start
- [ ] Benchmark Kimi K2.5 on the new matched DIMMs — expect significant improvement from the 30 GB/s baseline
- [ ] Monitor llama.cpp mainline for Eagle3 merge and ROCm tensor parallelism improvements
- [ ] Consider a DDR5 platform upgrade (EPYC Genoa/Turin). Roughly 3× memory bandwidth is a hardware planning estimate dependent on channel count, DIMM population, clocks, and sustained efficiency; it is not a forecast of 3× token throughput.
