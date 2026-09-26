# GLM-5.2 — Galactus results

## August baseline and MTP (2 TB, build 3653e6d6d, stock scheduler)

[Entry 12](lab-notebook/12-common-baseline-2tb.md) normalized all five models on one build. GLM-5.2 stock measured pp8192 95.99 ± 3.36 and tg128 5.30 ± 0.00 (t=32, no prefill patch; 119.36 remains the July patched-build figure).

MTP was repeated with llama-cli, the ZFS prompt, and greedy decoding on the same build:

| Draft depth | Decode (t/s) |
|---|---|
| n=1 | 5.9 |
| n=2 | 6.3 / 6.9 (6.6 ± 0.3) |
| n=3 | 6.3 |

The n=2 setting remained the selected configuration. Its reported gain was +25% ± 6 over baseline, compared with +31% in the earlier test, with more separation from n=1 than before.

The repeated speculative runs varied by about 9% despite identical token streams. The selected August configuration produced **6.6 ± 0.3 t/s** with MTP n=2.

## Earlier measurements

The specifications and configurations below describe the earlier runs. They do not replace the August conditions above.

**Model:** GLM-5.2, Unsloth UD-Q4_K_XL — `glm-dsa` arch, 753.86 B params, 435.19 GiB, 75 MoE layers, 256 experts / 8 active, MLA attention.
**System:** Galactus (EPYC 7713, 1 TB DDR4-2933 8-channel at the time, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md); the prefill patch: [../patches/prefill/README.md](../patches/prefill/README.md).
**Investigation:** July 13–21, 2026, plus an August MTP addendum.

### July investigation

Prefill rose from 37.63 to **119.36 t/s**, a factor of about 3.2: unclamping `n_ubatch` gave a factor of 2.6, and the three-edit scheduler patch added 13.7%. Decode rose from 5.15 to **7.1 t/s** (+38%), with the final figure coming from the later MTP test. These figures span builds and configurations; the controlled scheduler comparison was 104.97 to 119.36 t/s. DRAM bandwidth accounts for a substantial part of non-speculative decode time.

### Decode

| Configuration | Best decode | Conditions |
|---|---|---|
| Baseline (build 9942 + ZenDNN) | 5.15 t/s tg128 | -ngl 99 -ot exps=CPU, -t 64, ub clamped |
| Hybrid -ot exps=CPU (build 10001) | 5.53 t/s tg64 | -t 24–32 |
| Fitter (no manual placement, ~106 GiB VRAM filled) | 6.01 t/s tg128 | 15–16 expert layers resident |
| CPU-only denominator | 3.87 t/s tg64 | -ngl 0 |
| **MTP speculative decode (n=2)** | **7.1 t/s** | `--spec-type draft-mtp --spec-draft-n-max 2`, ~Aug 1 |

DRAM bandwidth bounds decode. Decode reads about 13.77 GB per token, and the two-term model (about 90 ms constant plus bytes ÷ 152 GB/s) predicted 5.5 / 6.2 / 3.9 t/s against measured 5.53 / 6.01 / 3.87. The thread sweep peaks at t=24–32 and collapses into SMT (t=96: 2.76; t=128: 1.29). The GPUs are worth +43% to +55% on decode over CPU-only.

### Prefill

The ubatch ladder (op_offload on, exps=CPU, -b 8192):

| n_ubatch | pp8192 (t/s) |
|---|---|
| 512 | 25.90 |
| 1024 | 41.68 |
| 2048 | 62.64 |
| 4096 | 84.62 |
| 8192 | 104.97 |

With the scheduler patch (see [../patches/prefill/README.md](../patches/prefill/README.md)):

| Build | pp8192 | pp16384 | pp32768 |
|---|---|---|---|
| Stock, ub 8192 | 104.97 | — | — |
| Edit 2 only (distribution) | 105.71 (null) | — | — |
| **Edit 2+3** | **119.36** | 86.11 | 55.76 |

The timing analysis attributes much of the remaining long-context prefill cost to arithmetic. Attention is quadratic (about 26.3 s of a pp32768 pass, or 72%), so the gains decay with depth.

### MTP (blk.78), corrected

At the July compilation, the loader flagged blk.78 as TENSOR_SKIP, so `--spec-type draft-mtp` could not work for glm-dsa. Upstream later added glm-dsa MTP support, and the blk.78 NextN head now loads from the existing Unsloth quant with no re-download. Measured around August 1, it reached **7.1 t/s at n=2 (+31%)**, above the 7–10 t/s reading-speed target used during the investigation. The depth curve was n=1 → 6.8, n=2 → 7.1, n=3 → 6.9. See the DSpark addendum in the lab notebook.

### Configurations selected during the earlier investigation

- For decode-first use (chat), use the fitter (no manual placement) at 6.01 t/s, or MTP n=2 at 7.1 t/s.
- For prefill-first use (long context, RAG, agents), use the patched build with `-ngl 99 -ot exps=CPU -b 8192 -ub 8192 -fa 1 -t 32`, which gives 119.36 t/s at pp8192.

### GLM-specific negative results

- Placing resident experts on ROCm1/2 with `-ot` ran 17% worse than all-CPU op_offload, because the resident-weight path is not the op_offload path.
- DFlash speculation was set aside because no GLM-5.2 draft model was available during the investigation; the projected 11 to 12.5 t/s was not measured.
- `llama-bench -d` crashes on KV restore at 16,384 cells (a `hipMemcpyAsync` illegal access); the notebook records it as unreported upstream at the time.
