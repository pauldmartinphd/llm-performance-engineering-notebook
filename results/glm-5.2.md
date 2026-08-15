# GLM-5.2 — Galactus results

**Model:** GLM-5.2, Unsloth UD-Q4_K_XL — `glm-dsa` arch, 753.86 B params, 435.19 GiB, 75 MoE layers, 256 experts / 8 active, MLA attention.
**System:** Galactus (EPYC 7713, 1 TB DDR4-2933 8-channel at the time, 4 × Radeon Pro V620). Platform: [../hardware/galactus/README.md](../hardware/galactus/README.md); method: [methodology.md](methodology.md); the prefill patch: [../patches/prefill/README.md](../patches/prefill/README.md).
**Investigation:** July 13–21, 2026, plus an August MTP addendum.

## Executive summary

Prefill went from 37.63 to **119.36 t/s** (≈ 3.2×): unclamping `n_ubatch` (2.6×) plus the three-edit scheduler patch (+13.7%). Decode went from 5.15 to **7.1 t/s** (+38%), the gain coming from MTP speculative decode; the non-speculative decode ceiling is the DDR4 bandwidth wall. Both ceilings are explained by measurement, not conjecture.

## Decode

| Configuration | Best decode | Conditions |
|---|---|---|
| Baseline (build 9942 + ZenDNN) | 5.15 t/s tg128 | -ngl 99 -ot exps=CPU, -t 64, ub clamped |
| Hybrid -ot exps=CPU (build 10001) | 5.53 t/s tg64 | -t 24–32 |
| Fitter (no manual placement, ~106 GiB VRAM filled) | 6.01 t/s tg128 | 15–16 expert layers resident |
| CPU-only denominator | 3.87 t/s tg64 | -ngl 0 |
| **MTP speculative decode (n=2)** | **7.1 t/s** | `--spec-type draft-mtp --spec-draft-n-max 2`, ~Aug 1 |

Decode is bounded by DRAM bandwidth: decode reads ≈ 13.77 GB/token, and the two-term model (~90 ms constant + bytes ÷ 152 GB/s) predicted 5.5 / 6.2 / 3.9 t/s vs measured 5.53 / 6.01 / 3.87. The thread sweep peaks at t=24–32 and collapses into SMT (t=96: 2.76; t=128: 1.29). The GPUs are worth +43–55% on decode over CPU-only.

## Prefill

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

The remaining prefill ceiling is arithmetic, not scheduling: attention is quadratic (≈ 26.3 s of a pp32768 pass, 72%), so the gains decay with depth.

## MTP (blk.78) — corrected

At the July compilation the loader flagged blk.78 TENSOR_SKIP, so `--spec-type draft-mtp` could not work for glm-dsa. **Upstream later added glm-dsa MTP support**; the blk.78 NextN head loads from the existing Unsloth quant (no re-download). Measured ~Aug 1: **7.1 t/s at n=2 (+31%)** — the first decode gain of the project, and above the 7–10 t/s reading-speed threshold. n=1 → 6.8, n=2 → 7.1, n=3 → 6.9. See the DSpark addendum in the lab notebook.

## Production configurations

- **Decode-first (chat):** the fitter (no manual placement) at 6.01 t/s, or MTP n=2 at 7.1 t/s.
- **Prefill-first (long context, RAG, agents):** patched build, `-ngl 99 -ot exps=CPU -b 8192 -ub 8192 -fa 1 -t 32` — 119.36 t/s pp8192.

## GLM-specific dead ends

- Resident experts on ROCm1/2 via `-ot`: 17% worse than all-CPU op_offload (resident-weight path ≠ op_offload path).
- DFlash speculation: parked — no GLM-5.2 draft model exists; projected ~11–12.5 t/s if one appears.
- `llama-bench -d` KV-restore crash at 16,384 cells (`hipMemcpyAsync` illegal access) — unreported upstream.
