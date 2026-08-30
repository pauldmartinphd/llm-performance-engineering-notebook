# P620 — platform *(codename TBD)*

x86 workstation in the fleet, same architectural family as [Borg](../borg/README.md) and [Galactus](../galactus/README.md) — a Threadripper Pro platform with discrete GPUs and 8-channel system DRAM. Of all the new machines, this one inherits the Galactus/Borg method most directly: same memory-and-offload shape, so the CPU-MoE method and the prefill patch should apply with the least adaptation.

**Not yet functional.** Acquired BIOS-corrupted for $400 and under repair. No fleet codename assigned yet (the fleet uses Marvel names); filed as `p620` until named. Figures marked planned are targets, not measurements.

## Compute

Lenovo ThinkStation P620. AMD Threadripper Pro on the WRX80 platform (8-channel). **Exact CPU model TBD** — the P620 shipped with Threadripper Pro 3945WX–3995WX (Zen 2); record the installed part once the board runs. A BIOS repair may also decide whether a 5000WX-series (Zen 3) upgrade is possible — worth checking, since Zen 3 would change the CPU-side constant in the decode model.

## Memory

**Planned: 512 GB DDR4-3200 ECC RDIMM, 8 channels** (8 × 64 GB, one DIMM per channel — the P620's 1DPC design tops out at 512 GB). **Not yet installed.**

Theoretical bandwidth at 8-channel DDR4-3200: **204.8 GB/s** (8 × 3200 MT/s × 8 bytes). That is ~9% above Galactus's DDR4-2933 theoretical (187.7 GB/s). At Galactus-like ~81% efficiency that would imply very roughly **~165 GB/s real** — an estimate only, not a measurement; the STREAM sweep (RFO-corrected, per methodology §1) is the prerequisite before any decode budget here is trusted.

## GPU

**TBD.** No inference GPU recorded yet. The WRX80 platform has the PCIe lanes for multiple discrete cards (the Galactus/Borg pattern); record cards and VRAM once chosen.

## Storage

**TBD.**

## Status

- **Under repair — BIOS corrupted at purchase.** First priority is a working board; no benchmarks are possible until then.
- Once running, the closest machine to Galactus and Borg: same 8-channel DDR4 + discrete-GPU shape, so the offload method, the split-histogram check, and the prefill patch should transfer with minimal change.

## Open items

- Repair the BIOS / recover the board.
- Install the 512 GB; confirm all 8 channels populate and train at DDR4-3200.
- Choose and record the inference GPU(s).
- STREAM baseline, then a shared-model comparison against Galactus and Borg (all three are the same architectural family — a clean cross-machine test of the two-term model).
- Assign a fleet codename.
