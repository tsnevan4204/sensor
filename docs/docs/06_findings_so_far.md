# 06 · Findings so far (validated)

Source: `results/2026-11-06_E007_validation/` (validation_log.txt, validation.png, validation_results.json).

## Correctness

- **Forward model:** the exact-star DEC matches the transfer-matrix reference to **9e-15**, for any cell count.
- **Gradients:** the adjoint Jacobian matches central finite differences to **1.9e-7** relative, which is the limit of the finite differences themselves.
- **40-digit check:** in 40-digit arithmetic the DEC and the transfer matrix agree to 3e-41.

## Identifiability (Fisher analysis at the true wall, 40 dB)

- **Every parameter free, drywall εr not shared:**
  - Relative singular values span 1 → 5e-5.
  - insulation t vs εr correlation is −0.999: only the optical thickness √ε·t is well determined.
  - Back drywall εr CRB = 0.088 (poor).
- **Drywall εr/tan δ shared:** the back drywall t CRB improves from 0.2 mm to 22 µm, but a discrete ambiguity remains (below).
- **Wet insulation:** the back of the wall is invisible (|∂Γ/∂t_ins| ≈ 9e-14, |∂Γ/∂t_back| ≈ 1e-14). Fix those parameters; don't estimate them.

## L1 dry wall: Monte Carlo, 10 random walls

- **Success:** 9 of 10 runs reach a noise-level fit (χ²/n = 0.84–1.05).
  - The failure (seed 4, front drywall about 2.7 mm off) had χ²/n = 1.58, so the misfit flagged it.
- **Front drywall:** 15–40 µm errors; err/CRB within about 1–2 except the failed run. εr to about 1–2%.
- **Standoff:** about 1 µm.
- **Insulation:** t errors up to 0.5 mm and εr up to 0.013, but the **optical thickness error is ≤ 0.07 mm**. Only one combination is identifiable from one band.
- **Back drywall:** **3–5 basins within Δχ² < 4**, spaced λ/(2n) ≈ 1.17 mm. The truth was inside the plausible set in 9 of 10 runs.
  - Cause: its faces are unresolved and its echo is weak (−24 dB).
  - The local CRB (22 µm) is the width of one basin, not the global uncertainty.

## L2 wet insulation (θ = 0.05)

- **Single scan, front drywall unknown:** the profile likelihood χ²(θ) has two equal minima, near 0.05 and at about 0.20.
  - The front drywall's t/εr can absorb the phase change of the drywall/insulation reflection.
  - Monte Carlo runs happened to land near 0.05 (θ = 0.047–0.052), but this is not guaranteed.
- **Front drywall known from a dry-patch scan** (differential measurement): a single sharp minimum, θ = 0.0500 ± 0.0011, matching the CRB of 0.0010.
- **Mismatch, water at 30 °C (model assumes 20 °C):** negligible bias (−0.0002).
- **Mismatch, insulation host εr 1.30 (model assumes 1.15):** **bias +0.0103 (20%)**, with χ²/n only 1.16. The dry-material model matters more than the water model, and χ² is a weak alarm.

## L3 wicked profile (8 cells of θ in the front drywall)

- **Singular values:** 1, 2.7e-4, 7e-5, … so the data supports about **one** number (roughly total water).
- **Smoothness (Tikhonov on first differences), λ by discrepancy principle (1e6):** monotone trend, rms error 0.035, mean water 0.030 vs 0.045 true.
- **Total variation (λ = 1e4):** a step, rms error 0.036, mean 0.023.
- **Conclusion:** neither recovers the shape. Use a 2-parameter physical profile (total water + decay length), or add bandwidth (a second band).

## Mistakes made and fixed (keep these in mind)

1. **A 1e-10 "round-off" residual was actually a constant mismatch:** μ₀ = 4π·1e-7 used next to η₀ = 376.730313668. Fixed with μ₀ = η₀/c.
2. **Plain random multi-start converged to the wrong wall in 23 of 24 starts** (cycle skipping). Fixed with:
   - time-gated layer stripping
   - deterministic lattice starts every half phase period
   - joint LM
   - cycle hopping
   - an ambiguity scan
3. **Standoff guessed from the range-profile peak was off by up to 6 mm**, because the merged front/back faces shift the peak, while the bounds were ±3 mm. Fixed by widening to ±10 mm and **profiling the standoff exactly**: it only enters as the phase ramp e^{−2jk₀d}, so the misfit on a fine grid costs one forward solve plus one matrix product.
4. **Benchmark truth equal to nominal** flattered lattice starts centred on nominal. Fixed by randomizing the truth.
5. **The wet case held θ fixed in the gated front stage**, but θ sets the front-interface reflection. Fixed by including θ in that stage.

## Compute (current, unoptimized, 1 CPU core)

- Forward + Jacobian: about 17 ms for 256 frequencies and ~100 cells (vectorized Thomas solve).
- One L1 inversion: **about 80–90 s**. That is about 66 gated LM runs from lattice starts, plus a joint LM, cycle hops and the ambiguity scan.
- **This is the main thing to cut** (see `08_low_compute_toolbox.md`).

## Update 2026-11-09 (E008–E012b, S001)

- **The benchmark was flattering.** In B1-dry-v1 the loss tangents equal the model's fixed values. With tan δ randomized and estimated (B1-dry-v2), the E007 method succeeds on 2/20 walls with 12/20 **silent** failures. With tan δ held at a wrong nominal value it succeeds on 0/20. Within the right basin the E007 error bars are well calibrated (err/CRB std 0.8–1.7). The old large err/CRB values were wrong-basin answers.
- **Search speed.** Coarse-frequency early-stopped search with optical-thickness coordinates, kept inside the prior box (E009 + E011b), reaches the same or better outcomes with 11× less work (about 12 s instead of 153 s per inversion). Search failures fall from 7 to 1 per 20 walls.
- **Information limit.** With tan δ unknown, one 77–81 GHz band and 40 dB, the front drywall is identified. The **cycle counts of the insulation and back drywall are not**: wrong basins 1–5 periods away fit as well as or better than the truth (Δχ² 0.2–6). Integer least squares cannot resolve this because the profiled χ² over the integers is nearly flat. A fixed Δχ² < 4 set under-covers when there are many basins (look-elsewhere effect).
- **Matrix pencil.** It super-resolves the front drywall's faces (within 1.73 periods on all 50 held-out walls). The insulation echo merges with the back-drywall echo at this SNR, so it does not help there.
- **Stability (S001, doc 11).**
  - 1D learned stars that are symmetric and cell-passive in their *modal* admittances give a uniquely solvable system, with a k- and mesh-independent bound.
  - In 2D, passivity alone is not sufficient. A full-side absorber or strict loss is needed, and β ≈ 1.2/k.
  - 2D needs learned stars because the local Trefftz space is infinite-dimensional. Exact fine-relative stars are dense Schur complements, and coarse learned stars need patch stencils (m ≈ log k).

## Update 2026-11-09, later (E013–E019): current best 1D method

**Best method: two bands (77–81 + 22–26 GHz) + E019** (`benchmark.py` method `e019dual`):
- coarse early-stopped search;
- optical-thickness coordinates kept inside the prior box;
- standoff box = one range-resolution cell;
- fallback cascade for χ²-flagged walls;
- no priors.

On 100 held-out honest-benchmark walls (loss unknown): **100/100 correct, 0 silent, 0 flagged.** Median errors:
- front drywall 7 µm, back drywall 6 µm, insulation 0.07 mm;
- εr 0.1–0.2 %;
- tan δ 1–2e-4;
- error bars exactly CRB-calibrated (err/CRB std ≈ 1);
- about 10 s per wall on 1 core.

**Single-band hardware (77–81 GHz only):** the best is E018b (E015b + range-cell box), which uses the drywall prior (ASTM C36) and an assumed stud-depth prior. It reaches 18/20 correct, 0 flagged, 2 silent; median errors are front 13 µm, back drywall 40 µm, insulation 1.1 mm. Without priors, one band cannot determine the cycle counts of the back layers (E009–E014).

**Lessons:**
- The second band is worth more than any amount of search or computation (E014: wrong-cycle basins vanish).
- Search bounds must come from physics (the range-resolution cell), not from convenience: seed 7's truth was outside a ±10 mm box.
