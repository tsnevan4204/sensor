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
