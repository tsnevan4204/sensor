# Experiment log

Newest entries go at the bottom. E000–E007 were reconstructed from the original chat sessions (Nov 3–6, 2026).

## E000 — Simulated world + reference solver  (2026-11-05)
**Change:** `wall_toy.py`: Debye water, CRIM mixing, transfer-matrix reference, radar model, range-profile baseline.
**Gates:** TMM vs half-space formula and Airy slab formula: exact; |Γ| ≤ 1 on all scenes.
**Results:** range profile cannot resolve drywall (optical 2.05 cm < 3.75 cm resolution). 5% wet insulation erases the back-wall echo. The graded-moisture scene produces a −33 dB false sidelobe peak.
**Next:** DEC forward model.

## E001 — Standard vs exact Hodge stars  (2026-11-05)
**Hypothesis:** exact (Trefftz) stars make the 1D DEC exact with one cell per layer.
**Change:** `exact_hodge.py`.
**Results:** exact stars at 4 nodes match the TMM (initially reported as 2.7e-10; see E004). Standard stars: 160 cells/layer gives 4.7e-2 error; 640 cells/layer gives 1.5e-2.
**Conclusion:** accepted. The standard star's phase error (κ−k ≈ k(kℓ)²/24) accumulates over about 300 rad of round-trip phase.

## E002 — Vectorized exact-star forward + adjoint Jacobian  (2026-11-05)
**Change:** `wall_inverse.forward`: a tridiagonal (Thomas) solve vectorized over frequencies, residual-checked with a LAPACK pivoting fallback; adjoint with a second right-hand side; analytic derivatives of the stars and CRIM.
**Gates:** adjoint vs FD relative difference ~2e-8 to 2e-7. Forward + Jacobian about 17 ms at 256 frequencies.
**Note:** wet insulation makes the back-wall Jacobian ~1e-14 (invisible). The first FD check "failed" only because both values were zero.

## E003 — Calibration + random multistart LM  (2026-11-05)
**Results:** 105 s; 23 of 24 starts converged to a wrong wall (χ² 1118 vs about 512 expected). Cycle skipping.
**Conclusion:** rejected as a global strategy.

## E004 — μ₀ constant bug  (2026-11-05)
**Finding:** the 1e-10 residual came from μ₀ = 4π·1e-7 used with η₀ = 376.730313668 (2019 SI mismatch), not round-off. With μ₀ = η₀/c the forward model matches to 3e-15.

## E005 — Time-gated layer stripping + cycle hopping + parameter tying  (2026-11-05)
**Results:**
- Untied, χ² = 447 ≈ noise, yet two parameters were about 7 CRB off. A genuine ambiguity: the alternative wall's prediction differs by 0.0014 rms, below the noise of 0.007.
- With drywall εr/tan δ tied: the back drywall t shows a discrete ambiguity, with ±k·1.17 mm basins all within Δχ² < 4.
**Change:** `Gate`, `Stage`, `strip_invert`, `cycle_hop`, `ambiguity_scan`, `P.tie`.
**Conclusion:** report plausible-basin sets, not a single number.

## E006 — Wet insulation (L2)  (2026-11-06)
**Findings:**
- θ was held fixed in the gated front stage, so the fit landed in a wrong basin (χ² 1107 vs truth 434). Fixed by putting θ in that stage.
- Then a physical degeneracy: θ = 0.20 fits as well as 0.05.
- Fix: a differential measurement, with the front drywall known from a dry-patch scan.
- Standoff lattice spaced a whole period could miss the basin. Fixed with quarter-period standoff starts.
- Benchmark truths at nominal flattered the method. Randomized via `scenarios.truth_random`.

## E007 — Full validation  (2026-11-06)
**First run:** only 3 of 10 random dry walls fit to noise. Cause: the range-peak standoff guess was off by up to 6 mm, outside the ±3 mm bounds. Fix: ±10 mm bounds plus exact standoff profiling (`best_standoff`).
**Final run:** see `docs/06_findings_so_far.md` and `results/2026-11-06_E007_validation/`.
- 9 of 10 at noise level; the failure was flagged (χ²/n = 1.58).
- Back drywall: 3–5 plausible basins; the truth was inside the set in 9 of 10.
- L2 with a known front: θ = 0.0500 ± 0.0011.
- Host εr mismatch: +20% bias.
- L3 profile: about one identifiable number.
**Compute:** about 85 s per L1 inversion. **Main target for the next experiments.**
**Next:** roadmap Phase 1 item 1 (cut search cost to ≤ 1 s), then item 2 (unknown layer count, B5-family benchmark).
