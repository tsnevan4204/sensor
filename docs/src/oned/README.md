# Layered-wall inversion with exact Hodge stars

| File | What it is |
|---|---|
| `wall_toy.py` | The simulated world: materials (Debye water, CRIM mixing), reference transfer-matrix solver, radar model (`measure`, `measure_metal_plate`), range-profile baseline. |
| `exact_hodge.py` | Teaching version: standard vs exact Hodge stars on the same DEC topology. |
| `wall_inverse.py` | The algorithm: exact-star forward model, adjoint Jacobian, calibration, Levenberg–Marquardt, time-gated layer stripping, cycle hopping, ambiguity scan, Fisher/CRB, regularizers, discrepancy principle. |
| `scenarios.py` | Wall definitions (truths and starting models). Add new scenarios here. |
| `run_validation.py` | Full validation suite; writes `validation_log.txt`, `validation.png`, `validation_results.json`. |

## Pipeline

1. **Calibrate.** Scan a metal plate at a known distance (Γ = −1 exactly) and divide it out: `yc, A = calibrate(y_wall, y_plate, f, d_plate)`.
2. **Describe the wall.** Build a `WallModel` from `LayerSpec`s. Every parameter is a `P(value, lo, hi, free, tie)`:
   - `free=False` fixes it (known from a datasheet or an earlier scan).
   - `tie="other.name"` makes it always equal another parameter (same material).
3. **Forward model.** `G, J = forward(model, f)`. This is the exact-star DEC solve. It matches the transfer-matrix reference to about 1e-14, and J comes from one adjoint solve, matching finite differences.
4. **Fit.** `strip_invert(model, f, yc, sigma, stages)` fits each echo group on its own time gate (`Gate(r_lo, r_hi)`), starting from a lattice of points spaced every half phase period so every basin gets a start. It then runs a joint LM on all data, then cycle hopping.
5. **Report honestly.**
   - `fisher(model, f, sigma)` gives the local error bars (CRB), correlations and singular values.
   - `ambiguity_scan(...)` gives the global picture: every thickness basin whose χ² is within 4 of the best. If more than one basin is listed, the data cannot decide between them.
6. **Profiles (L3).** Make each sub-layer's θ free, add `Regularizer(names, lam, "smooth" | "tv")`, and pick λ with `choose_lambda` (discrepancy principle: the largest λ whose data χ² is still at the noise level).

## Adding a scenario

Write a function in `scenarios.py` that returns a `WallModel` with your layers, bounds, free/fixed choices and ties. Pass it to `strip_invert` with stages whose gates bracket each echo group (optical range = standoff + Σ √εᵣ·t). Nothing in `wall_inverse.py` needs to change.

## Exactness

- The forward model has no discretization error for piecewise-constant layers: the stars are exact for any cell size.
- Cells are kept below 0.9π of phase only so the stars stay positive and away from the sin(kℓ) = 0 pole.
- Every tridiagonal solve is checked by its residual. Any failing frequency is re-solved with LAPACK's pivoting solver.
- What is *not* exact: the estimates. They carry noise (quantified by the CRB) and possible discrete ambiguity (quantified by the ambiguity scan). They also carry model mismatch when the true material law differs from the assumed one (tested in section 4 of the validation).
