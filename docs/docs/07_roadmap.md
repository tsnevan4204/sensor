# 07 · Roadmap: general 1D → 2D → 3D

Priority order within each phase is top to bottom. Each item should become one or more experiments (see `09_experiment_loop_protocol.md`).

## Definition of "very strong" (targets to converge to)

| Metric | 1D target | 2D target |
|---|---|---|
| Success rate on randomized benchmark (χ²/n ≤ 1.2 and truth inside the reported plausible set) | ≥ 98% | ≥ 90% |
| Silent failures (bad answer with χ²/n ≤ 1.2 and truth outside the set) | 0 | ≤ 1% |
| Error calibration: std of err/CRB within a basin | 0.8–1.5 | 0.7–2 |
| Inversion time on 1 CPU core | ≤ 1 s (from about 85 s now) | ≤ 60 s per cross-section |
| Learned parameters (total) | ≤ 10k | ≤ 50k |
| Training data | ≤ 1e4 simulated samples, ≤ 10 min CPU | ≤ 1e4 fine-mesh cell solves, ≤ 1 h CPU |
| Forward exactness | ≤ 1e-12 vs reference (piecewise-constant layers) | converged at the predicted rate; error < noise/10 |

## Phase 1: a 1D algorithm that generalizes

1. **Cut the cost of the global search** (85 s → ≤ 1 s). Ideas, cheapest first:
   - **Variable projection.** Standoff (done by grid), overall amplitude and a polynomial A(f) residual are "easy" parameters; eliminate them analytically or by grid.
   - **Echo-group decoupling with closed-form slab responses.** Grid-search the per-slab optical thickness with the exact slab formula (vectorized: 1e5 candidates in < 1 s), then a few LM runs.
   - **Fewer frequencies in the search phase** (16–32 frequencies give the same basins), then refine on 256.
   - **Amortized initialization:** a tiny model (≤ 5k parameters) that maps the calibrated spectrum or range profile to basin indices and starting values. It is trained only on simulated walls. LM then guarantees the final answer, so mistakes by the initializer cost time, not correctness.
   - Reuse factorizations across LM trial steps; low-rank updates when only one layer changes.
2. **Unknown number of layers** (model selection):
   - option (a): BIC / likelihood-ratio tests over N = 1…6;
   - option (b): a fine grid of exact cells with a sparse jump (TV / ℓ0-like) prior on ε(z), then merge cells.
   Benchmark families: drywall, plywood, OSB, brick, concrete, tile, air gaps, insulation types, water films, metal backing (PEC/impedance boundary).
3. **Second band and multi-band identifiability.** Add 3–10 GHz UWB or 57–64 GHz.
   - Quantify with the Fisher matrix whether it breaks the t/εr trade-off and the back-drywall cycle ambiguity.
   - Exact stars handle any band with no change.
4. **Oblique incidence, TE/TM, multi-angle (synthetic aperture):** exact stars with k_z = k cos θ and the matching wave impedances. This is a 1D model of real antenna beams.
5. **Physically parameterized profiles for L3:** total water + decay length (2 parameters) and wicking-front models. Compare with regularized cell profiles.
6. **Learned components (small, simulated data only):**
   - **Coarse DDEC star** for continuous profiles: s_θ, m_θ as functions of (kℓ, θ, ∂θ/∂z). Grade against exact thin-cell stars; test extrapolation outside the training band.
   - **Moisture closure** with Debye/Lorentz pole structure (causal and passive by construction) replacing CRIM. Later to be fit to lab samples.
   - **Sensor model:** low-dimensional A(f) and antenna pattern.
7. **Robustness:** model-mismatch suite (host εr wrong, mixing law wrong, temperature, surface roughness approximated as an effective layer, calibration-plate distance error). Detect mismatch beyond χ² (residual whiteness tests, profile-likelihood curvature).

**Exit criterion for Phase 1:** the 1D targets in the table above are met on the full randomized family benchmark.

## Phase 2: 2D cross-sections (TE and TM, frequency domain)

1. **2D DEC forward solver**, structured (Yee) mesh first, then Delaunay triangles meeting Hu's conditions.
   - Sparse direct solve per frequency (`scipy.sparse.linalg.splu`); reuse the LU for the adjoint (the matrix is complex symmetric).
   - **PML as complex-stretched Hodge stars** (topology stays exact).
2. **Exact references for validation:**
   - Mie series for a dielectric or PEC circular cylinder (analytic).
   - Layered half-space with the 1D code (reduces exactly).
   - A converged fine mesh.
   - Check convergence rates.
3. **Learned coarse cell stars (DDEC / data-driven Whitney forms):**
   - Training data = Schur complements of fine cell solves (no measurements).
   - Parameterize so the result is symmetric and passive and depends on frequency through a pole-residue (rational) form.
   - Measure coarse-grid phase error vs the standard star.
   - **This is the core DDEC contribution:** coarse cells larger than λ/10 with correct phase velocity.
4. **Hybrid:** layered-media Green's function or 1D exact stars for the planar wall, and 2D DEC only in the region of interest (studs, pipes, wet patches). Couple through Whitney traces (tangential fields).
5. **Inversion:** ε, σ maps or parametric shapes (level sets, circles for pipes, rectangles for studs), with adjoint gradients, layer stripping in time gates, and ambiguity scans. Multi-static or synthetic-aperture data.
6. **Model-order reduction across frequency** (Krylov/Padé or reduced basis): a few solves stand in for 256 frequencies.
7. **Stability of the indefinite discrete problem:** test inf-sup numerically; consult Kaibo Hu's theory.

## Phase 3: 3D (feasibility only, for now)

- Nédélec / Whitney 1-form edge elements (3D DEC); PML; sparse direct solvers become the bottleneck.
- Options: 2.5D (invariance along one axis with a Fourier transform), hybrid integral equations, reduced coarse cells learned in 2D/3D.
- Decide after Phase 2 whether 3D is tractable within the compute budget.
