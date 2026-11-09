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

## E008 — Honest benchmark B1-dry-v2: loss tangents randomized, basin-joint scoring  (2026-11-09)
**Roadmap item:** benchmark integrity (prerequisite for Phase 1.1). Benchmark-only change: the method is not modified.
**Why:** in B1-dry-v1 (`truth_random`) the true tan δ of drywall and insulation equal the model's fixed nominal values, so loss is known exactly; this violates "starting guesses must not equal the truth". Fisher check at the nominal wall: freeing tan δ raises the insulation-t CRB from 0.29 mm to 1.8 mm and insulation-εr from 0.0074 to 0.047. Also the E007 "truth inside plausible set" test used a fixed 0.3 mm window per thickness, which for the insulation tests a point on a continuous t/εr ridge, not a basin, and it checked parameters one at a time, not jointly.
**Hypothesis (written before the runs):** on v2 the E007 method with tan δ estimated will (a) still fit to noise (χ²/n ≤ 1.2) in ≥ 90% of runs, but (b) its insulation t/εr errors grow ~6× (CRB-consistent), and (c) the joint truth-in-set rate drops below the v1 rate because the ambiguity scan only hops one parameter at a time. With tan δ held at nominal (now wrong), χ²/n rises and errors become biased.
**Change:**
- `scenarios.truth_random_v2`: tan δ drywall ~ U(0.015, 0.045), fiberglass ~ U(0.005, 0.02) (assumed literature ranges), other draws as v1.
- `benchmark.py` (new): data generation, a method registry, one scorer for all methods. Outcomes: success (χ²/n ≤ 1.2 and truth in plausible set), flagged (χ²/n > 1.2), silent (χ²/n ≤ 1.2, truth outside). Truth in set = some plausible basin (Δχ² < 4) has every layer's optical thickness Re(n)·t within λc/4 of the truth (same phase basin in all layers jointly). err/CRB also in optical thickness. χ² at the truth recorded. Compute = wall time per inversion on 1 core (BLAS threads = 1) and forward-solve counts.
- `wall_inverse.ambiguity_scan(..., return_models=False)`: optional return of the full plausible basin models (default behaviour unchanged).
**Gates:** pass (13 tests incl. new `tests/test_search.py`). Regression: v1 + E007 method reproduces E007 exactly (same χ² per seed).
**Results** (seeds 1–10 for v1, 1–20 for v2; 1 core per run; machine shared, so wall time is approximate):

| benchmark / method | success | flagged | silent | best in truth basin | truth fits better (Δχ² < −4) | wall / run | forward solves (J) |
|---|---|---|---|---|---|---|---|
| v1 / E007 (tan δ = truth, fixed) | 9/10 | 1 | 0 | 5/10 | 1 | 131 s | 8122 |
| v2 / E007, tan δ estimated | 2/20 | 6 | **12** | 1/20 | 10 | 153 s | 11185 |
| v2 / E007, tan δ fixed at nominal (wrong) | 0/20 | 9 | 11 | 0/20 | 18 | 125 s | 11495 |

Median |error| of the optical thickness τ, in phase periods (λc/2), and fraction of runs in the right basin (< 0.5 period):

| | front drywall | insulation | back drywall |
|---|---|---|---|
| v1 / E007 | 0.00 (0.90) | 0.01 (0.90) | 0.51 (0.50) |
| v2 / E007 tan δ estimated | 0.02 (0.60) | 1.96 (0.10) | 1.94 (0.20) |
| v2 / E007 tan δ nominal | 0.01 (0.55) | 2.04 (0.25) | 2.98 (0.05) |

On v1 the corrected metric also shows that err/CRB *within the right basin* is well calibrated (std 0.8–1.7). E007's err/CRB std of 51–75 came entirely from wrong-basin answers.
**Failures examined:**
- **Optimizer** (truth fits better by Δχ² > 4): 10/20 with tan δ estimated. Example, seed 1: in the front gate, a wrong wall (front drywall 15.86 mm instead of 13.24) has gated χ² 5.97 vs 9.54 at the truth. With tan δ unknown, the unresolved front echo pair cannot fix (t, εr, tan δ, standoff) from the ~10 real degrees of freedom of the front gate (B·T ≈ 2.4). Stage 1 inherits the error, and the hops (one or two parameters at a time) do not reach the joint basin.
- **Enumeration/identifiability** (truth within the statistical spread of the best but outside the plausible set): the other 10. The truth's basin was never visited, because the per-parameter ambiguity scan does not try joint shifts (seed 1: insulation +1, back drywall −1).
**Conclusion:** accepted as the benchmark (B1-dry-v2 + `benchmark.py` scoring) for all later 1D experiments. The E007 method is **not** robust once loss is unknown: 12/20 silent failures. Cause is the search (layer stripping + single-parameter hops), not the forward model. Holding tan δ at a wrong nominal value is worse still (0/20). The loss tangent must be estimated.
**Next:** E009–E012 (pre-registered below). E012 (joint integer search) targets the silent failures directly.
**Artifacts:** results/2026-11-09_E008_benchmark/

### Pre-registered hypotheses for E009–E012 (written 2026-11-09, before any benchmark run of these methods)
All on B1-dry-v2 seeds 1–20 (tan δ free), each compared with the previous accepted method. Compute is reported as *work* = Σ forward solves × (frequencies / 256), which does not depend on machine load. Wall time is also reported, but this machine is shared with a background agent.
- **E009** (search LMs on every 8th frequency, early stop at Δχ² < 0.01; the 3 best distinct basins and every competitive hop re-converged on all 256 frequencies at tol 1e-12): work ≥ 4× lower than e007_tanfree, outcome counts (success / flagged / silent) within ±1 seed.
- **E010** (matrix-pencil seeding of the front-drywall stage: 9 starts on the measured τ ridge instead of 51 lattice starts; window ±2 periods set from held-out calibration seeds 1001–1050): work ≥ 1.5× lower than E009, outcomes within ±1 seed.
- **E011** (LM in optical coordinates τ = Re(n)·t): median LM iterations per run ≥ 30% lower, outcomes not worse.
- **E012** (integer least squares over joint period shifts of standoff and all thicknesses, candidates from a fitted quadratic, every candidate verified by a profiled LM): silent failures fewer than E011 (target 0); the reported P(best basin) is calibrated, i.e. its mean is within ±0.15 of the observed rate at which the best basin is the truth's; work ≤ 2× E011.

## S001 — Stability of the indefinite discrete problem with learned stars  (2026-11-09)
**Roadmap item:** Phase 2.7 (run in parallel with E008–E012 by a separate agent; reviewed by the main session: Thm A2 proof re-derived, headline numbers checked against results/2026-11-09_S001_stability/log.txt and figures).
**Full write-up:** docs/11_indefinite_stability.md

**Hypotheses** (set before running; threshold in brackets):
- H1: the 1D Green's-function bound of Thm A3 holds for exact and for passive learned stars [ratio ≤ 1 on all random walls].
- H2: the 2D DEC solver converges at $O(h^2)$ against an exact plane wave and against the Mie series [observed rate in [1.9, 2.1]].
- H3: $\beta_N\sim1/k$ for standard and blended stars [$k\beta_N$ varies < 2× over $k=5\ldots160$].
- H4: passivity alone does not imply invertibility in 2D [find a passive singular example with $\sigma_\min<10^{-12}$].
- H5: singular uniform star perturbations are always active [Im τ > 0 for all computed τ].

**Change.** New folder `src/stability/` (no existing file modified):
- `oned_stability.py` (1D theory tools)
- `dec2d.py` (2D Yee-grid DEC: TE $E_z$, impedance BC, PML stars, Whitney/Q1-consistent and blended stars, learned factors, real-form LU)
- `mie.py`, `infsup.py`, `cells.py`
- experiment scripts `run_s001_{1d,2d,cells}.py`

New tests in `tests/test_stability.py` (19 tests) and `tests/conftest.py` (pins BLAS to one thread). New doc `docs/11_indefinite_stability.md`.

**Results.**
- **H1 passed:** max ratio 0.185 (400 walls, exact) and 0.131 (233 walls, learned).
- **H2 passed:** plane-wave rates 2.04 → 2.0001 down to $h=1/640$; Mie rates 2.03, 2.01, 1.93.
- **H3 passed:** $k\beta_N\in[1.12,1.33]$.
- **H4 passed:** $\sigma_\min=3\cdot10^{-16}$ (absorber on a nodal line); $\sigma_\min\propto s_\min$ (pocket).
- **H5 passed:** at $k=10$, 20, 40, all τ have Im τ > 0, with $\lvert\tau_\min\rvert\approx4.5/k$.
- **Also:**
  - Proved and verified the mass-blend bound (B7) and the lossy bound (B5).
  - The PML is not passive ($\lambda_\min(\operatorname{Im}_HA)=-13.5$) but $\beta_N=0.14$.
  - The (½,½) blended star is 4th-order isotropic (phase error 3.3e-4 vs 1.7e-2 for DEC at 10 ppw); the error at 10 ppw stays 0.018–0.034 up to $k=160$, vs 0.04–1.14 for DEC.
  - 2D Trefftz certificate: rank 8 > 4; best-star defect ≈ 0.01(kH)².
  - Schur complements: passivity inherited, exactness 4.7e-15.
  - LOD-type localization: decay ≈ 0.2 per coarse cell; $m=3$–4 gives 1e-2 to 1e-3.
- **Compute:** 1D 25 s, 2D 234 s, cells 75 s, tests 8 s (one core).

**What failed or surprised.**
1. SciPy's complex SuperLU is ~800× slower than the real one. Fixed with the real 2×2-block form (`ComplexLU`).
2. OpenBLAS thread oversubscription made small dense SVDs 500× slower. Fixed by pinning threads (conftest).
3. The modal passivity condition is the right one: per-star $\operatorname{Re}b\ge0$ would exclude exact lossy stars ($\operatorname{Re}b/\lvert b\rvert$ down to −0.99999).
4. Learned 1D factors within ±25% of exact stay passband only if $\lvert k\ell\rvert\le\pi/2$.
5. LOD truncation breaks passivity of the consistent coarse operator, by $O(\beta^m)$.
6. Q1 with Ainsworth–Wajid mass blending is not 4th order in 2D; both stars must be blended.

**Next.**
- (a) Learned coarse cell stars in the R2–R4 form trained on Schur complements (roadmap Phase 2 item 3), starting from the (½,½) blend with a patch stencil $m=2$.
- (b) Test the passivity-restoring shift of C4.
- (c) Delaunay meshes: is unique continuation available (Q2)?
- (d) Ask Kaibo Hu Q1–Q6.
**Artifacts:** results/2026-11-09_S001_stability/
