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
- **E011b** (added 2026-11-09, before running; reason: in E011 the LM in τ-coordinates left the physical prior box, e.g. insulation t = 185 mm vs box 80–100 mm, and settled in worse basins): LM in optical coordinates, but any trial step whose physical parameters leave the box is rejected (damping raised, as for a cost increase), so iterates stay feasible. Hypothesis: search failures (truth-basin χ² < returned χ²) back to ≤ E010's count, outcomes not worse than E010, work lower than E010. **E012b** = E011b + integer least squares (same hypothesis as E012).

## E009–E012b — Faster search, pencil starts, optical coordinates, integer ambiguity  (2026-11-09)
**Roadmap item:** Phase 1.1 (cut search cost), evaluated on B1-dry-v2 seeds 1–20 (E008). Hypotheses pre-registered above.
**Change:** `search.py` (generalized LM: early stop `atol`, coarse-frequency search, optical coordinates with or without box), `pencil.py`, `ambiguity.py`, method registry in `benchmark.py`. Each method adds one switch to the previous one.
**Gates:** 33 tests pass. `search.lm` is step-for-step identical to `wall_inverse.levenberg_marquardt`. The optical-coordinate chain rule matches finite differences (< 1e-6). `search_base` reproduces `e007_tanfree` on seed 1 (χ² 428.0392480098).
**New diagnostic** (`results/2026-11-09_E008_benchmark/truth_basin.py`): χ² of LM started at the truth (all frequencies, tan δ free) = the best χ² inside the truth's basin. Each silent failure is then one of:
- **search:** the truth basin is better than the returned answer;
- **enumeration:** the truth basin is within Δχ² 4 but was not listed;
- **ambiguity:** the truth basin is worse by more than 4.
Check: from the truth, LM stays in the truth basin (τ errors ≤ 0.006 period).

**Results:**

| method | success | flagged | silent (search / enum / ambiguity) | work* | wall† | verdict |
|---|---|---|---|---|---|---|
| E008 baseline (E007 method, tan δ free) | 2 | 6 | 12 (7 / 5 / 0) | 11546 | 153 s | — |
| E009 coarse search (32 freq, Δχ² < 0.01 stop, 3 basins polished) | 6 | 2 | 12 (2 / 8 / 2) | 4387 | 48 s | partial: work 2.6× lower (target ≥ 4×), but search failures 7 → 2 |
| E010 + pencil starts (front drywall) | 7 | 2 | 11 (1 / 8 / 2) | 4674 | 51 s | rejected: no work saving (target ≥ 1.5×); harmless |
| E011 + optical coordinates | 2 | 2 | 16 (6 / 8 / 2) | 793 | 9.5 s | rejected: LM left the prior box (insulation 185 mm vs box 80–100) |
| E011b optical coordinates, box kept | 5 | 5 | 10 (1 / 6 / 3) | **1007** | **11.6 s** | partial: 11× less work than baseline; 3 more *flagged* front-stage failures than E010 |
| E012 + integer least squares (on E011) | 6 | 1 | 13 (4 / 5 / 4) | 2366 | 28 s | rejected |
| E012b + integer least squares (on E011b) | 7 | 2 | 11 (3 / 4 / 4) | 6068 | 65 s | rejected: silent not reduced, work 6× E011b, P(best) not calibrated (mean 0.38 vs observed 0.05) |

\*work = forward solves × frequencies/256. †1 core, shared machine.

**Failures examined (the main finding).** Per seed, the best answer's offset from the truth in whole periods (front, insulation, back), for E011b:
- **Front drywall:** offset 0 on 15/20 walls. The other 5 are front-stage failures, all flagged by χ² (χ²/n 3.4–8).
- **Insulation and back drywall:** off by 1–5 periods on most walls. The returned basin fits **better** than the truth basin by Δχ² 0.2–6.4 (seeds 1, 2, 5, 6, 8, 10, 18, 20).

So with tan δ unknown and a single 77–81 GHz band at 40 dB, the cycle counts of the insulation and back drywall are **not identified**. Many basins fit within noise, and which one fits best is decided by the noise. This is an information limit, not a search bug.

It also explains the E012 failure. Integer least squares assumes the profiled χ² is a well-curved quadratic over the integers. Here it is nearly flat: W was indefinite or ill-conditioned on most walls, so the closed-form success rates were undefined. The empirical posterior over a box-truncated set was overconfident.

Also found: with 5–20 near-equal basins, the maximum noise gain over basins routinely exceeds Δχ² = 4, a look-elsewhere effect. A fixed "Δχ² < 4" plausible set is therefore not a calibrated confidence set when there are many basins.

**Conclusion:**
- **Accepted** for speed: E009 + E011b. Search failures fall from 7 to 1, and work falls 11× (153 s → 12 s per inversion).
- **Rejected:** E010 (no gain), E011 (box bug), E012/E012b (resolution cannot come from enumeration when the data do not determine the integers).
- The remaining silent failures are ambiguity at the back of the wall. Only more information can fix them.

**Next:**
- **(E013)** A manufacturing prior on drywall thickness. The ASTM C36 tolerance is ±1/64 in nominal, ±1/32 in local (±0.8 mm), against a basin spacing of 1.2 mm. Expected to remove the back-drywall cycle ambiguity.
- **(E014)** More information for the insulation. Option 1: a second band, judged by the integer success rate rather than the CRB (E008 showed it does not break t/εr). Option 2: two incidence angles, which measure √(ε − sin²θ)·t and so separate t from εr.
- **(E015)** For flagged front-stage failures, fall back to the E010 search. This is cheap because only flagged runs pay.
- **(E016)** Calibrated reporting with many basins: replace the fixed Δχ² < 4 set by posterior mass over an enumerated basin grid, with an explicit prior.

**Artifacts:** results/2026-11-09_E009_coarse_search/, _E010_pencil/, _E011_optical_coords/, _E012_integer_ambiguity/

### Pre-registered: E013 / E013b (2026-11-09, before running)
- **E013** = current method (E009 + E011b) + Gaussian priors on both drywall thicknesses: mean 12.7 mm, sd 0.4 mm (the ASTM C36 nominal tolerance ±1/64 in, used as 1 sd). They enter as MAP residual rows, so all basin comparisons use χ² + prior penalty, and the χ²/n noise test uses the data χ² alone.
  - **Hypothesis:** back-drywall thickness median |error| falls from 2.3 mm to < 0.5 mm, and silent failures fall from 10 to ≤ 5 on B1-dry-v2 seeds 1–20, at work ≤ 1.5× the current method.
  - **Caveat:** the benchmark draws drywall t ~ U(12.1, 13.3) mm. That spread is comparable to the tolerance, so the prior is not wrong for this benchmark.
- **E013b** = E013 + a stud-depth prior on insulation t: mean 88.9 mm, sd 1.6 mm. The sd is **assumed**: PS 20 gives the minimum dressed size and I found no undersize tolerance. The benchmark draws U(85.9, 91.9) mm (sd 1.7), so this arm tests the *best case* for a cavity prior.
  - **Hypothesis:** insulation thickness median |error| falls from 3.5 mm to < 1.5 mm.

### Pre-registered: E014 (2026-11-09, before running)
**Information analysis (no inversion):** for 8 random B1-dry-v2 walls, the expected χ² gap Δ that the best wrong-cycle wall pays, after all other parameters are re-fitted on noiseless data. Hops are ±1/±2 periods of insulation and/or back drywall.
**Configurations:**
- A: today's 77–81 GHz at normal incidence.
- B, C, D: A plus a second band of 57–64, 22–26 or 3.1–10.6 GHz.
- E, F: A plus the same band at 30° (TE or TM).
- G: the same band measured twice (same data budget as B–F).

Exact oblique TMM, gated against the reference at 0°.
**Hypotheses:**
- (a) A has median min-gap < 4, consistent with the ambiguity seen in E009–E012.
- (b) G gains at most 2× (averaging alone does not resolve cycles).
- (c) At least one of B–F reaches worst-case gap > 25, i.e. P(flip) < 1e-2.

## E014 — Which extra measurement resolves the back-of-wall cycles? (information analysis)  (2026-11-09)
**Change:** `results/2026-11-09_E014_information/info_gap.py`: an oblique-incidence TMM (TE/TM; gate: equals `wall_toy.reflection_tmm` at 0° to < 1e-12). For each configuration and 8 random B1-dry-v2 walls, the fit starts at truth + whole-period shifts of insulation and/or back drywall, re-fits all 8 wall parameters (bounded least squares) on noiseless data, and records the χ² the wrong basin pays (Δ). Noise σ is the benchmark value, with 256 samples per band or angle. Runs in 21 s.
**Results** (median over walls of Δ per hop; "vanishes" = the fit from the shifted start slides back to the true wall, i.e. no wrong local minimum exists):

| configuration | insulation ±1 | back drywall ±1 | joint (+1, −1) | worst-wall min Δ | P(flip) worst |
|---|---|---|---|---|---|
| A 77–81 GHz (today) | 1.9–2.5 | **0.2** | 1.8 | 0.0 | 0.46 |
| G same band twice | 3.8–5.0 | 0.4 | 3.7 | 0.1 | 0.45 |
| E + 30° TE | 15.6 | 5.1–5.6 | 18–20 | 0.4 | 0.37 |
| F + 30° TM | 12.3 | 3.9–4.2 | 14–15 | 0.3 | 0.39 |
| B + 57–64 GHz | ~2000 | 246–292 | ~1800 | 61.7 | 4e-5 |
| C + 22–26 GHz | vanishes | vanishes | vanishes | ≥ 344 (±2 hops) | ~0 |
| D + 3.1–10.6 GHz | vanishes | vanishes | vanishes | no wrong basin | 0 |

**Hypotheses:**
- (a) **passed:** today's single band has median min-gap 0.2. It cannot tell the back-drywall cycle apart at all, which matches the E009–E012 failures.
- (b) **passed:** measuring the same band twice only doubles Δ.
- (c) **passed:** every second band resolves the cycles (B worst case Δ = 62; C and D remove the wrong minima entirely).

A 30° view helps the insulation (Δ ≈ 15) but not enough for the back drywall (Δ ≈ 5).
**Why** (math): a one-cycle shift changes the round-trip delay by exactly one carrier period, Δτ = 1/fc. At frequency f this changes the phase by 2πfΔτ = 2π + 2π(f − fc)/fc, so the residual is 2π(f − fc)/fc. Across 77–81 GHz it spans only ±0.16 rad, which is below the noise on the weak back echoes. A second band centred at f₂ adds 2π(fc − f₂)/fc: 0.47π for 57–64 GHz, 1.4π for 22–26 GHz, and 1.8π (varying across the band) for 3.1–10.6 GHz. Lower bands also lose less in the drywall, so the back echoes are stronger.
**Conclusion:** the remaining 1D silent failures are an **information** problem with a clear hardware answer: add a second band. 22–26 GHz (an ISM/automotive band) or UWB 3.1–10.6 GHz remove the ambiguity entirely; 57–64 GHz nearly does. A second angle is a weaker alternative.
**Next:** implement multi-band data in the benchmark and inversion (exact stars need no change), then rerun the full inversion with the chosen band to confirm in practice. The prior (E013) is the fallback for single-band hardware.
**Artifacts:** results/2026-11-09_E014_information/

## E013 / E013b — Manufacturing priors on thickness (single band)  (2026-11-09)
**Change:** `search.GaussianPrior` (MAP residual rows; gate `test_gaussian_prior_rows_and_penalty`). Methods `e013` (drywall priors) and `e013b` (+ stud-depth prior, sd assumed) in `benchmark.py`, built on the current method (E009 + E011b). 34 tests pass.
**Results** (B1-dry-v2 seeds 1–20; errors over walls with χ²/n ≤ 1.2):

| method | success | flagged | silent | front t | insulation t (90%) | back t (90%) | ins εr | back right cycle | ins right cycle | work |
|---|---|---|---|---|---|---|---|---|---|---|
| current (E009+E011b) | 5 | 5 | 10 | 0.015 mm | 3.48 (7.1) mm | 2.32 (4.1) mm | 3.5 % | 13 % | 27 % | 1007 |
| E013 drywall prior | 10 | 7 | 3 | 0.011 mm | 1.78 (10.8) mm | **0.03 (0.64) mm** | 7.0 % | 92 % | 46 % | 877 |
| E013b + stud prior (sd assumed) | **14** | 5 | **1** | 0.013 mm | **1.23 (3.3) mm** | **0.04 (0.21) mm** | 2.8 % | 100 % | 60 % | 665 |

**Hypotheses:**
- **E013 passed:** back-drywall median error 2.32 → 0.03 mm (target < 0.5); silent 10 → 3 (target ≤ 5); work 0.87× the current method.
- **E013b passed:** insulation median error 3.48 → 1.23 mm (target < 1.5).

**Failures examined:** all 5 flagged runs of E013b (seeds 7, 12, 14, 16, 18) are front-stage **search** failures. The truth fits better by Δχ² 97–3900 and χ²/n flags them, so none is silent. The single silent failure is an insulation cycle slip within the stud prior's spread.
**Caveats:**
- The drywall prior is physically sourced (ASTM C36).
- The stud-depth sd is assumed, and matches the benchmark's spread, so E013b is a best case.
- Priors only help when the true wall respects them. A non-standard wall (shimmed or double drywall) would be biased; χ² and the prior penalty must be monitored (see E016).
**Conclusion:** accepted. For single-band hardware, the drywall prior resolves the back-drywall cycles. The new best method is E013 (or E013b where stud depth is known). The insulation is resolved only partly; the full fix is a second band (E014).
**Next:**
- **(E015)** For χ²-flagged front-stage failures, rerun the search with the dense physical-coordinate lattice (E009/E010 settings). Only flagged walls pay. Target: flagged 5 → ≤ 1.
- **(E017)** Multi-band data in the benchmark and inversion (22–26 GHz), confirming E014 in the full pipeline without any prior.
**Artifacts:** results/2026-11-09_E013_priors/

### Pre-registered: E015 (2026-11-09, before running)
**Change:** a cascade on top of E013 (drywall prior) and E013b (+ stud prior). If the data χ²/n > 1.2 (the method's own alarm), rerun with the physical-coordinate coarse search (E009 settings). If still flagged, run the dense full-data search (E007 settings). Keep the lowest posterior cost. Only flagged walls pay.
**Hypothesis:** flagged runs fall from 7 → ≤ 2 (E013) and from 5 → ≤ 1 (E013b), with no increase in silent failures, and median work ≤ 1.3× E013/E013b (the median wall is unflagged).

### Pre-registered: E017 (2026-11-09, before running)
**Change:** benchmark `v2dual` has the same walls and the same 77–81 GHz data as v2, plus a 22–26 GHz band with its own metal-plate calibration and noise (`benchmark.DUAL_BAND`). Code support:
- `wall_inverse.band_segments` and `carrier`: block-diagonal time gates per band; cycle period from the highest band. Single-band paths are unchanged (gates `test_band_segments_and_block_gate`, `test_dual_benchmark_reuses_v2_truth_and_high_band`).
- The matrix-pencil start uses the highest band only. The bug found in the smoke test: the pencil assumes one uniform grid, and on concatenated bands it gave garbage starts.

Method: the current method (E009 + E011b), **no priors**.
**Hypothesis (from E014):** silent failures ≤ 1, back-drywall right cycle ≥ 90%, insulation right cycle ≥ 80%, insulation-t median error < 0.5 mm; work ≤ 2.5× the single-band method (2× the data).

### Pre-registered: E016 (2026-11-09, before running)
**Change:** E015 (drywall prior + cascade), but the plausible set comes from `search.posterior_scan` instead of the Δχ² < 4 scan:
- joint shifts of (insulation, back drywall) over [−3, 3]², plus front-drywall and standoff shifts;
- basins de-duplicated by integer cycle offsets;
- posterior w ∝ exp(−ΔC/2), with C = χ² + prior penalty, a flat prior over basins and equal Laplace volumes (an approximation);
- the reported set is the smallest one holding 95% of the posterior.

**Hypothesis:** truth basin inside the 95% set for ≥ 90% of non-flagged walls (silent ≤ 2/20); calibration: mean reported P(best basin) within ±0.15 of the observed rate at which the best basin is the truth's; work ≤ 2× E015.

## E015 — Cascade fallback for χ²-flagged walls  (2026-11-09)
**Change:** `benchmark.method_cascade`. If the primary method's data χ²/n > 1.2, rerun with the E009 physical-coordinate search; if still flagged, with the dense E007 search. Same priors throughout; the lowest posterior cost is kept.
**Results** (B1-dry-v2 seeds 1–20):

| method | success | flagged | silent | front t | insulation t | back t | work |
|---|---|---|---|---|---|---|---|
| E013 (drywall prior) | 10 | 7 | 3 | 0.011 mm | 1.78 mm | 0.03 mm | 877 |
| **E015** (E013 + cascade) | **15** | **1** | 4 | 0.011 mm | 3.47 mm | 0.065 mm | 1113 |
| E013b (+ stud prior) | 14 | 5 | 1 | 0.013 mm | 1.23 mm | 0.04 mm | 665 |
| **E015b** (E013b + cascade) | **17** | **1** | 2 | 0.013 mm | 1.64 mm | 0.044 mm | 963 |

(Thickness errors are medians over walls with χ²/n ≤ 1.2: 19/20 for both cascades vs 13 and 15 before.)

**Hypotheses:**
- **Passed** for flagged walls: 7 → 1 and 5 → 1.
- **Passed** for work: 1.27× and 1.45× (target ≤ 1.3×; E015b slightly over).
- **Failed** for "no new silent failures": E015 rose 3 → 4 and E015b 1 → 2. The rescued walls now fit to noise, but in several the insulation cycle is wrong while inside the Δχ² < 4 set's blind spot (seeds 11, 16, 19, 20 for E015). This is the reporting problem E016 targets.

Seed 7 stays flagged in every variant (χ²/n 1.20–1.23; the truth fits better by Δχ² 160–176). It is a front-stage search failure that none of the three searches finds; examine it separately.
**Conclusion:** accepted. The cascade turns flagged failures into fits at modest cost. The best single-band method is now E015 (sourced priors only) or E015b (with the assumed stud prior).
**Artifacts:** results/2026-11-09_E015_cascade/

## E017 — Two bands (77–81 + 22–26 GHz), no priors  (2026-11-09)
**Results** (v2dual: same walls and same 77–81 GHz data as v2; current method E009 + E011b, no priors, no cascade):

| | success | flagged | silent | best in truth basin | front t | insulation t (90%) | back t (90%) | εr drywall / insulation | right cycle front/ins/back | work |
|---|---|---|---|---|---|---|---|---|---|---|
| single band, same method | 5 | 5 | 10 | 1/20 | 0.015 mm | 3.48 (7.1) mm | 2.32 (4.1) mm | 0.2 % / 3.5 % | 0.80/0.27/0.13 | 1007 |
| **two bands** | **19** | 1 | **0** | **19/20** | **0.0065 mm** | **0.066 (0.34) mm** | **0.0057 (0.019) mm** | **0.10 % / 0.15 %** | **1.00/1.00/1.00** | 902 |

Within the right basin, err/CRB is calibrated (std 0.6–1.2), so the CRB is now the real error bar.
**Hypotheses:** all **passed**:
- silent 0 (target ≤ 1);
- back and insulation in the right cycle on 100% of fitted walls (targets 90% / 80%);
- insulation median error 0.066 mm (target < 0.5);
- work 0.9× single band (target ≤ 2.5×). Two bands cost about the same as one: less hopping, because the wrong basins do not exist.

**Failure examined:** seed 7 (χ²/n 234, flagged) also fails in every single-band method. Cause: the range-profile standoff guess is 10.8 mm off, while the standoff box is guess ± 10 mm. **The truth lies outside the search box**, the same mistake class as E007's first run. Fix in E018.
**Conclusion:** accepted. A second band turns the 1D layered-wall problem from ambiguous into well posed: µm-level thickness, 0.1% permittivity, no priors, no silent failures, about 10 s per wall. This confirms E014's information analysis in the full pipeline.
**Artifacts:** results/2026-11-09_E017_dual_band/

## E016 — Posterior credible sets (single band, drywall prior + cascade)  (2026-11-09)
**Results:** success 16, flagged 1, silent 3 (E015: 15/1/4). The truth basin is inside the 95% set for 16/19 non-flagged walls (84%). Mean reported P(best basin) 0.42 vs observed 0.30 (within ±0.15). Work 1414 (1.27× E015).
**Hypotheses:**
- **Failed:** coverage target ≥ 90% (silent ≤ 2): 3 silent remain (seeds 11, 16, 19).
- **Passed:** calibration within ±0.15 (0.42 vs 0.30).
- **Passed:** work ≤ 2× E015.

**Failures examined:** in all three, the truth's insulation cycle is off by more than the enumeration window, or its χ² + prior sits just outside the 95% mass. With one band the posterior over insulation cycles is wide, and the equal-volume Laplace approximation is crude.
**Conclusion:** partial. The posterior set is better calibrated than the Δχ² < 4 set, but full coverage with one band needs a wider enumeration (cost). E017 makes this moot for two-band hardware. Keep E016 reporting as the default for single-band use.
**Artifacts:** results/2026-11-09_E016_posterior/

### Pre-registered: E018 (2026-11-09, before running)
**Change:** standoff search box = range-profile guess ± one range-resolution cell, c/2B = 37.5 mm, replacing the arbitrary ±10 mm. Rationale: the peak of an unresolved echo group locates it only to within one resolution cell (seed 7: 10.8 mm off). Option `Opts.standoff_halfwidth`; the default stays 0.010 so earlier experiments reproduce.
**Hypothesis:** seed 7 becomes a noise-level fit on v2dual (E017 method) and with E015b on v2, and no other seed changes outcome.

## E018 — Standoff box = one range-resolution cell  (2026-11-09)
**Results:**
- **Single band, E015b + cascade (`e018b`):** success 18, flagged **0**, silent 2 (E015b: 17/1/2). Seed 7 fixed (χ²/n 0.86). Seed 11 silent → success, seed 15 success → silent: a noise-level insulation-cycle swap, expected with one band.
- **Two bands, no cascade (`e018dual`):** success 18, flagged 2, silent 0 (E017: 19/1/0). Seed 7 still flagged (χ²/n 76), and seed 17 newly flagged (50).

**Hypothesis:**
- **Passed** for single band with the cascade.
- **Failed** for the two-band method without a cascade. The wider box gives the front stage more room to go wrong, and nothing recovers it. Both failures are flagged, not silent.

**Conclusion:** the range-cell box is the right physical bound (it contains the truth), but it needs the cascade as a safety net → E019.
**Artifacts:** results/2026-11-09_E018_standoff_box/

### Pre-registered: E019 (2026-11-09, before running)
Two bands + resolution-cell standoff box + the E015 cascade, with no priors (fallbacks: E009 physical-coordinate search, then dense search).
**Hypothesis:** on v2dual seeds 1–20: success ≥ 19, silent 0, flagged ≤ 1; median work ≤ 1.3× E017.

## E019 — Two bands + range-cell standoff box + cascade, no priors  (2026-11-09)
**Results** (v2dual seeds 1–20): **success 20/20, flagged 0, silent 0, best model in the truth basin 20/20**; the cascade was used on 2 walls; median 10.8 s, work 920 (1.02× E017).

| quantity | median | 90% | max | err/CRB std |
|---|---|---|---|---|
| standoff | 0.4 µm | 0.9 µm | 1.9 µm | 1.00 |
| front drywall t | 6.7 µm | 18 µm | 22 µm | 1.08 |
| drywall εr | 0.11 % | 0.26 % | 0.37 % | 1.11 |
| drywall tan δ | 0.0002 | 0.0006 | 0.0011 | 1.02 |
| insulation t | 69 µm | 0.31 mm | 0.74 mm | 1.08 |
| insulation εr | 0.17 % | 0.68 % | 1.7 % | 1.09 |
| insulation tan δ | 0.0001 | 0.0002 | 0.0003 | 0.95 |
| back drywall t | 6.3 µm | 21 µm | 40 µm | 1.13 |

**Hypothesis passed** (success ≥ 19, silent 0, flagged ≤ 1, work ≤ 1.3×). The errors are exactly CRB-calibrated (err/CRB std 0.95–1.13, roadmap target 0.8–1.5). The insulation's larger absolute error is its t/εr trade-off, which the CRB itself reports.
**Note:** the Δχ² < 4 "plausible basins" list still shows 2–5 entries per wall. These are mostly duplicate convergences to the same basin from different hops (the old scan does not de-duplicate); `posterior_scan` (E016) de-duplicates. To clean up.
**Conclusion:** accepted as the **best 1D method: two bands + E019**. Confirmation on 100 held-out walls follows (seeds 21–120, never used during development).
**Artifacts:** results/2026-11-09_E019_dual_cascade/

### E019 held-out confirmation (seeds 21–120, never used in development)
**success 100/100, flagged 0, silent 0, best model in the truth basin 100/100.** χ²/n 0.91–1.10. Median 10.1 s per wall (1 core), work 881. err/CRB within the right basin: mean |bias| ≤ 0.18, std 0.89–1.11 for every parameter (target 0.8–1.5). 95% upper bound on the failure rate from 0/100 is about 3% (rule of three).
**Roadmap 1D targets** (doc 07), for this benchmark family (three-layer dry stud wall, loss unknown, two bands):
- **Met:** success ≥ 98% (100/100); silent failures 0; err/CRB std 0.8–1.5; forward exactness ≤ 1e-12.
- **Not met:** inversion time ≤ 1 s (10 s now).
- **Not yet tested:** generalization to the full B5 family (unknown layer count, other materials).

### Pre-registered: E020 — Reality-gap study (2026-11-09, before running)
**Change (data only; the inversion model is not changed):** `src/oned/realism.py` is a more physical data generator, with each effect a switch (gates `tests/test_realism.py`; all-off reproduces v2dual to < 1e-12):
- causal constant-Q dispersion (Kjartansson);
- ITU-style power-law conductivity;
- a Gaussian antenna beam (plane-wave-spectrum integral of oblique TE/TM reflections);
- antenna–wall multiple reflections (Lambot form, Hf);
- plate-distance error;
- Kirchhoff surface roughness;
- drywall face paper;
- an insulation air gap;
- calibration-to-scan drift;
- all of these together.

Methods: the best two-band method (e019dual) and the best single-band method (e018b), seeds 1–10 each.
**Hypotheses** (quantitative expectations to compare against):
- **(a)** Causal dispersion: drywall εr differs by 2.3% between 24 and 79 GHz, about 20× the εr CRB. Expect the two-band fit to be flagged by χ² or biased by > 3 CRB; the single-band method little affected.
- **(b)** Beam spreading changes the relative amplitude of later echoes by roughly r_front/r_back ≈ 0.7. Expect large εr biases.
- **(c)** Plate error of 0.5 mm: the standoff estimate shifts by about 0.5 mm (≫ CRB of 1 µm), with other parameters nearly unaffected.
- **(d)** At least one effect produces silent bias (χ²/n ≤ 1.2 with z > 3). These are the dangerous ones and set the priorities.

## E020 — Reality-gap study  (2026-11-09)
**Run note:** the first run (10 seeds, full 3-stage cascades) was stopped after 48 min with no output: flagged walls triggered the dense fallback on many runs. It was rerun with 8 seeds, 2-stage cascades (`e019dual_lite`, `e018b_lite`), incremental progress output, and one added effect (band phase-centre offset 0.3 mm). 1839 s.
**Results:** full table in `docs/12_reality_gap.md` §E and `results/2026-11-09_E020_reality_gap/gap_study.txt`. Summary:
- **Largest effect:** a finite antenna beam (z 30–90; εr bias 7–14%; flagged).
- **Silent** in the single band: roughness (εr 5%), face paper (εr 21%), plate error (standoff only), drift (mild).
- **Flagged** with two bands: causal dispersion, paper, air gap, band offset, antenna feedback.
- **All effects together:** flagged 8/8 in both methods.

**Hypotheses:**
- (a) passed: dispersion flags the two-band fit, z 8–18; single band little affected.
- (b) passed: beam bias is large.
- (c) passed: plate error equals the standoff error exactly.
- (d) passed: silent biases exist. They are mostly in the single band; two bands expose most model errors.

**Conclusion:** before real data, the forward model needs:
1. the antenna-beam angular-spectrum integral plus antenna functions;
2. causal dispersion;
3. thin facings and roughness;
4. a per-band phase-centre treatment;
5. a legal second band.

All keep the exact 1D Hodge stars. Priority list in doc 12 §F.
**Artifacts:** results/2026-11-09_E020_reality_gap/

### Pre-registered: E021 + E022 (2026-11-09, before running)
**E021 change:** `WallModel.law = "cq"`: host law ε(f) = εr(1 − j tan δ)(f/f_ref)^(−α), α = (2/π) arctan(tan δ), f_ref = 79 GHz. This is Kjartansson constant-Q: exactly causal and passive, with the same two parameters, and equal to the old law at f_ref. Gates: Jacobian vs FD < 1e-5; equal to the independent realism TMM to < 1e-12. Method `e021` = e019dual_lite + cq law.
**E022 change:** a legal second band in place of 22–26 GHz: 57–64 GHz (FCC 15.255) or 3.1–10.6 GHz (15.509 wall imaging). Benchmarks `real:<cq|itu>:<dual24|dual60|dualuwb>`, seeds 1–10.
**Hypotheses:**
- **E021:** on real:cq:dual24 (8/8 flagged in E020), flagged ≤ 1/10, silent 0, z ≤ 2.
- **E022:** both legal bands give success ≥ 9/10, silent 0. Dual60 is expected to be slightly weaker than dual24 (E014 worst-case gap 62), dualuwb as good or better.
- **ITU law** (not constant-Q, so a mild mismatch): flagged or z ≤ 3 (no silent z > 3).

### Pre-registered: E021b (2026-11-09, before running)
**Context:** E022 on the ITU-law data (real:itu:dual60) gave 8/10 flagged and 1 silent with the cq model. The ITU table form (ε′ constant, σ = c f^d) is not causal: its loss implies by Kramers–Kronig an ε′ drift of about 2.5% between bands. Real materials are causal but need not be exactly constant-Q.
**Change:**
- **Model:** Jonscher universal response, ε = εr[p + (1 − p − j tan δ)(f/f_ref)^(−α′)] with α′ = (2/π) arctan(tan δ/(1 − p)). It is causal and passive; p = ε∞/εr, and p = 0 gives constant-Q exactly. The drywall p is free (tied across sheets); the insulation stays constant-Q.
- **Truth:** `itucausal`, the ITU loss slope made causal (α′ = max(1 − d, α_cq)).
- Gates: reduction to cq < 1e-13; Jacobian vs FD < 1e-6; the truth law equals the model law with matching p to < 1e-12.

**Hypothesis:** on real:itucausal (dual60 and dualuwb), e021b gives success ≥ 9/10 and silent 0, while e021 (cq) is flagged on most walls. On real:cq:dual60, e021b loses at most 1 success against e021.

## E021 / E022 / E021b — Causal material laws and legal second bands  (2026-11-09)
**Results** (seeds 1–10; "real:" = realism.py data; outcomes success/flagged/silent; best-basin count):

| data | second band | model law | outcome | best basin | notes |
|---|---|---|---|---|---|
| constant-Q | 22–26 | const (E020) | 0/8/0 | — | z 8–18 |
| constant-Q | 22–26 | **cq** (E021) | **10/0/0** | 10 | err/CRB std 0.7–1.6 |
| constant-Q | **57–64** (legal) | cq | **10/0/0** | 9 | 8.0 s/wall |
| constant-Q | **3.1–10.6 UWB** (legal) | cq | **10/0/0** | 10 | insulation t error 85 µm |
| ITU table form (acausal) | 57–64 | cq | 1/8/1 | 6 | the acausal truth cannot be fitted by a causal law |
| ITU slope, causal (`itucausal`) | 57–64 | cq | 10/0/0 | 9 | biases < 1 CRB |
| itucausal | 57–64 | **Jonscher** (E021b) | 10/0/0 | 9 | biases < 0.6 CRB |
| itucausal | UWB | cq | 10/0/0 | 10 | **silent tan δ bias: mean err/CRB +3.1 (drywall), +3.5 (insulation)** |
| itucausal | UWB | **Jonscher** | **10/0/0** | 10 | all mean err/CRB ≤ 0.35 |
| constant-Q | 57–64 | Jonscher | 10/0/0 | 9 | the extra parameter costs nothing measurable |

**Hypotheses:**
- **E021 passed:** the constant-Q law removes the dispersion failure (8/8 flagged → 10/10 success).
- **E022 passed:** both legal bands give 10/10 with no silent failures. 57–64 GHz has 9/10 best-basin; UWB has 10/10 and the smallest insulation error.
- **E021b passed**, after a correction: the ITU failure was caused by the acausal form of the ITU table itself. ITU-R P.2040 §2.1.3 itself says ε′ falls above 5–10 GHz, while Table 3 sets b = 0 for simplicity. With the causally completed ITU slope, the narrow span (57–81 GHz) is fine even with constant-Q. **Over a wide span (3–81 GHz) constant-Q leaves silent tan δ biases of 3 CRB**, which the Jonscher law removes.

**Conclusion:** accepted.
- The default material law for multi-band data is **Jonscher**: drywall p free, insulation constant-Q. Constant-Q is adequate for a narrow span.
- Legal second band: **3.1–10.6 GHz UWB** gives the best accuracy (needs Part 15.509 eligibility), and **57–64 GHz** is the unlicensed alternative.
- Simulated data must themselves be causal: a material table is not a material law.

**Artifacts:** results/2026-11-09_E021_causal_bands/

### Pre-registered: E023 — Antenna beam model + multi-distance calibration (2026-11-09, before running)
**Change:**
- `wall_inverse.forward_oblique`: exact oblique stars (TE/TM, b = −jY/sin(k_zℓ), M = jY tan(k_zℓ/2)) with an adjoint Jacobian. Gates: equals an independent TMM to < 1e-12 at sin θ up to 0.8; Jacobian vs FD 6e-9.
- `forward_beam`: Gaussian-beam angular-spectrum integral by Gauss–Laguerre quadrature on a **rotated contour** u = v/(1 − jβ), β = 2d/(k₀w₀²) (steepest descent). Accuracy vs a 160-point Legendre truth: 1e-5 at K = 2, 6e-9 at K = 4, 1e-11 at K = 6. Without rotation, K = 20 was still 0.04 off for w₀ = 1 cm.
- `antenna.py`: multi-distance (5 plates, 0.20–0.30 m) Lambot calibration y = a + bS/(1 − cS), linear per frequency in (a, b − ac, c); w₀ by 1D search. It recovers w₀ = 2.00 cm, and 1.03/1.01 cm with H_f = 0.1 and H_i = 0.02.
- Method `e023` = Jonscher + beam (K = 2 search, K = 4 final polish) on calibrated beam responses. Bands 77–81 + 57–64 GHz (mmWave antennas; a UWB 3–10 GHz antenna is not a Gaussian beam and needs a measured pattern, which is open).

**Benchmarks:** `lambot:<cq|cqbeam2|cqbeam1|cqbeamhf|cqbeam1hf>:dual60`, seeds 1–10.
**Hypothesis:** e023 gives success ≥ 9/10 and silent 0 on all beam presets, with err/CRB std ≤ 1.5 (E020 with a plane-wave model: 8/8 flagged, z 30–90). On the beamless preset (cq) it is equivalent to e021b.

### Pre-registered: E024 — Facings, roughness, band phase-centre offset (2026-11-09, before running)
**Change:** `wall_inverse._effective`: sublayers with derivative maps. All gates pass (48), including:
- facing = core material is invisible (< 1e-12);
- facing equals the realism paper truth (< 1e-12);
- zero band offset is the identity (< 1e-14);
- Jacobian vs FD < 1e-5 with all features on, plane wave and beam;
- roughness taper vs Kirchhoff: best σ scale 0.9–1.1 and residual < 10% of the effect for σ ≤ 0.1 mm.

Model features:
- **paper facing** shared by all four drywall faces (t_p, εr_p, tan δ_p);
- **front roughness** as a graded taper of thickness h = 2√3σ/1.33 straddling the mean surface. Second-order equivalence: a graded layer reflects ~1 − (n̄k₀h)²/6, Kirchhoff ~1 − 2(k₀σ)²;
- **band phase-centre offset** δ (the second band sees d + δ).

Method `e024` = Jonscher + all three (plane wave). `e024beam` = the same + the beam model.
**Benchmarks:** real:<cqpaper|cqrough|cqoffset>:dual60 (E020: paper and offset 8/8 flagged; rough 5/8 flagged, 3 silent), plus lambot:all:dual60 (every E020 effect together incl. beam, H_f, H_i, plate error, drift, air gap) with e024beam. Seeds 1–10.
**Hypothesis:**
- e024 gives success ≥ 9/10 and silent 0 on paper, rough and offset.
- On `all`: success ≥ 8/10, silent 0. The standoff error equals the plate error by construction (it is measured relative to the calibration plane), so it is excluded from z.
- The 3 mm air gap is not modelled; expect it to cost some success.

## E023 — Antenna beam model + multi-distance plate calibration  (2026-11-09)
**Calibration fixes found during development:**
- (i) Plates 25 mm apart (≈ 13 half-wavelengths) make the per-frequency (a, b′, c) solve singular wherever the spacing is a whole number of half-wavelengths: a and c came out at 0.7 and 1.2 instead of 0. Fix: staggered positions (3 per base distance, λc/6 apart, at 0.20 and 0.30 m). Worst condition number 2–17.
- (ii) The calibrated data carry calibration noise (rms 1.4–1.6× the raw σ). Exact propagation: S_w = (y − a)/(b′ + cy), Var = σ²(|b|²/|D|⁴ + gᴴ(AᴴA)⁻¹g). The check gives |error|²/σ² = 1.9 vs an expected 2.

**Results** (lambot:<preset>:dual60, seeds 1–10):

| truth antenna | method | success / flagged / silent | best basin | notes |
|---|---|---|---|---|
| 2 cm beam | plane-wave model (e021b) | 0 / 10 / 0 | 1 | control: the beam must be modelled |
| **2 cm beam** | **beam model (e023)** | **9 / 0 / 1** | 8 | err/CRB std 0.5–1.2; 76 s per wall |
| no beam | e023 | 10 / 0 / 0 | 9 | the beam model does no harm (w₀ estimate → wide) |
| 1 cm beam | e023 | 2 / 6 / 2 | 1 | 300 s per wall |
| 1 cm beam + H_f 0.1, H_i 0.02 | e023 | 2 / 5 / 3 | 2 | |

**Failures examined (1 cm):**
- The truth fits the calibrated data at noise level with the calibrated w₀ (χ²/n 0.96–0.98 on all walls checked). So the beam model and the calibration are correct; the w₀-error hypothesis is rejected.
- The failures are search failures (flagged; the truth fits better) and ambiguity (silent; a wrong basin fits as well).
- Cause: a 1 cm waist at 30 cm diverges about 7–9°. The received echoes are about 11 dB (79 GHz) to 14 dB (60 GHz) weaker than for a plane wave (paraxial factor 1/√(1 + β²), β = 2d/(k₀w₀²) = 3.6–4.8), so the back-of-wall cycle information that the second band provided is lost again.

**Hypothesis:** passed for a 2 cm beam and for no beam; failed for a 1 cm beam.
**Conclusion:**
- The beam model (rotated-contour quadrature over exact oblique stars) and the multi-distance calibration are accepted.
- They do not make up for missing signal. Hardware requirement: an effective waist ≥ 2 cm at the operating distance (β ≲ 1), from an aperture/lens or a shorter standoff, or proportionally more SNR (averaging).
- Speed (76–300 s per wall) is now the binding constraint → E025.

**Artifacts:** results/2026-11-09_E023_beam/
