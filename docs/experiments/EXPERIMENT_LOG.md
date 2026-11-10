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

### Pre-registered: E025 — Speed: start-lattice density and hop sets with two bands (2026-11-09, before running)
**Profile (e021, one wall, real:cq:dual60, 7.4 s):** lattice stages 75% of the work (126 LM starts), cycle hops 12%, ambiguity scan 10%, joint LM 2%.
**Rationale:** with one band the basins are 1.9 mm (optical) apart and nearly degenerate, so starts every half period with 3 εr values were needed. With two bands the wrong basins are much weaker or absent (E014), so the lattice can be sparser.
**Variants** of the E021b method (Jonscher, cascade): `e025_base` (current), `s1` (1-period steps), `s2` (s1 + one εr start per stage), `s3` (2-period steps, one εr), `s4` (s2 + hops ±1 and ambiguity shifts ±1), `s5` (s3 + small hop sets).
**Benchmarks:** real:cq:dual60 and real:itucausal:dualuwb, seeds 1–20.
**Hypothesis:** at least one variant keeps success, silent and best-basin counts within ±1 of the base on both benchmarks, with ≥ 4× less work.

## E024 — Facings, roughness, band offset  (2026-11-09)
**Results** (dual60, seeds 1–10):

| data | method | success / flagged / silent | best basin | time/wall |
|---|---|---|---|---|
| face paper | e021b (no facing, control) | 0 / 10 / 0 | 9 | 37 s |
| face paper | **e024** (facing) | **10 / 0 / 0** | 9 | 23 s |
| roughness 0.1 mm | **e024** (taper) | **9 / 0 / 1** | 8 | 22 s |
| band offset 0.3 mm | e024 (δ fitted only in the joint LM) | 2 / 8 / 0 | 3 | 115 s |
| band offset | **e024b** (δ in the front stage) | 3 / 0 / 0 on the seeds 1–3 smoke | 3 | — |
| all effects + beam (lambot:all) | e024beam (before the stage fix) | 0 / 9 / 1 | 0 | 511 s |

**Failures examined:**
- **Band offset:** the front-stage fit held δ = 0, so both bands' front echoes disagreed and the stage locked into wrong basins. With δ in the front stage, 3/3.
- **All effects:** LM started from the truth reaches χ²/n 0.96, 1.18, 1.05 (seeds 1–3) without modelling the 3 mm air gap, and 0.95, 0.98, 0.97 with an explicit gap layer. So these are **search failures**, not model inadequacy. The run predates the stage fix: the front stage held δ, facing and roughness fixed although all three shape the front echo.
- Note: the automatic "χ² at truth" diagnostic is meaningless when the truth is not expressible in the stored parameters (plate-error shift, facing, unmodelled gap). It printed +6e4…+6e5 here. The protocol diagnostic must use LM-from-truth in such cases.

**Fix:** front stage now includes band_offset, facing.t, facing.εr and rough when enabled (`stud_stages`).
**Status:** paper and roughness accepted. Offset and all-effects are rerun below as E024b.

### Pre-registered: E024b (before running)
e024b on real:cqoffset:dual60 (seeds 1–10): success ≥ 9, silent 0. e024beam with the stage fix on lambot:all:dual60 (seeds 1–5): success ≥ 4/5, silent 0, standoff error ≈ plate error (excluded).

## E025 — Speed: start-lattice density and hop sets  (2026-11-09)
**Results** (seeds 1–20; success/flagged/silent; best basin; work in full 256-frequency solves; median wall time):

| variant | real:cq:dual60 | work | time | real:itucausal:dualuwb | work | time |
|---|---|---|---|---|---|---|
| base (½-period lattice, 3 εr, hops ±1/±2) | 20/0/0, 19 | 695 | 8.9 s | 20/0/0, 20 | 1487 | 17.4 s |
| s1 (1-period lattice) | 20/0/0, 19 | 562 | 7.0 s | 20/0/0, 20 | 1374 | 15.8 s |
| s2 (+ one εr start) | 19/1/0, 18 | 515 | 6.6 s | 20/0/0, 20 | 1263 | 14.6 s |
| s3 (2-period lattice, one εr) | 15/5/0, 14 | 809 | 9.8 s | 20/0/0, 20 | 1202 | 14.1 s |
| **s4** (s2 + hops ±1, ambiguity ±1) | 19/1/0, 18 | **413** | **4.8 s** | 20/0/0, 20 | 729 | 9.1 s |
| s5 (s3 + small hop sets) | 15/5/0, 14 | 653 | 8.2 s | **20/0/0, 20** | **693** | **8.3 s** |

**Hypothesis failed:** the best variant cuts work by 1.7× (60 GHz) or 2.1× (UWB), against a target of ≥ 4×.
- With UWB the lattice can be very sparse; wrong basins vanish, as E014 predicted.
- With 57–64 GHz a 2-period lattice loses 5 walls (flagged, which then pay for fallback).

**Conclusion:** partial. Adopt s4 as the default search (60 GHz) and s5 for UWB. Milliseconds need structural changes rather than tuning:
- (i) a compiled solver kernel;
- (ii) model evaluation on few frequencies with exact band-limited (DPSS) interpolation, justified by the Slepian count of E000/E006;
- (iii) a direct initializer (table lookup or small learned regressor on compressed data) plus a few Gauss–Newton steps, with the current search as a χ²-triggered fallback.

These are E026–E028.
**Artifacts:** results/2026-11-09_E025_speed/

## E026 + E027 — Compiled kernel and minimal pole-safe cells  (2026-11-09)
**Change:**
- **E026** (`fastkern.py`, numba): (i) a drop-in compiled Thomas solver; (ii) **per-sublayer stars**. All cells of a layer are identical, so b and M are computed once per (frequency, layer), and a compiled chain kernel returns the adjoint sums ΣQ₁ and ΣQ₂ per layer. Since Σ_cells ∂b·Q₁ = ∂b·ΣQ₁, the result is unchanged. Gate: compiled path equals the numpy path (Γ < 1e-13, J < 1e-12 relative).
- **E027** (`WallModel.min_cells`): the exact stars are exact for any cell count (doc 02 §7, doc 11 A5), and cells exist only to stay away from the poles of b (sin kℓ = 0) and M (cos(kℓ/2) = 0). New rule: the smallest n with |sin(kℓ/n)| ≥ 0.25 and |cos(kℓ/2n)| ≥ 0.25 at every frequency. That is 1–11 cells per layer instead of 13/58/13. Lossy layers need 1; lossless layers need more because they come closer to the poles. Gate: equal to the dense mesh (< 1e-12).

**Results:**
- forward + Jacobian, 512 frequencies: numpy 26.4 ms → compiled 4.3 ms → minimal cells 1.1–2.1 ms.
- Full inversion (E025 s4 settings, real:cq:dual60, seeds 1–20): **identical outcomes 19/1/0, best basin 18/20; median 4.8 s → 0.9 s.**

**Conclusion:** accepted. The remaining cost is about 800 evaluations per inversion. Milliseconds need about 10, which requires a direct initializer (E028).
**Artifacts:** results/2026-11-09_E026_speed_kernel/

### Pre-registered: E028 — Learned direct initializer + Gauss–Newton (before running)
**Change:**
- A small MLP (128 → 48 → 48 → 8, about 9k parameters, under the 10k budget) maps standoff-derotated coarse two-band data (32 frequencies per band; derotation by the matrix-pencil front-echo standoff on the top band) to (δd, front τ, εr, tan δ, insulation τ, εr, tan δ, back τ).
- It is trained on 20k simulated causal walls drawn from a **prior wider than the benchmark** (drywall t 9–17 mm, εr 2.0–3.6, tan δ 0.005–0.06; insulation t 80–100 mm, εr 1.0–1.4, tan δ 0.002–0.03), with rng seeds disjoint from the benchmark. Training takes minutes on a CPU.
- Inversion: MLP start → Jonscher LM on all data (min cells) → if χ²/n > 1.2, fall back to the e027_s4 search. The final answer is always the exact-model LM optimum; the network only chooses the starting point.

**Hypothesis:** on real:cq:dual60 seeds 1–20, outcomes within ±1 of e027_s4 (19/1/0), fallback on ≤ 3 walls, median time ≤ 50 ms.

## E024b — Band offset and all effects with the front-stage fix  (2026-11-09)
- **real:cqoffset:dual60, e024b:** **9 / 0 / 1** (success / flagged / silent), best basin 9/10. Before the fix: 2/8/0.
- **lambot:all:dual60** (beam 2 cm, H_f, H_i, plate error, drift, paper, roughness, air gap 3 mm, band offset), **e024beam with the fix, seeds 1–5:** **4 / 1 / 0**, best basin 4/5 (seed 4 χ²/n 1.23). Before the fix: 0/9/1. Median 256 s per wall.
- **But** err/CRB within the right basin has std 2–3 (means up to ±2.8): the estimates are biased beyond their error bars. Most likely cause: the unmodelled 3 mm insulation air gap (LM from the truth: χ²/n 0.96–1.18 without a gap layer, 0.95–0.98 with one).
- **Next:** a gap layer chosen by model selection, and an err/CRB-based acceptance test in addition to χ² (bias that χ² does not flag).

**Conclusion:**
- Paper, roughness and offset are solved individually.
- The all-effects case is solved for the basin (no silent failures), not yet for unbiased parameters.

**Artifacts:** results/2026-11-09_E024_facings/

## E028 — Learned direct initializer  (2026-11-09)
**Results** (real:cq:dual60, seeds 1–20; MLP ≤ 10k parameters trained on 20k prior-drawn causal walls, seeds disjoint from the benchmark; training 15–22 s):

| version | success / flagged / silent | direct (no fallback) | direct times | fallback median |
|---|---|---|---|---|
| spectrum features (32 frequencies per band), network + LM | 19 / 1 / 0 | 2/20 | 30, 49 ms | 1.1 s |
| hybrid (network + coarse back lattice by coarse χ²) | 16 / 1 / **3** | 13/20 | ~0.9 s | — (rejected: picks back basins without hop/ambiguity checks) |
| range-profile features (142 inputs, 9.6k parameters), network + LM | 18 / 1 / **1** | 3/20 | **16, 30, 54 ms** | 0.96 s |

**Network accuracy** (validation, within half a cycle):
- standoff δ 97–99%, front τ 73–75%;
- insulation τ 15% → 24% (median error 3.7 → 2.1 mm with range features);
- **back τ 16% (no better than the prior)**.

**Failures examined:**
- The back-drywall thickness is a cycle count (a periodic quantity) carried by two unresolved echoes. Smooth regression cannot represent the wrap-around.
- The fast path also skipped the ambiguity checks, which produced a silent failure (seed 9, χ²/n 1.03).

**Hypothesis failed** (the fast path is used on 2–3/20 walls, not ≥ 17/20). Where the start is right, a complete exact inversion takes **16–54 ms**, which shows the millisecond route works once the initializer is good.
**Next (E029):**
- Phase-aware outputs, a GNSS-style cascade: predict cos/sin of each layer's optical thickness on the **wide-lane** period c/(2Δf_carrier) ≈ 8.1 mm (79 vs 60.5 GHz) and the narrow-lane period 1.9 mm, plus a coarse value. Resolve the integers coarse → wide → narrow (the Teunissen/LAMBDA cascade).
- Always run a ±1-cycle ambiguity check (a few cheap LM runs) before accepting the fast path.

**Artifacts:** results/2026-11-09_E028_initializer/ (training logs), src/oned/init_net.py, init_net_weights.npz (9.6k parameters).

### Pre-registered: E029 — Phase-aware initializer + integer least squares (before running)
**Change:**
- **Network outputs**, 9.7k parameters: coarse values plus cos/sin of each layer's optical thickness on a wide lane P_w = c/(2Δf_c) = 8.10 mm and a narrow lane P_n = 1.90 mm. Cascade decoding coarse → wide → narrow.
- **Validation (within half a cycle):** front 51% → 92%. Insulation and back are not resolved by the cascade (wide-lane phase error 1.2–1.7 rad), **but their narrow-lane phases are accurate (0.25 / 0.58 rad)**: good float ambiguities.
- **Method e029:**
  1. Integer candidates N ∈ round(coarse) ± (4 insulation, 3 back), τ = P_n(N + φ).
  2. χ² at each candidate without optimisation (coarse data).
  3. Exact LM (min cells) on the best 3.
  4. ±1 ambiguity scan.
  5. If χ²/n > 1.2, fall back to e027_s4.

**Hypothesis:** real:cq:dual60 seeds 1–20: outcomes within ±1 of e027_s4 (19/1/0, best basin 18), the fast path taken on ≥ 15/20, median time ≤ 0.15 s. `e029_noamb` (without the ambiguity scan) shows what the scan costs and buys.

## E029 — Phase-aware initializer + integer least squares (results)  (2026-11-09)
**Results** (real:cq:dual60, seeds 1–20):

| method | success/flagged/silent | best basin | fast path | median time |
|---|---|---|---|---|
| e027_s4 (search) | 19/1/0 | 18 | — | 0.9 s |
| e029 | 19/0/**1** | 18 | 16/20 | 0.32 s |
| e029_noamb | 18/0/**2** | 18 | 14/20 | 0.24 s |

**Failure examined (seed 18, silent):** deterministic logic error.
- The integer search gave a bad start (χ² 1442, χ²/n 1.41). The ±1 scan then "rescued" it into a still-wrong basin (back drywall +1 cycle, χ² 1162, χ²/n 1.135). That passed the fixed χ²/n ≤ 1.2 alarm. The truth basin has χ² 1081.
- Lesson 1: a verification step must never be allowed to rescue an alarmed fast path; an alarm means fall back.
- Lesson 2: the fixed 1.2 alarm is a 4.4σ excess at n = 1024 (std of χ²/n = √(2/n) = 0.044). It was tuned for one band. Use χ² > n + 3√(2n).

**Conclusion:** partial (fast path on 16/20, but one silent failure and 0.32 s against a target of 0.15 s). Fix → E029b.

### Pre-registered: E029b (before running)
e029 + (i) the scaled alarm χ² > n + 3√(2n); (ii) an alarm before the scan triggers the fallback directly.
**Hypothesis:** real:cq:dual60 seeds 1–20: silent 0, success ≥ 19. Then held-out seeds 21–120: success ≥ 98, silent ≤ 1.

## E029b — Scaled alarm + verify-only scan; held-out check  (2026-11-09)
**Development seeds 1–20:** **20/0/0**, best basin 19/20, fast path 14/20, median 0.38 s.
**Held-out seeds 21–120:** **86 / 3 / 11** (success/flagged/silent), best basin 85/100, fast path 46/100 (median 273 ms), overall median 1.02 s. err/CRB within the right basin is calibrated (std 0.8–1.1).
**Silent failures examined (LM from the truth):**
- **Fallback search (e027_s4, sparse E025 settings): 3** (seeds 34, 36, 64). The insulation is +4 cycles off and the truth basin is better by Δχ² 50–200. χ²/n 1.02–1.04 looks statistically fine, so no alarm can catch this: a **search miss of the sparse lattice**.
- **Fast path: 8**, all in the **back drywall** (−1 or ±3 cycles). 5 are search misses (truth better by Δχ² 4–87); 3 are genuine ambiguity (truth worse).

**Conclusion: hypothesis failed (≥ 98 success, ≤ 1 silent).**
- (i) Speed shortcuts validated on 20 development walls (E025 s4, the E029 fast path) do not hold on 100 fresh walls. **Every speed variant must be validated on the held-out set before acceptance**; that is now part of the protocol.
- (ii) With 57–64 GHz as the second band, the back drywall's cycle is only marginally determined (E014 worst gap Δ = 62), so reduced search effort costs accuracy. UWB 3.1–10.6 GHz removes those basins (E014, E025).

**Next:**
- Held-out baselines for the dense (e025_base) and sparse (e027_s4) searches on dual60 (running).
- A back-drywall verification with ±1…±3 cycles polished on all data.
- The fast path trained and validated on the UWB second band.

### Pre-registered: E030 — Air-gap layer with model selection (before running)
**Change:** `WallModel.gap` splits the insulation into [batt t − g, air g], g ∈ [0.3, 10] mm. `insulation.t` stays the cavity depth. Gate: equals the realism air-gap truth to < 1e-12; Jacobian vs FD < 1e-6. Method `e030` fits with and without the gap (facing, roughness, band offset, Jonscher, min cells) and keeps the gap only if χ² drops by more than 6 (one extra parameter, about 3σ of a 1-dof χ²).
**Benchmarks:** real:cqgap:dual60 (3 mm gap only) and real:allpw:dual60 (roughness + paper + 3 mm gap + band offset, plane wave), seeds 1–10.
**Hypothesis:** on both, success ≥ 9, silent 0, err/CRB std ≤ 1.5 (E024b all-effects: std 2–3); the gap is selected on ≥ 8/10 walls.

### Pre-registered: E031 — FMCW chain of the prototype chip (TI IWR6843-like) (before running)
**Change:**
- **Data** (`realism.fmcw_band`): one chirp per scan with slope S. The target response is evaluated at the true instantaneous frequencies f_n + ε_n (quadratic nonlinearity, peak ε). It is multiplied by A(f) and filtered by the IF high-pass chain: two first-order sections, 175 and 350 kHz (IWR6843 HPF1/HPF2 options), bilinear, zero initial state, so the start transient is included. ADC noise is added. Bands 77–81 GHz (IWR1843) and 60–64 GHz (IWR6843).
- **Model** (`WallModel.chain`): per-band linear operator L_b = F_b·diag(Â_b), applied to Γ and J. Â is a smooth (degree-10 Chebyshev) fit from the plate scan through the same F. Gate: the truth fits the raw IF data at noise level; the Jacobian maps consistently.
- **Method e031:** search on chain-equalised data L⁻¹y, then the final exact LM and χ² on the raw IF data.
- **Baseline:** the classical calibration y/A_plate, which ignores the chain (fmcw:<preset>:naive with e027_s4).

**Physics:** an echo at delay τ beats at S·τ (200 kHz for the wall at S = 100 MHz/µs, at the HPF corners), so the HPF gain varies across the wall's echoes and differs between plate (0.25 m) and wall (0.30 m). At S = 250 MHz/µs the beat frequencies rise to ~500 kHz, where the HPF is flatter.

**Benchmarks:** fmcw:fm100 / fm250 / fm100nl (1 MHz nonlinearity, unmodelled), :naive (e027_s4) and :chain (e031), seeds 1–10.

**Hypotheses:**
- (a) naive at 100 MHz/µs: flagged or silently biased (z > 3) on most walls;
- (b) chain at 100 and 250 MHz/µs: success ≥ 9, silent 0, err/CRB std ≤ 1.5;
- (c) naive at 250 MHz/µs is less biased than naive at 100;
- (d) the unmodelled 1 MHz nonlinearity is flagged or mildly biased (characterises the need for chirp-linearity calibration).

### Held-out baselines on dual60 (seeds 21–120)  (2026-11-09)

| method | success/flagged/silent | best basin | median |
|---|---|---|---|
| e025_base (dense search) | **95 / 0 / 5** | 93 | 3.1 s |
| e027_s4 (sparse search) | 88 / 7 / 5 | 85 | 1.1 s |
| e029b (fast path) | 86 / 3 / 11 | 85 | 1.0 s |

**Conclusions:**
- With the 57–64 GHz second band, even the dense search leaves 5% silent failures (back-of-wall basins within noise: an information limit). With 22–26 GHz, E019 gave 100/100.
- The speed shortcuts add about 6 more.
- **Hardware implication:** UWB 3.1–10.6 GHz (wrong basins vanish, E014/E025) is the stronger second band. To be confirmed on held-out walls.

### E031 development notes
- (i) **Equalising** the data by L⁻¹ (to run the coarse search unchanged) fails: the IF high-pass inverse is ill-conditioned (cond ≈ 1.2e3) and its error grows along the sweep like a random walk, to 1.1–1.8 against a signal of 0.3. The model must predict filtered data; the filter is never inverted.
- (ii) The search therefore applies the operators on the full grid (search_step = 1).
- (iii) `best_standoff` must apply the standoff ramp **before** the IF chain (L·(Γ₀·ramp)); it applied it after. Fixed.
- Smoke (seeds 1–2): 1 success, 1 flagged (a search miss).

### Pre-registered: E032 — Held-out confirmation with the UWB second band (before running)
real:cq:dualuwb (77–81 + 3.1–10.6 GHz), seeds 21–120, dense (e025_base) and sparse (e027_s4) search.
**Hypothesis:** dense ≥ 98 success and ≤ 1 silent (vs 95/0/5 with 57–64 GHz); sparse within 2 of dense.

## E030 — Air-gap layer with model selection  (2026-11-09)
**Results** (dual60, seeds 1–10):

| data | success/flagged/silent | best basin | gap selected | estimated gap (true 3 mm) | insulation err/CRB std | time |
|---|---|---|---|---|---|---|
| real:cqgap | 10/0/0 | 10 | 7/10 | 0.6–0.8 mm | 2.2–2.3 | 62 s |
| real:allpw (roughness + paper + 3 mm gap + offset) | 6/2/2 | 7 | 6/10 | 0.7–9.3 mm | 1.8–2.1 | 75 s |

**Hypothesis failed:**
- The gap is selected (Δχ² > 6) but its size is not identified.
- Air (εr 1) behind a batt (εr ≈ 1.15) is a reflection of only ~3.5% in amplitude, and in optical thickness the gap trades almost one-for-one with the batt's εr.
- The insulation parameters stay biased by about 2 CRB.

**Conclusion:** a thin, low-contrast layer below the resolution cannot be estimated from this data. It should be treated as a **nuisance**: report cavity depth and insulation εr with uncertainty widened by marginalising over a plausible gap range (profile likelihood over g ∈ [0, 10] mm), rather than as a point estimate. Recorded as an identifiability limit (doc 12 §A7). The all-plane-wave-effects failures (2 flagged, 2 silent) need separate examination once the E031/E032 runs finish.
**Artifacts:** results/2026-11-09_E030_gap/

## E031 — FMCW chain of the prototype chip  (2026-11-09)
**Results** (77–81 GHz + 60–64 GHz, raw IF data, HPF 175/350 kHz, seeds 1–10):

| data | method | success/flagged/silent | best basin | time |
|---|---|---|---|---|
| S = 100 MHz/µs | classical calibration (chain ignored), e027_s4 | 0/10/0 | 6 | — |
| S = 250 MHz/µs | chain ignored | 0/10/0 | 6 | 3.4 s |
| S = 100 MHz/µs | **chain modelled**, 2-stage cascade | 7/3/0 | 7 | 11.6 s |
| S = 100 MHz/µs | **chain modelled**, 3-stage cascade | **9/0/1** | 9 | 13.6 s |
| **S = 250 MHz/µs** | **chain modelled**, 3-stage cascade | **10/0/0** | 10 | 14.8 s |
| S = 100 MHz/µs + 1 MHz chirp nonlinearity (unmodelled) | chain, 2-stage | 7/3/0 (identical to without) | 7 | — |

**Hypotheses:**
- (a) passed: ignoring the IF chain fails on every wall (flagged).
- (b) passed with the 3-stage cascade: 10/10 at 250 MHz/µs.
- (c) failed in form: both naive slopes fail completely, so the slope matters only once the chain is modelled. There, 250 MHz/µs is better: 10/0/0 vs 9/0/1, err/CRB std 0.9–1.3 vs 1.4.
- (d) passed: a 1 MHz quadratic chirp nonlinearity is negligible. Phase error 2π·ε·τ ≈ 0.013 rad at τ = 2 ns.

**Failures examined:** the 2-stage cascade's 3 flagged walls were search misses (the truth fits the raw IF data at χ²/n 0.92–1.00). The dense third stage fixes them. A crash on naive data (the pencil returned no starts) is fixed: an empty pencil start set falls back to the lattice, and non-finite candidates are dropped.
**Conclusion:** accepted.
- The IF high-pass chain enters the model as a fixed linear measurement operator (never inverted: cond ≈ 1e3).
- Use the maximum practical chirp slope (250 MHz/µs on the IWR6843) so the wall beat frequencies sit well above the HPF corners.

**Artifacts:** results/2026-11-09_E031_fmcw/

## E032 — Held-out confirmation with the UWB second band  (2026-11-09)
**Results** (real:cq:dualuwb, 77–81 + 3.1–10.6 GHz, seeds 21–120):
- dense search (e025_base): **100 / 0 / 0**, best basin 100/100, median 3.9 s;
- sparse search (e027_s4): **100 / 0 / 0**, best basin 100/100, median 1.3 s;
- err/CRB within the basin: mean |·| ≤ 0.17, std 0.95–0.98.

With 57–64 GHz the same methods gave 95/0/5 (dense) and 88/7/5 (sparse).
**Hypothesis passed.** The UWB second band removes the residual back-of-wall ambiguity, and the speed shortcuts lose nothing with it.
**Decision:** the second band of the prototype is **3.1–10.6 GHz UWB** (FCC §15.509 wall imaging for construction, with Part 90 eligibility, to be confirmed) alongside the IWR6843 at 60–64 GHz. The fast path (E029) is to be retrained and held-out-validated for this band pair.
**Artifacts:** results/2026-11-09_E032_uwb_heldout/

## E030 revisited — the air gap IS identifiable; E030 was a search failure  (2026-11-09)
- **Cramér–Rao bound** of a 3 mm gap (Fisher at the truth, gap free): **0.04 mm at 40 dB** for every band pair (57–64, UWB, prototype 60–64 + UWB). corr(gap, insulation εr) = −0.13 … −0.30. The information is there.
- **LM from the true gap** (real:cqgap:dual60): the gap is recovered as 2.96–3.01 mm, with **lower χ² than E030** on seeds 2–4 (by 21–235); seed 1 is a near-tie (Δχ² 4).
- **Cause:** the gap is a thickness with its own phase cycles (λ/2 ≈ 1.9 mm in air at 79 GHz), but it was in neither the start lattice nor the hops or ambiguity scan, so the search stayed in the wrong cycle.
- **E030's conclusion "not identifiable" is withdrawn.**

### Pre-registered: E033 — Gap in the cycle machinery (before running)
**Change:** the gap gets a phase period λc/2 (`_phase_periods`), so it enters the cycle hops and the ambiguity scan; stage 1 frees the gap with starts every half period (−1…+4 periods from 2 mm).
**Benchmarks:** real:cqgap:dual60, real:cqgap:dualuwb, real:allpw:dualuwb, seeds 1–10.
**Hypothesis:** gap error < 0.2 mm on ≥ 8/10, success ≥ 9, silent 0, insulation err/CRB std ≤ 1.5.

## E033 — Gap in the cycle machinery  (2026-11-09)
**Change (as run):** the first version (gap added to the stage-1 lattice, 825 starts) crashed on a variable-order bug and would have been 11× slower. Replaced by a **conditional gap stage**: after the back-wall stage, the gap is freed together with the stage-1 parameters, with starts every half phase period (−0.5…+3 periods from 2 mm). The gap is in the cycle hops and ambiguity scan (period λc/2). Model selection as in E030 (keep the gap only if Δχ² > 6).

| benchmark | outcome (s/f/silent) | gap estimates (truth 3.00 mm) | insulation εr err/CRB std |
|---|---|---|---|
| real:cqgap:dualuwb | 10/0/0 | 2.91–3.05 mm on **10/10** (gap selected 10/10) | 0.87 |
| real:cqgap:dual60 | 10/0/0 | 2.75–3.07 mm on 8/10; gap not selected on 2 (insulation absorbs it, still in the truth basin) | 0.61 |
| E030 on real:cqgap:dual60 (before) | 8/2/0 | 0.61–0.79 mm | 2.21 |

**Hypothesis** (gap error < 0.2 mm on ≥ 8/10, success ≥ 9, silent 0, insulation err/CRB std ≤ 1.5): **passed** on dualuwb (10/10 within 0.09 mm) and dual60 (7/10 within 0.2 mm, 8/10 within 0.25 mm, 0 silent). The air gap is measurable; the E030 failure was purely a search failure.
**Cost:** slow, because two full cascades run (with and without the gap): median 218 s (UWB), 86 s (dual60) per wall. Speed is addressed in E039.
**Artifacts:** results/2026-11-09_E033_gap_cycles/ (real:allpw:dualuwb still running at the time of writing).

### Pre-registered: E034 — Arbitrary stacks of arbitrary materials (before running)
**World:** `stack.py`. Random stacks of 1–5 layers from 15 materials: the 10 ITU-R P.2040-3 Table 3 building materials (concrete, brick, plasterboard, wood, glass, ceiling board, chipboard/OSB, plywood, marble, floorboard), made causal (Jonscher form, loss slope from the table's d), plus foam, fibreglass batt, ceramic tile, **air gaps** and **liquid water layers** (double-Debye water, Liebe–Hufford–Manabe 1991, one random wall temperature 5–35 °C). Backing: air or **metal** (σ = 10⁷ S/m). Material variability ±8 % in ε′ and ×0.6–1.6 in σ. Rejection sampling keeps only stacks whose last interface is visible (two-way loss < 45 dB at 5 GHz). Prototype bands 60–64 GHz + 3.1–10.6 GHz, 256 points each, 40 dB SNR, plane wave, calibrated.
**Forward gate:** exact-star DEC vs an independent transfer-matrix reference on 60 random stacks incl. water and metal: max |ΔΓ| = **1.6e-13**. Adjoint Jacobian vs central differences: worst relative 1e-4 (finite-difference noise on the weakest parameters; the analytic pinf derivative was checked separately to O(h²)).
**Inversion (knows no material names):** every layer is a causal Jonscher dielectric (t, ε′, tan δ at 10 GHz, dispersion shape p) or a water layer (CRIM θ = 1, double Debye, **water temperature fitted**). Back = air or metal, decided by χ². Steps: robust front-face detection (earliest peak above the far-range noise floor), with a standoff lattice; frequency continuation (low band, lowest third first: Bunks et al. 1995 multiscale principle against cycle skipping); greedy growth (residual-echo split/append, thin-layer insertion at every interface, retyping between dielectric/water/air), with a new layer accepted only if Δχ² > 4 ln n (Schwarz/BIC); then the high band, cycle hops, ambiguity scan, pruning and back-medium check; finally Gaussian-likelihood classification against the material library.
**Metrics:** structure correct (layer count, back medium); per-layer thickness error; ε′ error; class top-1; χ² alarm; silent = wrong structure without alarm.
**Hypothesis (first version, deliberately modest):** on 1–2-layer stacks, structure correct ≥ 8/10, silent ≤ 1; then scale to 5 layers.

### Pre-registered: E035 — No plate in the field (before running)
**Question:** can the sensor be used anywhere with only a one-time factory calibration?
**World:** factory Lambot/plate calibration, then a field scan whose radar chain has drifted: per band random gain (σ 0.5 dB), gain slope (0.3 dB), phase (0.3 rad), group delay (3 ps) (`realism.field_drift`). Variant "ref": the radar measures its own Tx→Rx leakage (−15 dB, 25 dB SNR), which passes the same chain, fits the same 4-parameter drift and divides it out; an untracked antenna part remains (0.1 dB, 0.02 rad).
**Methods:** e035_none (ignore drift), e035_gain (fit per-band gain + slope with the wall), e035_full (also phase + delay).
**Benchmarks:** real:cqdrift:proto and real:cqdriftref:proto (prototype pair 60–64 GHz + UWB), seeds 1–10.
**Hypothesis:** e035_none fails on cqdrift; with the leakage reference plus fitted gain, ≥ 9/10 success and 0 silent, err/CRB std ≤ 1.5.

## E035 — No plate in the field: factory calibration + self-calibration  (2026-11-09)
Prototype pair (60–64 GHz + UWB), seeds 1–10, e027_s4-type search. Drift between the factory calibration and the field scan: per band, gain σ 0.5 dB, gain slope 0.3 dB, phase 0.3 rad, delay 3 ps.

| world | ignore drift | fit gain + slope | fit gain, slope, phase, delay ("full") |
|---|---|---|---|
| no drift (cq) | 10/0/0 | 10/0/0 (εr CRB ×4) | not run |
| drift, no reference | 0/10/0 (all flagged, none silent) | 0/10/0 | **10/0/0**, err/CRB std 1.0 |
| drift + leakage reference (untracked 0.1 dB, 0.02 rad) | 8/2/0, err/CRB std up to 7 (biased) | 8/2/0, std 4 | **10/0/0**, err/CRB std 1.0–1.05 |

**Findings.**
1. A drifted chain is **never a silent failure**: the χ² alarm catches it.
2. The residual drift after a leakage reference (0.1 dB, 0.02 rad) is still far above the CRB: CRB-level accuracy needs gain known to ~0.02 dB and phase to ~0.005 rad, which no reference achieves. **The drift must be fitted.**
3. Fitting 4 smooth drift parameters per band with the wall works (10/10, unbiased, calibrated error bars), even **without any reference**. The price is information: CRB(εr front) 3.7e-3 → 1.7e-2 (×4.6), CRB(insulation t) 0.096 → 0.12 mm, CRB(back drywall t) 0.012 → 0.042 mm. Still sub-0.1 mm thickness and ~1 % εr.
**Hypothesis** (with reference + fitted gain ≥ 9/10, 0 silent, err/CRB ≤ 1.5): **failed** for gain-only (8/10, biased); **passed** with the full 4-parameter self-calibration (10/10, std 1.0).
**Decision:** the field workflow is a one-time factory calibration (antenna + lens, Lambot multi-distance plates), with no plate in the field and the per-band drift fitted with each wall. Next refinement: use the leakage reference as a Gaussian prior on the drift parameters (instead of dividing it out) to recover part of the ×4.6 information loss.
**Artifacts:** results/2026-11-09_E035_fieldcal/

## E037 — Signal-to-noise sweep  (2026-11-09)
Prototype pair (60–64 GHz + UWB), real:cq, 20 walls per level, e035_none (e027_s4-type search). SNR = that of a perfect reflector per frequency sample.

| SNR | outcome | εr front err/CRB std | CRB εr front | CRB insulation t | CRB back drywall t |
|---|---|---|---|---|---|
| 40 dB (E035) | 10/0/0 | 0.98 | 0.0037 | 0.096 mm | 0.012 mm |
| 35 dB | 20/0/0 | 0.96 | 0.0066 | 0.17 mm | 0.021 mm |
| 30 dB | 20/0/0 | 0.95 | 0.012 | 0.30 mm | 0.037 mm |
| 25 dB | 20/0/0 | 0.94 | 0.021 | 0.54 mm | 0.067 mm |
| 20 dB | 20/0/0 | 0.92 | 0.037 | 0.96 mm | 0.12 mm |

**Finding:** with the UWB band there is **no SNR cliff down to 20 dB**. The search finds the right basin on every wall, and the errors simply scale as 1/SNR (the CRB) with calibrated error bars. The cycle-ambiguity collapse that made single-band operation fragile does not appear, because UWB phase periods are ~6× longer than the noise-induced phase error. 15 and 10 dB are queued to find the break point. Averaging N chirps adds 10 log₁₀ N dB (100 chirps ≈ 2 ms ≈ +20 dB), so 20 dB per chirp is ample.
**Artifacts:** results/2026-11-09_E037_snr/

## E037 addendum — where SNR breaks
15 dB: 20/0/0 (εr front CRB 0.067, insulation t CRB 1.7 mm). **10 dB: 17/1/2**, 4/20 outside the truth basin, **2 silent**. Break point is between 10 and 15 dB per sweep, i.e. ≥ 15 dB needed; averaging 100 chirps (+20 dB) keeps a 40 dB radar far from it.

## E038 — Handheld motion (multi-frame joint fit)  (2026-11-09)
K frames of the same wall, each moved up to ±4 mm in standoff, fitted jointly (one wall, one standoff per frame; frame offsets started from range-profile peaks). Prototype pair.
| | outcome | εr front CRB | insulation t CRB | time |
|---|---|---|---|---|
| 1 frame, 25 dB | 20/0/0 | 0.021 | 0.54 mm | 2.0 s |
| 4 moving frames, 25 dB | 20/0/0 | 0.010 | 0.27 mm | 7.9 s |
| 4 moving frames, 40 dB | 10/0/0 | 0.0019 | 0.048 mm | 8.6 s |
**Finding:** motion costs nothing; 4 moving frames give the full √4 = 2× error reduction (+6 dB), err/CRB std 0.8–1.4. Each frame's standoff is estimated, so the hand need not be still. Cost grows ~linearly with K (to optimise: fit per-frame standoffs in a profile step).

## E039 — UWB-first fast path, held out  (2026-11-09)
real:cq:proto, seeds 21–120: **100/0/0, best basin 100/100, err/CRB std 0.93–0.95, median 1.1 s (lat_step 1) / 0.7 s (lat_step 2)** on a loaded CPU, plus a fast model copy (deepcopy was 40 % of the time). On real:allpw:proto (paper, roughness, gap, band offset) e039 without those nuisance terms: 0/40/0 — all flagged, none silent; the all-effects version needs facing/rough/gap/band-offset options added to the fast path (next).

## E040 — Stress tests of "hardware-only" assumptions (part 1)  (2026-11-09)
- **Antenna tilt** (Gaussian-beam monostatic factor exp(−(k₀w₀ sin θ)²/2), w₀ = 2 cm): 2° (−3.6 dB at 62 GHz): ignoring it 0/10/0 (flagged), **self-calibration (E035 full) 10/0/0**, err/CRB std 1.0–1.1. 4° (−14 dB): 0/10/0 for both — the self-cal gain box (±0.6 Np) was too narrow; widened to ±3 Np and rerunning.
- **Unknown noise level:** σ estimated from the far-range bins of a Hann-windowed range profile: σ̂/σ = 0.95–1.02 on 24 cases. Benchmark run crashed in the scorer (singular Fisher with a self-cal parameter at its bound); fixed with a pseudo-inverse fallback, rerunning.
- A container restart killed E034 (5-layer, after 6 seeds) and E036 before they ran; relaunched.

## E034 — 5-layer stacks, first 6 walls (partial)
2/6 structure right in front layers only; seed 4 (water/tile/plywood/water on metal) and seed 5 (batt/concrete/tile/chipboard: concrete + tile merged into one 49 mm layer, χ² 1039/1024) are **silent**. The greedy growth does not scale past ~3 layers; seed 5 also needs a CRB check of whether 7 mm tile behind 43 mm concrete is identifiable at all. Next: 3-layer statistics (running), then a better search (layer-stripping with time gates per interface, and multi-hypothesis growth instead of greedy).

### Pre-registered: E041 — Exact layer peeling for 7–12 layers in < 1 s (before running)
**Why:** greedy growth (E034) is exponential in the number of layers and fails past ~3.
**Math (verified):**
1. *Exact downward continuation.* For planar layers the reflection at the top of layer i depends on everything deeper only through Γᵢ₊₁: Γᵢ = (rᵢ + Γᵢ₊₁eᵢ)/(1 + rᵢΓᵢ₊₁eᵢ). Inverting this Möbius map, Γᵢ₊₁ = (Γᵢ − rᵢ)/(eᵢ(1 − rᵢΓᵢ)), turns the data into what a radar inside layer i would see. **Gate: 1e-9 relative error over 30 random stacks at every depth.** Noise is propagated exactly: σᵢ₊₁ = σᵢ|1 − r²|/|eᵢ(1 − rΓᵢ)²|.
2. *Exact fast forward.* The reflection recursion with a reverse-mode (adjoint) Jacobian, `tmm.py`, equals the exact-star DEC solve: **|ΔΓ| = 1.0e-14 and relative |ΔJ| = 3.6e-14 over 40 stacks of up to 8 layers** (with moisture, fitted water temperature, metal). 5× faster (1.7 ms vs 9.3 ms for 8 layers, 256 frequencies).
**Algorithm (`peel.py`):** per depth, a gated "layer over half-space" fit on the low band with window continuation (the widest time window still consistent with noise), ~18 physics-guided starts scored cheaply, top 4 optimised. When thin neighbours merge (no consistent window), *block peeling* fits two layers on all bands and keeps the top one. Beam search keeps the 3 best hypotheses; termination when the peeled remainder is noise (air behind) or a mirror (metal). Then one exact joint LM on the original data, a back-medium check and BIC pruning (κ = 4 ln n per layer).
**First results (3 layers, seeds 1–6, before block peeling):** batt/wood/brick exactly right (58.0/15.1/77.8 mm, εr 1.14/2.04/4.03). The failures, all *flagged*, are adjacent thin layers below UWB time resolution (chipboard 11 mm + marble 13 mm; floorboard 10 + tile 6 mm) and a weak foam front ahead of water. Time 2–80 s (scipy least squares with finite differences; to be replaced).
**Hypotheses (to test):** 3 layers ≥ 9/10 correct, 0 silent; 7 layers ≥ 8/10 correct-or-flagged, ≤ 1 silent; median time < 1 s after the speed work (analytic step Jacobians, compiled recursion).

## E041 — progress on 3-layer stacks (development seeds 1–6)  (2026-11-09)
Iterations and what each fixed (all exact physics; only the search changed):
1. Window continuation per depth (widest time window consistent with noise), fixed short windows (3/5/8 cm) accepted only when consistent (a short window looks "consistent" because it sees little data).
2. Block peeling for thin neighbours below UWB time resolution: two layers fitted together, UWB first then ±2 high-band cycles (bug found: the standoff box had lo = hi, so the block never ran).
3. Tail-cut pruning (drop the tail, either backing) and merging of identical neighbours.
4. **Bug: a free Jonscher shape p started exactly on its bound p = 0; in logit coordinates it can never move**, so dispersive materials (brick, plywood, wet layers) could only be imitated by fake extra layers. Releasing p early, however, drags thicknesses off-cycle. Final design: geometry with p = 0 (constant-Q shape) throughout, then release p at the end (starts 0.02 and 0.95), then merge/prune again.

**Result (dev seeds 1–6):** 4/6 structure and thicknesses exact (corrected from "5/6": two failures, both flagged), none silent. Reproduced after restoring this configuration (commit bc7151e) when later search tweaks (fewer/more starts, screening, wider beam) each made it worse: the per-depth search is fragile, see E041 notes.
| wall | truth (mm) | fit (mm) | εr fit |
|---|---|---|---|
| batt / wood / brick | 58.0 / 15.1 / 78.0 | 58.0 / 15.1 / 77.8 | 1.14 / 2.04 / 4.03 |
| batt / brick / wood | 33.3 / 93.7 / 19.1 | 33.3 / 93.5 / 19.1 | 1.11 / 3.68 / 2.03 |
| chipboard / marble / plywood | 18.8 / 14.9 / 16.0 | 18.8 / 14.9 / 16.0 | 2.64 / 6.66 / 2.48 |
| chipboard / marble / batt + metal | 11.4 / 13.2 / 54.2 | 11.4 / 13.3 / 54.4 | 2.54 / 6.51 / 1.09 |
| floorboard / tile / concrete | 10.4 / 6.2 / 94.7 | 10.5 / (tile + concrete merged 108.8) | flagged (χ²/n 5.3) |
| foam / water / tile (seed 4, earlier version) | — | weak foam front missed | flagged |
Time 7–124 s (prototype: scipy least squares with finite differences). Next: weak front faces, then speed (analytic Jacobians, own LM, compiled recursion), then held-out seeds and 5/7-layer benchmarks.

## E041 — held-out reality check  (2026-11-09)
Held-out 3-layer walls (seeds 101–108, never used for tuning):
- baseline peeling (commit bc7151e): **1/7 exact, 6 flagged, 0 silent**, 108–700 s per wall;
- + conduction-like dispersion kind in the per-depth fits: **1/8 exact, 7 flagged, 0 silent**, median 359 s; development walls fell from 4/6 to 3/6 with one **silent** failure (brick split again).
Failure pattern: water layers (2–7 mm) in 5 of 7 held-out failures, thin high-contrast glass/tile (3–8 mm), plywood fronts, thick lossy concrete split into pieces.
**Conclusion:** the development result (4/6) was optimistic; the greedy per-depth peeling is brittle on band-limited, lossy data with strong multiples (water, glass, tile), and search tweaks trade one wall for another. The information is there (CRB 0.01–0.3 mm at the truth, E041 check), so this is a search-design problem.
**Next design (not yet implemented):** replace the discrete per-depth decisions by a *global* continuous inversion, then segmentation. Fine sublayers (e.g. 2 mm optical) each with (ε′, tan δ, water fraction), exact TMM forward and adjoint Jacobian (O(F N), already built), total-variation (blocky) regularisation à la Occam/TV inversion (Constable et al. 1987), frequency continuation from low UWB to all bands; then merge sublayers into layers (change-point segmentation) and polish the blocky model exactly with BIC pruning. All depths are solved at once, so a wrong early choice cannot propagate.

## E042 / E043 — global blocky inversion and material-library prior  (2026-11-09)
**E042** (`profile.py`): all depths at once (thin slices, exact recursion with reverse-mode Jacobian: 1.6e-13 vs reference, Jacobian 1e-7), TV regularisation with frequency continuation, segmentation, exact polish and BIC pruning (bug fixed: stale standoff box during pruning). Held-out 3-layer walls 101–106: **1 exact, 2 flagged, 2 silent** (5 walls), 2–23 min per wall. Silent failures fit the data to the noise with 9+ implausible layers.
**E043** (`matlib.py`): + material-library prior (Mahalanobis distance of each non-water layer's (log ε′, log tan δ) at 10 GHz to the nearest ITU/library class; air class; sliver penalty; Jacobian 1e-7). Wall 102 exact (150 s); wall 103 still 13 layers (all now "plausible" materials), flagged. **E043b** (minimum-gradient-support focusing instead of TV in the last stage): profiles became more oscillatory (55–66 segments), water missed; **reverted**.

**What the E034 → E043 sequence establishes**
1. The physics is solved and fast: exact forward + adjoint for any stack (1.4 ms for 7 layers, 512 frequencies), verified to 1e-13–1e-14.
2. The information is there: at the true structure, CRB thickness 0.01–0.3 mm, εr 0.004–0.03 (E041 check).
3. The **global structure search** is the unsolved part: greedy growth, layer peeling and continuous blocky inversion all reach only 1–2 of 8 held-out 3-layer walls with thin high-contrast (glass, tile) or water layers. The data admit many-layer explanations that fit to noise; a material prior labels them but does not remove them.
**Next (proposed): dictionary proposals + exact verification.** Precompute the simulated response features of a large library of plausible stacks (materials × thicknesses, simulation only, no measured data); at query time, nearest-neighbour lookup proposes the top-K structures; each is polished by the exact LM with the material prior; BIC + χ² alarm decide. Offline-heavy, online-cheap (lookup + a few exact fits ⇒ sub-second plausible), non-learned (a lookup table), and the answer stays the exact optimum.

## E044 — known layer count N (the only user input)  (2026-11-10)
**Decision with the project owner:** N (the number of material layers) is a user input; thicknesses, materials and the backing are outputs (doc 14, accepted limitation E1).
**Method (`known_n.py`):** E042 slice profile → candidate A: optimal dynamic-programming segmentation into exactly N layers (gate: exact cut recovery 5/5) → exact fit under the material-library prior with cycle hops; candidate B: over-segmented exact fit reduced agglomeratively to exactly N (best drop/merge each step); the lower MAP cost wins; alarm on data χ² only.
**Held-out 3-layer walls (seeds 101–108, N given):** **2 exact (102, 107), 5 flagged (103, 104, 105, 106, 108), 0 silent, 1 timeout (101, killed after 90 min).** 5–36 min per wall.
**Findings:** knowing N removes silent failures (E042 had 2/5 silent), but accuracy is still poor. **Every failure has a thin, strongly reflecting layer** (glass 3–7 mm, tile 8 mm, water 2–7 mm); both exact walls have none. Speed is dominated by the agglomerative reduction (many exact fits per step).
**Next:** diagnose the thin high-contrast case directly (is it information or search? CRB at the truth vs. what the solver reaches), then speed.

## E046–E049 — the stated problem: known N, automatic gaps, three bands  (2026-11-10)
**Problem definition (with the owner):** input = number N of solid material layers + approximate radar-to-wall distance; output = each layer's thickness, material and moisture, every air gap (an optional gap slot at every interface, plus one in front of a metal backing), and the backing. Bands: 0.4–0.96 GHz (sub-GHz, 10 dB less SNR for FCC §15.209 power), 3.1–10.6 GHz UWB, 60–64 GHz. Generator `wallsolve.make_problem`; batched exact forward `gamma_batch` (== independent recursion to 2.8e-14).
**Global search attempts (E046):** differential evolution on (a) complex data on all bands, (b) low bands only, (c) echo-envelope misfit, (d) optical-thickness coordinates — all 0/4 on dev walls. Root causes found on the way: three search-space bugs that made the truth unreachable (water fraction floor 0.25 %, dispersion shape capped at p = 0.99 while ITU brick/plywood need ~0.996, water temperature fixed), after which the truth fits at χ²/n ≈ 1 but DE still does not reach it in ~19 dimensions.
**Echo-based solver (E049, `echoes.py`, `wallsolve.solve_echo`):** multiband sparse deconvolution with shared delays (UWB + 60 GHz) → candidate interface subsets with the front pinned to the known standoff, every gap pattern → vectorised exact screening (11k candidates in 5 s) → short exact fits of the best 40 → full exact fits with cycle hops; unobserved-layer check (a solid layer fitting as air or at the thickness limit); thin-layer split retry only when the χ² alarm fires.
**Stage diagnostics (`diagnose.py`)** located each failure: wall 4 = identifiability (low-density batt behind 4.4 mm water fits as air), wall 1 = merged echoes of a 6 mm plywood layer + poor amplitude-based starting materials for lossy/dispersive plywood.
**Dev walls (seeds 1–4), best configuration E049g:** 2 exact (33–38 s), 1 honestly reported unobservable, 1 flagged, **0 silent**. Later variants (material sweeps, interleaving, explicit alternative-basin refits, E049h–j) each traded one dev wall for another; switched off. Next: a held-out benchmark (20 fresh 3-layer walls) with E049g to measure, instead of tuning on four walls.
