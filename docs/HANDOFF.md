# HANDOFF — radar-ddec: known-N layered-wall inversion

*Written 2026-11-12 to carry the full context into a new conversation.*

- Branch: `claude/funny-cannon-mrihcy`.
- Everything below is committed and pushed.
- The detailed record of every experiment is `experiments/EXPERIMENT_LOG.md`, newest last.
- Known-good versions and revert instructions are in `VERSIONS.md`.
- Project rules are in `CLAUDE.md`.

---

## 1. The problem (as agreed with the owner)

**Goal.** A physics-exact, low-compute inversion algorithm. From a radar measurement of a wall, it recovers what is inside the wall, with honest uncertainty.

**Inputs**
- **N:** the number of SOLID material layers, counted front to back. For example, drywall | insulation | drywall is N = 3.
- **What N does not count:** the front free space (radar → wall) and the backing.
- **The approximate radar-to-wall distance:** used as a guess; the code allows ±3 mm.
- **The radar data:** complex reflection Γ(f) on three bands.

**Outputs**
- For every solid layer:
  - its **thickness**, with an honest error bar;
  - its **material**: a library label or label set, its measured ε′(10 GHz), tan δ and dispersion shape, plus an "unknown material" flag;
  - its **moisture**. In scope as an output, but water layers were removed from the current benchmark at the owner's request.
- **Any number of air gaps:** an optional gap slot at every interface between solids, plus one in front of a metal backing. Gaps under 5 mm are out of scope.
- **Backing:** air or metal.
- **Honesty outputs:**
  - a χ² alarm;
  - per-layer error bars (95 % profile-likelihood intervals);
  - "weak / unobserved / unresolved" flags;
  - alternative answers that fit within Δχ² < 4.

**Bands** (the code must work for any number of bands)

| Band | Regulation | Simulated SNR |
|---|---|---|
| 0.4–0.96 GHz | FCC §15.209 | 30 dB |
| 3.1–10.6 GHz UWB | §15.509, wall imaging | 40 dB |
| 57/60–64 GHz | §15.255 | 40 dB |

"+20 dB" means averaging about 100 chirps (about 10 ms). It is the recommended default.

**Accuracy target per wall ("OK")**
- every thickness within max(0.3 mm, 2 % of t);
- every gap within max(0.3 mm, 5 %);
- correct backing;
- the true material inside the reported label set.

**Success criteria**
- **Silent failures:** wrong answers reported with confidence must be ≈ 0.
- **Accepted outcomes:** "flagged" (χ² alarm) and "honestly unobservable / ambiguous" (truth inside the stated intervals or alternatives) are acceptable outcomes. They are counted separately.
- **Layer count:** the owner wants this to work comfortably up to N = 7.
- **Speed:** sub-second is the eventual goal. It is secondary to accuracy.

**Non-negotiables** (from `CLAUDE.md` and owner decisions)
- Exact physics, verified numerically against independent references.
- Low compute and low data. No ML unless small, and only if it helps. The ML experiment was run and dropped (E055).
- Honest reporting: CRB / profile error bars and global ambiguity.
- Randomised benchmarks, held-out validation, no tuning to the test set.
- Commit and push everything, and keep a version history.

**Accepted limitations.** N is a user input (doc 14, E1). Behind a strong absorber, "cannot see beyond layer k" counts as correct if it is reported. μr = 1.

---

## 2. The simulated world (benchmark)

**Generator:** `wallsolve.make_problem(seed, N)` (`docs/src/oned/wallsolve.py`).

**Solid materials** (13 library classes; liquid water is excluded): concrete, brick, plasterboard, wood, glass, ceiling_board, chipboard, plywood, marble, floorboard, foam, batt, tile.
- **Laws:** ITU-R P.2040-3 made causal (Jonscher law ε = εr[p + (1−p−j tanδ)(f/10 GHz)^(−a′)], a′ = (2/π)·atan(tanδ/(1−p))). Each instance is drawn randomly:
  - ε′ × U(0.92, 1.08);
  - σ × U(0.6, 1.6).
  - Brick and plywood need p ≈ 0.996.
- **Thickness:** log-uniform within each material's product range (`stack.thickness_range`), e.g. plasterboard 9.5–16 mm, brick 60–110 mm.

**Wall structure**
- Gaps: each slot has a 35 % chance of a gap, log-uniform 5–30 mm.
- Backing: metal with probability 0.3.
- Standoff ≈ 0.30 m, plus U(0, 4 mm).
- Noise: complex Gaussian at the band SNR; 256 points per band.

**Seeds in use**

| Seeds | Use |
|---|---|
| 400–449 | dev set, N = 3 |
| 600–649 | held-out #1, N = 3 |
| 700–749 | held-out #2, N = 3 |
| 900–919 | N = 5 and N = 7 |
| 100000+ | NN training (E055) |

**Material library prior.** `stack.library_stats()` gives each class's mean and covariance of (ln ε′, log10 tan δ) from the generator's variability. It stands in for a measured material database.

---

## 3. Code map (all in `docs/src/oned/`; run scripts from there)

| file | what it is |
|---|---|
| `wall_inverse.py`, `stack.py`, `tmm.py` | exact forward models (DEC exact-star ≡ reflection recursion to 1e-14), LM fitter, Fisher/CRB, material laws, `ST.build` model objects |
| `wallsolve.py` | problem generator `make_problem`, batched exact forward `gamma_batch` (P candidates at once), old echo-based solver (`solve_echo`, `solve`, `solve_full`), class table, `honest_report` |
| `pt.py` | **parallel tempering** (replica exchange) over the full continuous parameter vector; physics moves (random walk, library class swap, λ/2 cycle hop, interface shift, continuous thickness transfer solid↔solid and solid↔gap, gap open/close conserving the rear interface), early stopping, seeding (`init=`) |
| `xsolve.py` | **current main solver**, details below |
| `fastfwd.py` | numba version of `gamma_batch` (matches to 1e-14; only 1.5× faster, since complex exp/sqrt dominate) |
| `echoes.py`, `diagnose.py` | echo detection (old pipeline) and stage-attribution diagnostics |
| `e050_budget.py`, `e051_init.py`, `e055_nn.py`, `beam.py` | experiment scripts (information budget, initialisers, NN, beam search) |

**What `xsolve.py` contains**
- **Forward:** an exact TE oblique-incidence forward `gamma_ang`, verified at 0° vs `gamma_batch` (bit-exact) and at 30° vs an independent recursion (3e-16).
- **Multi-view problems:** any set of angles (`make_views`).
- **Polishing:** a bounded least-squares `Polisher` (all frequencies, library prior as residuals).
- **Cycle scans:** the mmWave cycle scan (`cycle_scan`).
- **Low-contrast alternatives:** the low-contrast structural alternatives (`low_contrast_variants`).
- **Labels:** Laplace class-evidence labels with label sets (`class_evidence`).
- **Honesty checks:** unresolved-interface and visibility-depth checks.
- **Error bars:** profile-likelihood intervals plus the global profile scan (`profile_t`, `profile_scan`).
- **Optional:**
  - per-view standoff and angle offsets (`VIEW_OFFSETS`);
  - fast config (`FAST`);
  - per-layer resampling (`LAYER_SWEEP`);
  - noise continuation (`SIGMA_FLOOR_SCALE`).

**How to run**

```bash
cd docs/src/oned
python3 -c "import xsolve as XS; v,tr=XS.make_views(404,3,(0.0,),20.0); r=XS.solve(v,3,tr['standoff']+0.001); print(XS.score(r,tr,3))"
```

**Benchmark runners** (resumable JSONL, one line per wall):
- `docs/results/2026-11-10_E061_xsolve/run.py`:
  - args: `N shard nshard seed0 seed1 angles boost_db out.jsonl`;
  - env: `FAST=1`, `OFFS=1 POS_ERR=… ANG_ERR=…`.
- `docs/results/2026-11-10_E060_heldout/run.py` and `run_list.py` (with `SEEDS=` env).

**Infrastructure warning.** The cloud container is reclaimed when the session is idle. Detached `nohup` jobs die then. Run long jobs under a *tracked* background task, and keep runners resumable or checkpointed.

---

## 4. Experiment history and the main learning from each

### Phase A — 1D DEC physics and a drywall/insulation wall (E000–E040)
- **E000–E004 (forward model is exact).**
  - Exact (Trefftz) Hodge stars make 1D DEC exact with one cell per layer.
  - Standard stars need hundreds of cells.
  - A μ₀ constant mismatch caused a fake 1e-10 error; with μ₀ = η₀/c the match is 3e-15.
  - Random-multistart LM fails because of cycle skipping.
- **E005–E008 (ambiguity and honest benchmarks).**
  - Some answers are genuinely ambiguous (Δχ² < 4 basins), so the solver reports basin sets.
  - Benchmarks must randomise every nuisance parameter (truths at nominal values flatter the method).
- **E009–E012 (one-band limits).** Faster search and integer-ambiguity handling help. With one 77–81 GHz band, however, back-of-wall cycles are not identifiable.
- **E014 (information analysis).** The single-band failures are an information problem. A second band (UWB or 22–26 GHz) removes the wrong minima. One-cycle-shift math: residual ≈ 2π(f−fc)/fc.
- **E013 / E015 / E016.** Manufacturing priors plus a cascade fallback reduce flagged failures. Credible sets are given.
- **E017–E019.** Two bands give 19/20 with no priors.
- **E020 (reality gap).** The biggest effect is the antenna beam. Others: dispersion, face paper, roughness, gaps, band offset.
- **E021 / E022.** A causal (Jonscher / constant-Q) material law is needed. The legal 57–64 GHz and UWB bands both give 10/10.
- **E023 (antenna).** Antenna beam model plus Lambot multi-distance calibration.
- **E024–E027 (effects and speed).** Facings, roughness and band offset are handled. A compiled kernel and minimal cells cut the time to 0.9 s.
- **E028–E029 (learned initialiser).** The learned initialiser works on dev but **fails held-out**: 86/3/11 on 100 walls. Lesson: **every speed shortcut must be validated held-out.**
- **E030–E033.** The air gap is identifiable. The FMCW chain model works.
- **E035 (calibration).** No calibration plate in the field: factory calibration plus fitted per-band drift (self-calibration) gives 10/10. A drifted chain is never silent.
- **E037 (SNR).** No SNR cliff down to 15 dB per sweep. +20 dB averaging puts a 40 dB radar far from the cliff.
- **E038 (motion).** Handheld motion with joint multi-frame fits costs nothing.
- **E039 (fast path).** UWB-first fast path, held out: 100/0/0 at about 1 s.
- **E040 (stress tests).** Antenna tilt is handled by self-calibration up to about 2°. The noise level can be estimated from far-range bins.

### Phase B — arbitrary stacks and the material library (E034, E041–E045)
- **E034.** Greedy layer growth fails beyond about 3 layers.
- **E041.**
  - Exact layer peeling (Möbius downward continuation) is exact and fast.
  - It fixed a key bug: a free Jonscher p started on its bound and could never move.
  - But it is brittle held-out: 1/8 exact.
- **E042 / E043.** A global blocky TV inversion plus the library prior: physics is solved, the information is present, and the **global structure search is the unsolved part.**
- **E044.** The owner decided **N is a user input**. Knowing N removes silent failures, but accuracy stayed poor.

### Phase C — the stated known-N problem (E046–E067)
- **E046.**
  - Differential evolution (several misfits and coordinate systems): 0/4.
  - Three search-space bugs that made the truth unreachable were fixed:
    - p started on its bound;
    - a 0.25 % water floor;
    - a p cap.
- **E049 (echo pipeline).** Multiband sparse deconvolution, then candidate interface subsets, then exact screening, then LM.
  - Held-out: 3/20 exact, 15 flagged, 2 honest, 0 silent.
  - Good at the front of the wall only.
- **E050 (information budget, CRB only).**
  - At +0 dB, 92 % of layers are measurable; at +20 dB, 100 %.
  - 3 angles: 96 % at +0 dB.
  - A 0.1–0.4 GHz band adds nothing.
  - → **Identifiability is not the bottleneck; search is.**
- **E051 (initialisers, isolated at the true geometry).**
  - amplitude starts: 14/30;
  - library sweeps: 24/30;
  - layer peel: 18/30.
- **E052 (50-wall dev set, echo + material rescue).**
  - 15/50 OK, 2 silent.
  - Stage attribution: 33/35 failures start at **echo detection**. Interfaces closer than the UWB resolution c/2B ≈ 2 cm merge.
- **E056 (exact layer-stripping beam search, Möbius map per horizon): negative.**
  - The pulse half-width (about 4 cm) exceeds the layer optical thickness. Range-sequential decisions are therefore not separable: the wall must be fitted as a whole.
  - Code kept in `beam.py`.
- **E054 (parallel tempering, PT).** PT over the full continuous vector with physics moves, tempered below 11 GHz (mmWave is too rugged), then exact LM on all bands. Rescued 14, then 21 more, of the 35 echo failures, giving 36/50 cumulative.
- **E053 / E061 (`xsolve`; PT is the main solver for every wall).**
  - **Dev N = 3, 0°: 41/50 OK.**
  - Key fixes along the way:
    - continuous thickness-transfer moves;
    - the **mmWave integer-ambiguity scan**: foam/gap boundaries sit 1 λ/2 ≈ 2.4 mm off;
    - **Laplace class-evidence labels**: refit with each class prior, starting from the class centre;
    - the unresolved-interface check: two adjacent same-class layers mean an unseen interface;
    - the visibility-depth check.
- **E061 (3 angles, 0/15/30°, ideal geometry).**
  - 43/50, **0 silent**.
  - Fixes the low-contrast and label cases.
  - 2.4× compute.
- **E059 / E061d (+20 dB).**
  - Plain +20 dB: 40/50, 0 silent; 5 walls broken by a sharper landscape.
  - With **σ-floor tempering** (temper at nominal SNR, polish at the true σ): **44/50, 0 silent.**
- **E058 (fast config).**
  - Settings: early stop, adaptive restart, 48 frequencies per band, 6 replicas.
  - **41/50, 0 silent, median 77 s**, against 44/50 at about 250 s.
  - The forward is 90 % of the time and already transcendental-bound. The lever is fewer evaluations.
- **E055 (tiny NN initialiser, 7.8k params, 20k simulated walls).**
  - Learns the front layer and the backing (99 %), not the deeper layers.
  - Alone: 0/50.
  - As PT seeds: 33 vs 34 without.
  - → **Dropped**; the solver stays fully first-principles.
- **E060 (held-out, N = 3, 0°, +20 dB).**
  - **#1 (seeds 600–649): 41 OK + 6 flagged/honest + 3 silent = 47/50 correct-or-honest (clean).**
  - #1 after the later fixes (E060b gap/foam split variants; E060d label sets = classes within an evidence score of Δ ≤ 9): 49/50, but slightly contaminated.
  - **#2 (seeds 700–749, final code, clean): 39 OK + 10 flagged + 1 silent = 49/50.**
  - Exact rate 78–82 %; honesty about 98 %; about 4 min per wall.
- **E064 (profile-likelihood intervals plus global profile scan).**
  - Each thickness is pinned at t̂(1 ± 30 %) and everything else is re-polished.
  - It adopts better basins: 725 was a missed basin, Δχ² +44.6 below the alarm.
  - It widens intervals to cover equal-fit basins.
  - 641 remains: its truth sits at Δχ² = 4.8, a statistically expected 95 %-interval miss.
- **E062 (sweeping sensor: 3 angles with unknown pose, σ 1 mm / 1°).**
  - With per-view standoff offsets: 14/20 flagged.
  - With per-view angles fitted too, and an air-path parametrisation that decouples offset from angle:
    - 401 fixed;
    - but only **5/20 OK vs 18/20 ideal** (search failures).
  - → **Not recommended** without pose tracking to ≲ 0.1 mm / 0.1°. A stationary sensor with averaging is recommended.
- **E063 (N = 5 and N = 7; seeds 900–919, 0°, +20 dB).**
  - **N = 5: 4/20 OK, 15 flagged, 1 weak, 0 silent.**
  - **N = 7: 0/7 OK (stopped).**
  - Diagnosis:
    - The truth fits at noise with CRB < 1 mm (902, 904, 910, 916): **information is present, search fails.**
    - 3× PT budget fixed 1 of 4 (916).
  - Things tried that did not fix it:
    - **E065** per-layer global resampling (every class × thickness for one layer at a time): no gain.
    - **E066b** single-layer cycle-slip moves: no gain.
    - A probe showed the residual errors are **coordinated multi-layer** thickness + material differences, not single slips.
  - **Key finding (902):**
    - The σ floor (E061d), which helps N = 3, divides the UWB evidence by about 100 at +20 dB.
    - The wrong fit has a UWB χ²/pt of 86.6 at the true σ (truth 1.08), but it looks perfect (0.48) to the tempering.
    - At N = 5 the floored data are ambiguous.
  - **E066 (stage-2 PT at the true σ, seeded from stage 1):** 910 went from χ²/n 877 to 84, with 4/5 layers exact. 902 and 904 were not fixed.
  - **E067 (noise continuation: floor ×1 → ×0.316 → true σ → true σ + mmWave, each stage seeded):** written and launched on 902/904/910, then **stopped by the owner before finishing**. The per-stage checkpoints are in `docs/results/2026-11-11_E063_Nlayers/ckpt/*_s316|strue|smm.npz`, possibly partial.

---

## 5. Where it stands right now

| setting | result | silent | time |
|---|---|---|---|
| N = 3, held-out, 0°, +20 dB (recommended config) | 78–82 % exact, **≈ 98 % correct-or-honest** | 1 / 50 | ~4 min/wall (fast config ~77 s, 41/50 dev) |
| N = 3, 3 angles, ideal pose | 43/50 dev, 0 silent | 0 | ~9 min |
| N = 3, 3 angles, realistic pose error | 5/20 | 1 | ~10 min |
| **N = 5** | **4/20 exact**, 15 flagged, 1 weak | 0 | ~11 min |
| **N = 7** | 0/7 (stopped) | 0 | — |

**Reading:** for N = 3 the algorithm is honest and mostly accurate. For N ≥ 5 it stays honest (no silent failures), but the **global search does not find the answer**, even though the data determine it (CRB ≪ tolerance). The owner's target is N ≤ 7 working comfortably, so **this is the main open research problem.**

---

## 6. What is left to do (precise)

### P1 — Scale the global search to N = 5–7 (blocking; the main research question)
**Facts to build on**
- The information is present (truth χ²/n ≈ 1, CRB < 1 mm).
- Wrong basins differ from the truth in several layers' thicknesses AND materials at once.
- More budget barely helps (1/4).
- Single-layer resampling or slip moves don't help.
- Tempering below 11 GHz with the σ floor hides the discriminating UWB evidence. True-σ stage-2 tempering helped one wall a lot.

**Candidates, in order**
1. **Finish E067** (noise continuation) on 902/904/910. If it helps, add it to `xsolve` and rerun 20 walls at N = 5.
2. **Layer-growing search:**
   - solve with k = 2, 3, … front layers, the rest of the wall modelled as a half-space or "unknown tail";
   - each stage is seeded by the previous one and tempered over all k layers at the true σ.
   - Data gating by depth is NOT separable locally (E056), so each stage must be a global PT over all current layers, not a local decision.
3. **Better PT design for coupled moves:**
   - multi-layer block moves (resample 2–3 adjacent layers jointly: classes × thicknesses);
   - more replicas and temperatures scaled with N;
   - optionally, structure-level moves (swap two layers' materials).
4. **Compute.** The forward is already near hardware speed; multicore or numba prange PT is possible. Budget per wall grows with N.

**Acceptance:** N = 5 dev (900–919) at ≥ 80 % correct, 0 silent. Then N = 7. Then a fresh held-out set.

### P2 — Validate N = 3 once more with the final code
A clean held-out set with the final code including E064b (profile scan); E060 #2 predates E064b. Count "within-bars" separately.

### P3 — Realism in this benchmark
The knobs exist from Phase A (`realism.py`, E020–E040) but are OFF in the known-N benchmark. Turn them on and re-measure:
- antenna beam;
- per-band drift / self-calibration;
- FMCW chain;
- roughness;
- face paper.

### P4 — Speed toward sub-second
- Fast config: 77 s at N = 3.
- Options: fewer and cheaper PT evaluations (adaptive stopping, a coarse-to-fine frequency schedule), multicore, a compiled PT loop.
- Validate every speedup held-out (E029 lesson).

### P5 — Smaller open items
- Calibrate label-set probabilities (E060d uses a fixed evidence window Δ ≤ 9).
- Out-of-library materials ("unknown" flag; `wallsolve.free_refit` and `random_causal_material` exist but are untested in `xsolve`).
- Moisture output in `xsolve`. Water layers are excluded; wet porous hosts with CRIM are not exercised in this benchmark.
- Optional:
  - multi-angle with pose tracking;
  - a pose-aware search, if a sweeping sensor is wanted.

---

## 7. Recommended hardware / measurement design (current evidence)
- **Measurement:** three bands, a stationary sensor at 0°, +20 dB by chirp averaging (about 100 chirps, about 10 ms).
- **Calibration:** a one-time factory calibration plus fitted per-band drift.
- **Angles:** they help only with accurate pose (≲ 0.1 mm / 0.1°). Otherwise they hurt, with the current search.
