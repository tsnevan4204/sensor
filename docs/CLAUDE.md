# CLAUDE.md — radar-ddec

You are continuing a research project that already has working code and validated results. Read this file, then `docs/00_INDEX.md`, before changing anything.

## Mission

Build a **physics-exact, low-compute inversion algorithm** that recovers hidden structure *and material properties* (permittivity, conductivity, moisture) behind walls and other occluders from radar measurements (RF / mmWave).
- It should generalize across geometries: 1D layered walls first, then 2D cross-sections, then 3D if feasible.
- The mathematical backbone is **data-driven exterior calculus (DDEC) and data-driven Whitney forms** (Trask lab, UPenn PIMI). Topology (exterior derivatives, incidence matrices) is kept exact. Only the metric/material part (Hodge stars, closures, sensor model) is approximated or learned.
- The goal is an algorithm that can be trained on many kinds of *simulated* data and becomes very strong, while staying **cheap in data and compute**.

## Non-negotiable constraints (from the project owner)

1. **Exactness first.** Every claim of exactness must be verified numerically.
   - Forward models must match an independent reference (transfer matrix, Mie series, converged fine mesh) to machine precision where the theory says they are exact.
   - Otherwise they must converge at the predicted rate, with an error bound below the noise floor.
2. **Low compute and low data.** Prefer mathematical and physics tricks over bigger models or more data.
   - Learned components should be small (aim for well under 10k parameters) and trainable on simulated data on a CPU in minutes.
   - Real measurements are only for low-dimensional laws physics can't give: material closures and the sensor response.
3. **Structure guarantees are kept by construction, not by penalty.** Exterior derivatives stay integer incidence matrices. Learned Hodge stars are parameterized to be positive/passive (and causal if frequency dependent). The discrete energy balance must hold for any parameter value.
4. **Frequency domain is the default.** Radar data are frequency samples, water dispersion is exact as complex ε(ω), and in 1D the Hodge stars can be made exact. Time domain only with a stated reason.
5. **Honest reporting.**
   - Report local error bars (Cramér–Rao bound) **and** global ambiguity (all basins within Δχ² < 4).
   - Never tune the benchmark to make results look better. True walls must be randomized, and starting guesses must not equal the truth.
   - A failure the algorithm flags itself through χ² is acceptable and should be counted. A silent failure is a bug.
6. The owner prefers **math over analogies** in explanations.

## Repo layout

```
CLAUDE.md                 this file
docs/                     all context (start at docs/00_INDEX.md)
src/oned/                 1D layered-wall code (run scripts from this folder)
  wall_toy.py             simulated world: materials, reference TMM solver, radar + calibration scans
  exact_hodge.py          teaching: standard vs exact Hodge stars on the same DEC topology
  wall_inverse.py         the algorithm: exact-star DEC forward, adjoint Jacobian, LM, gating, lattice starts,
                          standoff profiling, cycle hopping, ambiguity scan, Fisher/CRB, regularizers
  scenarios.py            wall definitions (truths + starting models); add scenarios here
  run_validation.py       validation suite (about 26 min on 2 cores)
tests/test_gates.py       correctness gates; MUST pass before any experiment result is trusted
experiments/EXPERIMENT_LOG.md   every experiment, newest last (historical E000–E007 already logged)
results/<date>_<ID>/      outputs of each experiment
papers/                   key papers (DDEC, Whitney forms, DEC convergence, MIT mmWave papers, Gross talk)
```

## How to work: the experiment loop

Follow `docs/09_experiment_loop_protocol.md`. In short:
1. Pick the highest-priority open item in `docs/07_roadmap.md` (or a failure from the last experiment).
2. Write a falsifiable hypothesis with a metric and a pass/fail threshold *before* running.
3. Implement it on a branch. Run `pytest tests/` (the gates) first.
4. Run the benchmark with randomized truths. Record compute (wall time, number of forward solves) alongside accuracy.
5. Append an entry to `experiments/EXPERIMENT_LOG.md`: hypothesis, change, results table, what failed, why, next step. Save outputs to `results/<date>_<ID>/`.
6. Commit with a message referencing the experiment ID. Then pick the next item.

When a result surprises you, find the cause before moving on. Every major fix so far came from this:
- the μ₀ constant mismatch
- standoff bounds that excluded the truth
- a benchmark truth that sat at the nominal starting point
- moisture held fixed in the wrong stage

## Coding conventions

- Time dependence e^{+jωt}, so lossy ε = ε' − jε''. SI units. μ₀ = η₀/c (2019 SI). Never use 4π·10⁻⁷ next to η₀ = 376.730313668.
- Every new forward model ships with:
  - an exactness or convergence test against an independent reference
  - an adjoint Jacobian checked against central finite differences (relative difference ≲ 1e-6)
- Linear solves: verify by residual, and fall back to a pivoting solver (see `_tridiag_solve`).
- Keep parameters adjustable through spec objects (`WallModel`, `LayerSpec`, `P(value, lo, hi, free, tie)`), not hard-coded.
- Plots go to `results/`, never only to the screen.

## Quick start

```bash
pip install -r requirements.txt
pytest -q tests/                      # gates (< 1 min)
cd src/oned && python3 wall_toy.py    # sanity demo
python3 run_validation.py             # full 1D validation (~26 min, 2 cores)
```
