# 09 · Experiment loop protocol

The point of the loop: iterate quickly but honestly, so each round either improves a target metric in `07_roadmap.md` or explains a failure.

## One iteration

1. **Select.** Take the top open roadmap item, or the most important failure from the previous experiment.
2. **Hypothesis (write it before running).**
   - "Change X will move metric M from a to b on benchmark B, at compute cost ≤ C."
   - State the pass/fail threshold.
3. **Gates.** `pytest -q tests/` must pass: forward exactness, adjoint vs FD, solver residual check. Any new forward model must add its own gate.
4. **Run.**
   - Randomized truths (never nominal); fixed seeds for reproducibility.
   - Record accuracy **and** compute: wall time, number of forward solves, parameter count, training samples.
5. **Analyze.**
   - Look at every failure individually. Check χ² at the truth vs at the estimate:
     - truth fits better → optimizer problem;
     - fits equally → identifiability or ambiguity;
     - truth fits worse → model or benchmark bug.
   - Check err/CRB calibration.
6. **Log.** Append to `experiments/EXPERIMENT_LOG.md` using the template below. Save artifacts to `results/<YYYY-MM-DD>_<ID>/`.
7. **Commit.** Use `E0xx: <one-line result>`. Update `docs/06_findings_so_far.md` when a finding is established.
8. **Decide** the next iteration. If three consecutive iterations fail on the same item, step back: re-check identifiability (Fisher), the benchmark, and assumptions.

## Benchmarks

| Name | Contents | Typical size |
|---|---|---|
| B1-dry | Random three-layer dry walls (as in `scenarios.truth_random`) | 10–50 seeds |
| B2-wet | Wet insulation, dry-patch-known and unknown variants | 10–20 |
| B3-mismatch | Wrong host εr, temperature, mixing law, plate-distance error | 4 per case |
| B4-profile | Graded moisture profiles | 5–10 |
| B5-family (to build) | Random N-layer walls from a material library, unknown N | 100+ |
| 2D benchmarks | Mie cylinders, stud + pipe cross-sections | to build in Phase 2 |

## Log template (copy into `experiments/EXPERIMENT_LOG.md`)

```markdown
## E0xx — <short title>  (<date>)
**Roadmap item:** <phase.item>
**Hypothesis:** <change> will move <metric> from <a> to <b> on <benchmark>; compute ≤ <C>.
**Change:** <files/functions, key idea, math>
**Gates:** pass/fail (<numbers>)
**Results:**
| metric | before | after |
|---|---|---|
**Failures examined:** <seed: cause (optimizer / identifiability / model / benchmark)>
**Conclusion:** accepted / rejected / partial — why
**Next:** <next experiment>
**Artifacts:** results/<date>_E0xx/
```

## Rules that keep it honest

- Don't change the benchmark and the method in the same experiment.
- Don't set starting points or bounds from the truth.
- Report silent failures separately from flagged failures.
- Keep the compute budget visible in every result table.
