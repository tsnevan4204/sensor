# 10 · Owner preferences and decisions already made

## Preferences

- **Exact and guaranteed modelling is the priority.** Prefer exact or certified methods over approximate ones, and verify exactness numerically.
- **Low data, low compute.** Avoid learning anything that physics can supply. Learned components must be small and trained on simulated data.
- **Math over analogies** in explanations: show derivations.
- **Generality:** the algorithm should work across many wall and geometry configurations. Make everything adjustable through specs (no hard-coding).
- **Iterate experimentally** and learn from mistakes; keep going until the algorithm is strong.

## Decisions (with reasons)

| Decision | Reason |
|---|---|
| Frequency domain, not time domain | Data are frequency samples; exact water dispersion; exact 1D stars; no time-stepping error. A time trace, if needed, comes from an inverse FFT |
| Exact Hodge stars for the 1D toy | No discretization error; no training; fully general within 1D piecewise-constant layers |
| Metal-plate calibration | Removes A(f) with one scan; Γ = −1 exactly |
| Adjoint gradients | Cost is independent of parameter count; one extra right-hand side |
| Damped least squares (Levenberg–Marquardt) | About 10 parameters, least-squares misfit, fast local convergence |
| Bounded parameters through a logistic transform | Keeps iterates physical without constraint handling |
| Regularizers: smoothness and TV, λ by discrepancy principle | No validation-set tuning. Result: the data supports about one number for the profile, so a parametric profile is preferred |
| Validation must use randomized truths and report CRB and ambiguity sets | Earlier nominal truths flattered the method |
| Tie both drywall sheets' εr and tan δ by default (`same_drywall=True`) | Removes the worst ambiguity; it is realistic |
| Differential (dry-patch) measurement for moisture | Removes the θ ↔ drywall degeneracy |
| Noise is absolute (40 dB for a perfect reflector) | Weak echoes get realistically lower SNR |

## Open questions to raise with Prof. Trask / Kaibo Hu

- Stability theory for DDEC with indefinite (Helmholtz/Maxwell) operators. Which positivity and inf-sup conditions are needed for learned frequency-dependent stars?
- Learnable partition of unity for coarse cells in wave problems: cells must preserve phase velocity, not only fluxes.
- Mesh conditions (Delaunay, well-centered) for 2D DEC with PML.
