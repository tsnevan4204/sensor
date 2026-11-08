# 08 · Toolbox: keeping data and compute low

Each entry: what it is, why it reduces cost, and its status in this repo.

## Exactness that removes learning or fine meshes

- **Exact (Trefftz) Hodge stars in 1D.** Closed-form, frequency-dependent cell stars. One cell per layer is exact, so no fine mesh and no training data. *Used.*
- **Schur-complement cell stars.** Eliminate cell interiors exactly from a fine local solve, computed once per cell type offline. Gives data-free coarse stars in 2D. *Planned (Phase 2).*
- **Exact boundaries.** In 1D the characteristic/impedance condition is exact. In 2D/3D use a PML written as a Hodge-star modification. *1D used.*
- **Analytic regions.** Air (and ideal layered walls) via exact propagation factors or layered Green's functions; discretize only where needed. *Standoff used.*

## Structure that makes learning cheap and safe

- **Learn only the metric.** Exterior derivatives stay integer matrices, so the learned part is small: scalar factors per cell type, or a small partition-of-unity network.
- **Positivity / passivity / causality by construction** (softplus factors, pole-residue frequency dependence). There are no constraint penalties to tune, and every iterate is physical.
- **Grade learned parts against known truths** (exact 1D stars, Schur complements) before using them. This catches generalization failures without real data.
- **Conditioning on Z** (conditional Whitney forms): one small model covers many frequencies, materials and moistures.

## Optimization tricks (fewer forward solves)

- **Adjoint gradients:** one extra right-hand side per frequency gives gradients for all parameters. The matrix is complex symmetric, so the same factorization serves. *Used.*
- **Levenberg–Marquardt** on a least-squares misfit: second-order convergence near the optimum with about 10 parameters. *Used.*
- **Variable projection / profiling:** eliminate parameters that enter simply. The standoff (a pure phase ramp) is profiled exactly on a grid at the cost of one matrix product. *Used.* Next candidates: overall amplitude, a polynomial residual in A(f).
- **Phase-period lattices:** false minima sit at known spacings λ/(2 Re n), so start once per basin instead of randomly. *Used.* Cycle hopping between basins. *Used.*
- **Time gating / layer stripping:** echo groups separated in time are fitted separately. The same linear gate is applied to model and data, so there is no bias. *Used.*
- **Coarse-to-fine in frequency count:** search basins with 16–32 frequencies, refine with 256. *Planned.*
- **Amortized initialization:** a tiny simulated-data model predicts basins and starting values; exact LM finishes. *Planned.*
- **Model-order reduction across frequency** (Padé/Krylov, reduced basis). *Planned (2D).*
- **Factorization reuse / low-rank updates** when only a few cells change between LM trials. *Planned.*

## Statistics that prevent wasted effort

- **Fisher information / CRB before experimenting:** if a parameter is unidentifiable (singular values ≪ 1), fix it, tie it, or add information; don't optimize harder. *Used.*
- **Profile likelihood and ambiguity scans** expose discrete non-uniqueness that the CRB cannot see. *Used.*
- **Discrepancy principle** for regularization weights: no grid search over validation sets. *Used.*
- **χ²/n as a built-in failure detector.** *Used.*

## Measurement-design tricks (physics, not compute)

- **Calibration scans** (metal plate, Γ = −1): remove the sensor response with one scan. *Used.*
- **Differential measurement** (dry patch first): removes the θ ↔ drywall ambiguity. *Used.*
- **Second band or multiple angles:** add information exactly where the Fisher analysis shows it is missing. *Planned.*
- **Parameter tying from prior knowledge** (same product, standard thicknesses). *Used (shared drywall material).*
