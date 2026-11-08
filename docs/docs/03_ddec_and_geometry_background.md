# 03 · DDEC, Whitney forms, and geometric-numerics background

PDFs are in `papers/`. Re-read the originals before citing specifics; the summaries below may simplify.

## Trask, Huang, Hu (2022). *Enforcing exact physics in scientific machine learning: a data-driven exterior calculus on graphs.* JCP 456, 110969.

- Builds a graph exterior calculus: coboundary operators δ_k are signed incidence matrices, so δ_{k}∘δ_{k−1} = 0 exactly.
- **Learns the metric**: the inner products / Hodge stars and nonlinear fluxes, as a perturbation of a Hodge Laplacian.
- Conservation and exact-sequence properties hold independently of training error or data quantity.
- Well-posedness theory for nonlinear perturbations of div–grad and div–curl problems.
- Trained by **PDE-constrained optimization** (physics as an equality constraint, no penalty weights).
- Shows commuting diagrams preserved under coarsening.
- **Limitation for us:** the analysis is for elliptic, positive-definite problems. Time-harmonic Maxwell (curl-curl − ω²ε) is indefinite.

## Actor, Hu, Huang, Roberts, Trask (2024). *Data-driven Whitney forms for structure-preserving control volume analysis.* JCP 496, 112520.

- Replaces barycentric coordinates in Whitney forms with a **learnable partition of unity (POU)**. This learns control volumes, i.e. a coarse mesh adapted to the physics, while keeping the de Rham structure.
- Produces structure-preserving reduced-order models without a full-order model.
- Example: transport through microstructure reproduced to under 1% with 8 data-driven Whitney forms.
- Relevance: learned coarse cells for walls with sub-cell structure (studs, insulation fibers, wet patches). This is homogenization done structure-preservingly.

## Kinch, Shaffer, Armstrong, Meehan, Hewson, Trask (2025). *Structure-preserving digital twins via conditional neural Whitney forms.* arXiv 2508.06981.

- Conditions the reduced FEEC model on a latent variable Z (sensor readings or parameters). Conditional attention learns both the reduced basis (POU) and a nonlinear conservation law.
- Guarantees well-posedness and exact conservation regardless of data sparsity or optimization error.
- Supports real-time calibration.
- Claimed: discretization invariance and generalization across geometries.
- Mapping to RF: Z = (frequency, wall type, moisture, cell geometry). The learned object is the frequency-dependent cell star / closure.
- Positive-definiteness again; the wave problem needs an extension.

## Zhu, Christiansen, Hu, Hirani (2025). *Convergence and stability of DEC for the Hodge–Laplace problem in 2D.* arXiv 2505.08966.

- Proves DEC convergence on non-degenerate Delaunay, shape-regular 2D meshes by relating DEC to lowest-order FEEC.
- Includes a Poincaré inequality and a discrete inf-sup condition.
- The DEC diagonal Hodge star is a mass-lumped Whitney/FEEC star; DEC on such meshes is linked to Yee's scheme.
- In 1D every mesh is trivially well-centered. **In 2D these conditions decide which meshes are allowed.** It is also the theory to extend to indefinite (Helmholtz/Maxwell) problems; Kaibo Hu is the person to ask.
- Other Hu topics that may matter:
  - Regge/metric finite elements, to parameterize an anisotropic ε tensor as a metric (transformation-optics view).
  - Helicity- and topology-preserving magnetic relaxation (structure-preserving time stepping).
  - Graph Hodge Laplacians.
  - See `papers/Kaibo_Hu_research_page.txt`.

## Oliver Gross (with Albert Chern): geometry-first simulation

See `papers/Gross_talk_transcript_geometry_and_physics_simulation.txt`.

- **Force-free fields are conformally geodesic** (Chern & Gross, SIAM J. Appl. Algebra Geom. 2024, arXiv 2312.05252).
  - Field lines are geodesics of a conformally rescaled metric. Equivalently, light rays in an index of refraction n follow geodesics of the optical metric n²g.
  - **Use for us:** in a hybrid solver, the ray (geometric-optics) region and the full-wave DEC region can share **one material field ε(x)**. The rays are geodesics of $n^2 g$, with $n=\sqrt{\varepsilon_r}$, and the full-wave part uses $\star_\varepsilon$. Inverting the ray region is travel-time tomography, related to the boundary-rigidity problem.
- **Discrete elastic curves via an isoperimetric characterization.**
  - Elastic curves are critical points of length with area and volume vectors fixed. This is a lower-order (first-derivative) problem, much better conditioned than the bending-energy formulation.
  - **Lesson:** look for lower-order, better-conditioned reformulations of the inverse problem.
  - Possible use: elastic or isoperimetric shape priors for reconstructed boundaries (pipes, studs) instead of generative fill-in.
- **Differentiable lifts with cheap adjoints** (shape-change locomotion): the same machinery as our adjoint inversion.

## Other relevant techniques (not yet used)

- **Integral equations with layered-media Green's functions:**
  - The classic through-the-wall radar approach (Moeness Amin, Villanova). Discretize only the scatterer; the wall is analytic. Contrast-source inversion, distorted Born iteration.
  - The strongest classical baseline in 2D/3D, and a natural partner in a hybrid scheme.
- **Trefftz methods:** element bases that solve the PDE exactly (plane waves). Exact in 1D; exponentially convergent in 2D/3D. This is the idea behind the exact stars.
- **Numerical homogenization / multiscale FEM:** cell response via local solves (Schur complements). Structure-preserving versions give data-free coarse stars.
- **Diffusion priors with a physics likelihood** (diffusion posterior sampling; e.g. Back-Projection Diffusion arXiv 2408.02866; plug-and-play latent diffusion for EM inverse scattering arXiv 2509.04860). The prior proposes, the exact physics scores.
- **Unrolled / learned iterative reconstruction:** small networks interleaved with forward and adjoint solves.
- **Perfectly matched layers as Hodge-star modifications:** complex coordinate stretching only changes the metric, so D stays exact. Use this for 2D/3D open boundaries.
