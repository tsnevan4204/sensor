# 02 · Physics derivation (1D layered wall → DEC → Hodge stars)

Conventions: time dependence $e^{+j\omega t}$; lossy permittivity $\varepsilon = \varepsilon' - j\varepsilon''$ (fold conductivity in as $\varepsilon \to \varepsilon - j\sigma/\omega$); SI units; $\mu_0 = \eta_0/c$.

## 1. Maxwell's equations split into topology and metric

In differential forms:

$$
\underbrace{dE = -\partial_t B,\quad dH = \partial_t D + J,\quad dB = 0,\quad dD=\rho}_{\text{topological: exact on any mesh}}
\qquad
\underbrace{D=\star_\varepsilon E,\quad B=\star_\mu H,\quad J=\star_\sigma E}_{\text{metric / constitutive: the material}}
$$

| Field | Units | Type | Integrated over | Physical reading |
|---|---|---|---|---|
| E | V/m | 1-form | curves | voltage along a path |
| B | Wb/m² | 2-form | surfaces | magnetic flux through a loop |
| H | A/m | 1-form (dual / twisted) | dual curves | enclosed current |
| D | C/m² | 2-form (dual) | dual surfaces | enclosed charge |
| J | A/m² | 2-form (dual) | dual surfaces | current through a surface |
| ρ | C/m³ | 3-form (dual) | dual volumes | charge in a volume |

The two de Rham complexes:
- **Primal:** $\varphi \xrightarrow{\text{grad}} E \xrightarrow{\text{curl}} B \xrightarrow{\text{div}} 0$
- **Dual:** $H \xrightarrow{\text{curl}} (D, J) \xrightarrow{\text{div}} \rho$

Hodge stars map a primal k-form to a dual (3−k)-form. **Recovering the wall equals recovering the Hodge stars.**

## 2. Cochains and exact discrete derivatives

A k-cochain stores **integrals** of a k-form over the k-cells of a mesh: one number per vertex, edge, face or cell. By Stokes' theorem, $\int_S d\omega = \oint_{\partial S}\omega$, so $d$ becomes a signed incidence matrix with entries in $\{0,\pm 1\}$, with no approximation.

Example on one triangle (counterclockwise face orientation):

$$
\mathsf D_0=\begin{pmatrix}-1&1&0\\0&-1&1\\-1&0&1\end{pmatrix},\qquad
\mathsf D_1=\begin{pmatrix}1&1&-1\end{pmatrix},\qquad
\mathsf D_1\mathsf D_0=0 .
$$

Approximation enters **only** through the Hodge star. A cochain holds integrals over primal cells, but the star must output integrals over dual cells, and that requires the field's shape *inside* cells, which the cochain does not contain. Quadrature does not fix this: quadrature integrates an *assumed* shape exactly, but the assumed shape is wrong.

## 3. Reduction to 1D (normal incidence on planar layers)

Take $\mathbf E = E_x(z)\hat x$ and $\mathbf H = H_y(z)\hat y$:

$$
\partial_z E_x = -\mu\,\partial_t H_y,\qquad -\partial_z H_y = \varepsilon\,\partial_t E_x + \sigma E_x .
$$

**Why the form degree drops by one.** Nothing varies in x. Take a strip of unit width in x.
- E's natural number is a line integral across the unit x-rung at $z_i$: $e_i=\int_0^1E_x\,dx=E_x(z_i)$. On the z-line it is a **0-cochain** (a value at a node).
- B's natural number is the flux through the panel between rungs: $b_{i+\frac12}=\iint B_y\,dx\,dz$. On the z-line it is a **1-cochain** (a value on an edge).
- H, D and J live on a y–z strip shifted half a cell: H at dual nodes (edge midpoints), D and J on dual cells around the nodes.

Both $E_x$ and $H_y$ are tangential to the interfaces (planes z = const), so both are **continuous across every interface**.

## 4. The 1D DEC model

Integrate Faraday's law over primal edges and Ampère's law over dual cells. Both are exact:

$$
\dot{\mathbf b} = -\mathsf D_0\,\mathbf e,\qquad \dot{\mathbf d} + \mathbf j = \mathsf D_0^{\top}\mathbf h,\qquad \mathsf D_0\in\{0,\pm1\}^{N\times(N+1)} .
$$

Close the system with the Hodge stars, $\mathbf d=\star_\varepsilon\mathbf e$, $\mathbf h=\star_{\mu^{-1}}\mathbf b$, $\mathbf j=\star_\sigma\mathbf e$:

$$
\star_\varepsilon\dot{\mathbf e}+\star_\sigma\mathbf e = \mathsf D_0^{\top}\star_{\mu^{-1}}\mathbf b,\qquad
\star_\varepsilon\ddot{\mathbf e}+\star_\sigma\dot{\mathbf e}+\mathsf D_0^{\top}\star_{\mu^{-1}}\mathsf D_0\,\mathbf e=\text{sources}.
$$

Equivalent circuit: an LC ladder, with shunt C = $\star_\varepsilon$, shunt G = $\star_\sigma$ and series L = $\star_{\mu^{-1}}^{-1}$.

**Why keep H, D and J at all?** Eliminating them to get a wave equation in E alone puts ε and μ inside the curl equations, so those equations are no longer exact on a mesh. Keeping them separate isolates all of the material in the stars. One step of the scheme goes $\mathbf e\to\mathbf b\to\mathbf h\to(\mathbf d,\mathbf j)\to\mathbf e$.

**Energy (discrete Poynting).** Let $W=\tfrac12\mathbf e^\top\star_\varepsilon\mathbf e+\tfrac12\mathbf b^\top\star_{\mu^{-1}}\mathbf b$. Then

$$\dot W=-\mathbf e^\top\star_\sigma\mathbf e+\text{boundary power},$$

for **any** symmetric positive (semi)definite stars. The coupling terms cancel because the Ampère operator is $\mathsf D_0^\top$. Constraints on learned stars: $\star_\varepsilon\succ0$, $\star_{\mu^{-1}}\succ0$, $\star_\sigma\succeq0$, and causality (Debye or Lorentz pole forms) if they depend on frequency.

## 5. Boundaries (exact in 1D)

On the radar side (air, incident amplitude $a$), with $E=ae^{-jkz}+re^{jkz}$, the field satisfies $H(0)=(2a-e_0)/\eta_0$. Behind the wall, outgoing waves only: $H(L)=e_N/\eta_{\text{back}}$.

Contributions to the system: conductance $1/\eta_0$ on node 0, source $2a/\eta_0$, conductance $1/\eta_{\text{back}}$ on node N.

Measurement: $\Gamma=(e_0-a)/a$. The air standoff $d$ is handled analytically by the factor $e^{-2jk_0d}$. This is the "hybrid" in its simplest form: exact physics in air, discretized physics in the wall.

## 6. Frequency domain (the default)

$$
\big(\mathsf K-\omega^2\star_\varepsilon(\omega)+j\omega\star_\sigma+\mathsf B\big)\hat{\mathbf e}=\hat{\mathbf s},\qquad \mathsf K=\mathsf D_0^\top\star_{\mu^{-1}}\mathsf D_0 .
$$

- **Tridiagonal**, so O(N) per frequency.
- **Complex symmetric**, so the adjoint reuses the factorization.
- **Indefinite.** This is fine in 1D, but it is the open stability issue in 2D.
- **Uniquely solvable for every ω > 0.** If $A\mathbf e=0$, the imaginary part of the energy identity forces $e_0=0$ through the boundary conductance, and the tridiagonal recursion with nonzero off-diagonals then forces $\mathbf e=0$.
- **Dispersion is exact:** use the complex ε(ω) of water directly.

**Time domain (leapfrog/Yee)** is explicit, sparse and symplectic for σ = 0, and stable under the CFL condition $\Delta t\le\min_i\ell_i\sqrt{\bar\varepsilon_i\bar\mu_i}$. It is never exact, though: spatial and temporal dispersion accumulate as phase drift, and Debye water needs an auxiliary-equation polarization ODE. If a time trace is needed, inverse-FFT the frequency-domain result instead.

## 7. Standard vs exact vs DDEC Hodge stars (one edge, length ℓ, constant ε, μ, $k=\omega\sqrt{\varepsilon\mu}$)

Every cell model is a 2×2 map $K[e_0;e_1]=[H(0);-H(\ell)]$. A symmetric matrix with equal diagonal entries splits uniquely as

$$
K=\begin{pmatrix}a&-b\\-b&a\end{pmatrix}=b\,\mathsf D_0^\top\mathsf D_0+(a-b)\,I,\qquad S:=b=\tfrac{1}{j\omega}\star_{\mu^{-1}},\quad M:=a-b=j\omega\star_\varepsilon^{(\text{half cell})}.
$$

**Exact.** The solution inside the cell is $E(z)=[e_0\sin k(\ell-z)+e_1\sin kz]/\sin k\ell$ and $H=-E'/(j\omega\mu)$, giving

$$
S_{\text{exact}}=\frac{1}{j\omega\mu\ell}\cdot\frac{k\ell}{\sin k\ell}=\frac{-jk}{\omega\mu\sin k\ell},\qquad
M_{\text{exact}}=j\omega\frac{\varepsilon\ell}{2}\cdot\frac{\tan(k\ell/2)}{k\ell/2}=\frac{jk\tan(k\ell/2)}{\omega\mu}.
$$

**Standard (Whitney, lumped).** Assume E linear and H constant on the edge, and lump the half-cell integral:

$$
S_{\text{std}}=\frac{1}{j\omega\mu\ell},\qquad M_{\text{std}}=j\omega\frac{\varepsilon\ell}{2}.
$$

**Ratios.** $S_{\text{exact}}/S_{\text{std}}=k\ell/\sin k\ell=1+(k\ell)^2/6+\dots$ and $M_{\text{exact}}/M_{\text{std}}=\tan(k\ell/2)/(k\ell/2)=1+(k\ell)^2/12+\dots$

**Discrete dispersion.** On a uniform chain, $\cos\kappa\ell=a/b$:
- Standard: $\sin(\kappa\ell/2)=k\ell/2$, so $\kappa\ne k$ with error $\approx k(k\ell)^2/24$.
- Exact: $\kappa=k$ exactly.

At 20 cells per wavelength the error is 0.4% per cell. Over the roughly 300 rad round-trip phase through 89 mm of insulation at 79 GHz, that adds up to more than 1 rad, so the standard star needs about 640 cells per layer just to reach about 1e-2 error in Γ. The exact star reaches 1e-14 with **one cell per layer**.

**Positivity.** The exact star is positive only for kℓ < π (sin kℓ = 0 is a pole). The code uses cells with |k|ℓ ≤ 0.9π. **The result is exact for any cell count**; the cell rule is only for positivity and conditioning.

**DDEC.**
$$
S_\theta=S_{\text{std}}\cdot s_\theta(\xi),\qquad M_\theta=M_{\text{std}}\cdot m_\theta(\xi),\qquad \xi=(\omega,\varepsilon,\ell,\theta_w,\dots)\ \text{or}\ k\ell .
$$
Constrain $s_\theta, m_\theta > 0$ (softplus) and fit by PDE-constrained least squares. On the training range the result should approach the exact factors. In 1D the exact factors are known in closed form, so learned stars can be graded against the truth.

**General definition of the exact cell star** (any dimension, any internal structure). Split a cell's fine-mesh unknowns into boundary (b) and interior (i) and eliminate the interior with the Schur complement:

$$K_{\text{exact,cell}}(\omega)=A_{bb}-A_{bi}A_{ii}^{-1}A_{ib}.$$

In 2D this is dense, depends on ω and has no closed form. A DDEC star is a low-parameter, structure-preserving approximation of it.

**What generalizes.** The exact star is a *formula* in (ω, ε, μ, ℓ), not a fit. It works for any wall, band or radar that satisfies its assumptions: 1D, piecewise-constant cells, single frequency. It does not cover continuous ε(z) inside a cell (use thin sub-layers), or 2D/3D (no finite exact star exists, because the local solution space is infinite-dimensional).

## 8. Measurement model, calibration, inversion

- **Measurement:** $y(f)=A(f)\,\Gamma(f;p)+n$. The radar response $A(f)$ is removed with a metal-plate scan ($\Gamma=-1$ exactly) at a known distance.
- **Adjoint gradient:** $A$ is symmetric, so $\partial\Gamma/\partial p=-\lambda^\top(\partial A/\partial p)\,e$ with $A\lambda=\mathbf 1_0$. That is one extra right-hand side per frequency for **all** parameters. Per-cell sensitivities:
  - $\partial/\partial p=\sum_c\big[\partial_p b\,(\lambda_c-\lambda_{c+1})(e_c-e_{c+1})+\partial_p M(\lambda_ce_c+\lambda_{c+1}e_{c+1})\big]$
  - $\partial_k b=\frac{-j}{\omega\mu}\frac{\sin x-x\cos x}{\sin^2x}$, $\partial_\ell b=\frac{jk^2\cos x}{\omega\mu\sin^2x}$
  - $\partial_kM=\frac{j}{\omega\mu}\big(\tan\frac x2+\frac x2\sec^2\frac x2\big)$, $\partial_\ell M=\frac{jk^2\sec^2(x/2)}{2\omega\mu}$, with $x=k\ell$
  - $\partial k/\partial\varepsilon_r=k_0/(2n)$
- **Moisture closure (CRIM):** $\sqrt{\varepsilon}=(1-\theta)\sqrt{\varepsilon_{\text{host}}}+\theta\sqrt{\varepsilon_w(\omega)}$.
- **Water (Debye):** $\varepsilon_w=\varepsilon_\infty+\frac{\varepsilon_s-\varepsilon_\infty}{1+j\omega\tau}$, with τ ≈ 9.4 ps at 20 °C. This gives about 11.5 − 20.8j at 77 GHz.
