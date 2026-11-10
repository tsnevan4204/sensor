# 11 · Stability of the indefinite discrete problem with learned Hodge stars

**Question (doc 10):** which conditions on learned, possibly frequency-dependent, complex Hodge stars guarantee that the time-harmonic system $A(\omega)e=s$ is uniquely and stably solvable, with constants explicit in $k$ and $h$? And why does 2D need learned or approximate stars when 1D has exact ones?

Code: `src/stability/`. Gates: `tests/test_stability.py`. Outputs: `results/2026-11-09_S001_stability/`. The tests run in about 8 s, the experiments in about 6 min on one core.

**Labels.** **[proved]** means a full proof is given here; every proved inequality is also checked numerically. **[literature]** means the result is quoted from a paper and was not re-proved. **[observed]** means a numerical fact, not proved. **[conjecture]** is an open claim, with the evidence for it.

## 0. Summary

| # | Statement | Status | Numerical check |
|---|---|---|---|
| A2 | 1D: a learned star that is symmetric, passive cell by cell ($\operatorname{Re}Y_\pm\ge0$), has $b_c\neq0$, and has $\operatorname{Re}G_0>0$ gives a uniquely solvable system | proved | 400 random walls (exact stars) + 233 with learned stars; every singular cell shift is active (max $\operatorname{Re}\delta\,\eta_0=-0.59$) |
| A3 | 1D: $\lvert(A^{-1})_{ij}\rvert\le 2\sqrt{\eta_i\eta_j}\,c_G\,P$ with $P=\prod_c\lVert\check T_c\rVert^2$, for **any** learned symmetric passive star | proved | max ratio to the bound 0.185 (exact stars), 0.131 (learned stars) |
| A4 | Lossless passband cells: $P\le$ product of effective-impedance mismatches. This is **independent of k and of the mesh**. Standard star: $\eta_\text{eff}=\eta/\sqrt{1-(k\ell/2)^2}$, passband iff $k\ell<2$ | proved | $P/\Pi_\text{eff}\in[0.46,1]$; the $\eta_\text{eff}$ formula holds to 6e-16 |
| A5 | Exact stars: $A^{-1}_{ij}=g(z_i,z_j)$ (sampled Green's function), so $A^{-1}$ is mesh-independent. The pole $\sin k\ell=0$ affects $\lVert A\rVert$ (conditioning), not $\lVert A^{-1}\rVert$ (stability) | proved | 1.3e-14 vs closed form; 6.7e-13 under refinement; $\lVert A^{-1}\rVert\le\eta_0$ while $\lVert A\rVert$ reaches $4408/\eta_0$ |
| A6 | "Positive stars" is a time-domain condition. At a fixed ω only passivity matters: exact passive cells can have $\operatorname{Re}b<0$ or $M/M_\text{std}<0$. Causality needs positive-real (Foster) frequency dependence | proved | $\operatorname{Re}b/\lvert b\rvert$ down to −0.99999 in passive cells; Foster series hold to 1e-5 |
| B2 | 2D tensor (Yee) grid, diagonal passive stars, $S_e\ne0$, impedance on one **full** side: the system is invertible for every $k$ | proved | one-side cavity: $\min\beta_N=1.9\cdot10^{-4}>0$ on a 301-point k-scan |
| B3 | Passivity alone is **not** sufficient: there are passive singular systems | proved (explicit) | $\sigma_\min=3\cdot10^{-16}$; pocket: $\sigma_\min\propto s_\min$ |
| B4 | Perturbation: $\beta_\theta\ge\beta_\text{ref}-\delta$ with $\delta=\max(\lVert s_\theta-s_\text{ref}\rVert_\infty,\lVert m_\theta-m_\text{ref}\rVert_\infty)$ | proved | sharp in $k$: singular uniform active scalings have $\lvert\tau\rvert\approx4.5/k$, while $\beta\approx1.2/k$ |
| B5 | Strictly lossy stars: $\lVert A^{-1}\rVert_{M_0^{-1}\to N}\le C_\tau/k$ with explicit $C_\tau$, for any learned star in the stated Loewner class | proved | holds in all 10 cases; tight within 1.09 at $k=80,\tau=0.2$ |
| B6 | PML (complex-stretched stars) is **not** passive. The energy theorems do not cover it | proved | $\lambda_\min(\operatorname{Im}_HA)=-13.5$; still $\beta_N=0.14$ |
| B7 | Mass blending changes $\beta$ by at most $\alpha_0[\tfrac29(kh)^2+\tfrac13kh]$ | proved | operator norm 0.049 vs bound 0.060 |
| B8 | $\beta_N\approx1.2/k$ for DEC, blended $(\tfrac12,\tfrac12)$ and Q1 stars at 10 ppw and at $k^3h^2$ fixed. Dispersion correction improves accuracy, not stability | observed | $k\beta_N\in[1.12,1.33]$ for $k=5\ldots160$ |
| B9 | Blended stars $(\alpha_1,\alpha_0)=(\tfrac12,\tfrac12)$ give 4th-order **isotropic** dispersion. Q1 with Ainsworth–Wajid mass $(1,\tfrac12)$ is only 2nd order in 2D | proved (symbol) | phase error at 10 ppw: DEC 1.7e-2, Q1+AW 8.7e-3, $(\tfrac12,\tfrac12)$ 3.3e-4 |
| C1–C2 | 1D: the local Trefftz space has dimension 2, equal to the number of DOFs, so an exact 2×2 star exists. 2D: the local Trefftz space is infinite-dimensional, so no finite exact star exists; the best 4×4 vertex star has defect ≈ 0.01 (kH)² | proved (1D); numerical certificate (2D) | rank 2 = 2 in 1D vs rank 8 > 4 in 2D; defect slope 1.98 |
| C3 | Fine-relative exact star = cell Schur complement. It is dense, of size ~4H/h, rational in $k^2$ with rank-one PSD residues at the cell Dirichlet eigenvalues, inherits passivity, and satisfies $\sigma_\min(S)\ge\sigma_\min(A)$ | proved | static condensation 4.7e-15; $S^{-1}=(A^{-1})_{bb}$ to 6e-16 |
| C4 | LOD: the exact coarse vertex operator is global but decays exponentially (factor 0.17–0.26 per coarse cell, k-independent at fixed kH). Patch $m=2$ is the first stable truncation and $m=3$ to 4 reaches 1e-2 to 1e-3 error. Only real correctors make passivity hold by construction, and those lose consistency | literature + observed | table C4 |
| — | Discrete $\beta\gtrsim1/k$ for every passive learned star in a bounded class (discrete Rellich identity) | conjecture | B5 passive perturbations up to 60% keep $\beta/\beta_\text{ref}\ge0.84$ |

**Recommended constraints, short form (§D):**
1. Keep the stars symmetric (reciprocity).
2. Make every cell passive, as a matrix inequality on the anti-Hermitian part. In 1D this is a condition on the two **modal** admittances, not on $b$ and $M$ separately.
3. Make the frequency dependence positive-real by construction (Foster / pole-residue form with nonnegative coefficients).
4. Give edge stars a strict lower bound, $\operatorname{Re}s_\theta\ge s_\min>0$, not just $>0$.
5. Put an absorber on a unique-continuation set (a full side on tensor grids), or make the medium strictly lossy everywhere.
6. Keep a certificate $\delta<\beta_\text{ref}\approx1/k$ wherever the stars are not passive (PML, truncated LOD).
7. For coarse 2D stars: use $kH\lesssim1$ and a patch stencil with $m\approx C\log k$ coarse layers, not a cell-local stencil.

---

## 1. Setting and notation

**1D (admittance form, as in `src/oned/wall_inverse.py`).** Nodes $0..N$; cell $c$ lies between nodes $c-1$ and $c$. Each cell is a symmetric 2-port:

$$
Y_c=\begin{pmatrix}b_c+M_c&-b_c\\-b_c&b_c+M_c\end{pmatrix},\qquad
A=\sum_c P_c^\top Y_cP_c+G_0\,\mathbf e_0\mathbf e_0^\top+G_N\,\mathbf e_N\mathbf e_N^\top .
$$

Here $b_c=\tfrac1{j\omega}\star_{\mu^{-1}}$ and $M_c=j\omega\star_\varepsilon^{(1/2)}$. $Y_c=b_c\,\mathsf D^\top\mathsf D+M_cI$ is normal, with eigenvectors $(1,1)/\sqrt2$ and $(1,-1)/\sqrt2$. Its **modal admittances** are

$$Y_{c,+}=M_c,\qquad Y_{c,-}=M_c+2b_c .$$

A cell is **passive** if $\operatorname{Re}(v^HY_cv)\ge0$ for all $v$, which is equivalent to $\operatorname{Re}Y_{c,\pm}\ge0$. Learned stars have the DDEC form $b_c=b^\text{std}_c s_\theta$, $M_c=M_c^\text{std}m_\theta$ (doc 02 §7), and $(s_\theta,m_\theta)$ may be complex.

**2D (Helmholtz form, `src/stability/dec2d.py`).** On a Yee grid of squares of side $h$, $u$ is a 0-cochain on primal nodes. The system is

$$
A(k)=\mathsf D_0^\top S\,\mathsf D_0-k^2M+jkB .
$$

This equals $j\omega\mu_0$ times the admittance form. $S$ is the edge star $\lvert\tilde e\rvert/\lvert e\rvert/\mu_r$ (diagonal for DEC, Whitney-consistent if $\alpha_1>0$); $M$ is the node star $\lvert\tilde v\rvert\varepsilon_r$ (lumped, or Q1-consistent if $\alpha_0>0$); $B\succeq0$ is the first-order impedance boundary, $\partial_nu+jku=0$.

Write $\operatorname{Im}_HX:=(X-X^H)/(2j)$. Then

$$
\operatorname{Im}(u^HAu)=(\mathsf Du)^H\operatorname{Im}_H(S)(\mathsf Du)-k^2u^H\operatorname{Im}_H(M)u+k\,u^HBu .
\tag{1.1}
$$

**Passive** means $\operatorname{Im}_HS\succeq0$, $\operatorname{Im}_HM\preceq0$ and $B\succeq0$, which gives $\operatorname{Im}(u^HAu)\ge0$. For diagonal stars this reads $\operatorname{Im}s_\theta\ge0$ and $\operatorname{Im}m_\theta\le0$. A lossy permittivity $\varepsilon'-j\varepsilon''$ gives $\operatorname{Im}M<0$, as it should.

**Norms.** $S_0$ and $M_0$ are the geometric (real, standard, $\varepsilon_r=1$) stars, and $N:=\mathsf D^\top S_0\mathsf D+k^2M_0$ is the discrete $H^1_k$ norm. The inf-sup constant is

$$
\beta_N(A)=\inf_u\sup_v\frac{\lvert v^HAu\rvert}{\lVert u\rVert_N\lVert v\rVert_N}=\sigma_\min(N^{-1/2}AN^{-1/2}).
$$

`infsup.beta` computes it with ARPACK on $f\mapsto NA^{-H}NA^{-1}f$, using one LU. It agrees with a dense SVD to 1e-14.

---

## Part A. One dimension

### A1. Cell energy identity for the exact star [proved]

**Lemma A1.** Let $\mu>0$, $\varepsilon=\varepsilon'-j\varepsilon''$ with $\varepsilon''\ge0$, and $\sin k\ell\ne0$. For every $v=(E(0),E(\ell))$, let $E$ be the unique solution of $E''+k^2E=0$ on $(0,\ell)$ with these end values, and $H=-E'/(j\omega\mu)$. Then

$$
v^HY_\text{ex}v=\frac{-j}{\omega\mu}\int_0^\ell\lvert E'\rvert^2+j\omega\int_0^\ell\varepsilon\lvert E\rvert^2,\qquad
\operatorname{Re}(v^HY_\text{ex}v)=\omega\int_0^\ell\varepsilon''\lvert E\rvert^2\ \ge0 .
$$

*Proof.* By definition, $Y_\text{ex}v=(H(0),-H(\ell))$. So $v^HY_\text{ex}v=\bar E(0)H(0)-\bar E(\ell)H(\ell)=[E'\bar E]_0^\ell/(j\omega\mu)$. Integrate by parts and use $E''=-k^2E$:

$$[E'\bar E]_0^\ell=\int\lvert E'\rvert^2-k^2\int\lvert E\rvert^2 .$$

Finally $k^2/(\omega\mu)=\omega\varepsilon$. ∎

The exact star is therefore passive. It is lossless (purely imaginary modal admittances) iff $\varepsilon''=0$.

### A2. Unique solvability for any passive learned star [proved]

**Theorem A2.** Suppose:
- **(P1)** every cell is passive: $\operatorname{Re}Y_{c,\pm}\ge0$;
- **(P2)** $b_c\in\mathbb C\setminus\{0\}$ (finite and nonzero);
- **(P3)** $\operatorname{Re}G_0>0$ and $\operatorname{Re}G_N\ge0$.

Then $A$ is invertible. No sign of $\operatorname{Im}$ (reactance) and no smallness of $k\ell$ is needed.

*Proof.* Let $Ae=0$. Then

$$0=\operatorname{Re}(e^HAe)=\sum_c\operatorname{Re}(v_c^HY_cv_c)+\operatorname{Re}G_0\lvert e_0\rvert^2+\operatorname{Re}G_N\lvert e_N\rvert^2 .$$

Every term is $\ge0$, so $e_0=0$. Row 0 of $Ae=0$ reads $(b_1+M_1+G_0)e_0-b_1e_1=0$, which gives $e_1=0$ because $b_1\ne0$. Row $c$ reads $-b_ce_{c-1}+(\dots)e_c-b_{c+1}e_{c+1}=0$, which gives $e_{c+1}=0$. By induction $e=0$. ∎

**Each hypothesis is needed.**
- **(P1).** Take a random lossless wall with exact stars and a random cell $c$. Every complex $\delta$ that makes $\det(A+\delta E_c)=0$ (where $E_c$ adds $\delta$ to both modal admittances of cell $c$) has $\operatorname{Re}\delta<0$, i.e. it injects power. Over 200 random walls the largest $\operatorname{Re}\delta\,\eta_0$ was −0.59 (`A_active_roots`; test `test_1d_singular_perturbations_are_active`).
- **(P2).** With $b_c=0$ the chain decouples. With a reactive back boundary ($\operatorname{Re}G_N=0$), the right part is a closed lossless resonator and is singular at its resonances.

**What is *not* the right condition.** Cell-wise "$\operatorname{Re}b\ge0$ and $\operatorname{Re}M\ge0$" is sufficient but not necessary. For a lossy exact cell ($\varepsilon_r=4(1-0.6j)$ scaled), $\operatorname{Re}b/\lvert b\rvert$ reaches −0.99999 while $\operatorname{Re}Y_\pm>0$ (`A_Re_b_negative_but_passive`). Parameterizing $b$ and $M$ separately with positive real parts would exclude the exact star. **Parameterize the modal admittances instead.**

### A3. Explicit stability bound for any passive learned star [proved]

**Transfer matrices.** Let $h$ denote the current flowing in $+z$. Cell $c$ maps $(e_{c-1},h^+_{c-1})$ to $(e_c,h^-_c)$ by

$$
T_c=\begin{pmatrix}a/b&-1/b\\ b-a^2/b&a/b\end{pmatrix},\qquad a=b+M,\qquad \det T_c=1 .
$$

Kirchhoff's current law at a node with injected current $s_i$ gives $h_i^+-h_i^-=\pm s_i$. At node 0, $h^+_0=-G_0e_0$.

**Weighted norm.** For a reference impedance $\eta_i>0$ at node $i$, let

$$N(\eta)=\operatorname{diag}(\eta^{-1/2},\eta^{1/2}),\qquad W_i=\lvert e_i\rvert^2/\eta_i+\eta_i\lvert h_i\rvert^2=\lVert N(\eta_i)(e_i,h_i)\rVert^2,$$

$$\check T_c=N(\eta_c)\,T_c\,N(\eta_{c-1})^{-1},\qquad \kappa_c=\lVert\check T_c\rVert_2^2\ \ (\ge\lvert\det\check T_c\rvert=1).$$

**Theorem A3.** Assume (P1), (P2), $\operatorname{Re}G_0>0$ and $\operatorname{Re}G_N>0$. Fix $\eta_0=1/\lvert G_0\rvert$, $\eta_N=1/\lvert G_N\rvert$, and choose any interior references $\eta_1,\dots,\eta_{N-1}>0$. Then for all $i,j$:

$$
\lvert(A^{-1})_{ij}\rvert\le2\sqrt{\eta_i\eta_j}\;c_G\;P,\qquad P=\prod_{c=1}^N\kappa_c,\qquad c_G=\max\Big(\frac{\lvert G_0\rvert}{\operatorname{Re}G_0},\frac{\lvert G_N\rvert}{\operatorname{Re}G_N}\Big).
$$

In particular, in the current-to-voltage norm, $\lVert e\rVert_\infty\le2\eta_{\max}c_GP\lVert s\rVert_1$.

*Proof.* Let $e=A^{-1}(s\,\mathbf e_j)$.

1. **Power balance.** By (P1),
$$\operatorname{Re}(\bar e_js)=\operatorname{Re}(e^HAe)\ge\operatorname{Re}G_0\lvert e_0\rvert^2 .$$
2. **Start of the chain.** If $j\ne0$, then $h^+_0=-G_0e_0$, so $W_0=2\lvert G_0\rvert\lvert e_0\rvert^2\le2c_G\lvert e_j\rvert\lvert s\rvert$. If $j=0$, the bound follows directly from step 1.
3. **Propagation.** Between node 0 and node $j$ there is no source, so $(e,h)$ propagates by the $T_c$. Hence for $i\le j$, $W_i\le\big(\prod_{c\le i}\kappa_c\big)W_0\le PW_0$, using $\kappa_c\ge1$; for $i=j$ take $W_j^-$.
4. **Diagonal entry.** Since $\lvert e_i\rvert^2\le\eta_iW_i$, taking $i=j$ gives $\lvert e_j\rvert\le2\eta_jc_GP\lvert s\rvert$.
5. **Off-diagonal entries.** For $i<j$, $\lvert e_i\rvert^2\le2\eta_ic_GP\lvert e_j\rvert\lvert s\rvert\le4\eta_i\eta_jc_G^2P^2\lvert s\rvert^2$.
6. **Right side.** For $i>j$, run the same argument from node $N$ using $G_N$, and note $\lVert\check T_c^{-1}\rVert=\lVert\check T_c\rVert$ because $\det\check T_c=1$. ∎

**Checks.** Over 400 random walls (half lossless, a quarter with one exact cell per layer and so $\lvert k\ell\rvert\in(\pi,2\pi)$ allowed), $\max_{ij}\lvert A^{-1}_{ij}\rvert/\text{bound}\le0.185$ (median 0.033). Over 233 walls with random passive complex learned factors around the exact ones (factors ×[0.3,3], added loss up to 0.5), the ratio is $\le0.131$ and $\sigma_\min(A)\,\eta_0\ge0.12$.

### A4. Corollaries: k- and mesh-independence [proved]

Call a cell **lossless passband** if $\operatorname{Re}Y_\pm=0$ and $\operatorname{Im}Y_+\,\operatorname{Im}Y_-<0$. Then:

$$\eta_\text{eff}:=(Y_+Y_-)^{-1/2}>0,\qquad \tan(\varphi/2)=-j\eta_\text{eff}Y_+\in\mathbb R .$$

Using the identities $(\cot x-\tan x)/(\cot x+\tan x)=\cos2x$ and $2/(\cot x+\tan x)=\sin2x$, one finds

$$T_c=\begin{pmatrix}\cos\varphi&-j\eta_\text{eff}\sin\varphi\\-j\sin\varphi/\eta_\text{eff}&\cos\varphi\end{pmatrix}.$$

So $N(\eta_\text{eff})T_cN(\eta_\text{eff})^{-1}$ is **unitary**: every such learned cell is exactly a uniform line section. Changing the reference from $\eta$ to $\eta'$ costs a factor $\rho(\eta,\eta')=\max(\eta/\eta',\eta'/\eta)$. Choosing $\eta_i=\eta_{\text{eff},i+1}$ gives

$$P\le\Pi_\text{eff}:=\rho(\eta_0,\eta_{\text{eff},1})\prod_{c=1}^{N-1}\rho(\eta_{\text{eff},c},\eta_{\text{eff},c+1})\;\rho(\eta_{\text{eff},N},1/G_N).$$

(Measured: $P/\Pi_\text{eff}\in[0.46,1.0]$ over 200 walls.)

- **(a) Exact stars, lossless.** $\eta_\text{eff}=\eta_\text{layer}$, so $\Pi_\text{eff}$ is the product of the physical interface mismatches. It does not depend on $k$, $\omega$ or the number of cells, and holds also for cells with $k\ell\in(\pi,2\pi)$ (both reactances change sign). Measured: $P/\Pi\in[0.92,1]$; over a 0.5–200 GHz sweep of a four-layer wall, $\max\lvert A^{-1}\rvert/\eta_0\le1.34$ while $P\le\Pi=7.84$ (fig_A_1d.png).
- **(b) Standard stars.** $Y_+=j\omega\varepsilon\ell/2$ and $Y_-=j(\omega\varepsilon\ell/2-2/(\omega\mu\ell))$. So
$$\eta_\text{eff}=\eta/\sqrt{1-(k\ell/2)^2},\qquad\text{passband}\iff k\ell<2$$
(verified to 6e-16; stopband confirmed at $k\ell=2.1$). With $k\ell\le\theta<2$ in every cell, $P$ is bounded uniformly in $k$. The bound degrades only through the impedance mismatch $1/\sqrt{1-(k\ell/2)^2}$.
- **(c) Lossy exact cells.** With $\eta=\sqrt{\mu/\varepsilon'}$, $W$ satisfies $W'=-2\omega\varepsilon''\eta\operatorname{Re}(\bar EH)$ inside the cell, so $\lvert W'\rvert\le\omega\varepsilon''\eta W$. Grönwall then gives
$$\kappa_c\le\exp(k'\tan\delta\,\ell),\qquad k'=\omega\sqrt{\mu\varepsilon'}$$
(test `test_1d_lossy_exact_cell_transfer_growth_bound`, 200 random cells). The bound is explicit but pessimistic, because loss actually helps.
- **(d) Learned factors near the exact ones.** The passband margin of the exact star is $(k\ell)^2<x^2/\sin^2(x/2)$, which tends to equality as $x\to\pi$. Learned perturbations of ±25% stayed passband in 6 of 200 walls with the cell rule $\lvert k\ell\rvert\le0.9\pi$, and in 200 of 200 with $\lvert k\ell\rvert\le\pi/2$. **For learned 1D stars, use $\lvert k\ell\rvert\le\pi/2$.**

### A5. Exact stars sample the Green's function; the pole is conditioning, not stability [proved]

**Proposition A5.** If every cell uses the exact star and $\sin k_c\ell_c\ne0$, then $(A^{-1})_{ij}=g(z_i,z_j)$. Here $g$ is the continuous Green's function: the voltage at $z_i$ due to a unit current sheet at $z_j$, with the same radiation conditions.

*Proof.* The continuous solution $E$ with nodal current sources solves the homogeneous equation in every cell. By the definition of $Y_\text{ex}$ (Lemma A1), its end currents equal $Y_\text{ex}$ applied to its end values. Summing currents at each node reproduces $Ae=s$ with $e_i=E(z_i)$. Theorem A2 gives uniqueness. ∎

**Consequences.**
- Refining an exact mesh only adds rows and columns of a fixed matrix: the coarse $A^{-1}$ is a principal submatrix of the fine one (6.7e-13).
- For a matched homogeneous line, $A^{-1}=\tfrac{\eta_0}{2}e^{-jk\lvert z_i-z_j\rvert}$ (1.3e-14).
- As $k\ell\to n\pi$, $b\to\infty$, so $\lVert A\rVert\to\infty$, while $A^{-1}=g$ stays continuous in ω. In the one-cell sweep through $k\ell=\pi$, $\lVert A^{-1}\rVert_2\le\eta_0$ throughout while $\lVert A\rVert_2\eta_0$ reaches 4408.
- So the cell rule $\lvert k\ell\rvert\le0.9\pi$ in `wall_inverse.py` is a conditioning rule, as doc 02 says. It is not a stability requirement.
- The pole is the Dirichlet eigenvalue of the cell: $k^2=(n\pi/\ell)^2$. This is the 1D case of Proposition C3(i).

### A6. Positivity vs passivity vs causality [proved]

At a single real ω, Theorems A2 and A3 need only passivity, symmetry and $b\ne0$.
- The exact star with $k\ell\in(\pi,2\pi)$ has $M_\text{ex}/M_\text{std}=\tan(x/2)/(x/2)<0$, i.e. a negative "$\star_\varepsilon$". It is still exact and passive.
- The requirement $\star\succ0$ in doc 02 §4 is the condition for a positive **time-domain** energy $W$. In the frequency domain it is replaced by **positive-realness** in $s=j\omega$: analytic for $\operatorname{Re}s>0$ and $\operatorname{Re}Y(s)\ge0$ there. This implies passivity at every ω and causality.

**Foster forms of the exact cell.** With $\tau=\ell\sqrt{\varepsilon\mu}/2$ (half-cell delay) and $\eta=\sqrt{\mu/\varepsilon}$, the Mittag-Leffler expansions of $\tan$ and $\cot$ give

$$
\eta Y_+(s)=j\tan\omega\tau=\sum_{n\ge1}\frac{(2/\tau)\,s}{s^2+((n-\frac12)\pi/\tau)^2},\qquad
\eta Y_-(s)=-j\cot\omega\tau=\frac1{s\tau}+\sum_{n\ge1}\frac{(2/\tau)\,s}{s^2+(n\pi/\tau)^2}
$$

(test `test_1d_foster_expansions_of_exact_modal_admittances`). The leading term of $Y_-$ is the inductor $1/(s\,\mu\ell/2)=2b_\text{std}$.

**Recommended 1D parameterization**, positive-real by construction:

$$
Y^\theta_\pm(s)=G_\pm+sC_\pm+\frac1{sL_\pm}+\sum_i\frac{r_is}{s^2+a_is+w_i^2},\qquad G,C,L^{-1},r_i,a_i,w_i^2\ge0\ (\text{softplus}).
$$

Each term is positive-real: $\operatorname{Re}\frac{rj\omega}{w^2-\omega^2+ja\omega}=\frac{ra\omega^2}{(w^2-\omega^2)^2+a^2\omega^2}\ge0$, with poles in $\operatorname{Re}s\le0$. Then set $M=Y_+$ and $b=(Y_--Y_+)/2$. Debye/Lorentz moisture fits the $a_i>0$ terms.

---

## Part B. Two dimensions

### B1. Solver verification (exactness gate)

| Check | Result |
|---|---|
| $\mathsf D_0^\top W_1\mathsf D_0$ = bilinear Q1 stiffness (Whitney edge star, $\alpha_1=1$) | 5.6e-17 |
| Discrete plane wave (with the discrete $\kappa$ from the symbol) solves the interior equations | residual ≤ 9.5e-15 for all three star families |
| Plane wave, Robin data, $k=10$, $h=1/20\ldots1/640$: DEC | errors 4.9e-2 → 4.6e-5; rates 2.04, 2.01, 2.00, 2.00, 2.00 |
| same: blended $(\tfrac12,\tfrac12)$ / Q1 | rate 2.000 / 2.000 (the boundary Robin quadrature limits blended to $O(h^2)$, with a 4× smaller constant) |
| Mie series, dielectric cylinder $\varepsilon_r=2.25$, $a=0.4$, $k=4\pi$; DEC + PML stars, scattered-field source, dual-cell-averaged $\varepsilon$; error on $[-1,1]^2$ | 6.5e-2, 1.6e-2, 3.9e-3, 1.0e-3 for $h=0.026\ldots0.00325$; rates 2.03, 2.01, 1.93 (staircase/PML floor starting) |
| Mie reference itself: continuity at $r=a$; Helmholtz residual by finite differences | 1.6e-8; 7e-5 (FD-limited) |
| Adjoint $\partial J/\partial\varepsilon_i=k^2\lvert\tilde v_i\rvert\lambda_iu_i$ vs central FD | 6.4e-7 (experiment), < 1e-6 (test) |

**Implementation note.** SciPy's complex SuperLU was ~800× slower than the real one on these matrices. `dec2d.ComplexLU` factors the real 2×2-block form instead: 0.24 s for 26k unknowns, 4.3 s for 231k. OpenBLAS thread oversubscription made 441×441 dense SVDs 500× slower (56 s vs 0.11 s), so `tests/conftest.py` and the scripts pin BLAS to one thread.

### B2. Unique solvability on tensor grids with a full-side absorber [proved]

**Theorem B2.** On a rectangular Yee grid, suppose:
- $S$ and $M$ are diagonal and passive (with any complex values);
- $S_e\ne0$ on every $x$-edge;
- $B_i>0$ on every node of the side $x=0$.

Then $A(k)$ is invertible for every $k>0$.

*Proof.* Let $Au=0$. By (1.1) with passivity, $0=\operatorname{Im}(u^HAu)\ge k\sum_iB_i\lvert u_i\rvert^2$, so $u=0$ on column $i=0$. The 5-point row at node $(0,j)$ then reduces to $-S_{(0,j)(1,j)}\,u_{1,j}=0$, so column 1 vanishes. The row at $(i,j)$ involves column $i+1$ only through $-S_{(i,j)(i+1,j)}u_{i+1,j}$, so by induction every column vanishes. ∎

**Remarks.**
- The proof is a discrete unique-continuation (sweep) argument. It uses only the 5-point structure and $S_e\ne0$.
- For 9-point (Whitney/Q1, or learned non-diagonal) stars, the step from column $i$ to $i+1$ needs the column-coupling block to be invertible. For Q1 that block is tridiagonal Toeplitz with symbol $\propto(1+2\cos\theta)$ plus mass terms, which can vanish. So **uniqueness for non-diagonal learned stars is not automatic** (open question Q2).
- **Quantitatively**, the theorem says nothing about $\beta$. With impedance on one side only (the other three sides natural, i.e. a lossless cavity open on one side), $\min_k\beta_N=1.9\cdot10^{-4}$ ($k\beta_N=5\cdot10^{-3}$, at $k=28.2$). With impedance on all sides, $\min_k\beta_N=0.042$ ($k\beta_N=1.23$). This is a 230× loss from near-resonances (fig_B_cavity_scan.png).

### B3. Passivity is not sufficient [proved by explicit construction]

1. **Absorber on a nodal set.** Take a unit square with natural boundary everywhere except impedance at the two boundary midpoints $(n/2,0)$ and $(n/2,n)$, and the standard stars. The 1D DEC Neumann eigenvector $\cos(\pi i/n)$ (half dual cells at the ends) has eigenvalue $4n^2\sin^2(\pi/2n)$, and the 2D tensor mode $u=\cos\pi x$ vanishes on $x=\tfrac12$. So $Bu=0$ and $(K-k^2M)u=0$ exactly. $A$ is passive (λ_min of $\operatorname{Im}_HA$ = 0) and singular (residual 3.9e-16, $\sigma_\min=3.1\cdot10^{-16}$).
2. **Learned edge star vanishing on a cut.** Set $s_\theta=\epsilon$ on the edges around a 5×5 interior block. The block becomes a closed lossless cavity. At its first Neumann eigenvalue, $\sigma_\min=0.0111,\ 1.1\cdot10^{-4},\ 1.1\cdot10^{-8},\ 4\cdot10^{-16}$ for $\epsilon=10^{-2},10^{-4},10^{-8},0$. So $\sigma_\min\propto\epsilon$: a softplus that may underflow toward 0 gives arbitrarily bad conditioning. **Use a floor $s_\theta\ge s_\min>0$.**

### B4. Perturbation (Schatz-type in the discrete setting) [proved]

**Theorem B4.** Let $A_\text{ref}$ have $\beta_N(A_\text{ref})=\beta_\text{ref}>0$. Let $A_\theta$ have diagonal stars $S_0s_\theta$, $M_0m_\theta$ and the same $B$, with

$$\delta:=\max\big(\lVert s_\theta-s_\text{ref}\rVert_\infty,\lVert m_\theta-m_\text{ref}\rVert_\infty\big).$$

Then $\beta_N(A_\theta)\ge\beta_\text{ref}-\delta$. If $\delta<\beta_\text{ref}$, $A_\theta$ is invertible with $\lVert A_\theta^{-1}\rVert_{N^{-1}\to N}\le(\beta_\text{ref}-\delta)^{-1}$.

*Proof.* $\lvert v^H(A_\theta-A_\text{ref})u\rvert\le\delta\lVert\mathsf Du\rVert_{S_0}\lVert\mathsf Dv\rVert_{S_0}+\delta k^2\lVert u\rVert_{M_0}\lVert v\rVert_{M_0}\le\delta\lVert u\rVert_N\lVert v\rVert_N$ by Cauchy–Schwarz in $\mathbb R^2$. Since $\sigma_\min$ is 1-Lipschitz, the claim follows. ∎

For non-diagonal stars the same holds with $\delta$ the smallest constant in $\lvert w^H(S_\theta-S_\text{ref})w'\rvert\le\delta\lVert w\rVert_{S_0}\lVert w'\rVert_{S_0}$ (and likewise for $M$).

**Sharpness in $k$** (B5 in `results_2d.json`). Scale the node star uniformly, $m=1+\tau$. Then $A$ is singular iff $k^2(1+\tau)$ is an eigenvalue of the pencil $(K+jkB,M)$.

| $k$ (10 ppw) | $\beta_\text{ref}$ | $k\beta_\text{ref}$ | $\min\lvert\tau\rvert$ singular | $k\min\lvert\tau\rvert$ | all singular τ active ($\operatorname{Im}\tau>0$)? |
|---|---|---|---|---|---|
| 10 | 0.129 | 1.29 | 0.432 | 4.32 | yes |
| 20 | 0.0617 | 1.23 | 0.228 | 4.55 | yes |
| 40 | 0.0295 | 1.18 | 0.117 | 4.66 | yes |

So a uniform **active** error of relative size $\approx4.5/k$ in $\star_\varepsilon$ makes the system singular. Theorem B4 guarantees invertibility below $\approx1.2/k$. The bound has the right $k$-scaling, within a factor ≈ 3.8.

**Passive** random perturbations ($s=1\pm\delta$, $m=(1\pm\delta)(1-j[0,0.2]\delta)$, 12 draws each) never broke invertibility, even for $\delta=0.6\gg\beta_\text{ref}$, and kept $\beta/\beta_\text{ref}\in[0.84,1.37]$. This is consistent with B2. Theorem B4 is informative only for $\delta<\beta_\text{ref}$; beyond that, passivity together with B2 is what protects invertibility.

### B5. Strictly lossy stars: explicit k-bound [proved]

**Theorem B5.** Assume, as matrix inequalities (so non-diagonal learned stars are allowed):
- $\operatorname{Re}_HS_\theta\succeq s_-S_0$ and $\operatorname{Im}_HS_\theta\succeq0$;
- $-\operatorname{Im}_HM_\theta\succeq\tau M_0$ with $\tau>0$;
- $\lvert w^HM_\theta w\rvert\le m_+w^HM_0w$;
- $B\succeq0$.

Then for $Au=f$:

$$
\lVert u\rVert_N\le\frac{C_\tau}{k}\lVert f\rVert_{M_0^{-1}},\qquad C_\tau=\Big(\frac1\tau\Big[\frac{1+m_+/\tau}{s_-}+\frac1\tau\Big]\Big)^{1/2}.
$$

*Proof.*
1. By (1.1), $\tau k^2\lVert u\rVert_{M_0}^2\le\operatorname{Im}(u^HAu)\le\lvert u^Hf\rvert\le\lVert u\rVert_{M_0}\lVert f\rVert_{M_0^{-1}}$. Hence $\lVert u\rVert_{M_0}\le\lVert f\rVert/(\tau k^2)$.
2. The real part gives the Gårding inequality $s_-\lVert\mathsf Du\rVert^2_{S_0}-m_+k^2\lVert u\rVert^2_{M_0}\le\operatorname{Re}(u^HAu)\le\lVert u\rVert_{M_0}\lVert f\rVert$. Hence $s_-\lVert\mathsf Du\rVert^2\le\lVert f\rVert^2(1+m_+/\tau)/(\tau k^2)$.
3. Add $k^2\lVert u\rVert^2_{M_0}$. ∎

This bound is uniform in $h$ and in the geometry (trapping does not matter). The price is $\tau^{-1}$-type constants.

| $k$ | $\tau=0.05$: $\lVert A^{-1}\rVert$ / bound | $\tau=0.2$: $\lVert A^{-1}\rVert$ / bound |
|---|---|---|
| 5 | 0.535 / 5.73 | 0.451 / 1.49 |
| 20 | 0.446 / 1.43 | 0.255 / 0.372 |
| 80 | 0.263 / 0.358 | 0.0854 / 0.0931 |

### B6. PML stars are not passive [proved]

With $s_x=1-j\sigma_x/k$ and $s_y=1-j\sigma_y/k$, the $x$-edge star is multiplied by $s_y/s_x$. In a layer with $\sigma_x=0<\sigma_y$ (a horizontal PML strip), $\operatorname{Im}(s_y/s_x)=-\sigma_y/k<0$, which violates $\operatorname{Im}S\ge0$.

Measured: $\lambda_\min(\operatorname{Im}_HA)=-13.5$ ($k=10$) and $-6.3$ ($k=20$), yet $\beta_N=0.138$ and $0.080$. So B2 and B5 do not apply to PML. Its solvability must be certified differently:
- by B4 relative to a passive reference;
- by the continuous PML theory [literature: Lassas–Somersalo, Bramble–Pasciak; not checked here];
- or by replacing it with a passive graded-conductivity absorber when a guarantee is required.

### B7. Mass blending is a small perturbation [proved]

On each square cell, $M_L-M_c=\tfrac{h^2}{36}(9I-\text{circ}(4,2,1,2))$ has eigenvalues $\tfrac{h^2}{36}\{0,6,8,6\}$: it is PSD, annihilates constants, and is $\le\tfrac{2h^2}{9}$. The cell part of $\lVert\mathsf Du\rVert^2_{S_0}$ is a 4-cycle Laplacian with weights ½ (eigenvalues 0, 1, 1, 2). So $k^2\lvert u^H(M_L-M_c)v\rvert\le\tfrac29(kh)^2\lVert\mathsf Du\rVert\lVert\mathsf Dv\rVert$. The boundary blend $B_L-B_c=\tfrac h6\begin{psmallmatrix}1&-1\\-1&1\end{psmallmatrix}$ per boundary edge contributes $\tfrac{kh}3$. Hence

$$\lvert\beta(0,\alpha_0)-\beta(0,0)\rvert\le\lVert\Delta\rVert\le\alpha_0\big[\tfrac29(kh)^2+\tfrac13kh\big].$$

Measured operator norms: 0.049 (bound 0.060) and 0.080 (bound 0.111). The same argument does **not** apply to the edge-star blend: $W_1-S_\text{diag}$ is $O(1)$ on grid-scale oscillations.

### B8. Inf-sup vs k (numerics) [observed]

On the unit square with impedance on all sides (fig_B_convergence_infsup.png):

| $k$ | $n$ (10 ppw) | $k\beta_N$ DEC | $k\beta_N$ blended $(\tfrac12,\tfrac12)$ | $k\beta_N$ Q1 | $k\beta_N$ DEC at $k^3h^2\approx3.9$ |
|---|---|---|---|---|---|
| 5 | 8 | 1.23 | 1.24 | 1.25 | 1.19 |
| 10 | 16 | 1.29 | 1.33 | 1.33 | 1.29 |
| 20 | 32 | 1.23 | 1.25 | 1.27 | 1.26 |
| 40 | 64 | 1.18 | 1.19 | 1.21 | 1.21 |
| 80 | 128 | 1.15 | 1.15 | 1.18 | 1.18 |
| 160 | 255 | 1.15 | 1.12 | 1.14 | — |

At $k=20$, $\beta_N$ under refinement is 0.0530, 0.0617, 0.0637, 0.0642, 0.0643, 0.0643 for $n=16\ldots512$, converging to a continuous value with $k\beta\approx1.29$.

So $\beta_N\approx1.2/k$ for all three star families, already at 10 points per wavelength. This matches the continuous wavenumber-explicit theory for star-shaped domains with impedance boundary [literature: Melenk 1995; Esterhazy–Melenk 2012: inf-sup $\sim1/k$ in $H^1_k$]. **Dispersion correction does not change stability; it changes accuracy.**

Plane-wave error at 10 ppw for $k=5\ldots160$:
- DEC: 0.041 → 1.14 (∝ $k$, the pollution $k^3h^2$);
- Q1: 0.035 → 1.07;
- blended $(\tfrac12,\tfrac12)$: 0.018 → 0.034 (nearly flat).

### B9. Dispersion of blended stars [proved by symbol expansion]

With $X=\xi_x^2h^2$, $Y=\xi_y^2h^2$ and $R=X+Y$, the symbol of $\mathsf D^\top S_{\alpha_1}\mathsf D-k^2M_{\alpha_0}$ gives

$$
(kh)^2=R+R^2\Big(\frac{\alpha_0}6-\frac1{12}\Big)+XY\Big(\frac16-\frac{\alpha_1}3\Big)+O(h^6).
$$

So the error is $O(h^4)$ and isotropic iff $\alpha_0=\alpha_1=\tfrac12$. Blending only the mass (Ainsworth–Wajid, $\alpha_0=\tfrac12$) on Q1 stiffness ($\alpha_1=1$) leaves $-XY/6$: 4th order along the axes only. Test `test_dispersion_fourth_order_for_half_half_blend` measures rate 2 (DEC) vs 4 (blend).

Max phase error over directions:

| ppw | 4 | 6 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| DEC (0,0) | 1.5e-1 | 5.2e-2 | 1.7e-2 | 4.2e-3 | 1.0e-3 |
| Q1 (1,1) | 8.0e-2 | 4.1e-2 | 1.6e-2 | 4.1e-3 | 1.0e-3 |
| Q1 + AW mass (1,½) | 7.7e-2 | 2.7e-2 | 8.7e-3 | 2.1e-3 | 5.2e-4 |
| blend (½,½) | 1.5e-2 | 2.6e-3 | 3.3e-4 | 2.0e-5 | 1.3e-6 |

The (½,½) star is a non-diagonal but **data-free** DEC star: half DEC-lumped, half Whitney-consistent, in both $\star_0$ and $\star_1$. It is symmetric and as passive as its ingredients. Learned stars should start from it.

---

## Part C. Why 2D needs approximate (learned) stars

### C1. 1D: the local Trefftz space is finite-dimensional [proved]

On a homogeneous cell, $\mathcal T_1=\{E:E''+k^2E=0\}=\operatorname{span}\{e^{\pm jkz}\}$ has dimension 2, which equals the number of DOFs (2 nodes). The trace map $\mathcal T_1\to\mathbb C^2$, $E\mapsto(E(0),E(\ell))$, has determinant $-2j\sin k\ell$. It is an isomorphism iff $k^2\ne(n\pi/\ell)^2$, i.e. iff $k^2$ is not a Dirichlet eigenvalue of the cell. The exact star is then the 2×2 Dirichlet-to-Neumann map (Lemma A1), and Proposition A5 makes the assembled system exact.

Numerical certificate: on 50 random local solutions, rank[values; fluxes] = rank[values] = 2, and the best-star defect is ≤ 1e-14 for all kH.

### C2. 2D: no finite exact star exists [proved for the space; numerical certificate for vertex DOFs]

On a square cell $T$, $\mathcal T_2=\{u:\Delta u+k^2u=0\}$ contains the plane waves $e^{-jk\,d\cdot x}$ for every $d\in S^1$. Distinct directions are linearly independent (distinct exponentials), so $\dim\mathcal T_2=\infty$. For any finite set of DOF functionals, the trace map has an infinite-dimensional kernel. An exact cell-local star $Y$ (fluxes $=Y\cdot$DOFs) would have to annihilate the fluxes of every kernel element.

For the 4 vertex DOFs and the DEC fluxes $q_v$ (normal derivative integrated over the two half-edges at $v$), sample 400 plane waves:

| $kH$ | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|
| rank values / rank [values; fluxes] (tol 1e-10) | 4/6 | 4/7 | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| best 4×4 star, relative flux defect | 2.6e-5 | 1.0e-4 | 4.1e-4 | 2.5e-3 | 9.0e-3 | 2.6e-2 | 0.63 |

Rank 8 > 4 certifies (with singular values far above roundoff for $kH\ge0.2$) that no 4×4 star is exact. The defect scales as $\approx0.01(kH)^2$ (fitted slope 1.98).

### C3. The exact star relative to the fine mesh is the cell Schur complement [proved]

Let $A_T(k)$ be a cell's fine DEC matrix (natural boundary), with fine boundary nodes $b$ and interior nodes $i$. Define

$$S(k)=A_{bb}-A_{bi}A_{ii}^{-1}A_{ib}.$$

1. **Existence.** $S$ exists iff $A_{ii}$ is invertible. For a lossless cell this means $k^2\notin\sigma(K_{ii},M_{ii})$, the fine Dirichlet spectrum of the cell. This is the 2D analogue of $\sin k\ell=0$. The first pole of the 16×16 cell is at $kH=4.436$ (continuum $\pi\sqrt2=4.443$); fig_C_cells.png also shows the double pole at 6.99.
2. **Symmetry.** $S$ is complex symmetric (6.6e-17).
3. **Passivity is inherited.** For $v_b$ let $v=(v_b,-A_{ii}^{-1}A_{ib}v_b)$. Then $Av=(Sv_b,0)$, so $v_b^HSv_b=v^HAv$ and $\operatorname{Im}(v_b^HSv_b)\ge0$. ∎ Measured: $\lambda_\min(\operatorname{Im}_HS)=-8.6\cdot10^{-17}$ (lossless) and $+8.8\cdot10^{-3}$ (lossy).
4. **Exact and at least as stable as the fine system.** Assembling the cell Schur complements plus boundary terms gives the skeleton Schur complement (1.1e-16), and solving it reproduces the fine solution on the skeleton (4.7e-15). Moreover $S_\text{skel}^{-1}=(A^{-1})_{\text{skel},\text{skel}}$ (6.2e-16), so $\sigma_\min(S_\text{skel})\ge\sigma_\min(A)$: condensation never loses $\ell^2$ stability.
5. **Rational structure.** For a lossless cell with diagonal $M$, $A_{bi}=K_{bi}$ does not depend on $k$. With $K_{ii}\varphi_j=\lambda_jM_{ii}\varphi_j$,
$$S(k)=K_{bb}-k^2M_{bb}-\sum_j\frac{r_jr_j^\top}{\lambda_j-k^2},\qquad r_j=K_{bi}\varphi_j\in\mathbb R^{\lvert b\rvert}.$$
This is a Foster form with **rank-one PSD residues**: the matrix generalization of A6.
6. **Size and density.** $S$ is dense (88% of entries nonzero for a 16×16 cell; the zeros are symmetry cancellations), of size about $4H/h$, which grows without bound under refinement.

**The precise statement.**
- In 1D, $\dim\mathcal T_1=2=\#\text{DOF}$. So the exact cell star exists, is local and 2×2, is a closed-form function of $(\omega,\varepsilon,\mu,\ell)$, and makes $A^{-1}$ equal to the sampled continuous Green's function, independently of the mesh.
- In 2D, $\dim\mathcal T_2=\infty$. A cell star is exact only relative to a fine mesh. It is then the Schur complement: dense, of size growing as $h\to0$, rational in $k^2$ with poles at the cell's Dirichlet spectrum, and with no closed form.
- Any coarse star with $O(1)$ DOFs per cell is a **compression**. It can be cell-local but approximate (defect ≈ 0.01(kH)² for 4 DOFs), or exact but non-local (C4).
- Learning is needed to pick the compression that keeps phase velocity, from data (the fine solves).

**What this implies for stability.**
- **(i) Real compressions inherit passivity.** Any compression $Y=P^\top SP$ with **real** $P$ is passive; with complex $P$ use $P^HSP$. The vertex star $Y=P^\top SP$ with piecewise-linear boundary prolongation has $\lambda_\min(\operatorname{Im}_HY)\ge-1.3\cdot10^{-16}$ in all 12 cases (C3 table below). Theorem B2 then gives invertibility.
- **(ii) Stability is not inherited.** Its constant comes from the coarse geometry; the measured $\beta$ is comparable to standard DEC on the same coarse grid.
- **(iii) Accuracy.** The compressed vertex star halves the error of standard coarse DEC at $kH=1$ and cuts it by up to 2.8× at $kH=1.5$, but it is not pollution-free:

| $N_c$ | $kH$ | err compressed $P^\top SP$ | err std DEC (coarse) | $\beta_N$ compressed / std |
|---|---|---|---|---|
| 8 | 0.5 | 0.013 | 0.021 | 0.31 / 0.30 |
| 8 | 1.0 | 0.077 | 0.175 | 0.17 / 0.16 |
| 16 | 1.0 | 0.147 | 0.349 | 0.077 / 0.073 |
| 16 | 1.5 | 0.453 | 1.286 | 0.047 / 0.038 |

### C4. Localization: what LOD proves and what it implies for coarse stars

**[literature] Verfürth (2019), indefinite $H(\mathrm{curl})$.**
- Under the resolution condition $\omega H\lesssim1$, $B$ is inf-sup stable on $W=\ker\pi^E_H$ (Falk–Winther) with a constant $\alpha$ independent of ω (Lemma 3.1, via a regular decomposition).
- Correctors decay like $\beta^m$ (Prop. 4.1).
- If $m\gtrsim\lvert\log(\gamma(\omega)/\sqrt{C_{ol,m}})\rvert/\lvert\log\beta\rvert$, the localized scheme is inf-sup stable with $\gamma_\text{LOD}\approx\gamma(\omega)$ (Thm 4.5). If $\gamma\sim\omega^{-q}$, this means $m\approx\log\omega$, and the error is $\lesssim(H+\beta^m\gamma^{-1})\lVert f\rVert$ (Thm 4.6).

**[literature] Ohlberger–Verfürth (2017), two-scale Helmholtz (Petrov–Galerkin LOD).**
- Under $k(H_c+h_c)\lesssim1$, $B$ is coercive on the kernel of the quasi-interpolation (Lemma 4.1).
- With the oversampling condition (4.6), $m\ge[(q+1)\log k+\text{const}]/\lvert\log\beta\rvert$, the method is inf-sup stable with constant $(C_\text{LOD}k^{q+1})^{-1}$ (Thm 4.3) and quasi-optimal (Thm 4.5). This overcomes pollution with a few DOFs per wavelength.

**[observed] Our discrete version.** Use coarse vertices, the kernel of nodal interpolation, and bilinear hats $\varphi_z$. The correctors $q_z\in W$ solve $A_{WW}q=-A_W\varphi_z$ on $m$-patches, and the coarse operator is $Y_m=\Psi^\top A\Psi$. Error is measured at coarse vertices against the fine solution (fig_C_cells.png, right).

| $N_c$, $kH$ ($k$) | ideal $Y$ decay per coarse cell | err $m=1$ | $m=2$ | $m=3$ | $m=4$ | $m=\infty$ | $\beta$ ($m\ge2$) | $\lambda_\min(\operatorname{Im}_HY_m)/\lVert Y\rVert$, $m=1\to4$ | real correctors: err ($m=4$), passive? |
|---|---|---|---|---|---|---|---|---|---|
| 16, 0.5 (8) | 0.17–0.20 | 0.96 | 0.16 | 5.9e-3 | 1.2e-3 | 1.1e-13 | 0.16–0.17 | −2.7e-3 → −3.8e-7 | 0.12, yes |
| 16, 1.0 (16) | 0.22–0.26 | 1.26 | 0.029 | 0.010 | 2.4e-3 | 4.7e-14 | 0.082 | −1.1e-2 → −1.2e-5 | 0.27, yes |
| 24, 1.0 (24) | 0.19–0.26 | 1.29 | 0.022 | 6.4e-3 | 1.3e-3 | 5.9e-14 | 0.054 | −7.3e-3 → −3.2e-6 | 0.27, yes |

**Implications for a learned coarse star.**
1. With $O(1)$ DOFs per coarse vertex, the exact (fine-relative) coarse operator is global. It is, however, quasi-local: it decays by a factor ≈ 0.2–0.26 per coarse cell, independent of $k$ at fixed $kH$. Its exactness at the vertices (1e-13) and $\beta$ are reached from $m=2$ on.
2. A **cell-local** 4-DOF star ($m=0$, or $m=1$ here) has no stability or accuracy guarantee: $m=1$ gives an error ~100%. **The stencil of a learned coarse star should be a patch of $m\approx C\log k$ coarse layers** (in practice $m=2$–3 at $kH\le1$), with $kH\lesssim1$.
3. **Passivity versus exactness.**
   - The A-orthogonal (complex) correctors make $\Psi^\top A\Psi$ exact for $m=\infty$, but non-passive for finite $m$, with a violation that decays like $\beta^m$.
   - **Real** correctors (computed from the lossless real part) make $\Psi^\top A\Psi=\Psi^HA\Psi$ passive **for every $m$**, but consistency near the impedance boundary saturates at 12–27%.
   - The sesquilinear $\Psi^HA\Psi$ with complex correctors is passive but inconsistent (42–85%).
   - **Recommended:** use the A-orthogonal correctors and restore passivity by the minimal shift $Y_m+j\,c\,M_c$ with $c=\max(0,-\lambda_\min(\operatorname{Im}_HY_m))$, which is $O(\beta^m)$. The shift is part of the parameterization, so passivity holds by construction and accuracy loses only $O(\beta^m)$. [The accuracy part is a conjecture; not yet measured.]

---

## D. Recommended constraints on learned stars (structure by construction)

| # | Constraint | How to parameterize | Guarantees |
|---|---|---|---|
| R1 | Topology exact | $\mathsf D_0,\mathsf D_1$ integer incidence matrices | conservation; $\mathsf D_1\mathsf D_0=0$ |
| R2 | Reciprocity | $Y_c=Y_c^\top$ (complex symmetric). 1D: two modal admittances. 2D: $Y=\sum_jw_jw_j^\top$-type forms | adjoint = forward factorization |
| R3 | Passivity (cell-wise) | $\operatorname{Im}_HS_\theta\succeq0$, $\operatorname{Im}_HM_\theta\preceq0$ (Helmholtz form), i.e. $\operatorname{Re}Y_{c,\pm}\ge0$ (admittance form). Diagonal: $s_\theta=s_r+js_i$ with $s_i\ge0$; $m_\theta=m_r-jm_i$ with $m_i\ge0$. Coarse compressions: $P^\top SP$ with real $P$ | 1D unique solvability (A2) and bound (A3); 2D uniqueness with R6 (B2) |
| R4 | Causality | PR pole-residue form in $s=j\omega$ with nonnegative coefficients (A6); 2D cells: $K_0-k^2M_0-\sum_jr_jr_j^\top/(\lambda_j-k^2)$ with $\lambda_j>0$, $r_j$ real, plus loss terms (C3) | passivity at every ω, Kramers–Kronig |
| R5 | Strict floor | $\operatorname{Re}s_\theta\ge s_\min>0$ (softplus + floor) | unique continuation in B2; avoids B3(2) |
| R6 | Absorption on a UCP set | impedance on a full side of a tensor grid (B2), or $-\operatorname{Im}m_\theta\ge\tau>0$ everywhere (B5) | invertibility; with B5 an explicit $C_\tau/k$ bound |
| R7 | Certificate where passivity fails (PML, truncated LOD) | check $\delta<\beta_\text{ref}$ (B4); $\beta_\text{ref}\approx1.2/k$ on open domains, much smaller on cavities | invertibility with $\lVert A^{-1}\rVert\le(\beta_\text{ref}-\delta)^{-1}$ |
| R8 | Resolution and stencil | 1D: $\lvert k\ell\rvert\le\pi/2$ for learned cells (passband margin, A4d); exact cells: any $\ell$ off the poles. 2D: $kH\lesssim1$, patch stencil $m\approx C\log k$ (C4) | k-uniform 1D bound; LOD-type inf-sup ≈ γ(k) |
| R9 | Start from the best data-free star | 2D: blended $(\tfrac12,\tfrac12)$ (B9); 1D: exact | 4th-order isotropic dispersion; learned part is a small correction, which helps R7 |

**Not recommended:** the constraint "$s_\theta,m_\theta>0$ real" alone. It is passive but lossless only, so it cannot represent a lossy exact star, and it allows $s_\theta\to0$ (B3).

---

## E. Open questions for Kaibo Hu

1. **Discrete Rellich / Morawetz identity for DEC.** For star-shaped domains with impedance boundary, is there a DEC multiplier identity (a discrete $x\cdot\nabla u$) giving $\beta_N\ge C/k$ for every passive star in a Loewner class $s_-S_0\preceq\operatorname{Re}S_\theta\preceq s_+S_0$ (and likewise for $M$)? Numerically $k\beta_N\approx1.15$–1.33 for DEC, Q1 and the blend, and passive perturbations up to 60% keep $\beta/\beta_\text{ref}\ge0.84$. [conjecture]
2. **Discrete unique continuation on Delaunay meshes and for non-diagonal (learned or Whitney-blended) stars.** What sign or M-matrix structure guarantees continuation from a boundary portion? The sweep proof of B2 fails for 9-point stars.
3. **Bounded cochain projections for data-driven Whitney forms.** Do the POU-based coarse spaces of Actor et al. / Kinch et al. admit a Falk–Winther-type bounded commuting projection, so that Verfürth's "inf-sup on $\ker\pi_H$ for $kH\lesssim1$" carries over? This would give learned coarse stars a wavenumber-independent guarantee.
4. **Dispersive Hodge stars.** Is there a FEEC/DEC theory of positive-real, matrix-valued, frequency-dependent Hodge stars (the Foster form of C3 with rank-one PSD residues), including causality and a discrete Kramers–Kronig relation?
5. **Stability of DEC with non-passive PML stars.** Is there a Gårding plus complex-symmetric argument, or a result for the discrete PML on Delaunay meshes?
6. **Blended stars on simplices.** Does the 4th-order $(\tfrac12,\tfrac12)$ blend of lumped and consistent stars have a Delaunay-mesh analogue? Is its inf-sup constant still controlled (B7 covers only the mass blend)?

---

## Log entry S001

**S001 · 2026-11-09 · Stability of the indefinite discrete problem with learned stars (roadmap Phase 2 item 7)**

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
