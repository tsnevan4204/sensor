# 16 · Structure search for the known-N wall: algorithm, what is proved, what is measured

Written 2026-11-12 (E050–E053). Code: `src/oned/bnb.py`, `fastbatch.py`, `polish.py`, `wallsearch.py`, `wallsolve2.py`. Evidence: `experiments/EXPERIMENT_LOG.md` (E050–E053), `results/2026-11-12_E05*/`.

## 1. The problem and why the earlier searches failed

Unknown: N solid layers (library class, thickness, material within the class), an optional air gap behind every solid layer, the backing (air or metal), the standoff. Known: N, the standoff to about a millimetre.
Data: complex reflection $y(f_m)=\Gamma(f_m)+n_m$ on three bands (0.4–0.96, 3.1–10.6, 60–64 GHz), white Gaussian noise of known σ.

This is a mixed problem: a discrete structure (class sequence × gap pattern × backing; about 3·10⁴ structures for N = 3) and continuous parameters inside each structure. E034–E049 attacked it with continuous or greedy searches. Measured on 12 fresh walls (E050), the last of them (E049d) solves 0/12.

Two measurements (E051, `rank_test.txt`) say where the difficulty is.
1. With the **true classes**, the exact misfit over **all** interface positions on a 2.5 mm optical grid has its minimum at or next to the true wall (4 of 5 walls).
2. With the **true positions**, the exact misfit over **all** class sequences ranks the true sequence first (4 of 5).
3. The exception is a wall whose material sample is far from its class's nominal member.

So the landscape is benign once the structure is discrete and enumerated, and the earlier methods failed because they searched a continuous relaxation (many-layer explanations, cycle skipping) or decided depth by depth. Section 3 shows that the second is impossible in principle for thin walls.

## 2. Exact building blocks

Reflection recursion (same mathematics as the exact-star DEC solve, gate 1e-14):
$$G_i=\frac{r_i+G_{i+1}e_i}{1+r_iG_{i+1}e_i},\quad r_i=\frac{n_{i-1}-n_i}{n_{i-1}+n_i},\quad e_i=e^{-2jk_0n_it_i},\quad \Gamma=G_1e^{-2jk_0d}.$$

`fastbatch.py` evaluates it for class-discrete walls from a refractive-index table $n[c,f]$ (gate: 2.4e-15 against `stack.gamma_stack` on 50 random stacks with water and metal).

**Lemma 1 (stability of the recursion, lossless interfaces).** Write $G_i=M_{r_i}(G_{i+1}e_i)$ with $M_r(z)=(r+z)/(1+rz)$. For real $|r|<1$, $M_r$ is an automorphism of the unit disc, and multiplication by $e_i$, $|e_i|\le1$, maps the disc into itself. Hence
- $|G_i|\le1$ at every depth (passivity is kept by the recursion, for any parameter values);
- in the pseudo-hyperbolic distance $\rho(z,w)=|z-w|/|1-\bar wz|$ no stage expands (Schwarz–Pick): a perturbation of the wall behind interface $i$ is never amplified, in this metric, on its way to the front;
- in the Euclidean metric, from $M_r'(z)=(1-r^2)/(1+rz)^2$,
$$\Big|\frac{\partial\Gamma}{\partial t_i}\Big|\le 2k_0|n_i|\prod_{j\le i}\frac{1+|r_j|}{1-|r_j|}.$$

*Proof.* $M_r$ maps the unit circle to itself and $M_r(0)=r$ lies inside, so it is a disc automorphism; Schwarz–Pick gives the non-expansion. The derivative bound follows from $|1+rz|\ge1-|r|$ on the disc and the chain rule through the stages $j\le i$, with $\partial(G_{i+1}e_i)/\partial t_i=-2jk_0n_iG_{i+1}e_i$. ∎

Use: a grid of optical step $h$ contains a wall whose response differs from the true wall's by at most $\sum_i L_i h/2$ with $L_i$ the bound above. For lossy layers $r_i$ is complex and the clean statement is lost; the solver does not rely on the bound but on the measured discrepancy model of Section 5.

## 3. Why depth-by-depth decisions cannot work on thin walls (dimension count)

A time gate keeps the part of the data that arrived before the round trip to optical depth $T$. On a band of width $B$ sampled at $F$ points, the gate is the span of the sampled spectra whose range-domain energy lies in the window $[a,T]$ up to a fraction ε: the eigenvectors of
$$A_{mn}=\int_a^T e^{-j4\pi(f_m-f_n)\rho/c}\,d\rho$$
with eigenvalue above $1-\varepsilon$ (discrete prolate spheroidal sequences; Slepian & Pollak 1961, Landau & Pollak 1962). Their number is
$$K(T)\approx \frac{2B\,(T-a)}{c}-O(\log)\quad\text{(the 2WT theorem).}$$

Measured (`treesolve.slepian`, UWB band, ε = 10⁻⁴, window starting 50 mm in front of the wall): K = 0 for the first 40 mm of optical depth, 1 at 40 mm, 4 at 100 mm, 6 at 160 mm.

Consequence. A wall of 84 mm optical depth with five interfaces behind the front face (development wall 202) offers a gate of 3–4 complex numbers by the time its last interface is reached. Any method that fixes the front layers before looking at the rest (E034 growth, E041 peeling, a breadth-first beam) decides with no data. E051 confirms it: a beam of 600–1300 survivors loses the true prefix at 35 mm. The first interfaces of a thin wall can only be decided jointly with everything behind them. The branch-and-bound below does exactly that: it prunes nothing where K = 0 and everything it can where K > 0.

## 4. Branch-and-bound over the grid problem

**Grid problem.** All walls with N solid layers from the class table (nominal members; near-identical classes share a representative), any admissible gap pattern, backing air or metal, every interface at optical depth $d_0+h(q+1)$, $q=0..m-1$. Cost $C(w)=\sum_f|\Gamma_w(f)-y(f)|^2/\sigma_{\rm eff}(f)^2$ (Section 5).

**Tree.** A node is a prefix: $k$ complete layers and the class of the half-space behind them, with response $G_{\rm node}$. Children add one interface at a later grid index $q$ and a class behind it.

**Lemma 2 (gate bound).** Let $w$ be any completion of a node whose next interface is at grid index $q$. Let $Q_q$ have orthonormal rows in the whitened data space, and let $A_q\ge\|Q_q(\Gamma_w-G_{\rm node})/\sigma\|$. Then
$$C(w)\ \ge\ \big(\|Q_q(y-G_{\rm node})/\sigma\|-A_q\big)_+^2 .$$

*Proof.* $Q_q$ has orthonormal rows, so $C(w)=\|(y-\Gamma_w)/\sigma\|^2\ge\|Q_q(y-\Gamma_w)/\sigma\|^2$, and the triangle inequality gives $\|Q_q(y-\Gamma_w)/\sigma\|\ge\|Q_q(y-G_{\rm node})/\sigma\|-\|Q_q(\Gamma_w-G_{\rm node})/\sigma\|$. ∎

The physics is in $A_q$. By causality $\Gamma_w-G_{\rm node}$ consists only of echoes that have travelled at least to the new interface: $\Gamma_w-G_{\rm node}=\sum_j a_j(f)e^{-2jk_0\rho_j}$ with $\rho_j\ge d_0+h(q+1)$. Choosing the rows of $Q_q$ as the Slepian vectors of the window ending there makes its gain on such echoes small. `bnb.build_gates` computes
$$A_q=a_{\rm tail}\Big(\sum_{\rm bands}\sup_{\rho\ge T_q}\|Q_q\,e(\rho)/\sigma\|^2\Big)^{1/2}$$
numerically over one unambiguous range period. This is a valid $A_q$ **if the later echoes have frequency-independent amplitudes with $\sum_j|a_j|\le a_{\rm tail}$**.

What is not proved: with loss and dispersion the amplitudes $a_j(f)$ vary slowly with frequency, so the bound is exact only for non-dispersive tails. A weaker assumption cannot be avoided by any gate bound: finitely many frequency samples never certify causality by themselves (any finite set of boundary values is interpolated by a bounded causal transfer function, Nevanlinna–Pick), so some model of the tail is always needed. The solver therefore measures bound violations on the benchmark with the truth in hand (Section 7).

**Proposition (exhaustiveness).** Suppose every bound used is valid. Run the depth-first search with pruning threshold $\min(\theta,\rho\,c_{\rm best})$, $\rho\ge1$, where $c_{\rm best}$ is the best leaf cost so far. If $\theta\ge\rho c^*$ ($c^*$ the optimum of the grid problem), every grid wall with cost $\le\rho c^*$ is evaluated; with $\rho=1$ the search returns $c^*$.

*Proof.* A subtree is discarded only when its bound exceeds the threshold at that moment, which is at least $\rho c^*$ because $c_{\rm best}\ge c^*$ and $\theta\ge\rho c^*$; then no leaf of the subtree costs $\le\rho c^*$. Closing a node at index $q$ discards the children at $q'\ge q$ as well: their gates contain... the same node response is exact for them up to a later time, and the bound is evaluated per $q$ in increasing order, so the node is closed at the first failing $q$ and the later ones are assumed to fail (monotone in the nested gates, checked numerically). Leaf sums are abandoned only after they exceed the threshold (partial sums of squares are increasing), so no leaf under the threshold is lost. ∎

**Threshold continuation.** The optimum is not known in advance and depth-first search with a poor incumbent prunes badly (measured: 1.8·10⁷ after 2·10⁶ nodes on a wall whose optimum is 3·10³). The search therefore runs with an absolute threshold θ = 0.125 n, doubled until a leaf is found (iterative deepening on the bound, as in IDA*, Korf 1985). Each failed pass is cheap because it prunes hard; the first successful pass ends at the grid optimum; one more pass collects everything within ρ.

## 5. Model-discrepancy weights

The grid problem is not the real problem: positions are quantised to $h$, materials are nominal, the standoff is a guess. On development wall 202 the raw χ² of the best grid wall is above 64 n while the true wall at its true positions has 3 n, so raw χ² on the grid does not discriminate.

Treat the discrepancy as a second noise term (Kennedy & O'Hagan 2001). For an echo $a\,e^{-2jk_0\tau}$ whose interface lies uniformly within $\pm h/2$ of its grid point,
$$\mathbb E\,|e^{-2jk_0\delta}-1|^2=2\Big(1-\frac{\sin k_0h}{k_0h}\Big),$$
and class variability changes the amplitude by a relative $\epsilon_{\rm mat}\approx0.07$, the standoff guess adds $(2k_0\delta_d)^2$. With $\sum|a|^2\approx\overline{|y|^2}$ per band,
$$\sigma_{\rm eff}^2(f)=\sigma^2+\tfrac12\,\overline{|y|^2}\Big[2\Big(1-\frac{\sin k_0h}{k_0h}\Big)+(2k_0\delta_d)^2+\epsilon_{\rm mat}^2\Big].$$

Effect (h = 5 mm): σ_eff/σ ≈ 1–2 below 1 GHz, 6–8 at 3 GHz, 19–26 at 10.6 GHz. The high frequencies, which the grid cannot follow, are down-weighted automatically; the cost of the true wall's grid representative becomes 0.2–0.8 n and thresholds are scale-free. The polish and the final fit use the measurement σ.

## 6. From the grid optimum to the answer

1. Best leaf of every distinct (class sequence, backing) within ρ of the optimum; cluster members expanded.
2. Compiled Levenberg–Marquardt polish (`polish.py`) of thicknesses, materials under the class prior, standoff and water temperature: first under σ_eff (smooth), then under σ.
3. Light block moves on the best three: all class sequences at fixed positions, insertion/removal of gaps (thin gaps below the grid step), one layer at a time.
4. Exact adjoint LM on all bands with cycle hops and free dispersion shape (existing `search.lm`, `cycle_hop`).
5. Model order: each gap is one parameter; candidates with a gap closed are refitted and the Schwarz criterion decides (a gap must lower the MAP cost by more than ln n).
6. Alarm: data χ² > n + 3√(2n). Alternatives within Δcost 10 are reported.

## 7. What is measured (observability)

Per wall, `results/*/…jsonl` records: nodes, pruned subtrees, leaves, passes and their thresholds, whether the search was truncated by the node budget, grid optimum, number of sequences kept, polished and refined costs, final χ²/n, Cramér–Rao bars, alternatives. `devbnb.py` adds, with the truth in hand: the cost of the truth's grid representative relative to the grid optimum, the number of leaves that beat it, and the rank of the true class sequence. A gate-bound violation shows up as "truth representative within ρ of the optimum but absent from the leaves".

## 8. Two-point limits: what no algorithm can do

Let $A$, $B$ be two walls with responses $\Gamma_A,\Gamma_B$ and $\delta^2=\sum_f|\Gamma_A-\Gamma_B|^2/(2\sigma^2)$ (σ per real component). Deciding between them from $y$ is a Gaussian shift problem.

**Proposition.** Any decision rule has average error probability at least $\Phi(-\delta/2)$ (equal priors), and choosing the smaller χ² attains it. $\Delta\chi^2=\chi^2_B-\chi^2_A$ is distributed $\mathcal N(\pm\delta^2,4\delta^2)$ under $A$/$B$.

*Proof.* The log-likelihood ratio is $\Delta\chi^2/2$, a linear function of the Gaussian data with the stated mean and variance; Neyman–Pearson gives the optimal test and its error. ∎

Uses.
- **Ambiguity is a property of the wall, not of the solver.** If the best-fitting wall of another structure lies at δ ≲ 3 from the truth, every algorithm errs with probability ≥ 7 %. The solver's list of alternatives (Δcost < 10) is the set this bound speaks about.
- **Silent failures.** A wrong structure whose best fit is at distance δ from the data's truth passes the alarm with probability about $\Phi\big((3\sqrt{2n}-\delta^2)/\sqrt{2n+4\delta^2}\big)$; for n = 1536 this is below 10⁻³ once δ² > 350. A silent failure is therefore either an ambiguity (small δ) or impossible.

## 9. Related work used

- Sequence detection over channels with memory: Forney 1972 (Viterbi MLSE), Anderson & Mohan 1984 (M-algorithm), sphere decoding / branch-and-bound for integer least squares (Fincke & Pohst 1985; Agrell et al. 2002; cf. Teunissen 1999 in `papers/`). The wall is the "channel", the class sequence the "symbols", causality the trellis order.
- Time–frequency concentration: Slepian & Pollak 1961; Landau & Pollak 1962; Thomson 1982 (multitaper).
- Iterative deepening on a bound: Korf 1985 (IDA*).
- Model discrepancy as noise: Kennedy & O'Hagan 2001.
- Finite-alphabet ("blocky") impedance inversion in reflection seismology, and super-resolution of thin layers in GPR (Le Bastard et al. 2007, in `papers/`).
- Schwarz–Pick lemma; Nevanlinna–Pick interpolation (why finite data cannot certify causality).
