"""
Integer treatment of the phase-cycle ambiguity (E012), after Teunissen's integer least squares
(LAMBDA) and success-rate theory.

Model. Let p* be the best fit and a in Z^K the vector of whole phase-period shifts of the K
thickness-like parameters (standoff, drywall_front.t, insulation.t, drywall_back.t), each period
lambda_c / (2 Re n). Profiling out the continuous parameters (LM from the shifted point) gives
    q(a) := chi2_min(a) - chi2_min(0).
Because a one-period shift is tiny compared with the range resolution (1.9 mm vs 3.75 cm), the
group-delay part of the misfit is quadratic in the shift, so
    q(a) ~= (a - a_hat)^T W (a - a_hat) - a_hat^T W a_hat,      W = Q^{-1},
where a_hat is the real-valued ("float") ambiguity and Q its covariance in cycles^2
(chi2 = -2 log L + const, so W is the Fisher information of a).
Steps
  1. Evaluate q at +-e_i and e_i + e_j (2K + K(K-1)/2 profiled LM runs) and fit (W, b = W a_hat)
     by least squares (the system is square-determined plus redundancy if more points are used).
  2. Integer least squares: enumerate all a in a box, keep those with predicted q < margin.
     This includes joint moves (e.g. +1 in one layer and -1 in the next) automatically.
  3. Verify every candidate by a profiled LM on all frequencies. Plausible set = verified
     candidates with q < 4. If a candidate beats the incumbent, re-centre and repeat.
  4. Success rate of integer estimation for covariance Q:
       bootstrapped lower bound  P_B = prod_i [2 Phi(1 / (2 sqrt(d_i))) - 1],
       with Q_z = Z Q Z^T = L D L^T after an integer (unimodular) decorrelation Z, and
       a Monte Carlo estimate of the integer-least-squares success rate.
"""
from __future__ import annotations
import itertools
from math import erf, sqrt
import numpy as np
import wall_inverse as I
import search as SR


def _names(model, f):
    per = I._phase_periods(model, f)
    order = ["standoff", "drywall_front.t", "insulation.t", "drywall_back.t"]
    return [n for n in order if n in per], per


def _profiled(model, f, y, sigma, opts, a, names):
    trial = SR._shift(model, f, [(n, int(k)) for n, k in zip(names, a) if k != 0])
    if trial is None:
        return None, np.inf
    if opts.search_step > 1 or opts.search_atol > 0:
        fs, ys = SR._subset(f, y, opts.search_step)
        trial, _ = SR.lm(trial, fs, ys, sigma, atol=opts.search_atol, coords=opts.coords)
    return SR.lm(trial, f, y, sigma, coords=opts.coords)


def fit_quadratic(points, values):
    """Least-squares fit of q(a) = a^T W a - 2 a^T b (no constant: q(0) = 0)."""
    K = len(points[0])
    iu = [(i, j) for i in range(K) for j in range(i, K)]
    A = []
    for a in points:
        a = np.asarray(a, float)
        A.append([a[i] * a[j] * (1 if i == j else 2) for i, j in iu] + list(-2 * a))
    sol = np.linalg.lstsq(np.array(A), np.array(values), rcond=None)[0]
    W = np.zeros((K, K))
    for (i, j), w in zip(iu, sol[:len(iu)]):
        W[i, j] = W[j, i] = w
    b = sol[len(iu):]
    return W, b


def decorrelate(Q):
    """Pairwise integer Gauss reduction: repeatedly subtract integer multiples of one row of Z
    from another while that lowers a diagonal entry of Z Q Z^T. Every step is unimodular
    (det Z = 1), so z = Z a stays integer. Simpler than the full LAMBDA Z-transform; adequate for
    K <= 4. Decorrelation only tightens the bootstrap bound, it never invalidates it."""
    K = Q.shape[0]
    Z = np.eye(K, dtype=int)
    for _ in range(50):
        changed = False
        Qz = Z @ Q @ Z.T
        for i in range(K):
            for j in range(K):
                if i != j:
                    mu = int(np.round(Qz[i, j] / Qz[j, j]))
                    if mu != 0:
                        Zn = Z.copy()
                        Zn[i] -= mu * Z[j]
                        Qn = Zn @ Q @ Zn.T
                        if Qn[i, i] < Qz[i, i] - 1e-15:
                            Z, Qz, changed = Zn, Qn, True
        if not changed:
            break
    return Z


def bootstrap_success(Q):
    """Teunissen's bootstrapped success rate (a lower bound on integer least squares)."""
    Z = decorrelate(Q)
    Qz = Z @ Q @ Z.T
    # LDL^T via sequential conditioning: d_i = conditional variance of z_i given z_1..z_{i-1}
    K = Qz.shape[0]
    d = np.zeros(K)
    for i in range(K):
        if i == 0:
            d[i] = Qz[0, 0]
        else:
            A = Qz[:i, :i]
            c = Qz[:i, i]
            d[i] = Qz[i, i] - c @ np.linalg.solve(A, c)
    Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    return float(np.prod([2 * Phi(1 / (2 * sqrt(max(di, 1e-300)))) - 1 for di in d]))


def ils_success_mc(Q, n=20000, box=3, rng=0):
    """Monte Carlo success rate of integer least squares for float error ~ N(0, Q)."""
    K = Q.shape[0]
    rng = np.random.default_rng(rng)
    Lc = np.linalg.cholesky(Q + 1e-15 * np.eye(K))
    e = rng.standard_normal((n, K)) @ Lc.T
    W = np.linalg.inv(Q)
    grid = np.array(list(itertools.product(range(-box, box + 1), repeat=K)), float)
    ok = 0
    for chunk in np.array_split(e, max(1, n // 2000)):
        d = chunk[:, None, :] - grid[None, :, :]
        qd = np.einsum("nki,ij,nkj->nk", d, W, d)
        best = grid[np.argmin(qd, axis=1)]
        ok += np.sum(np.all(best == 0, axis=1))
    return ok / n


def resolve(model, f, y, sigma, opts, box=2, margin=8.0, dchi2=4.0, max_recentre=3):
    """Integer-least-squares ambiguity resolution. Returns best model, cost, plausible basins
    [(summary, chi2)], and an info dict (W, a_hat, success rates, LM runs)."""
    n_lm = 0
    best = model
    best_cost = SR.cost_of(best, f, y, sigma)
    for rec in range(max_recentre + 1):
        names, per = _names(best, f)
        K = len(names)
        pts = [tuple(int(i == k) * s for i in range(K)) for k in range(K) for s in (1, -1)]
        pts += [tuple(int(i in (j, k)) for i in range(K)) for j in range(K) for k in range(j + 1, K)]
        evaluated = {(0,) * K: (best, best_cost)}
        vals = []
        for a in pts:
            mo, c = _profiled(best, f, y, sigma, opts, a, names)
            n_lm += 1
            evaluated[a] = (mo, c)
            vals.append(c - best_cost if np.isfinite(c) else 1e6)
        finite = [(a, v) for a, v in zip(pts, vals) if v < 1e5]
        W, b = fit_quadratic([a for a, _ in finite], [v for _, v in finite])
        # predicted q on the box; propose candidates
        grid = list(itertools.product(range(-box, box + 1), repeat=K))
        G = np.array(grid, float)
        qpred = np.einsum("ki,ij,kj->k", G, W, G) - 2 * G @ b
        cands = [tuple(a) for a, q in zip(grid, qpred) if q < margin and tuple(a) not in evaluated]
        for a in cands:
            mo, c = _profiled(best, f, y, sigma, opts, a, names)
            n_lm += 1
            evaluated[a] = (mo, c)
        winner = min(evaluated.items(), key=lambda kv: kv[1][1])
        if winner[1][1] < best_cost * (1 - 1e-9):
            best, best_cost = winner[1]
            continue                                   # re-centre on the better basin
        break
    plaus = sorted([(m.summary(), c) for a, (m, c) in evaluated.items()
                    if m is not None and c <= best_cost + dchi2], key=lambda t: t[1])
    try:
        Wp = 0.5 * (W + W.T)
        ev = np.linalg.eigvalsh(Wp)
        if ev.min() <= 0:
            raise np.linalg.LinAlgError
        Q = np.linalg.inv(Wp)
        a_hat = Q @ b
        p_boot = bootstrap_success(Q)
        p_mc = ils_success_mc(Q)
    except np.linalg.LinAlgError:
        Q, a_hat, p_boot, p_mc = None, None, float("nan"), float("nan")
    # posterior basin probabilities under a flat prior over the plausible set
    cs = np.array([c for _, c in plaus])
    post = np.exp(-(cs - cs.min()) / 2)
    post = (post / post.sum()).tolist()
    info = dict(ils_names=names, W=W.tolist(), a_hat=None if a_hat is None else a_hat.tolist(),
                p_success_bootstrap=p_boot, p_success_ils_mc=p_mc, ils_lm_runs=n_lm,
                basin_posterior=post, recentred=rec,
                # probability that the reported (best) basin is the right one, from the verified
                # plausible set under a flat prior over basins; defined even when W is not positive
                # definite (an unconstrained ambiguity), where the closed-form rates are undefined
                p_best_basin=post[0], n_plausible=len(plaus))
    return best, best_cost, plaus, info
