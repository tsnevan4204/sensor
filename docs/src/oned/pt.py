"""
E054: parallel tempering (replica exchange) over the full continuous known-N parameter vector, exact
batched forward (wallsolve.gamma_batch), no echo detection.

Target  pi_beta(x) ~ exp(-beta * E(x)),  E = chi2/2 + library prior/2 + thickness-range prior/2.
Coordinates per solid layer: log optical thickness, log(eps'-1), log tan d, u (p = 1 - e^-u); gaps
g (g_eff = max(g, 0), so gaps open and close continuously); standoff d. Water is out of scope.
All chains (temperatures x replicas) are one batched forward call per step.
Moves (physics-informed, each a symmetric Metropolis proposal):
  rw     : Gaussian random walk, step ~ beta^-1/2 (capped)
  class  : one layer's (eps', tan d, p) <- a random library class (+ jitter); optical thickness kept
  hop    : one layer's optical thickness +- lambda/2 at the UWB centre (cycle skip)
  shift  : one interface moved by +- lambda/2 (layer i thicker, layer i+1 thinner, optical)
  gap    : one gap toggled open (U[5, 30] mm) / closed
Adjacent temperatures swap states with the usual exchange probability.
"""
from __future__ import annotations
import time
import numpy as np
import wallsolve as WS
import stack as ST

C0 = WS.C0
T_WIDEN = (0.7, 1.4)
FMAX = 11e9
SUB = 96          # frequencies per band used while tempering (all are used in the LM polish)


class Problem:
    def __init__(self, f, y, s, N, back, d_guess, d_tol=3e-3, fmax=FMAX):
        sel = f < fmax                                   # mmWave phase is too rugged to temper over
        f, y, s = f[sel], y[sel], s[sel]
        if SUB:                                          # E054c: temper on a subset of each band
            import wall_inverse as I
            idx = []
            for a, b in I.band_segments(f):
                idx += list(np.linspace(a, b - 1, min(SUB, b - a)).astype(int))
            f, y, s = f[idx], y[idx], s[idx]
        self.f, self.y, self.s, self.N, self.back = f, y, s, N, back
        self.lo, self.hi, self.NG = WS.layout(N, back)
        self.lo[0], self.hi[0] = d_guess - d_tol, d_guess + d_tol
        self.D = self.lo.size
        self.lib = WS._lib_arrays()
        tab = WS.class_table()
        self.cls = np.array([(np.log(er - 1), np.log(td), -np.log(max(1 - p, 1e-3))) for _, er, td, p in tab])
        self.trng = np.array([ST.thickness_range(nm) for nm, *_ in tab])
        self.cer = np.array([er for _, er, _, _ in tab])
        self.lam2 = C0 / (2 * 6.85e9)
        self.d_guess = d_guess

    def energy(self, X):
        G = WS.gamma_batch(X, self.f, self.N, self.back)
        chi = np.sum(np.abs((G - self.y[None, :]) / self.s[None, :]) ** 2, axis=1)
        pen = WS._prior_batch(X, self.N, self.lib)
        # thickness prior: physical t inside the (widened) range of the nearest class in eps'
        for i in range(self.N):
            lt, la = X[:, 1 + 5 * i], X[:, 2 + 5 * i]
            er = 1 + np.exp(la)
            k = np.argmin(np.abs(np.log(er)[:, None] - np.log(self.cer)[None, :]), axis=1)
            t = np.exp(lt) / np.sqrt(er)
            lo, hi = T_WIDEN[0] * self.trng[k, 0], T_WIDEN[1] * self.trng[k, 1]
            pen += (np.maximum(0, np.log(lo / t)) ** 2 + np.maximum(0, np.log(t / hi)) ** 2) / 0.05 ** 2
        out = 0.5 * (chi + pen)
        bad = np.any(X < self.lo[None, :] - 1e-12, axis=1) | np.any(X > self.hi[None, :] + 1e-12, axis=1)
        return np.where(np.isfinite(out) & ~bad, out, np.inf), chi

    def sample_prior(self, P, rng):
        X = np.zeros((P, self.D))
        X[:, 0] = rng.uniform(self.lo[0], self.hi[0], P)
        X[:, -1] = 20.0
        for i in range(self.N):
            k = rng.integers(0, len(self.cls), P)
            la, lb, u = self.cls[k].T
            er = 1 + np.exp(la)
            t = np.exp(rng.uniform(np.log(self.trng[k, 0]), np.log(self.trng[k, 1])))
            X[:, 1 + 5 * i: 6 + 5 * i] = np.stack([np.log(t * np.sqrt(er)), la, lb, u, np.full(P, -6.0)], 1)
        for j in range(self.NG):
            g = rng.uniform(5e-3, 30e-3, P)
            X[:, 1 + 5 * self.N + j] = np.where(rng.random(P) < 0.35, g, -5e-3)
        return np.clip(X, self.lo, self.hi)


STEP = None


def propose(pb, X, beta, rng):
    P, D = X.shape
    Y = X.copy()
    mv = rng.random(P)
    N = pb.N
    sc = np.minimum(1.0 / np.sqrt(beta), 30.0)[:, None]
    step = np.zeros(D)
    step[0] = 0.15e-3
    for i in range(N):
        step[1 + 5 * i: 6 + 5 * i] = [0.004, 0.02, 0.08, 0.15, 0.0]
    step[1 + 5 * N: 1 + 5 * N + pb.NG] = 0.3e-3
    rw = mv < 0.5
    Y[rw] += rng.standard_normal((rw.sum(), D)) * step[None, :] * sc[rw]
    li = rng.integers(0, N, P)
    j0 = 1 + 5 * li
    cl = (mv >= 0.5) & (mv < 0.7)
    k = rng.integers(0, len(pb.cls), P)
    for a in range(3):
        Y[cl, j0[cl] + 1 + a] = pb.cls[k[cl], a] + rng.standard_normal(cl.sum()) * (0.03, 0.15, 0.2)[a]
    hp = (mv >= 0.7) & (mv < 0.85)
    sg = rng.choice((-1.0, 1.0), P)
    tau = np.exp(Y[np.arange(P), j0])
    Y[hp, j0[hp]] = np.log(np.maximum(tau[hp] + sg[hp] * pb.lam2, 1e-4))
    sh = (mv >= 0.85) & (mv < 0.95) & (li < N - 1)
    tau2 = np.exp(Y[np.arange(P), np.minimum(j0 + 5, D - 1)])
    Y[sh, j0[sh]] = np.log(np.maximum(tau[sh] + sg[sh] * pb.lam2, 1e-4))
    Y[sh, j0[sh] + 5] = np.log(np.maximum(tau2[sh] - sg[sh] * pb.lam2, 1e-4))
    if pb.NG:
        gp = mv >= 0.95
        gj = 1 + 5 * N + rng.integers(0, pb.NG, P)
        cur = Y[np.arange(P), gj]
        new = np.where(cur > 0, -5e-3, rng.uniform(5e-3, 30e-3, P))
        Y[gp, gj[gp]] = new[gp]
    return Y


def run(pb, n_temp=20, n_rep=8, steps=3000, beta_min=1e-4, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    betas = np.geomspace(1.0, beta_min, n_temp)
    beta = np.repeat(betas, n_rep)                     # chain c: temperature c // n_rep
    X = pb.sample_prior(beta.size, rng)
    E, chi = pb.energy(X)
    best = (np.inf, None, None)
    acc = np.zeros(5)
    for it in range(steps):
        Y = propose(pb, X, beta, rng)
        Ey, chy = pb.energy(Y)
        a = np.log(rng.random(beta.size)) < -beta * (Ey - E)
        X[a], E[a], chi[a] = Y[a], Ey[a], chy[a]
        # replica exchange between adjacent temperatures (same replica index)
        for t in range(it % 2, n_temp - 1, 2):
            i = np.arange(t * n_rep, (t + 1) * n_rep); j = i + n_rep
            la = (betas[t] - betas[t + 1]) * (E[i] - E[j])
            sw = np.log(rng.random(n_rep)) < la
            ii, jj = i[sw], j[sw]
            X[ii], X[jj] = X[jj].copy(), X[ii].copy()
            E[ii], E[jj] = E[jj].copy(), E[ii].copy()
            chi[ii], chi[jj] = chi[jj].copy(), chi[ii].copy()
        k = int(np.argmin(E))
        if E[k] < best[0]:
            best = (float(E[k]), X[k].copy(), float(chi[k]))
        if verbose and it % 500 == 0:
            print(f"   it {it}  best E {best[0]:.0f}  chi2n {best[2] / (2 * pb.f.size):.2f}  cold median {np.median(chi[:n_rep]) / (2 * pb.f.size):.1f}", flush=True)
    # distinct cold-ish states for polishing: best of each replica at the 3 coldest temperatures + global
    cand = [best[1]] + [X[c] for c in np.argsort(E)[:12]]
    return best, cand


def x_to_lay(x, N, back):
    lay, slots = [], []
    NG = N - 1 + (1 if back == "metal" else 0)
    for i in range(N):
        lt, la, lb, u, c = x[1 + 5 * i: 6 + 5 * i]
        er = float(1 + np.exp(la))
        lay.append(dict(t=float(np.exp(lt) / np.sqrt(er)), er=er, td=float(np.exp(lb)), water=False,
                        pinf=float(min(1 - np.exp(-u), 0.99))))
        slots.append(("solid", i))
        if i < NG and x[1 + 5 * N + i] > 1e-3:
            lay.append(dict(t=float(x[1 + 5 * N + i]), er=1.0001, td=1e-5, water=False)); slots.append(("gap", i))
    return lay, slots


def solve(f, y, sigma, N, d_guess, steps=4000, n_polish=10, restarts=2, verbose=False, seed=0):
    import matlib as ML, search as SR, wall_inverse as I
    t0 = time.time()
    runs = []
    for back, rs in [(b, r) for b in ("air", "metal") for r in range(restarts)]:
        pb = Problem(f, y, sigma, N, back, d_guess)
        best, cand = run(pb, steps=steps, verbose=verbose, seed=seed + 101 * rs)
        if verbose:
            print(f"  PT {back}: chi2n {best[2] / (2 * f.size):.2f}  {time.time() - t0:.0f}s", flush=True)
        runs += [(pb.energy(c[None, :])[0][0] / pb.f.size, c, back) for c in cand]
    runs.sort(key=lambda r: r[0])
    pri = ML.LibPrior(); old = SR._PRIORS; SR._PRIORS = (pri,)
    best = None
    try:
        seen = 0
        for E, x, back in runs:
            if seen >= n_polish:
                break
            seen += 1
            lay, slots = x_to_lay(x, N, back)
            mod, c = WS._fit_start(lay, slots, float(x[0]), back, f, y, sigma)
            G, _ = I.forward(mod, f, jacobian=False)
            chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
            if best is None or chi < best[0]:
                best = (chi, mod, slots, back)
    finally:
        SR._PRIORS = old
    chi, mod, slots, back = best
    n_all = 2 * f.size
    return dict(model=mod, slots=slots, back=back, chi2=chi, chi2n=chi / n_all, alarm=bool(chi > ST.chi2_alarm(n_all)),
                wall_s=time.time() - t0, labels=pri.labels(mod), weak=[False] * len(mod.layers), unobserved=[],
                route="pt", t_err=[np.nan] * len(mod.layers), alternatives=[], ambiguous=False)
