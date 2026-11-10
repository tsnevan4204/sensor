"""
E046+: the stated problem. Known: the number N of SOLID material layers and (approximately) the
radar-to-wall distance. Unknown (outputs): each layer's thickness, material (library class) and
moisture; any number of air gaps (an optional gap at every interface between solid layers, and
in front of a metal backing; thickness 0 = no gap); the backing (air or metal).

  make_problem(seed, N, bands, ...)   truth generator (independent of the solver)
  gamma_batch(X, f, N, back)          exact reflection recursion for P candidate walls at once
  solve(f, y, sigma, N, d_guess)      global search (differential evolution over all unknowns,
                                      library prior, low-frequency subset) -> exact refinement
  score(result, truth)                per-layer thickness / material / gap / backing checks
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import differential_evolution
import wall_toy as W
import stack as ST
import search as SR
import wall_inverse as I
import matlib as ML

C0 = W.C0
F_REF = ST.F_REF
SOLIDS = [m for m in ST.ALL_MATERIALS if m != "air"]
BANDS3 = ((0.4e9, 0.96e9), (3.1e9, 10.6e9), (60e9, 64e9))     # sub-GHz imaging + UWB + mmWave
SNR3 = (30.0, 40.0, 40.0)        # sub-GHz is limited to FCC 15.209 levels: assume 10 dB less SNR
NPTS = 256
GAP_MIN = 0.2e-3                 # a gap below this is reported as "no gap"


# ============================================================================ truth generator
def make_problem(seed, N, bands=BANDS3, snr=SNR3, p_gap=0.35, p_wet=0.0, p_metal=0.3, vis_db=50.0):
    rng = np.random.default_rng(seed)
    while True:
        names = []
        for _ in range(N):
            names.append(str(rng.choice([m for m in SOLIDS if not names or m != names[-1]])))
        back = "metal" if rng.random() < p_metal else "air"
        Tw = rng.uniform(5, 35)
        layers, desc, gaps = [], [], []
        for i, nm in enumerate(names):
            lo, hi = ST.thickness_range(nm)
            t = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            e, dsc = ST.material_law(nm, rng, Tw)
            if nm in ST.POROUS and rng.random() < p_wet:
                th = float(rng.uniform(0.01, 0.12))
                e = ST.wet_law(e, th, Tw)
                dsc = dict(dsc, theta=th)
            layers.append((t, e)); desc.append(dict(name=nm, t=t, **dsc))
            last = i == N - 1
            if (not last or back == "metal") and rng.random() < p_gap:
                g = float(np.exp(rng.uniform(np.log(1e-3), np.log(30e-3))))
                layers.append((g, lambda f: np.ones(np.shape(f), complex)))
                gaps.append(g)
            elif not last or back == "metal":
                gaps.append(0.0)
        if _visible(layers, back, vis_db):
            break
    d = 0.30 + 0.004 * rng.random()
    fs, ys, ss = [], [], []
    for k, ((a, b), sn) in enumerate(zip(bands, snr)):
        f = np.linspace(a, b, NPTS)
        G = ST.gamma_stack(f, layers, back) * np.exp(-2j * 2 * np.pi * f / C0 * d)
        sig = 10 ** (-sn / 20) / np.sqrt(2)                       # per real component
        nr = np.random.default_rng(seed + 7919 * (k + 1))
        ys.append(G + sig * (nr.standard_normal(f.size) + 1j * nr.standard_normal(f.size)))
        fs.append(f); ss.append(np.full(f.size, sig))
    truth = dict(layers=desc, gaps=gaps, back=back, standoff=d, T=Tw)
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ss), truth


def _visible(layers, back, vis_db):
    f = np.array([0.7e9, 5e9])
    k0 = 2 * np.pi * f / C0
    loss = np.zeros(2)
    for t, e in (layers[:-1] if back == "air" else layers):
        n = np.sqrt(e(f) + 0j)
        loss += 2 * k0 * np.abs(n.imag) * t * 20 / np.log(10)
    return bool(np.min(loss) < vis_db)


# ============================================================================ batched forward
# parameter vector: [d, (lt, la, lb, p, c) x N, g x NG]   (NG = N-1 (+1 if metal))
#   t = exp(lt); eps' = 1 + exp(la); tan d = exp(lb); p in [0, .99]; theta = sigmoid(c); g_eff = max(g, 0)
def layout(N, back):
    NG = N - 1 + (1 if back == "metal" else 0)
    lo = [0.0] + [np.log(3e-4), np.log(0.02), np.log(1e-4), 0.0, -6.0] * N + [-10e-3] * NG
    hi = [0.0] + [np.log(0.25), np.log(11.0), np.log(0.5), 0.99, 6.0] * N + [40e-3] * NG
    return np.array(lo), np.array(hi), NG


def gamma_batch(X, f, N, back, T=20.0):
    """X (P, D) -> Gamma (P, F) including the standoff phase; exact reflection recursion."""
    P = X.shape[0]
    k0 = 2 * np.pi * f / C0
    w = W.eps_water_lhm(f, T)[None, :]
    sw = np.sqrt(w)
    lg = -np.log(f / F_REF)[None, :]
    NG = N - 1 + (1 if back == "metal" else 0)
    ns, ts = [], []
    for i in range(N):
        lt, la, lb, p, c = (X[:, 1 + 5 * i + j][:, None] for j in range(5))
        er, td, th = 1 + np.exp(la), np.exp(lb), 1 / (1 + np.exp(-c))
        a = 2 / np.pi * np.arctan(td / (1 - p))
        h = er * (p + (1 - p - 1j * td) * np.exp(a * lg))
        s = (1 - th) * np.sqrt(h) + th * sw
        n = np.where(s.imag > 0, -s, s)
        ns.append(n); ts.append(np.exp(lt))
        if i < NG:
            g = np.maximum(X[:, 1 + 5 * N + i], 0.0)[:, None]
            ns.append(np.ones((P, f.size), complex)); ts.append(g)
    nb = np.sqrt(np.full((1, f.size), ST.METAL_EPS)) if back == "metal" else np.ones((1, f.size), complex)
    G = (ns[-1] - nb) / (ns[-1] + nb)
    for i in range(len(ns) - 1, -1, -1):
        nprev = ns[i - 1] if i > 0 else np.ones_like(ns[0])
        e = np.exp(-2j * k0[None, :] * ns[i] * ts[i])
        r = (nprev - ns[i]) / (nprev + ns[i])
        G = (r + G * e) / (1 + r * G * e)
    return G * np.exp(-2j * k0[None, :] * X[:, :1])


def _prior_batch(X, N, lib_arr, w=1.0):
    """Library-prior penalty for each candidate: nearest-class Mahalanobis distance of every
    non-water layer's (log eps', log10 tan d) at 10 GHz (eps' = er (p + (1-p)) = er there)."""
    pen = np.zeros(X.shape[0])
    mus, Ls = lib_arr
    for i in range(N):
        la, lb, c = X[:, 1 + 5 * i + 1], X[:, 1 + 5 * i + 2], X[:, 1 + 5 * i + 4]
        x = np.stack([np.log(1 + np.exp(la)), np.exp(lb) / np.log(10) * 0 + lb / np.log(10)], axis=1)
        d = x[:, None, :] - mus[None, :, :]                        # (P, C, 2)
        z = np.einsum("pci,cij->pcj", d, Ls)
        m = np.min(np.sum(z ** 2, axis=2), axis=1)
        dry = 1 / (1 + np.exp(c)) > 0.5
        pen += w * np.where(dry, m, 0.0)
    return pen


def _lib_arrays():
    lib = ST.library_stats()
    mus = np.array([mu for mu, C in lib.values()])
    Ls = np.array([np.linalg.cholesky(np.linalg.inv(C)) for mu, C in lib.values()])
    return mus, Ls


# ============================================================================ solver
def _subset(f, y, s, per_band=48, fmax=None):
    idx = []
    for a, b in I.band_segments(f):
        if fmax is not None and f[a] > fmax:
            continue
        idx += list(np.linspace(a, b - 1, min(per_band, b - a)).astype(int))
    idx = np.array(sorted(set(idx)))
    return f[idx], y[idx], s[idx]


def global_search(f, y, s, N, back, d_guess, d_tol=0.01, seed=0, maxiter=1500, popsize=15):
    lo, hi, NG = layout(N, back)
    lo[0], hi[0] = d_guess - d_tol, d_guess + d_tol
    lib_arr = _lib_arrays()

    def obj(Xt):                                     # scipy vectorised: Xt (D, S)
        X = Xt.T
        G = gamma_batch(X, f, N, back)
        r = (G - y[None, :]) / s[None, :]
        chi = np.sum(r.real ** 2 + r.imag ** 2, axis=1)
        out = chi + _prior_batch(X, N, lib_arr)
        return np.where(np.isfinite(out), out, 1e30)
    res = differential_evolution(obj, list(zip(lo, hi)), vectorized=True, updating="deferred",
                                 maxiter=maxiter, popsize=popsize, tol=1e-8, seed=seed,
                                 init="sobol", polish=False, mutation=(0.5, 1.0), recombination=0.9)
    return res.x, float(res.fun)


def _to_stack(x, N, back):
    """DE vector -> stack.py layer list (gaps as air layers), standoff, gap slot map."""
    lay, slots = [], []
    NG = N - 1 + (1 if back == "metal" else 0)
    for i in range(N):
        lt, la, lb, p, c = x[1 + 5 * i: 6 + 5 * i]
        th = 1 / (1 + np.exp(-c))
        if th > 0.6:
            lay.append(dict(t=float(np.exp(lt)), er=1.0, td=0.0, water=True))
        elif th > 0.03:
            lay.append(dict(t=float(np.exp(lt)), er=float(1 + np.exp(la)), td=float(np.exp(lb)), water=False,
                            moist=True, theta=float(th), pinf=float(p)))
        else:
            lay.append(dict(t=float(np.exp(lt)), er=float(1 + np.exp(la)), td=float(np.exp(lb)), water=False,
                            pinf=float(p)))
        slots.append(("solid", i))
        if i < NG:
            g = float(x[1 + 5 * N + i])
            if g > GAP_MIN:
                lay.append(dict(t=g, er=1.0001, td=1e-5, water=False))
                slots.append(("gap", i))
    return lay, float(x[0]), slots


def solve(f, y, sigma, N, d_guess, d_tol=0.01, seed=0, verbose=False):
    t0 = time.time()
    fs, ys, ss = _subset(f, y, sigma, per_band=64, fmax=20e9)   # global search on the smooth (low) bands only
    cands = []
    for back in ("air", "metal"):
        x, val = global_search(fs, ys, ss, N, back, d_guess, d_tol, seed=seed)
        cands.append((val, x, back))
        if verbose:
            print(f"  DE {back}: cost {val:.1f} ({time.time() - t0:.1f}s)", flush=True)
    cands.sort(key=lambda t: t[0])
    pri = ML.LibPrior()
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    best = None
    try:
        for val, x, back in cands:
            lay, d, slots = _to_stack(x, N, back)
            ST.SBOX = (d - 0.005, d + 0.005)
            mod = ST.build(lay, d, back=back)
            ST.SBOX = None
            for (kind, _), L in zip(slots, mod.layers):
                if kind == "gap":                     # gap layers are air: only thickness free
                    L.eps_r.free = L.tan_d.free = False
                    L.eps_r.value, L.tan_d.value = 1.0, 0.0
                elif L.theta.value < 0.5:
                    L.pinf.free = True
            mod, c = ST._fit(mod, f, y, sigma)
            mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))   # high-band cycles
            if best is None or c < best[1]:
                best = (mod, c, slots, back)
    finally:
        SR._PRIORS = old
    mod, c, slots, back = best
    G, _ = I.forward(mod, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
    n_all = 2 * f.size
    res = dict(model=mod, slots=slots, back=back, chi2=chi, chi2n=chi / n_all,
               alarm=bool(chi > ST.chi2_alarm(n_all)), wall_s=time.time() - t0,
               labels=pri.labels(mod))
    return res


def report(res, N):
    """Per solid layer (t, label, eps', tan d, theta) and per gap slot thickness (0 if absent)."""
    mod, slots = res["model"], res["slots"]
    solids, gaps = [], {}
    for (kind, i), L, lab in zip(slots, mod.layers, res["labels"]):
        if kind == "solid":
            solids.append(dict(t=L.t.value, label=lab, er=L.eps_r.value, td=L.tan_d.value,
                               theta=L.theta.value))
        else:
            gaps[i] = L.t.value
    NG = N - 1 + (1 if res["back"] == "metal" else 0)
    return solids, [gaps.get(i, 0.0) for i in range(NG)]


def score(res, truth, N, t_tol=(0.3e-3, 0.02), g_tol=(0.3e-3, 0.05)):
    solids, gaps = report(res, N)
    ok_back = res["back"] == truth["back"]
    tl = truth["layers"]
    t_ok = [abs(s["t"] - L["t"]) <= max(t_tol[0], t_tol[1] * L["t"]) for s, L in zip(solids, tl)]
    m_ok = [s["label"] == L["name"] for s, L in zip(solids, tl)]
    tg = list(truth["gaps"]) + [0.0] * (len(gaps) - len(truth["gaps"]))
    fg = list(gaps) + [0.0] * (len(tg) - len(gaps))
    g_ok = [abs(a - b) <= max(g_tol[0], g_tol[1] * a) for a, b in zip(tg, fg)]
    ok = ok_back and all(t_ok) and all(g_ok) and len(solids) == N
    return dict(ok=bool(ok), back_ok=ok_back, t_ok=t_ok, mat_ok=m_ok, gap_ok=g_ok,
                t_err_mm=[1e3 * (s["t"] - L["t"]) for s, L in zip(solids, tl)],
                gaps_true_mm=[1e3 * g for g in tg], gaps_fit_mm=[1e3 * g for g in fg])
