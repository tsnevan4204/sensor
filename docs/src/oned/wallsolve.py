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
    truth = dict(layers=desc, gaps=gaps, back=back, standoff=d, T=Tw, _layers=layers)
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
# parameter vector: [d, (lt, la, lb, u, c) x N, g x NG, T]   (NG = N-1 (+1 if metal))
#   t = exp(lt); eps' = 1 + exp(la); tan d = exp(lb); p = 1 - exp(-u) (u in [0,7]: p up to 0.999,
#   brick/plywood need p ~ 0.996); theta = sigmoid(c); g_eff = max(g, 0); T = water temperature (C)
def layout(N, back):
    NG = N - 1 + (1 if back == "metal" else 0)
    lo = [0.0] + [np.log(5e-4), np.log(0.02), np.log(1e-4), 0.0, -6.0] * N + [-10e-3] * NG + [0.0]
    hi = [0.0] + [np.log(0.6), np.log(11.0), np.log(0.5), 7.0, 6.0] * N + [40e-3] * NG + [40.0]
    return np.array(lo), np.array(hi), NG


def _theta(c):
    """Water fraction from its search coordinate, EXACTLY 0 below c ~ -4.6 (a floor of 0.25 %
    liquid water shifts eps' by ~0.05 - far above the noise: E046 bug)."""
    return np.clip((1 / (1 + np.exp(-c)) - 0.01) / 0.98, 0.0, 1.0)


def _nopt(er, th):
    """Index converting optical thickness to thickness: CRIM mix of sqrt(eps') and water (~8.6)."""
    return (1 - th) * np.sqrt(er) + th * 8.6


def gamma_batch(X, f, N, back):
    """X (P, D) -> Gamma (P, F) including the standoff phase; exact reflection recursion."""
    P = X.shape[0]
    k0 = 2 * np.pi * f / C0
    w = W.eps_water_lhm(f[None, :], X[:, -1:])                    # (P, F): fitted water temperature
    sw = np.sqrt(w)
    lg = -np.log(f / F_REF)[None, :]
    NG = N - 1 + (1 if back == "metal" else 0)
    ns, ts = [], []
    for i in range(N):
        lt, la, lb, u, c = (X[:, 1 + 5 * i + j][:, None] for j in range(5))
        er, td, th, p = 1 + np.exp(la), np.exp(lb), _theta(c), 1 - np.exp(-u)
        a = 2 / np.pi * np.arctan(td / (1 - p))
        h = er * (p + (1 - p - 1j * td) * np.exp(a * lg))
        s = (1 - th) * np.sqrt(h) + th * sw
        n = np.where(s.imag > 0, -s, s)
        ns.append(n); ts.append(np.exp(lt) / _nopt(er, th))       # lt = log optical thickness
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
        dry = _theta(c) < 0.5
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


class _Envelope:
    """E046b: envelope (|range profile|) misfit per band. Echo envelopes move smoothly with
    thickness, so the global search sees no cycle skipping (seismic envelope inversion: Bozdag et
    al. 2011; Wu, Luo & Wu 2014). Range window: 3 cm before the front guess to 1 m behind it;
    noise level of the envelope from the data's far-range bins."""
    def __init__(self, f, y, s, d_guess, pad=4):
        self.segs, self.w, self.sel, self.Pd, self.sn = [], [], [], [], []
        for a, b in I.band_segments(f):
            F = b - a
            w = np.hanning(F)
            n = pad * F
            r = np.arange(n) * C0 / (2 * (f[a + 1] - f[a]) * n)
            sel = (r > d_guess - 0.03) & (r < d_guess + 1.0)
            P = np.abs(np.fft.ifft(y[a:b] * w, n))
            far = (r > d_guess + 1.5) & (r < r.max() - 0.5)
            sn = np.std(P[far]) if np.any(far) else np.median(s[a:b]) * np.sqrt(np.sum(w ** 2)) / F
            self.segs.append((a, b)); self.w.append(w); self.sel.append((n, sel)); self.Pd.append(P[sel])
            self.sn.append(max(sn, 1e-12))

    def misfit(self, G):
        out = np.zeros(G.shape[0])
        for (a, b), w, (n, sel), Pd, sn in zip(self.segs, self.w, self.sel, self.Pd, self.sn):
            Pm = np.abs(np.fft.ifft(G[:, a:b] * w[None, :], n, axis=1))[:, sel]
            out += np.sum((Pm - Pd[None, :]) ** 2, axis=1) / sn ** 2
        return out


def global_search(f, y, s, N, back, d_guess, d_tol=0.01, seed=0, maxiter=1500, popsize=20):
    lo, hi, NG = layout(N, back)
    lo[0], hi[0] = d_guess - d_tol, d_guess + d_tol
    lib_arr = _lib_arrays()

    env = _Envelope(f, y, s, d_guess)

    def obj(Xt):                                     # scipy vectorised: Xt (D, S)
        X = Xt.T
        G = gamma_batch(X, f, N, back)
        out = env.misfit(G) + _prior_batch(X, N, lib_arr)
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
        lt, la, lb, u, c = x[1 + 5 * i: 6 + 5 * i]
        th, p = float(_theta(c)), 1 - np.exp(-u)
        lt = np.log(np.exp(lt) / _nopt(1 + np.exp(la), th))          # back to physical thickness
        if th > 0.6:
            lay.append(dict(t=float(np.exp(lt)), er=1.0, td=0.0, water=True))
        elif th > 0.005:
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
    return lay, float(x[0]), slots, float(x[-1])


def solve(f, y, sigma, N, d_guess, d_tol=0.01, seed=0, verbose=False):
    t0 = time.time()
    fs, ys, ss = f, y, sigma                       # envelope misfit is smooth on every band: use all data
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
            lay, d, slots, Tw = _to_stack(x, N, back)
            ST.SBOX = (d - 0.005, d + 0.005)
            mod = ST.build(lay, d, back=back, water_T=Tw)
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


# ============================================================================ E048: measured eps(f), unknown flag
EPS_REPORT_F = np.array([0.5e9, 0.8e9, 3.1e9, 5e9, 7.5e9, 10e9, 60e9, 62e9, 64e9])
UNKNOWN_D2 = 13.8        # chi2(2 dof) 99.9 % quantile: farther than this from every class = unknown


def free_refit(res, f, y, sigma):
    """Refit the solved wall with the material-library prior OFF: every layer's eps(f) is what the
    data say. Returns per-solid-layer measured eps(f) at EPS_REPORT_F with 1-sigma errors (Fisher,
    delta method) and the nearest library class with its Mahalanobis distance (unknown flag)."""
    mod = res["model"]
    old = SR._PRIORS
    SR._PRIORS = ()
    try:
        mod, c = ST._fit(mod, f, y, sigma)
    finally:
        SR._PRIORS = old
    try:
        F = I.fisher(mod, f, sigma)
        cov, names = F["cov"], F["names"]
    except Exception:
        cov, names = None, mod.free_names()
    lib = ST.library_stats()
    T = mod.water_Tp.value if mod.water_Tp is not None else mod.water_T
    out = []
    for (kind, i), L in zip(res["slots"], mod.layers):
        if kind != "solid":
            continue
        eps, d = I.layer_eps_and_derivs(L, EPS_REPORT_F, T, mod.law, mod.f_ref, mod.water_law)
        err = np.full(EPS_REPORT_F.size, np.nan)
        if cov is not None:                      # delta method on |eps| components
            g = np.zeros((len(names), EPS_REPORT_F.size), complex)
            for key in ("eps_r", "tan_d", "pinf", "theta"):
                o = mod.owner(f"{L.name}.{key}")
                if o is not None and o in names:
                    g[names.index(o)] += d[key]
            err = np.sqrt(np.abs(np.einsum("if,ij,jf->f", g.real, cov, g.real))
                          + np.abs(np.einsum("if,ij,jf->f", g.imag, cov, g.imag)))
        x = np.array([np.log(max(L.eps_r.value, 1.0001)), np.log10(max(L.tan_d.value, 1e-6))])
        dists = {nm: float((x - mu) @ np.linalg.solve(C, x - mu)) for nm, (mu, C) in lib.items()}
        nearest = min(dists, key=dists.get)
        water = L.theta.value > 0.5 and not L.theta.free
        out.append(dict(t=L.t.value, eps=eps, eps_err=err, theta=L.theta.value,
                        label="water" if water else (nearest if dists[nearest] <= UNKNOWN_D2 else "unknown"),
                        nearest=nearest, d2=dists[nearest]))
    G, _ = I.forward(mod, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
    return dict(model=mod, chi2=chi, layers=out)


def random_causal_material(rng):
    """Out-of-library truth material: random Jonscher host (eps' 1.5-9, tan d 1e-3-0.2, shape p)
    plus, half the time, one Cole-Cole relaxation (centre 0.5-30 GHz, strength 0.2-3, alpha 0-0.4).
    Exactly causal and passive; NOT representable by the solver's family when the relaxation is on."""
    er = float(np.exp(rng.uniform(np.log(1.5), np.log(9.0))))
    td = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2))))
    p = float(rng.uniform(0, 0.9))
    relax = rng.random() < 0.5
    de, f0, al = float(rng.uniform(0.2, 3.0)), float(np.exp(rng.uniform(np.log(0.5e9), np.log(30e9)))), float(rng.uniform(0, 0.4))

    def eps(f):
        f = np.asarray(f, float)
        a = 2 / np.pi * np.arctan(td / (1 - p))
        e = er * (p + (1 - p - 1j * td) * (f / F_REF) ** (-a))
        if relax:
            e = e + de / (1 + (1j * f / f0) ** (1 - al))
        return e
    return eps, dict(name="custom", eps10=er, tan10=td, p=p, relax=relax, d_eps=de, f0=f0, alpha=al)


# ============================================================================ E049: echo-subset solver
def _impedance(amps_low, idx):
    """Interval indices from the chosen echoes' amplitudes (lowest band), recursion with two-way
    transmission correction."""
    a = np.real(np.asarray(amps_low)[list(idx)])
    n_prev, T2, ns = 1.0, 1.0, []
    for k in range(len(idx) - 1):
        r = float(np.clip(a[k] / max(T2, 1e-3), -0.95, 0.95))
        n = float(np.clip(n_prev * (1 - r) / (1 + r), 1.0, 9.0))
        ns.append(n); T2 *= (1 - r ** 2); n_prev = n
    return ns


def _candidates(taus, amps, N, front):
    """front: index of the echo that is the wall's front face (pinned by the known standoff)."""
    import itertools
    M_all = len(taus)
    out = []
    later = [k for k in range(M_all) if k > front]
    for M in range(N + 1, min(2 * N + 1, len(later) + 1) + 1):
        for rest in itertools.combinations(later, M - 1):
            idx = (front,) + rest
            ns = _impedance(amps[0], idx)
            dt = np.diff(np.asarray(taus)[list(idx)])
            n_gap = (M - 1) - N
            if n_gap < 0:
                continue
            for gaps in itertools.combinations(range(M - 1), n_gap):
                if any(g + 1 < M - 1 and g + 1 in gaps for g in gaps):
                    continue                              # two gaps in a row = one gap
                lay, slots, k_solid = [], [], 0
                for k in range(M - 1):
                    if k in gaps:
                        lay.append(dict(t=max(dt[k], 3e-4), er=1.0001, td=1e-5, water=False))
                        slots.append(("gap", k_solid - 1))
                    else:
                        n = max(ns[k], 1.02)
                        if n > 6.0:
                            lay.append(dict(t=max(dt[k] / 8.6, 3e-4), er=1.0, td=0.0, water=True))
                        else:
                            lay.append(dict(t=max(dt[k] / n, 3e-4), er=n ** 2, td=0.02, water=False))
                        slots.append(("solid", k_solid)); k_solid += 1
                if slots and slots[0][0] == "gap":
                    continue                              # nothing before the first solid layer
                out.append((float(taus[idx[0]]), lay, slots))
    return out


SPLIT = 5e-3            # optional extra interfaces around the strongest echoes (thin layers)
N_SPLIT = 3             # only the N_SPLIT strongest echoes behind the front are split
AMBIG_DCHI2 = 10.0      # structurally different answers within this data-chi2 of the best are reported


def _cand_to_x(d, lay, slots, N, back):
    """Candidate (layer list with gap layers) -> gamma_batch parameter vector, or None if the
    candidate does not fit the layout (e.g. a gap behind the last layer with an air backing)."""
    lo, hi, NG = layout(N, back)
    x = np.zeros(lo.size)
    x[0], x[-1] = d, 20.0
    si = 0
    for (kind, i), L in zip(slots, lay):
        if kind == "solid":
            if L["water"]:
                x[1 + 5 * si: 6 + 5 * si] = [np.log(L["t"] * 8.6), 0.0, np.log(1e-3), 0.0, 6.0]
            else:
                er = max(L["er"], 1.0001)
                x[1 + 5 * si: 6 + 5 * si] = [np.log(L["t"] * np.sqrt(er)), np.log(er - 1), np.log(max(L["td"], 1e-4)), 0.0, -6.0]
            si += 1
        else:
            gi = si - 1
            if gi >= NG:
                return None
            x[1 + 5 * N + gi] = L["t"]
    return x if si == N else None


def _screen_batch(cands, f, y, sigma, N, chunk=2000):
    """Score every candidate (both backings) with ONE exact batched forward per chunk."""
    out = []
    for back in ("air", "metal"):
        xs, meta = [], []
        for d, lay, slots in cands:
            x = _cand_to_x(d, lay, slots, N, back)
            if x is not None:
                xs.append(x); meta.append((d, lay, slots))
        for a in range(0, len(xs), chunk):
            X = np.array(xs[a:a + chunk])
            G = gamma_batch(X, f, N, back)
            c = np.sum(np.abs((G - y[None, :]) / sigma[None, :]) ** 2, axis=1)
            out += [(float(ci), *meta[a + j], back) for j, ci in enumerate(c)]
    out.sort(key=lambda t: t[0])
    return out


def solve_echo(f, y, sigma, N, d_guess, n_refine=8, verbose=False):
    import echoes as EC
    t0 = time.time()
    taus, amps, _ = EC.find_echoes(f, y, sigma, d_guess)
    # E049b: the radar-to-wall distance is KNOWN: the front face is the echo nearest d_guess
    # (within 3 mm); if the front face is too weak to be found (foam, batt), one is inserted there
    k = int(np.argmin(np.abs(taus - d_guess))) if len(taus) else -1
    if k < 0 or abs(taus[k] - d_guess) > 3e-3:
        taus = np.sort(np.append(taus, d_guess))
        taus, amps, _ = EC._refine(EC._bands(f, y, sigma), taus, iters=0)
        k = int(np.argmin(np.abs(taus - d_guess)))
    # E049d: thin layers merge two interfaces into one echo (6 mm plywood: 18.1/28.5 mm -> 24.3 mm);
    # every echo behind the front also spawns optional interfaces at +-SPLIT, so a merged echo can split
    later = [(abs(amps[0][j]), t) for j, t in enumerate(taus) if t > taus[k] + 2e-3]
    strong = [t for _, t in sorted(later, reverse=True)[:N_SPLIT]]          # bounded: combinatorics
    extra = [t + dd for t in strong for dd in (-SPLIT, SPLIT) if t + dd > taus[k] + 2e-3]
    taus = np.sort(np.concatenate([taus, extra]))
    taus, amps, _ = EC._refine(EC._bands(f, y, sigma), taus, iters=0)
    k = int(np.argmin(np.abs(taus - d_guess)))
    cands = _candidates(taus, amps, N, k)
    scored = _screen_batch(cands, f, y, sigma, N)          # E049e: vectorised screening
    if verbose:
        print(f"  echoes {len(taus)}, candidates {len(cands)}, screen {time.time() - t0:.1f}s", flush=True)
    pri = ML.LibPrior()
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    best = None
    try:
        # E049c: two-stage screening - a short exact fit of the top 40 before ranking (raw starting
        # values rank the true structure poorly when a low-contrast gap or layer is present)
        short = []
        for c0, d, lay, slots, back in scored[:40]:
            ST.SBOX = (d - 0.004, d + 0.004)
            mod = ST.build(lay, d, back=back, water_T=20.0)
            ST.SBOX = None
            for (kind, _), L in zip(slots, mod.layers):
                if kind == "gap":
                    L.eps_r.free = L.tan_d.free = False
                    L.eps_r.value, L.tan_d.value = 1.0, 0.0
            ST.set_dispersion_free(mod, True)
            mod, c = ST._fit(mod, f, y, sigma, atol=1e-2, max_iter=15)
            short.append((c, d, lay, slots, back))
        short.sort(key=lambda t: t[0])
        refined = []
        for c0, d, lay, slots, back in short[:n_refine]:
            ST.SBOX = (d - 0.004, d + 0.004)
            mod = ST.build(lay, d, back=back, water_T=20.0)
            ST.SBOX = None
            for (kind, _), L in zip(slots, mod.layers):
                if kind == "gap":
                    L.eps_r.free = L.tan_d.free = False
                    L.eps_r.value, L.tan_d.value = 1.0, 0.0
            ST.set_dispersion_free(mod, True)
            mod, c = ST._fit(mod, f, y, sigma)
            mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
            for L in mod.layers:                          # release the dispersion shape
                if L.theta.value < 0.5 and L.eps_r.free:
                    L.pinf.value, L.pinf.free = max(L.pinf.value, 0.02), True
            mod, c = ST._fit(mod, f, y, sigma)
            refined.append((c, mod, slots, back))
            if best is None or c < best[1]:
                best = (mod, c, slots, back)
    finally:
        SR._PRIORS = old
    mod, c, slots, back = best
    G, _ = I.forward(mod, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
    n_all = 2 * f.size
    # E049c: per-layer thickness error bars (CRB); a layer whose error bar exceeds 20 % of its
    # thickness is reported as poorly constrained (e.g. a weak layer behind strongly absorbing water)
    crb = {}
    try:
        F = I.fisher(mod, f, sigma)
        crb = dict(zip(F["names"], F["crb_std"]))
    except Exception:
        pass
    t_err = [crb.get(f"{L.name}.t", np.nan) for L in mod.layers]
    # E049d: ambiguity report - structurally different answers that fit (almost) as well as the best
    def _sig(m, sl, bk):
        sol = [(round(L.t.value * 1e3), lab) for (kind, _), L, lab in zip(sl, m.layers, pri.labels(m)) if kind == "solid"]
        gp = [round(L.t.value * 1e3) for (kind, _), L in zip(sl, m.layers) if kind == "gap"]
        return (tuple(sol), tuple(gp), bk)
    alts, seen = [], {_sig(mod, slots, back)}
    for cc, mm, sl, bk in sorted(refined, key=lambda t: t[0]):
        Gm, _ = I.forward(mm, f, jacobian=False)
        cm = float(np.sum(np.abs((Gm - y) / sigma) ** 2))
        sg = _sig(mm, sl, bk)
        if cm <= chi + AMBIG_DCHI2 and sg not in seen:
            seen.add(sg)
            alts.append(dict(chi2=cm, solids=[(t, lab) for t, lab in sg[0]], gaps=list(sg[1]), back=bk))
    weak = [bool(np.isfinite(e) and e > 0.2 * L.t.value) for e, L in zip(t_err, mod.layers)]
    return dict(model=mod, slots=slots, back=back, chi2=chi, chi2n=chi / n_all,
                alarm=bool(chi > ST.chi2_alarm(n_all)), wall_s=time.time() - t0, labels=pri.labels(mod),
                n_echoes=len(taus), n_cands=len(cands), t_err=t_err, weak=weak,
                ambiguous=bool(alts), alternatives=alts)
