"""
E041: exact layer peeling for many-layer walls (target: 7-12 layers in < 1 s).

Principle (exact for planar layers at normal incidence). The reflection at the top of layer i,
seen from medium i-1, depends on everything deeper only through Gamma_{i+1}:
    Gamma_i = (r_i + Gamma_{i+1} e_i) / (1 + r_i Gamma_{i+1} e_i),
    r_i = (n_{i-1} - n_i)/(n_{i-1} + n_i),  e_i = exp(-2j k0 n_i t_i).
This Moebius map is invertible (downward continuation):
    Gamma_{i+1} = (Gamma_i - r_i) / (e_i (1 - r_i Gamma_i)),
so once layer i is known, the data become exactly what a radar inside layer i would measure
(checked: 1e-9 relative over 30 random stacks at all depths). An N-layer problem becomes N small
problems; noise is carried along exactly (|dGamma_{i+1}/dGamma_i| = |1 - r^2| / |e| |1 - r Gamma|^2).

Per depth (Bruckstein-Levy-Kailath layer peeling, made robust for band-limited, lossy data):
  - time gate the current data to the first two echoes (top and bottom face of the layer),
  - fit "layer over half-space" (5 params, closed form) from ~18 physics-guided starts
    (eps from the leading-echo amplitude, thickness from the second echo's delay, water option),
  - keep the best few hypotheses (beam search), peel each, stop when the remainder is noise
    (air behind) or a mirror (metal behind).
Then ONE exact joint fit of the whole wall (exact-star DEC forward, adjoint Jacobian, LM) on the
original data: the peeling is the initializer; the answer is the exact optimum.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
import wall_toy as W
import wall_inverse as I
import stack as ST

C0 = W.C0
F_REF = ST.F_REF


def _segs(f):
    return I.band_segments(f)


def n_of(er, td, f, water=False, T=20.0):
    if water:
        n = np.sqrt(W.eps_water_lhm(f, T) + 0j)
    else:
        a = 2 / np.pi * np.arctan(td)
        n = np.sqrt(er * (1 - 1j * td) * (f / F_REF) ** (-a) + 0j)
    return np.where(n.imag > 0, -n, n)


def _gate(f, r_hi, r_lo=-0.01):
    return I.Gate(r_lo, r_hi).op(f)


def _profile(f, G):
    w = np.hanning(f.size)
    P = np.abs(np.fft.ifft(G * w, 4 * f.size))
    r = np.arange(4 * f.size) * C0 / (2 * (f[1] - f[0]) * 4 * f.size)
    return r, P


def _echo_ranges(f, G, sig, k=3):
    """Optical one-way ranges of the strongest echoes after the leading one (low band)."""
    r, P = _profile(f, G)
    noise = np.median(P[(r > 2.0) & (r < 4.0)]) if np.any((r > 2.0) & (r < 4.0)) else 0
    sel = (r > 0.004) & (r < 0.6)
    rr, pp = r[sel], P[sel]
    pk = [i for i in range(1, len(pp) - 1) if pp[i] >= pp[i - 1] and pp[i] >= pp[i + 1] and pp[i] > 4 * noise]
    pk = sorted(pk, key=lambda i: -pp[i])
    out = []
    for i in pk:
        if all(abs(rr[i] - q) > 0.006 for q in out):
            out.append(rr[i])
        if len(out) >= k:
            break
    return out


class Bands:
    """Current (peeled) data per band: f, Gamma at the top of the current layer, noise std per f."""
    def __init__(self, fs, Gs, ss):
        self.fs, self.Gs, self.ss = fs, Gs, ss

    def low(self):
        return int(np.argmin([f[0] for f in self.fs]))


def _cols(p):
    p = np.asarray(p, float)
    return [p[..., k, None] for k in range(p.shape[-1])]


def _model_half(p, f, n_prev, water):
    """Layer (er, td, t) over a half-space (erb, tdb): reflection at the layer's top face.
    p: (P,) or (B, P) -> (F,) or (B, F) (batched for the small Gauss-Newton fits)."""
    if water:
        T, t, erb, tdb = _cols(p)
        n = n_of(0, 0, f, True, T)
    else:
        er, td, t, erb, tdb = _cols(p)
        n = n_of(er, td, f)
    nb = n_of(erb, tdb, f)
    r = (n_prev - n) / (n_prev + n)
    rb = (n - nb) / (n + nb)
    e = np.exp(-2j * 2 * np.pi * f / C0 * n * t)
    return (r + rb * e) / (1 + r * rb * e)


class _Sol:
    def __init__(self, x, c):
        self.x, self.cost = x, c / 2


def _lsq(res_b, q0, lo, hi, max_nfev=60):
    """scipy trust-region least squares (its path finds the right basin from rough starts more
    often than plain LM: E041 check) with a BATCHED forward-difference Jacobian: res_b maps
    (B, P) -> (B, M), so one Jacobian is one vectorised evaluation of P+1 parameter vectors."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    q0 = np.clip(np.asarray(q0, float), lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))

    def fun(q):
        return res_b(q[None])[0]

    def jac(q):
        h = 1e-7 * np.maximum(np.abs(q), 1e-3)
        h = np.where(q + h > hi, -h, h)
        R = res_b(np.vstack([q[None], q[None] + np.diag(h)]))
        return ((R[1:] - R[0][None]) / h[:, None]).T
    return least_squares(fun, q0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev)


def _fit_step(B, n_prevs, starts, r_win, d_free=None, top=4):
    """Least squares of the gated half-space model; all starts are scored without optimisation and
    only the `top` best are optimised. Returns sorted (cost, p, water, d, ndof)."""
    ops = [_gate(f, r_win) for f in B.fs]
    out = []

    def make_res(water):
        def res(Q):                                   # Q: (B, P) -> (B, M)
            Q = np.atleast_2d(Q)
            if d_free is not None:
                d, p = Q[:, :1], Q[:, 1:]
            else:
                d, p = None, Q
            rs = []
            for f, G, s, Op, npv in zip(B.fs, B.Gs, B.ss, ops, n_prevs):
                M = _model_half(p, f, npv, water)
                if d is not None:
                    M = M * np.exp(-2j * 2 * np.pi * f / C0 * d)
                rr = ((M - G) / s) @ Op.T
                rs += [rr.real, rr.imag]
            return np.concatenate(rs, axis=1)
        return res
    scored = []
    for water in (False, True):                       # score all starts of a kind in one batch
        grp = [st for st in starts if st[0] == water]
        if not grp:
            continue
        Q = np.array([([d_free[0]] if d_free is not None else []) + list(st[1]) for st in grp], float)
        R = make_res(water)(Q)
        cs = np.sum(R ** 2, axis=1)
        scored += [(c if np.isfinite(c) else np.inf, st) for c, st in zip(cs, grp)]
    starts = [st for _, st in sorted(scored, key=lambda t: t[0])[:top]]
    for water, p0, lo, hi in starts:
        res = make_res(water)
        q0 = np.array(([d_free[0]] if d_free is not None else []) + list(p0), float)
        ql = np.array(([d_free[1]] if d_free is not None else []) + list(lo), float)
        qh = np.array(([d_free[2]] if d_free is not None else []) + list(hi), float)
        q0 = np.clip(q0, ql + 1e-9, qh - 1e-9)
        try:
            sol = _lsq(res, q0, ql, qh)
        except Exception:
            continue
        c = 2 * sol.cost
        if np.isfinite(c):
            d = sol.x[0] if d_free is not None else None
            p = sol.x[1:] if d_free is not None else sol.x
            out.append((c, p, water, d, sum(Op.shape[0] for Op in ops) * 2))
    out.sort(key=lambda t: t[0])
    return out


T_LO, T_HI = 3e-4, 0.25


def _starts(B, n_prev_low, r_echo):
    st = []
    for er in (1.05, 2.0, 2.8, 4.5, 6.5):
        for rr in r_echo:
            t = rr / np.sqrt(er)
            if T_LO < t < T_HI:
                for erb in (1.0, 2.8, 6.0):
                    st.append((False, [er, 0.02, t, erb, 0.02], [1.0, 0.0, T_LO, 1.0, 0.0], [12.0, 0.5, T_HI, 12.0, 0.5]))
    for rr in r_echo:
        t = rr / 8.5
        if T_LO < t < 0.08:
            for erb in (1.0, 2.8, 6.0):
                st.append((True, [20.0, t, erb, 0.02], [0.0, T_LO, 1.0, 0.0], [40.0, 0.08, 12.0, 0.5]))
    if not st:
        for er in (2.8, 5.0):
            st.append((False, [er, 0.05, 0.05, 2.8, 0.05], [1.0, 0.0, T_LO, 1.0, 0.0], [12.0, 0.5, T_HI, 12.0, 0.5]))
    return st


def _model_block(p, f, n_prev):
    """Two dielectric layers (er1, td1, t1, er2, td2, t2) over a half-space (erb, tdb); batched."""
    er1, td1, t1, er2, td2, t2, erb, tdb = _cols(p)
    n1, n2, nb = n_of(er1, td1, f), n_of(er2, td2, f), n_of(erb, tdb, f)
    k0 = 2 * np.pi * f / C0
    G = (n2 - nb) / (n2 + nb)
    r2 = (n1 - n2) / (n1 + n2)
    e2 = np.exp(-2j * k0 * n2 * t2)
    G = (r2 + G * e2) / (1 + r2 * G * e2)
    r1 = (n_prev - n1) / (n_prev + n1)
    e1 = np.exp(-2j * k0 * n1 * t1)
    return (r1 + G * e1) / (1 + r1 * G * e1)


def _fit_block(B, n_prevs, echoes, d_free=None, top=4, keep=2):
    """Cascade: low band only (few phase cycles), then all bands from the best low-band blocks with
    +-2 high-band cycles on each thickness (GNSS-style wide-lane -> narrow-lane)."""
    lo = B.low()
    Bl = Bands([B.fs[lo]], [B.Gs[lo]], [B.ss[lo]])
    res_l = _fit_block1(Bl, [n_prevs[lo]], echoes, d_free, top=top, keep=3, full=True)
    if len(B.fs) == 1 or not res_l:
        return [(c, p[:5], w, d, nd) for c, p, w, d, nd in res_l[:keep]]
    fh = max(B.fs, key=lambda x: x[0])
    lam = C0 / fh.mean()
    seeds = []
    for c, p, w, d, nd in res_l:
        n1, n2 = np.sqrt(p[0]), np.sqrt(p[3])
        for k1 in (-2, -1, 0, 1, 2):
            for k2 in (-2, -1, 0, 1, 2):
                q = np.array(p, float)
                q[2] += k1 * lam / (2 * n1); q[5] += k2 * lam / (2 * n2)
                if q[2] > T_LO and q[5] > T_LO:
                    seeds.append((q, d))
    return _fit_block1(B, n_prevs, echoes, d_free, top=top, keep=keep, seeds=seeds)


def _fit_block1(B, n_prevs, echoes, d_free=None, top=4, keep=2, full=False, seeds=None):
    """Window continuation for the block: windows end just past each echo (ascending); use the
    widest window whose best fit is consistent with noise, else the best per-dof fit."""
    best_ok, best_any = None, None
    echo_wins = {round(e + 0.012, 6) for e in echoes[:3]}
    for win in sorted(set([0.03, 0.05, 0.08]) | echo_wins):
        ech = [e for e in echoes if e < win] or [win - 0.012]
        r = _fit_block_win(B, n_prevs, ech, win, d_free, top, keep, full, seeds)
        if not r:
            continue
        cd = r[0][0] / r[0][4]
        if (round(win, 6) in echo_wins or not echoes) and (best_any is None or cd < best_any[0]):
            best_any = (cd, r)
        if cd < 2.5:
            best_ok = (cd, r)
    pick = best_ok or best_any
    return [] if pick is None else pick[1]


def _fit_block_win(B, n_prevs, echoes, win, d_free=None, top=4, keep=2, full=False, seeds=None):
    """E041 block peeling: thin neighbours whose echoes merge are fitted together on ALL bands
    (the high band's phase resolves what the low band's time gate cannot); only the top layer
    is kept and peeled."""
    ops = [_gate(f, win) for f in B.fs]
    lo_b = [1, 0, T_LO, 1, 0, T_LO, 1, 0]
    hi_b = [12, .5, T_HI, 12, .5, T_HI, 12, .5]
    starts = []
    tots = [e for e in echoes[:3]] or [0.03]
    for tot in tots:
        for frac in (0.3, 0.5, 0.7):
            for e1 in (1.1, 2.0, 2.8, 4.5, 6.5):
                for e2 in (1.1, 2.8, 4.5, 6.5):
                    if e1 == e2:
                        continue
                    starts.append([e1, .02, tot * frac / np.sqrt(e1), e2, .02, tot * (1 - frac) / np.sqrt(e2), 2.8, .02])

    def res_fn(Q):
        Q = np.atleast_2d(Q)
        d, p = (Q[:, :1], Q[:, 1:]) if d_free is not None else (None, Q)
        rs = []
        for f, G, s, Op, npv in zip(B.fs, B.Gs, B.ss, ops, n_prevs):
            M = _model_block(p, f, npv)
            if d is not None:
                M = M * np.exp(-2j * 2 * np.pi * f / C0 * d)
            rr = ((M - G) / s) @ Op.T
            rs += [rr.real, rr.imag]
        return np.concatenate(rs, axis=1)
    pre = lambda p0, dd=None: ([d_free[0] if dd is None else dd] if d_free is not None else []) + list(p0)
    cand = [(p0, None) for p0 in starts] if seeds is None else seeds
    lo_q = np.array(([d_free[1]] if d_free is not None else []) + lo_b, float)
    hi_q = np.array(([d_free[2]] if d_free is not None else []) + hi_b, float)
    Q0 = np.clip(np.array([pre(p0, dd) for p0, dd in cand], float), lo_q, hi_q)
    cs = np.sum(res_fn(Q0) ** 2, axis=1) if len(Q0) else np.array([])
    sc = sorted([(c, q) for c, q in zip(cs, Q0) if np.isfinite(c)], key=lambda t: t[0])
    out = []
    ql = np.array(([d_free[1]] if d_free is not None else []) + lo_b, float)
    qh = np.array(([d_free[2]] if d_free is not None else []) + hi_b, float)
    for _, q0 in sc[:top]:
        q0 = np.clip(np.array(q0, float), ql + 1e-9, qh - 1e-9)
        try:
            sol = _lsq(res_fn, q0, ql, qh)
        except Exception:
            continue
        d = sol.x[0] if d_free is not None else None
        p = sol.x[1:] if d_free is not None else sol.x
        ndof = sum(2 * Op.shape[0] for Op in ops)
        # keep the TOP layer only, in the half-space format (er, td, t, erb, tdb)
        out.append((2 * sol.cost, np.array(p) if full else np.array([p[0], p[1], p[2], p[3], p[4]]), False, d, ndof))
    out.sort(key=lambda t: t[0])
    return out[:keep]


def _fit_layer(B, n_prevs, sig_low, d_free=None, keep=2):
    """Window continuation: candidate windows end just past each echo (ascending range); the fit
    of the widest window that is still consistent with noise (chi2/dof < 2.5) is used. Only the
    lowest band is used while peeling (fewest cycles); all bands return in the final exact fit."""
    lo = B.low()
    Bl = Bands([B.fs[lo]], [B.Gs[lo]], [B.ss[lo]])
    npl = [n_prevs[lo]]
    echoes = sorted(_echo_ranges(Bl.fs[0], Bl.Gs[0], sig_low, k=5)) or [0.02]
    best_ok, best_any = None, None
    echo_wins = {round(e + 0.012, 6) for e in echoes}
    for win in sorted(set([0.03, 0.05, 0.08]) | echo_wins):
        ech = [e for e in echoes if e < win] or [win - 0.012]
        res = _fit_step(Bl, npl, _starts(Bl, npl[0], ech), win, d_free)
        if not res:
            continue
        cd = res[0][0] / res[0][4]
        # fixed short windows only count when consistent: per-dof cost favours windows that see little
        if round(win, 6) in echo_wins and (best_any is None or cd < best_any[0]):
            best_any = (cd, res)
        if cd < 2.5:
            best_ok = (cd, res)                # widest consistent window wins
    if best_ok is None:                        # no single-layer window is consistent: thin neighbours
        blk = _fit_block(B, n_prevs, echoes, d_free, keep=keep)
        if blk and (best_any is None or blk[0][0] / blk[0][4] < best_any[0]):
            return blk
    pick = best_ok or best_any
    return [] if pick is None else pick[1][:keep]


def _peel(B, n_prevs, layer):
    er, td, t, water, T = layer
    fs, Gs, ss = [], [], []
    for f, G, s, npv in zip(B.fs, B.Gs, B.ss, n_prevs):
        n = n_of(er, td, f, water, T)
        r = (npv - n) / (npv + n)
        e = np.exp(-2j * 2 * np.pi * f / C0 * n * t)
        den = e * (1 - r * G)
        Gn = (G - r) / den
        sn = s * np.abs(1 - r ** 2) / np.abs(den * (1 - r * G))
        if np.median(sn) < 0.3:               # band still informative at this depth
            fs.append(f); Gs.append(Gn); ss.append(sn)
    return Bands(fs, Gs, ss), [n_of(er, td, f, water, T) for f in fs] if fs else []


def _remainder(B):
    """chi2/dof of the remainder against 'nothing' (air) and against a mirror (metal), gated to 0.6 m."""
    ca = cm = n = 0.0
    for f, G, s in zip(B.fs, B.Gs, B.ss):
        Op = _gate(f, 0.6)
        a = Op @ (G / s)
        m = Op @ ((G + 1) / s)
        ca += np.sum(np.abs(a) ** 2); cm += np.sum(np.abs(m) ** 2); n += 2 * Op.shape[0]
    return ca / max(n, 1), cm / max(n, 1)


def _merge_prune(model, cost, f, y, sigma, n_all):
    """Merge identical neighbours (same type, eps within 3 %), then drop/merge/cut-tail any layer
    whose removal raises chi2 by less than kappa = 4 ln n (Schwarz)."""
    kappa = 4 * np.log(n_all)
    lay = ST.describe(model)
    i = 0
    while i + 1 < len(lay):
        a, b = lay[i], lay[i + 1]
        if a["water"] == b["water"] and abs(a["er"] - b["er"]) <= 0.03 * max(a["er"], b["er"]):
            mm = ST.rebuild(model, layers=lay[:i] + [dict(a, t=a["t"] + b["t"])] + lay[i + 2:])
            for Lm, Lo in zip(mm.layers, model.layers[:i + 1] + model.layers[i + 2:]):
                Lm.pinf.value, Lm.pinf.free = Lo.pinf.value, Lo.pinf.free
            mm, cm = ST._fit(mm, f, y, sigma)
            if cm - cost < kappa:
                model, cost, lay = mm, cm, ST.describe(mm)
                continue
        i += 1
    return model, cost


def _release_dispersion(model, cost, f, y, sigma, n_all):
    """Last stage: free each dielectric layer's Jonscher shape p (starts 0.02 and 0.95, inside the
    box), keep the better fit, then merge/prune again (a dispersive layer that was imitated by
    several pieces can now be one layer)."""
    best = (model, cost)
    for p0 in (0.02, 0.95):
        mm = model.copy()
        for L in mm.layers:
            if L.theta.value < 0.5:
                L.pinf.value, L.pinf.free = p0, True
        mm, cm = ST._fit(mm, f, y, sigma)
        if cm < best[1]:
            best = (mm, cm)
    return _merge_prune(*best, f, y, sigma, n_all)


def peel_invert(f, y, sigma, nmax=12, beam=3, keep=2, verbose=False):
    """Returns (model, chi2, info). y = Gamma e^{-2j k0 d} + noise (calibrated plane-wave data)."""
    t0 = time.time()
    d0 = ST.front_guess(f, y)
    segs = _segs(f)
    fs = [f[a:b] for a, b in segs]
    k0s = [2 * np.pi * x / C0 for x in fs]
    ys = [y[a:b] for a, b in segs]
    # hypotheses: (score, layers, Bands, n_prevs, d, back or None)
    hyps = []
    # ---- depth 1 (with the standoff)
    B0 = Bands(fs, [yy * np.exp(2j * k * d0) for yy, k in zip(ys, k0s)], [np.full(x.size, sigma, float) for x in fs])
    ones = [np.ones(x.size, complex) for x in fs]
    res = _fit_layer(B0, ones, sigma, d_free=(0.0, -0.03, 0.01), keep=keep)
    for c, p, water, dd, ndof in res[:keep]:
        lay = (0, 0, p[1], True, p[0]) if water else (p[0], p[1], p[2], False, 20.0)
        Bd = Bands(fs, [yy * np.exp(2j * k * (d0 + dd)) for yy, k in zip(ys, k0s)], B0.ss)
        Bn, nps = _peel(Bd, ones, lay)
        hyps.append((c / ndof, [lay], Bn, nps, d0 + dd, None))
    done = []
    # ---- deeper
    for depth in range(1, nmax + 1):
        new = []
        for score, lays, Bc, nps, d, _ in hyps:
            if not Bc.fs:
                done.append((score, lays, d, "air"));
                continue
            ca, cm = _remainder(Bc)
            if ca < 1.3 or depth == nmax:
                done.append((score, lays, d, "air" if ca <= cm else "metal")); continue
            if cm < 1.3:
                done.append((score, lays, d, "metal")); continue
            lo = Bc.low()
            res = _fit_layer(Bc, nps, np.median(Bc.ss[lo]), keep=keep)
            for c, p, water, _, ndof in res[:keep]:
                lay = (0, 0, p[1], True, p[0]) if water else (p[0], p[1], p[2], False, 20.0)
                Bn, npn = _peel(Bc, nps, lay)
                new.append((score + c / ndof, lays + [lay], Bn, npn, d, None))
        if not new:
            break
        new.sort(key=lambda h: h[0] / len(h[1]))
        hyps = new[:beam]
    done += [(h[0], h[1], h[4], "air") for h in hyps if h[1] not in [x[1] for x in done]]
    # ---- exact joint polish of the best few complete hypotheses (DEC forward, LM), pick by BIC
    done.sort(key=lambda h: h[0] / len(h[1]))
    best = None
    n_all = 2 * f.size
    for score, lays, d, back in done[:beam]:
        L = [dict(t=l[2], er=l[0] if not l[3] else 1.0, td=l[1] if not l[3] else 0.0, water=l[3]) for l in lays]
        Tw = next((l[4] for l in lays if l[3]), 20.0)
        ST.SBOX = (d - 0.01, d + 0.01)
        m = ST.build(L, d, back=back, water_T=Tw)
        ST.set_dispersion_free(m, True)
        for bk in (back, "metal" if back == "air" else "air"):
          for p0 in (0.0,):
            mm = ST.rebuild(m, back=bk) if bk != back else m.copy()
            ST.set_dispersion_free(mm, True)      # p = 0 sits on its logit bound: constant-Q shape (geometry first)
            mo, c = ST._fit(mm, f, y, sigma)
            bic = c + 4 * len(lays) * np.log(n_all)
            if best is None or bic < best[2]:
                best = (mo, c, bic)
    model, cost, _ = best
    model = ST._drop_trailing_air(ST._drop_leading_air(model))
    ST.set_dispersion_free(model, True)
    model, cost = ST._fit(model, f, y, sigma)
    # identical neighbours (same type, eps within 3 %) are one physical layer: merge, keep if the exact
    # refit still fits (chi2 rise < kappa)
    lay = ST.describe(model)
    i = 0
    while i + 1 < len(lay):
        a, b = lay[i], lay[i + 1]
        if a["water"] == b["water"] and abs(a["er"] - b["er"]) <= 0.03 * max(a["er"], b["er"]):
            merged = lay[:i] + [dict(a, t=a["t"] + b["t"])] + lay[i + 2:]
            mm = ST.rebuild(model, layers=merged)
            ST.set_dispersion_free(mm, True)
            mm, cm = ST._fit(mm, f, y, sigma)
            if cm - cost < 4 * np.log(n_all):
                model, cost, lay = mm, cm, ST.describe(mm)
                continue
        i += 1
    # exact pruning (Schwarz/BIC): drop or merge any layer whose removal costs < kappa
    kappa = 4 * np.log(n_all)
    changed = True
    while changed and len(model.layers) > 1:
        changed = False
        lay = ST.describe(model)
        trials = []
        for i in range(len(lay)):
            trials.append(lay[:i] + lay[i + 1:])                                   # drop
            if i + 1 < len(lay):
                trials.append(lay[:i] + [dict(lay[i + 1], t=lay[i]["t"] + lay[i + 1]["t"])] + lay[i + 2:])  # merge
        tails = [(lay[:k], bk) for k in range(1, len(lay)) for bk in ("air", "metal")]   # cut the tail, either backing
        best_t = None
        for L2 in trials + tails:
            bk = None
            if isinstance(L2, tuple):
                L2, bk = L2
            mm = ST.rebuild(model, layers=L2, back=bk)
            ST.set_dispersion_free(mm, True)
            mm, cm = ST._fit(mm, f, y, sigma, atol=1e-2)
            if best_t is None or cm < best_t[1]:
                best_t = (mm, cm)
        if best_t is not None:
            mm, cm = ST._fit(best_t[0], f, y, sigma)
            if cm - cost < kappa:
                model, cost, changed = mm, cm, True
    model, cost = _release_dispersion(model, cost, f, y, sigma, n_all)
    info = dict(wall_s=time.time() - t0, chi2=cost, alarm=bool(cost > ST.chi2_alarm(n_all)),
                n_hyp=len(done))
    info["classes"] = ST.classify(model, f, sigma)
    return model, cost, info
