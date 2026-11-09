"""
E042: global blocky inversion for many-layer walls (replaces the per-depth searches of E034/E041).

Why: per-depth decisions (greedy growth, layer peeling) on band-limited, lossy data with strong
multiples (water, glass, tile) are brittle: one wrong early choice corrupts everything below
(E041 held-out: 1/8). The information is there (CRB 0.01-0.3 mm), so all depths are solved at once.

Model: K thin slices of physical thickness DZ from z0 (3 cm before the front-face guess) to the
deepest echo, behind them air. Slice k has three unknowns, each in a coordinate that keeps it
physical:
    eps'_k  = 1 + exp(a_k)            (never below air)
    tan d_k = exp(b_k)                (passive)
    theta_k = sigmoid(c_k)            (liquid-water volume fraction, CRIM mixing)
host h = eps' (1 - j tan d) (f/f_ref)^(-alpha), alpha = (2/pi) atan(tan d)   (causal constant-Q)
eps = ((1 - theta) sqrt(h) + theta sqrt(eps_water(f, 20 C)))^2              (water: LHM 1991)

Forward: exact reflection recursion over slices (same physics as the exact-star DEC / tmm.py),
reverse-mode Jacobian (one backward sweep).
Objective: chi2 + lambda * sum_k sum_p sqrt((m_{p,k+1} - m_{p,k})^2 + eps^2)   (smoothed total
variation: blocky profiles with sharp interfaces; Rudin-Osher-Fatemi / blocky geophysical
inversion), minimised by Gauss-Newton with lagged-diffusivity reweighting (IRLS) and LM damping.
lambda is lowered geometrically until chi2 reaches the noise level (Morozov discrepancy).
Frequency continuation: 3.1-5 GHz, then the whole low band, then all bands (Bunks et al. 1995).
Then: change-point segmentation -> discrete layers -> exact layered polish + BIC pruning
(stack.py / peel.py machinery), so the reported answer is the exact optimum of a discrete model.
"""
from __future__ import annotations
import time
import numpy as np
import wall_toy as W
import stack as ST

C0 = W.C0
F_REF = ST.F_REF
DZ = 2.0e-3
A_LO, A_HI = np.log(1e-3), np.log(80.0)
B_LO, B_HI = np.log(1e-5), np.log(1.0)
C_LO, C_HI = -9.0, 9.0
TV_EPS = 0.05


def _sig(c):
    return 1 / (1 + np.exp(-c))


def forward(m, f, z0, dz=DZ, back_eps=1.0 + 0j, jacobian=True, water_T=20.0):
    """m: (3, K) [a, b, c]. Returns y (F,) and J (F, 3K) (columns ordered a_0..a_K-1, b_..., c_...)."""
    a, b, c = m
    K = a.size
    f = np.asarray(f, float)
    k0 = 2 * np.pi * f / C0
    er = 1 + np.exp(a)[:, None]                          # (K, 1)
    td = np.exp(b)[:, None]
    th = _sig(c)[:, None]
    alpha = 2 / np.pi * np.arctan(td)
    lg = -np.log(f / F_REF)[None, :]
    g = np.exp(alpha * lg)                               # (K, F)
    h = er * (1 - 1j * td) * g
    w = W.eps_water_lhm(f, water_T)[None, :]
    sh, sw = np.sqrt(h), np.sqrt(w)
    s = (1 - th) * sh + th * sw
    n = s.copy()
    flip = n.imag > 0                                    # decaying branch for e^{+jwt}
    n[flip] = -n[flip]
    sgn = np.where(flip, -1.0, 1.0)
    nb = np.sqrt(back_eps + 0j)
    n_all = np.vstack([np.ones((1, f.size), complex), n])   # n_all[0] = air in front
    # recursion from the back
    r = (n_all[:-1] - n_all[1:]) / (n_all[:-1] + n_all[1:])  # r[k]: top of slice k   (K, F)
    rB = (n[-1] - nb) / (n[-1] + nb)
    e = np.exp(-2j * k0[None, :] * n * dz)
    G = np.empty((K + 1, f.size), complex)
    G[K] = rB
    for k in range(K - 1, -1, -1):
        G[k] = (r[k] + G[k + 1] * e[k]) / (1 + r[k] * G[k + 1] * e[k])
    ph = np.exp(-2j * k0 * z0)
    y = G[0] * ph
    if not jacobian:
        return y, None
    # backward sweep: a_k = dy/dG_k
    dr = np.empty((K, f.size), complex)
    de = np.empty((K, f.size), complex)
    av = ph.copy()
    for k in range(K):
        D = 1 + r[k] * G[k + 1] * e[k]
        dr[k] = av * (1 - (G[k + 1] * e[k]) ** 2) / D ** 2
        de[k] = av * G[k + 1] * (1 - r[k] ** 2) / D ** 2
        av = av * e[k] * (1 - r[k] ** 2) / D ** 2
    drB = av
    # dy/dn_k: n_k is the lower medium of r[k], the upper medium of r[k+1] (or rB), and in e[k]
    dn = dr * (-2 * n_all[:-1] / (n_all[:-1] + n_all[1:]) ** 2) + de * (-2j * k0[None, :] * dz * e)
    up = np.empty_like(dn)
    up[:-1] = dr[1:] * (2 * n_all[2:] / (n_all[1:-1] + n_all[2:]) ** 2)
    up[-1] = drB * (2 * nb / (n[-1] + nb) ** 2)
    dn = dn + up
    ds = dn * sgn                                        # n = sgn * s
    # chain to (a, b, c)
    dh_der = h / er
    dalpha = 2 / np.pi / (1 + td ** 2)
    dh_dtd = er * (-1j * g + (1 - 1j * td) * g * lg * dalpha)
    ds_dh = (1 - th) / (2 * sh)
    Ja = ds * ds_dh * dh_der * np.exp(a)[:, None]
    Jb = ds * ds_dh * dh_dtd * td
    Jc = ds * (sw - sh) * th * (1 - th)
    J = np.hstack([Ja.T, Jb.T, Jc.T])                    # (F, 3K)
    return y, J


def _tv(m, eps=TV_EPS):
    d = np.diff(m, axis=1)
    return np.sqrt(d ** 2 + eps ** 2)


def _gn(m, f, y, sigma, z0, lam, wts, iters=12, back_eps=1.0 + 0j):
    """Gauss-Newton / IRLS with LM damping on chi2 + lam * sum wts_p TV(m_p)."""
    K = m.shape[1]
    Dm = np.diff(np.eye(K), axis=0)                      # (K-1, K)
    lo = np.concatenate([np.full(K, A_LO), np.full(K, B_LO), np.full(K, C_LO)])
    hi = np.concatenate([np.full(K, A_HI), np.full(K, B_HI), np.full(K, C_HI)])

    def obj(mm):
        yy, _ = forward(mm, f, z0, back_eps=back_eps, jacobian=False)
        rr = (yy - y) / sigma
        chi = float(np.sum(rr.real ** 2 + rr.imag ** 2))
        return chi + lam * float(np.sum(wts[:, None] * _tv(mm))), chi

    phi, chi = obj(m)
    mu = 1e-2
    for _ in range(iters):
        yy, Jc = forward(m, f, z0, back_eps=back_eps)
        rr = (yy - y) / sigma
        rv = np.concatenate([rr.real, rr.imag])
        J = np.vstack([Jc.real, Jc.imag]) / sigma
        A = J.T @ J
        g = J.T @ rv
        beta = 1 / _tv(m)                                # lagged diffusivity (IRLS)
        Rm = np.zeros_like(A)
        gm = np.zeros(3 * K)
        for p in range(3):
            Bp = Dm.T @ (beta[p][:, None] * Dm) * wts[p]
            sl = slice(p * K, (p + 1) * K)
            Rm[sl, sl] = Bp
            gm[sl] = Bp @ m[p]
        H = A + lam * Rm
        gt = g + lam * gm
        ok = False
        for _ in range(8):
            try:
                step = np.linalg.solve(H + mu * np.diag(np.diag(H) + 1e-9), -gt)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            mn = np.clip(m.reshape(-1) + step, lo, hi).reshape(3, K)
            phn, chin = obj(mn)
            if np.isfinite(phn) and phn < phi:
                m, phi, chi, ok = mn, phn, chin, True
                mu = max(mu / 3, 1e-6)
                break
            mu *= 5
        if not ok:
            break
    return m, chi


def invert_profile(f, y, sigma, d0=None, dz=DZ, verbose=False, back_eps=1.0 + 0j):
    """Continuous blocky profile. Returns (m, z0, dz, info)."""
    t0 = time.time()
    if d0 is None:
        d0 = ST.front_guess(f, y)
    z0 = d0 - 0.03
    lowsel = f < 20e9
    fl = f[lowsel]
    r, P = W.range_profile(fl, y[lowsel])
    noise = np.median(P[(r > 2.0) & (r < 4.5)])
    sig = (r > d0) & (r < d0 + 0.7) & (P > 5 * noise)
    r_last = r[sig].max() if np.any(sig) else d0 + 0.05
    depth = min(max(r_last - z0 + 0.03, 0.05), 0.35)
    K = int(np.ceil(depth / dz))
    m = np.vstack([np.full(K, np.log(1.0)), np.full(K, np.log(0.02)), np.full(K, -6.0)])
    wts = np.array([1.0, 0.3, 1.0])
    stages = [f < 5.0e9, f < 11e9, np.ones(f.size, bool)]
    log = []
    for si, sel in enumerate(stages):
        fs, ys = f[sel], y[sel]
        n = 2 * fs.size
        lam = 30.0 * n / K
        for _ in range(8):
            m, chi = _gn(m, fs, ys, sigma, z0, lam, wts, back_eps=back_eps)
            log.append((si, lam, chi / n))
            if verbose:
                print(f"  stage {si} lam {lam:.3g} chi2/n {chi / n:.2f}")
            if chi / n < 1.3:
                break
            lam /= 4
    return m, z0, dz, dict(log=log, wall_s=time.time() - t0, K=K)


def segment(m, z0, dz, tol=(0.25, 1.0, 1.5)):
    """Change points where any channel jumps by more than tol (a, b, c units); returns
    [(z_top, thickness, eps', tan d, theta)] with leading/trailing air removed."""
    K = m.shape[1]
    cuts = [0]
    for k in range(1, K):
        if any(abs(m[p, k] - m[p, cuts[-1]:k].mean()) > tol[p] for p in range(3)):
            cuts.append(k)
    cuts.append(K)
    segs = []
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        mm = m[:, a_:b_].mean(axis=1)
        segs.append([z0 + a_ * dz, (b_ - a_) * dz, 1 + np.exp(mm[0]), np.exp(mm[1]), _sig(mm[2])])
    air = lambda s_: s_[2] < 1.04 and s_[4] < 0.2
    while segs and air(segs[0]):
        segs.pop(0)
    while segs and air(segs[-1]):
        segs.pop()
    return segs


def invert(f, y, sigma, verbose=False):
    """Profile -> segments -> exact layered model (stack.py) -> polish, back check, BIC pruning."""
    import peel as PL
    t0 = time.time()
    m, z0, dz, info = invert_profile(f, y, sigma, verbose=verbose)
    segs = segment(m, z0, dz)
    if not segs:
        segs = [[z0 + 0.03, 0.01, 2.5, 0.02, 0.0]]
    d = segs[0][0]
    lay = []
    for z, t, er, td, th in segs:
        if th > 0.6:
            lay.append(dict(t=t, er=1.0, td=0.0, water=True))
        elif th > 0.05:
            lay.append(dict(t=t, er=max(er, 1.05), td=td, water=False, moist=True, theta=th))
        else:
            lay.append(dict(t=t, er=max(er, 1.0001), td=min(td, 0.49), water=False))
    ST.SBOX = (d - 0.006, d + 0.006)
    n_all = 2 * f.size
    best = None
    for back in ("air", "metal"):
        mod = ST.build(lay, d, back=back)
        mod, c = ST._fit(mod, f, y, sigma)
        if best is None or c < best[1]:
            best = (mod, c)
    model, cost = best
    model, cost = PL._release_dispersion(model, cost, f, y, sigma, n_all)
    info.update(wall_s=time.time() - t0, chi2=cost, alarm=bool(cost > ST.chi2_alarm(n_all)),
                n_segments=len(segs), segments=segs)
    info["classes"] = ST.classify(model, f, sigma)
    return model, cost, info
