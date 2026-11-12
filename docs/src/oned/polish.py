"""
E051: compiled batch polish (bounded Levenberg-Marquardt, finite-difference Jacobian) of many
candidate walls at once, used to RANK candidates during the structure search. The final answer is
always refitted by the exact adjoint LM (search.lm); this kernel only has to be a faithful, fast
local optimiser of the same cost:

    cost = sum_f |Gamma(f; x) - y(f)|^2 / sigma_f^2  +  class prior  +  sliver penalty

Layer kinds: 0 dry solid (causal Jonscher law with the class's dispersion shape pp fixed),
1 liquid water (double Debye, Liebe-Hufford-Manabe 1991, temperature fitted), 2 air (gap), 3 metal.
Free parameters: standoff d, every thickness (log), (log eps', log10 tan d) of every dry solid,
the water temperature if the wall has water.
Class prior (dry solids): squared Mahalanobis distance of (ln eps', log10 tan d) at 10 GHz to the
layer's class (matlib.LibPrior uses the nearest class; here the class is the candidate's).
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange

F_REF = 10e9
T_MIN, T_SIG = 0.8e-3, 0.3e-3
LMAX = 12


@njit(cache=True, fastmath=True)
def _nrow(kind, er, td, pp, Tw, f, out):
    F = f.shape[0]
    if kind == 0:
        ap = 2.0 / np.pi * np.arctan(td / (1.0 - pp))
        for i in range(F):
            g = np.exp(-ap * np.log(f[i] / F_REF))
            n = np.sqrt(er * (pp + (1.0 - pp - 1j * td) * g))
            if n.imag > 0:
                n = -n
            out[i] = n
    elif kind == 1:
        th = 300.0 / (273.15 + Tw)
        e0 = 77.66 + 103.3 * (th - 1.0)
        e1 = 0.0671 * e0
        e2 = 3.52
        g1 = 20.20 - 146.0 * (th - 1.0) + 316.0 * (th - 1.0) ** 2
        g2 = 39.8 * g1
        for i in range(F):
            fg = f[i] / 1e9
            n = np.sqrt((e0 - e1) / (1.0 + 1j * fg / g1) + (e1 - e2) / (1.0 + 1j * fg / g2) + e2)
            if n.imag > 0:
                n = -n
            out[i] = n
    elif kind == 2:
        for i in range(F):
            out[i] = 1.0 + 0.0j
    else:
        n = np.sqrt(1.0 - 1e8j)
        if n.imag > 0:
            n = -n
        for i in range(F):
            out[i] = n


@njit(cache=True, fastmath=True)
def _forward(kind, er, td, pp, t, nl, back, d, Tw, f, k0, nbuf, G):
    """Gamma(f) of one wall; nbuf (LMAX + 1, F) work array."""
    F = f.shape[0]
    for j in range(nl):
        _nrow(kind[j], er[j], td[j], pp[j], Tw, f, nbuf[j])
    _nrow(back, 1.0, 0.0, 0.0, Tw, f, nbuf[nl])
    for i in range(F):
        nb = nbuf[nl, i]
        nlast = nbuf[nl - 1, i] if nl > 0 else 1.0 + 0.0j
        g = (nlast - nb) / (nlast + nb)
        for j in range(nl - 1, -1, -1):
            nj = nbuf[j, i]
            npv = nbuf[j - 1, i] if j > 0 else 1.0 + 0.0j
            e = np.exp(-2j * k0[i] * nj * t[j])
            r = (npv - nj) / (npv + nj)
            ge = g * e
            g = (r + ge) / (1.0 + r * ge)
        G[i] = g * np.exp(-2j * k0[i] * d)


@njit(cache=True, fastmath=True)
def _unpack(x, kind, cls, nl, has_w, er, td, t):
    """x -> physical arrays; returns (d, Tw)."""
    d = x[0]
    k = 1
    for j in range(nl):
        t[j] = np.exp(x[k]); k += 1
        if kind[j] == 0:
            er[j] = np.exp(x[k]); td[j] = 10.0 ** x[k + 1]; k += 2
    Tw = x[k] if has_w else 20.0
    return d, Tw


@njit(cache=True, fastmath=True)
def _resid(x, kind, cls, pp, nl, back, has_w, f, k0, y, isig, mu, Li, w_prior, er, td, t, nbuf, G, r):
    d, Tw = _unpack(x, kind, cls, nl, has_w, er, td, t)
    _forward(kind, er, td, pp, t, nl, back, d, Tw, f, k0, nbuf, G)
    F = f.shape[0]
    chi = 0.0
    for i in range(F):
        a = (G[i].real - y[i].real) * isig[i]
        b = (G[i].imag - y[i].imag) * isig[i]
        r[2 * i] = a; r[2 * i + 1] = b
        chi += a * a + b * b
    m = 2 * F
    k = 1
    for j in range(nl):
        tj = t[j]
        r[m] = (T_MIN - tj) / T_SIG if tj < T_MIN else 0.0
        m += 1
        k += 1
        if kind[j] == 0:
            c = cls[j]
            a0 = x[k] - mu[c, 0]
            a1 = x[k + 1] - mu[c, 1]
            r[m] = w_prior * (Li[c, 0, 0] * a0 + Li[c, 1, 0] * a1)
            r[m + 1] = w_prior * (Li[c, 0, 1] * a0 + Li[c, 1, 1] * a1)
            m += 2
            k += 2
    return chi, m


@njit(cache=True, fastmath=True)
def _polish_one(x, lo, hi, npar, kind, cls, pp, nl, back, has_w, f, k0, y, isig, mu, Li, w_prior, iters):
    F = f.shape[0]
    R = 2 * F + 3 * LMAX
    er = np.ones(LMAX); td = np.zeros(LMAX); t = np.zeros(LMAX)
    nbuf = np.empty((LMAX + 1, F), np.complex128)
    G = np.empty(F, np.complex128)
    r = np.zeros(R); r2 = np.zeros(R)
    J = np.zeros((R, npar))
    chi, m = _resid(x, kind, cls, pp, nl, back, has_w, f, k0, y, isig, mu, Li, w_prior, er, td, t, nbuf, G, r)
    cost = 0.0
    for i in range(m):
        cost += r[i] * r[i]
    lam = 1e-2
    xt = x.copy()
    for it in range(iters):
        for p in range(npar):
            h = 1e-6 if p == 0 else (1e-2 if (has_w and p == npar - 1) else 1e-5)
            xp = x[p]
            x[p] = xp + h
            _resid(x, kind, cls, pp, nl, back, has_w, f, k0, y, isig, mu, Li, w_prior, er, td, t, nbuf, G, r2)
            x[p] = xp
            for i in range(m):
                J[i, p] = (r2[i] - r[i]) / h
        A = np.zeros((npar, npar)); g = np.zeros(npar)
        for i in range(m):
            for p in range(npar):
                g[p] += J[i, p] * r[i]
                for q in range(p, npar):
                    A[p, q] += J[i, p] * J[i, q]
        for p in range(npar):
            for q in range(p):
                A[p, q] = A[q, p]
        improved = False
        for tries in range(8):
            B = A.copy()
            for p in range(npar):
                B[p, p] += lam * A[p, p] + 1e-9
            dx = np.linalg.solve(B, -g)
            for p in range(npar):
                v = x[p] + dx[p]
                if v < lo[p]:
                    v = lo[p]
                if v > hi[p]:
                    v = hi[p]
                xt[p] = v
            chi_t, m = _resid(xt, kind, cls, pp, nl, back, has_w, f, k0, y, isig, mu, Li, w_prior, er, td, t, nbuf, G, r2)
            ct = 0.0
            for i in range(m):
                ct += r2[i] * r2[i]
            if ct < cost:
                rel = (cost - ct) / cost
                for p in range(npar):
                    x[p] = xt[p]
                for i in range(m):
                    r[i] = r2[i]
                cost = ct; chi = chi_t
                lam = max(lam / 4.0, 1e-9)
                improved = True
                break
            lam *= 5.0
        if (not improved) or rel < 1e-7:
            break
    return cost, chi


@njit(parallel=True, cache=True, fastmath=True)
def polish_batch(X, LO, HI, npar, kind, cls, pp, nl, back, has_w, f, k0, y, isig, mu, Li, w_prior, iters):
    """X (P, NP) packed parameters (in place); returns MAP cost and data chi2 per candidate."""
    P = X.shape[0]
    cost = np.empty(P); chi = np.empty(P)
    for p in prange(P):
        try_x = X[p, :npar[p]].copy()
        c, ch = _polish_one(try_x, LO[p, :npar[p]], HI[p, :npar[p]], npar[p], kind[p], cls[p], pp[p], nl[p],
                            back[p], has_w[p], f, k0, y, isig, mu, Li, w_prior, iters)
        X[p, :npar[p]] = try_x
        cost[p] = c; chi[p] = ch
    return cost, chi


@njit(cache=True, fastmath=True)
def gamma_one(kind, er, td, pp, t, nl, back, d, Tw, f, k0):
    nbuf = np.empty((LMAX + 1, f.shape[0]), np.complex128)
    G = np.empty(f.shape[0], np.complex128)
    _forward(kind, er, td, pp, t, nl, back, d, Tw, f, k0, nbuf, G)
    return G
