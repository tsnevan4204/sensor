"""
E051: compiled, batched, exact reflection recursion for CLASS-DISCRETE candidate walls.

A candidate is a sequence of up to LMAX layers; layer j has a class index c (row of a refractive
index table n[c, f], precomputed once per problem) and a physical thickness t. Behind the last
layer: a half-space of class `back` (air, metal, or any material class: partial walls during the
tree search end in a half-space of the next material).

  Gamma = G_0 exp(-2j k0 d),  G_i = (r_i + G_{i+1} e_i) / (1 + r_i G_{i+1} e_i),
  r_i = (n_{i-1} - n_i)/(n_{i-1} + n_i),  e_i = exp(-2j k0 n_i t_i)

Same mathematics as tmm.forward / stack.gamma_stack (gate: test_fastbatch, |diff| < 1e-12).
Kernels:
  gamma_batch(...)        -> Gamma (P, F)
  chi_batch(...)          -> sum_f |Gamma - y|^2 w_f                       (full misfit)
  proj_batch(...)         -> || Q (Gamma - y) ||^2  with Q (K, F) complex  (gated / projected misfit)
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange

LMAX = 12


@njit(cache=True, fastmath=True)
def _gamma_one(cls, t, nl, back, d, ntab, k0, out):
    F = k0.shape[0]
    for f in range(F):
        if nl == 0:
            nlast = 1.0 + 0.0j
        else:
            nlast = ntab[cls[nl - 1], f]
        nb = ntab[back, f]
        G = (nlast - nb) / (nlast + nb)
        for i in range(nl - 1, -1, -1):
            ni = ntab[cls[i], f]
            if i > 0:
                npv = ntab[cls[i - 1], f]
            else:
                npv = 1.0 + 0.0j
            e = np.exp(-2j * k0[f] * ni * t[i])
            r = (npv - ni) / (npv + ni)
            Ge = G * e
            G = (r + Ge) / (1.0 + r * Ge)
        out[f] = G * np.exp(-2j * k0[f] * d)


@njit(parallel=True, cache=True, fastmath=True)
def gamma_batch(cls, t, nl, back, d, ntab, k0):
    P = cls.shape[0]
    F = k0.shape[0]
    out = np.empty((P, F), np.complex128)
    for p in prange(P):
        _gamma_one(cls[p], t[p], nl[p], back[p], d[p], ntab, k0, out[p])
    return out


@njit(parallel=True, cache=True, fastmath=True)
def chi_batch(cls, t, nl, back, d, ntab, k0, y, w):
    P = cls.shape[0]
    F = k0.shape[0]
    out = np.empty(P)
    for p in prange(P):
        g = np.empty(F, np.complex128)
        _gamma_one(cls[p], t[p], nl[p], back[p], d[p], ntab, k0, g)
        s = 0.0
        for f in range(F):
            dr = g[f].real - y[f].real
            di = g[f].imag - y[f].imag
            s += (dr * dr + di * di) * w[f]
        out[p] = s
    return out


@njit(parallel=True, cache=True, fastmath=True)
def proj_batch(cls, t, nl, back, d, ntab, k0, Qy, Q):
    """|| Q Gamma - Qy ||^2 for each candidate; Q (K, F), Qy (K,) = Q y (precomputed)."""
    P = cls.shape[0]
    F = k0.shape[0]
    K = Q.shape[0]
    out = np.empty(P)
    for p in prange(P):
        g = np.empty(F, np.complex128)
        _gamma_one(cls[p], t[p], nl[p], back[p], d[p], ntab, k0, g)
        s = 0.0
        for k in range(K):
            z = -Qy[k]
            for f in range(F):
                z += Q[k, f] * g[f]
            s += z.real * z.real + z.imag * z.imag
        out[p] = s
    return out


@njit(parallel=True, cache=True, fastmath=True)
def joint_enum(seqs, backs, combos, h, d, nref, ntab, k0, y, w, keep):
    """Exhaustive scan: for every class sequence seqs[s] (N layers) and backing backs[b], the
    misfit of every interface-position combination combos[c] (N increasing grid indices; interface
    i sits at optical depth h * (combos[c, i] + 1) behind the front face at d).
    Layer thickness i = h * dq_i / nref[class], so its propagation factor is E_i^dq_i with
    E_i = exp(-2j k0 n_i h / nref_i): powers are tabulated once per sequence, and the inner loop
    has no transcendental function. Returns the `keep` best (chi2, combo index) per (s, b)."""
    S, N = seqs.shape
    B = backs.shape[0]
    Cn = combos.shape[0]
    F = k0.shape[0]
    Q = 0
    for c in range(Cn):
        if combos[c, N - 1] + 1 > Q:
            Q = combos[c, N - 1] + 1
    best_chi = np.full((S, B, keep), np.inf)
    best_idx = np.zeros((S, B, keep), np.int64)
    ph = np.empty(F, np.complex128)
    for f in range(F):
        ph[f] = np.exp(-2j * k0[f] * d)
    for s in prange(S):
        E = np.empty((N, Q + 1, F), np.complex128)
        r = np.empty((N, F), np.complex128)
        for i in range(N):
            ci = seqs[s, i]
            for f in range(F):
                ni = ntab[ci, f]
                npv = ntab[seqs[s, i - 1], f] if i > 0 else 1.0 + 0.0j
                r[i, f] = (npv - ni) / (npv + ni)
                base = np.exp(-2j * k0[f] * ni * h / nref[ci])
                E[i, 0, f] = 1.0
                for q in range(1, Q + 1):
                    E[i, q, f] = E[i, q - 1, f] * base
        for b in range(B):
            rb = np.empty(F, np.complex128)
            for f in range(F):
                nl_ = ntab[seqs[s, N - 1], f]
                nb = ntab[backs[b], f]
                rb[f] = (nl_ - nb) / (nl_ + nb)
            for c in range(Cn):
                chi = 0.0
                worst = best_chi[s, b, keep - 1]
                for f in range(F):
                    G = rb[f]
                    for i in range(N - 1, -1, -1):
                        dq = combos[c, i] - (combos[c, i - 1] if i > 0 else -1)
                        Ge = G * E[i, dq, f]
                        G = (r[i, f] + Ge) / (1.0 + r[i, f] * Ge)
                    G = G * ph[f]
                    dr = G.real - y[f].real
                    di = G.imag - y[f].imag
                    chi += (dr * dr + di * di) * w[f]
                    if chi > worst:
                        break
                if chi < worst:
                    j = keep - 1
                    while j > 0 and best_chi[s, b, j - 1] > chi:
                        best_chi[s, b, j] = best_chi[s, b, j - 1]
                        best_idx[s, b, j] = best_idx[s, b, j - 1]
                        j -= 1
                    best_chi[s, b, j] = chi
                    best_idx[s, b, j] = c
    return best_chi, best_idx
