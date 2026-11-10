"""
E049: echo-based initialisation for the known-N problem (wallsolve.py).

The calibrated response of a layered wall is, to first order, a sum of echoes, one per interface,
at optical depths tau_k (radar range of interface k):  y_b(f) ~ sum_k a_kb exp(-2j k0 tau_k).
Multiband sparse deconvolution (greedy pursuit on a fine tau grid with per-band complex amplitudes
and SHARED delays across bands, then continuous joint refinement of all delays - "newtonized" OMP)
finds the interfaces; super-resolution comes from the shared delays across the UWB and mmWave bands
(cf. Le Bastard et al. 2007, thin pavement layers).
From the echo signs and sizes, the impedance recursion r_k = (n_{k-1} - n_k)/(n_{k-1} + n_k),
corrected for two-way transmission through the layers above, gives each interval's index n_k;
intervals with n ~ 1 are air gaps. With N solid layers given, (#echoes - 1) - N intervals are gaps.
This is only an initialiser: the exact layered fit (multiples, loss, dispersion) decides.
"""
from __future__ import annotations
import numpy as np
import wall_inverse as I
import wall_toy as W

C0 = W.C0


def _bands(f, y, s, fmin=2e9):
    out = []
    for a, b in I.band_segments(f):
        if f[a] >= fmin:
            out.append((f[a:b], y[a:b], s[a:b]))
    return out


def _atoms(fb, taus):
    return np.exp(-2j * 2 * np.pi * fb[:, None] / C0 * np.asarray(taus)[None, :])


def _ls(bands, taus):
    """Per-band complex amplitudes for fixed delays; returns amplitudes and total chi2."""
    amps, chi = [], 0.0
    for fb, yb, sb in bands:
        A = _atoms(fb, taus) / sb[:, None]
        a, *_ = np.linalg.lstsq(A, yb / sb, rcond=None)
        r = yb / sb - A @ a
        amps.append(a)
        chi += float(np.vdot(r, r).real)
    return amps, chi


def _refine(bands, taus, iters=30):
    """Continuous joint refinement of all delays (Gauss-Newton on the variable-projection cost)."""
    taus = np.array(taus, float)
    amps, chi = _ls(bands, taus)
    for _ in range(iters):
        g = np.zeros(taus.size)
        H = np.zeros((taus.size, taus.size))
        for (fb, yb, sb), a in zip(bands, amps):
            k = 2 * 2 * np.pi * fb / C0
            A = _atoms(fb, taus)
            r = (yb - A @ a) / sb
            dA = (-1j * k[:, None]) * A * a[None, :] / sb[:, None]          # d model / d tau_k
            g += -2 * np.real(dA.conj().T @ r)
            H += 2 * np.real(dA.conj().T @ dA)
        try:
            step = -np.linalg.solve(H + 1e-9 * np.eye(taus.size) * np.trace(H), g)
        except np.linalg.LinAlgError:
            break
        lam = 1.0
        while lam > 1e-4:
            tn = taus + lam * step
            an, cn = _ls(bands, tn)
            if cn < chi:
                taus, amps, chi = tn, an, cn
                break
            lam /= 2
        if lam <= 1e-4 or abs(lam * step).max() < 1e-7:
            break
    order = np.argsort(taus)
    amps = [a[order] for a in amps]
    return taus[order], amps, chi


def find_echoes(f, y, s, d_guess, max_echoes=12, window=(-0.03, 0.8), dtau=2e-4, bic=None):
    bands = _bands(f, y, s)
    grid = np.arange(d_guess + window[0], d_guess + window[1], dtau)
    n_data = 2 * sum(b[0].size for b in bands)
    kappa = (1 + 2 * len(bands)) * np.log(n_data) if bic is None else bic
    taus, chi = [], sum(float(np.sum(np.abs(yb / sb) ** 2)) for _, yb, sb in bands)
    res = [yb.copy() for _, yb, _ in bands]
    hist = []
    for _ in range(max_echoes):
        score = np.zeros(grid.size)
        for (fb, yb, sb), rb in zip(bands, res):
            A = _atoms(fb, grid) / sb[:, None]
            c = A.conj().T @ (rb / sb)
            score += np.abs(c) ** 2 / np.sum(np.abs(A) ** 2, axis=0)
        cand = taus + [grid[int(np.argmax(score))]]
        tn, an, cn = _refine(bands, cand)
        if chi - cn < kappa:
            break
        taus, chi = list(tn), cn
        hist.append((len(taus), cn))
        res = [yb - _atoms(fb, tn) @ a for (fb, yb, sb), a in zip(bands, an)]
    if not taus:
        return np.array([]), [], chi
    tn, an, cn = _refine(bands, taus)
    return tn, an, cn


def layers_from_echoes(taus, amps, f, N, back_guess="air"):
    """Impedance recursion from the lowest band's amplitudes (least loss): index of each interval,
    then N solid layers + (M-1-N) gaps (the intervals closest to n = 1 become gaps).
    Returns standoff d, list of (kind, t, eps') with kind in {solid, gap}."""
    if len(taus) < 2:
        return None
    a = np.real(amps[0])                         # lowest band used: smallest loss and dispersion
    a = a / np.abs(a[0]) * np.abs(a[0])
    n_prev, T2 = 1.0, 1.0
    ns = []
    for k in range(len(taus) - 1):
        r = np.clip(a[k] / max(T2, 1e-3), -0.95, 0.95)
        n = n_prev * (1 - r) / (1 + r)
        n = float(np.clip(n, 1.0, 9.0))
        ns.append(n)
        T2 *= (1 - r ** 2)
        n_prev = n
    dt = np.diff(taus)
    M1 = len(dt)
    n_gaps = max(0, M1 - N)
    gap_idx = set(np.argsort(np.abs(np.array(ns) - 1.0))[:n_gaps]) if n_gaps else set()
    out = []
    for k in range(M1):
        kind = "gap" if k in gap_idx else "solid"
        n = 1.0 if kind == "gap" else max(ns[k], 1.02)
        out.append((kind, dt[k] / n, n ** 2))
    while sum(1 for o in out if o[0] == "solid") < N:     # merged echoes: split the thickest solid
        i = max((j for j, o in enumerate(out) if o[0] == "solid"), key=lambda j: out[j][1], default=None)
        if i is None:
            out.append(("solid", 0.01, 2.5)); continue
        kd, t, e = out[i]
        out = out[:i] + [(kd, t / 2, e), (kd, t / 2, e)] + out[i + 1:]
    return float(taus[0]), out
