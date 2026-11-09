"""
E023: antenna model and multi-distance metal-plate calibration (after Lambot et al. 2004).

Measurement model per band (realism.py generates data the same way):
    y(f) = a(f) + b(f) S / (1 - c(f) S),     S = beam-received signal of the target (wall_inverse
                                             forward_beam), a = A H_i (leakage), b = A (response),
                                             c = H_f (antenna-target multiple reflections).
For a metal plate (Gamma = -1 for TE and TM) at distance d_p, S_p(f; d_p, w0) is known exactly given
the beam waist w0. Rearranging, y = a + (b - a c) S + c (y S): LINEAR in (a, b' = b - a c, c), so
per frequency a least-squares solve over N >= 3 plate distances. With N > 3 the residual depends
on w0, which is chosen by a 1D search. The wall's beam response then follows exactly:
    S_w = (y_w - a) / (b + c (y_w - a)).
"""
from __future__ import annotations
import numpy as np
import wall_inverse as I

C0 = I.C0


def plate_response(f, d, w0, K=10):
    """Beam-received signal of a PEC plate at distance d: S = -sum_q w_q exp(-2j k_z0 d)."""
    kr, wt = I.beam_nodes(f, w0, K, d_ref=d)
    k0 = 2 * np.pi * f[:, None] / C0
    kz0 = np.sqrt(k0**2 - kr**2 + 0j)
    kz0 = np.where(kz0.imag > 0, -kz0, kz0)
    return -np.sum(wt * np.exp(-2j * kz0 * d), axis=1)


def solve_abc(f, y_plates, d_plates, w0, with_cov=False):
    """Per-frequency least squares for (a, b', c); returns a, b, c, the total residual and (with
    with_cov) the per-frequency (A^H A)^-1, i.e. the noise covariance of (a, b', c) per unit sigma^2."""
    Sp = np.array([plate_response(f, d, w0) for d in d_plates])          # (N, F)
    Y = np.asarray(y_plates)                                             # (N, F)
    a = np.zeros(f.size, complex); bp = np.zeros(f.size, complex); c = np.zeros(f.size, complex)
    covs = np.zeros((f.size, 3, 3), complex)
    res = 0.0
    for m in range(f.size):
        Amat = np.stack([np.ones(len(d_plates)), Sp[:, m], Y[:, m] * Sp[:, m]], axis=1)
        x, *_ = np.linalg.lstsq(Amat, Y[:, m], rcond=None)
        r = Amat @ x - Y[:, m]
        res += float(np.sum(np.abs(r) ** 2))
        a[m], bp[m], c[m] = x
        if with_cov:
            covs[m] = np.linalg.inv(Amat.conj().T @ Amat)
    b = bp + a * c
    if with_cov:
        return a, b, c, res, bp, covs
    return a, b, c, res


def calibrate(f, y_wall, y_plates, d_plates, w0_grid=None):
    """Estimate w0 (1D search on the calibration residual), (a, b, c), and return the wall's
    calibrated beam response S_w plus the estimates."""
    if w0_grid is None:
        w0_grid = np.geomspace(0.004, 0.2, 60)
    res = np.array([solve_abc(f, y_plates, d_plates, w)[3] for w in w0_grid])
    i = int(np.argmin(res))
    lo, hi = w0_grid[max(i - 1, 0)], w0_grid[min(i + 1, len(w0_grid) - 1)]
    for _ in range(30):                                                  # golden-section refine
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if solve_abc(f, y_plates, d_plates, m1)[3] < solve_abc(f, y_plates, d_plates, m2)[3]:
            hi = m2
        else:
            lo = m1
    w0 = 0.5 * (lo + hi)
    a, b, c, r, bp, covs = solve_abc(f, y_plates, d_plates, w0, with_cov=True)
    # S_w = (y - a) / D with D = b' + c y  (exact rearrangement of (y - a)/(b + c (y - a)))
    D = bp + c * y_wall
    S_w = (y_wall - a) / D
    # noise propagation per unit raw sigma^2: wall-scan noise + noise of the estimated (a, b', c)
    g = np.stack([-1 / D, -S_w / D, -S_w * y_wall / D], axis=1)          # dS/d(a, b', c)
    var = np.abs(b) ** 2 / np.abs(D) ** 4 + np.real(np.einsum("fi,fij,fj->f", g.conj(), covs, g))
    return S_w, dict(w0=w0, a=a, b=b, c=c, residual=r, var_per_sigma2=var)
