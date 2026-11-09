"""
1D stability theory for the indefinite frequency-domain DEC system (docs/11, Part A).

Admittance normalization, identical to src/oned/wall_inverse.py:
  A = sum_c [ b_c (D0^T D0)_c + M_c I_c ] + G0 e_0 e_0^T + GN e_N e_N^T ,
  b_c = S_c = (1/jw) *mu^-1 on edge c,   M_c = jw *eps (half cell, lumped on each end node).
Each cell is the symmetric 2-port  Y_c = [[b+M, -b], [-b, b+M]]  (node voltages -> currents
into the cell).  Its two modal admittances are
  Y_plus  = M          (even mode, eigenvector (1, 1)),
  Y_minus = M + 2 b    (odd mode,  eigenvector (1,-1)).
Everything here is plain numpy; the module does not touch src/oned.
"""
from __future__ import annotations
import numpy as np

C0 = 299_792_458.0
ETA0 = 376.730313668
MU0 = ETA0 / C0
EPS0 = 1 / (MU0 * C0**2)


def nbranch(eps):
    """Refractive index with Im(n) <= 0 (e^{+jwt}, decaying wave)."""
    n = np.sqrt(np.asarray(eps, complex))
    return np.where(n.imag > 0, -n, n)


# ----------------------------------------------------------------------------- stars
def std_stars(eps_r, ell, w, mu_r=1.0):
    """Standard (Whitney, lumped) stars. Returns (b, M) arrays."""
    eps_r = np.asarray(eps_r, complex); ell = np.asarray(ell, float)
    b = 1 / (1j * w * MU0 * mu_r * ell)
    M = 1j * w * EPS0 * eps_r * ell / 2
    return b, M


def exact_factors(x):
    """s_exact = x / sin x, m_exact = tan(x/2)/(x/2), x = k l (complex allowed)."""
    x = np.asarray(x, complex)
    return x / np.sin(x), np.tan(x / 2) / (x / 2)


def exact_stars(eps_r, ell, w, mu_r=1.0):
    eps_r = np.asarray(eps_r, complex); ell = np.asarray(ell, float)
    k = w / C0 * nbranch(eps_r * mu_r)
    s, m = exact_factors(k * ell)
    b, M = std_stars(eps_r, ell, w, mu_r)
    return b * s, M * m


def learned_stars(eps_r, ell, w, s_theta, m_theta, mu_r=1.0):
    """DDEC form of docs/02 sec. 7: S = S_std * s_theta, M = M_std * m_theta."""
    b, M = std_stars(eps_r, ell, w, mu_r)
    return b * np.asarray(s_theta, complex), M * np.asarray(m_theta, complex)


def modal(b, M):
    return M, M + 2 * b


# ----------------------------------------------------------------------------- assembly
def assemble(b, M, G0, GN):
    b = np.asarray(b, complex); M = np.asarray(M, complex)
    n = b.size + 1
    A = np.zeros((n, n), complex)
    i = np.arange(b.size)
    A[i, i] += b + M
    A[i + 1, i + 1] += b + M
    A[i, i + 1] -= b
    A[i + 1, i] -= b
    A[0, 0] += G0
    A[-1, -1] += GN
    return A


def wall_cells(layers, w, cells_per_layer=None, kind="exact", max_kl=0.9 * np.pi):
    """layers: list of (thickness, eps_r complex). Returns per-cell (eps_r, ell)."""
    eps, ell = [], []
    for j, (t, e) in enumerate(layers):
        if cells_per_layer is None:
            k = w / C0 * abs(nbranch(e))
            n = max(1, int(np.ceil(k * t / max_kl)))
        else:
            n = cells_per_layer[j] if np.ndim(cells_per_layer) else int(cells_per_layer)
        eps += [e] * n
        ell += [t / n] * n
    return np.array(eps, complex), np.array(ell)


def node_positions(ell):
    return np.concatenate([[0.0], np.cumsum(ell)])


# ----------------------------------------------------------------------------- transfer matrices
def transfer(b, M):
    """T_c with (e_R, h_R) = T_c (e_L, h_L);  h = current flowing in +z along the chain.
    h_L = current entering the cell at its left node, h_R = current leaving at its right node.
    det T_c = 1 (reciprocity)."""
    a = b + M
    return np.array([[a / b, -1 / b], [b - a * a / b, a / b]])


def normalizer(eta):
    """N(eta) maps (e, h) to (e / sqrt(eta), h sqrt(eta)); W = |e|^2/eta + eta |h|^2 = |N v|^2."""
    return np.diag([1 / np.sqrt(eta), np.sqrt(eta)])


def eta_eff(b, M):
    """Effective line impedance of a cell: eta_eff^2 = 1 / (Y_plus Y_minus).
    Real positive iff the cell is lossless and 'passband' (Im Y+ Im Y- < 0)."""
    Yp, Ym = modal(b, M)
    e = 1 / np.sqrt(Yp * Ym + 0j)
    return np.where(e.real < 0, -e, e)


def stability_constant(b, M, G0, GN, eta_ref=None):
    """Theorem A3 bound.  Returns (P, eta_r, cG) with
         |(A^{-1})_{ij}| <= 2 sqrt(eta_r[i] eta_r[j]) * P * cG
    where P = prod_c ||N(eta_r[c+1]) T_c N(eta_r[c])^{-1}||_2^2, eta_r[0] = 1/|G0|,
    eta_r[N] = 1/|GN|, and cG = max(|G|/Re G) over the two ends.
    Any positive interior reference impedances are allowed (the bound holds for all);
    the default is the geometric mean of |eta_eff| of the two neighbouring cells."""
    b = np.asarray(b, complex); M = np.asarray(M, complex)
    Nc = b.size
    if eta_ref is None:
        ee = np.abs(eta_eff(b, M))
        eta_r = np.empty(Nc + 1)
        eta_r[1:-1] = np.sqrt(ee[:-1] * ee[1:])
    else:
        eta_r = np.array(eta_ref, float)
    eta_r[0] = 1 / abs(G0)
    eta_r[-1] = 1 / abs(GN)
    P = 1.0
    for c in range(Nc):
        Tn = normalizer(eta_r[c + 1]) @ transfer(b[c], M[c]) @ np.linalg.inv(normalizer(eta_r[c]))
        P *= np.linalg.norm(Tn, 2) ** 2
    cG = max(abs(G0) / G0.real if np.iscomplexobj(G0) else 1.0, abs(GN) / np.real(GN))
    return P, eta_r, cG


def passivity_margin(b, M):
    """min over cells of Re Y_plus, Re Y_minus  (>= 0  <=>  every cell passive)."""
    Yp, Ym = modal(b, M)
    return min(Yp.real.min(), Ym.real.min())


def green_bound_ratio(b, M, G0, GN):
    """max_ij |A^-1_ij| / (2 sqrt(eta_i eta_j) P cG): Theorem A3 says <= 1."""
    A = assemble(b, M, G0, GN)
    Ai = np.linalg.inv(A)
    P, eta_r, cG = stability_constant(b, M, G0, GN)
    den = 2 * np.sqrt(np.outer(eta_r, eta_r)) * P * cG
    return np.abs(Ai / den).max(), P, Ai


# ----------------------------------------------------------------------------- continuous-bound pieces
def interface_product(etas):
    """prod of max(eta_i/eta_{i+1}, eta_{i+1}/eta_i) over consecutive real impedances."""
    e = np.asarray(etas, float)
    r = e[1:] / e[:-1]
    return float(np.prod(np.maximum(r, 1 / r)))


def random_wall(rng, lossless=False, nlayers=None):
    nl = nlayers or rng.integers(1, 7)
    layers = []
    for _ in range(nl):
        t = rng.uniform(2e-3, 60e-3)
        er = rng.uniform(1.5, 9.0)
        td = 0.0 if lossless else rng.choice([0.0, rng.uniform(0, 0.3)])
        layers.append((t, er * (1 - 1j * td)))
    return layers
