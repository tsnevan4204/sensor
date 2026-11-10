"""
Why 2D needs approximate (learned) stars: local Trefftz spaces and cell Schur complements.

trefftz_rank_1d / trefftz_rank_2d
    Sample many local solutions (plane waves) of the homogeneous Helmholtz equation in one cell and
    stack [boundary DOF values ; exact boundary fluxes]. A linear exact star Y with flux = Y value
    exists on that family iff rank([values; fluxes]) == rank(values).
    1D (2 nodes): rank 2 == 2.   2D square cell with 4 vertex DOFs: rank 8 > 4  (no exact 4x4 star).
cell_matrix / schur
    Fine-mesh DEC matrix of one cell (natural boundary) and its Schur complement onto the cell's
    fine boundary nodes: the cell star that is exact relative to the fine mesh.
"""
import numpy as np
import scipy.sparse as sp
from dec2d import Grid, Problem


def trefftz_rank_1d(k, ell, ndir=50, tol=1e-10):
    # local solutions span{e^{-jkz}, e^{+jkz}}, sampled as random combinations
    rng = np.random.default_rng(0)
    c = rng.standard_normal((2, ndir)) + 1j * rng.standard_normal((2, ndir))
    val = np.array([[1, 1], [np.exp(-1j * k * ell), np.exp(1j * k * ell)]]) @ c
    # fluxes into the cell: -u'(0), +u'(ell)
    du0 = np.array([-1j * k, 1j * k]) @ c
    dul = np.array([-1j * k * np.exp(-1j * k * ell), 1j * k * np.exp(1j * k * ell)]) @ c
    flux = np.vstack([-du0, dul])
    sv = lambda X: np.linalg.svd(X, compute_uv=False)
    rk = lambda X: int(np.sum(sv(X) > tol * sv(X)[0]))
    return rk(val), rk(np.vstack([val, flux])), sv(np.vstack([val, flux]))


def trefftz_rank_2d(kH, ndir=400, nq=64, tol=1e-10):
    """Square cell [0,1]^2 (lengths in units of H, wavenumber kH). DOFs: 4 vertex values.
    Exact fluxes: q_v = integral of the outward normal derivative over the two half-edges at
    vertex v (the part of the cell boundary that belongs to v's dual cell)."""
    th = np.linspace(0, 2 * np.pi, ndir, endpoint=False)
    d = np.stack([np.cos(th), np.sin(th)])
    verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    val = np.exp(-1j * kH * (verts @ d))                     # (4, ndir)
    # Gauss-Legendre on [0, 1/2]
    xg, wg = np.polynomial.legendre.leggauss(nq)
    t = 0.25 * (xg + 1); w = 0.25 * wg
    flux = np.zeros((4, ndir), complex)
    # edges: bottom (y=0, n=(0,-1)), right (x=1, n=(1,0)), top (y=1,n=(0,1)), left (x=0,n=(-1,0))
    edges = [((0, 0), (1, 0), (0, -1)), ((1, 0), (1, 1), (1, 0)), ((1, 1), (0, 1), (0, 1)), ((0, 1), (0, 0), (-1, 0))]
    for e, (p, q, n) in enumerate(edges):
        p, q, n = np.array(p, float), np.array(q, float), np.array(n, float)
        dn = -1j * kH * (n @ d)                               # d_n of the plane wave, per direction
        for vtx, s in ((e, t), ((e + 1) % 4, 1 - t)):        # half of the edge next to each end vertex
            pts = p[None, :] + s[:, None] * (q - p)[None, :]
            pw = np.exp(-1j * kH * (pts @ d))                 # (nq, ndir)
            flux[vtx] += (w[:, None] * pw).sum(0) * dn
    sv = lambda X: np.linalg.svd(X, compute_uv=False)
    rk = lambda X: int(np.sum(sv(X) > tol * sv(X)[0]))
    return rk(val), rk(np.vstack([val, flux])), sv(np.vstack([val, flux]))


def cell_problem(nf, H=1.0, eps=None):
    g = Grid(nf, nf, H / nf)
    P = Problem(g, impedance=np.zeros(g.nn, bool))
    if eps is not None:
        P.eps_node = eps(P.X, P.Y) + 0j
    return P


def schur(A, bnd):
    """S = A_bb - A_bi A_ii^{-1} A_ib (dense)."""
    A = A.toarray() if sp.issparse(A) else A
    i = np.setdiff1d(np.arange(A.shape[0]), bnd)
    Abb, Abi, Aib, Aii = A[np.ix_(bnd, bnd)], A[np.ix_(bnd, i)], A[np.ix_(i, bnd)], A[np.ix_(i, i)]
    return Abb - Abi @ np.linalg.solve(Aii, Aib)


def boundary_nodes(P):
    return np.where(P.on_bnd)[0]


def interior_dirichlet_eigs(P, nev=3):
    """Smallest generalized eigenvalues k^2 of (K_ii, M_ii): poles of the cell Schur complement."""
    K = (P.D.T @ sp.diags(P.ratio) @ P.D).toarray()
    i = np.where(~P.on_bnd)[0]
    from scipy.linalg import eigh
    w = eigh(K[np.ix_(i, i)], np.diag(P.area[i] * P.eps_node[i].real), eigvals_only=True)
    return w[:nev]


def vertex_prolongation(P):
    """Bilinear-on-the-boundary (piecewise linear along each side) map from the 4 cell vertices
    to the fine boundary nodes; columns ordered (0,0), (H,0), (H,H), (0,H)."""
    g = P.grid
    H = g.nx * g.h
    bnd = boundary_nodes(P)
    x, y = P.X[bnd] / H, P.Y[bnd] / H
    Pm = np.stack([(1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y], axis=1)
    return bnd, Pm


def star_defect(val, flux):
    """Relative residual of the best linear star on the sampled family:
    min_Y ||flux - Y val||_F / ||flux||_F = ||flux (I - Pi_val)||_F / ||flux||_F."""
    Q, _ = np.linalg.qr(val.conj().T)          # orthonormal basis of the row space of val
    R = flux - (flux @ Q) @ Q.conj().T
    return np.linalg.norm(R) / np.linalg.norm(flux)


def trefftz_family_2d(kH, ndir=400, nq=64):
    """(vertex values, half-edge fluxes) of plane waves in the unit cell, as in trefftz_rank_2d."""
    th = np.linspace(0, 2 * np.pi, ndir, endpoint=False)
    d = np.stack([np.cos(th), np.sin(th)])
    verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    val = np.exp(-1j * kH * (verts @ d))
    xg, wg = np.polynomial.legendre.leggauss(nq)
    t = 0.25 * (xg + 1); w = 0.25 * wg
    flux = np.zeros((4, ndir), complex)
    edges = [((0, 0), (1, 0), (0, -1)), ((1, 0), (1, 1), (1, 0)), ((1, 1), (0, 1), (0, 1)), ((0, 1), (0, 0), (-1, 0))]
    for e, (p, q, n) in enumerate(edges):
        p, q, n = np.array(p, float), np.array(q, float), np.array(n, float)
        dn = -1j * kH * (n @ d)
        for vtx, s in ((e, t), ((e + 1) % 4, 1 - t)):
            pts = p[None, :] + s[:, None] * (q - p)[None, :]
            flux[vtx] += (w[:, None] * np.exp(-1j * kH * (pts @ d))).sum(0) * dn
    return val, flux
