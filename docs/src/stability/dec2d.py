"""
Minimal 2D DEC solver for the scalar (E_z) time-harmonic problem on a structured Yee grid.

Normalization (dimensionless, Helmholtz form; equals jw mu0 times the admittance form of src/oned):
    A(k) = D0^T S D0  -  k^2 M  +  j k B
  D0 : node -> edge incidence (entries 0, +-1; exact topology)
  S  : edge Hodge star  *_{mu^-1}  (|dual edge| / |edge| / mu_r)   [diagonal DEC, or Whitney-consistent]
  M  : node Hodge star  *_eps      (|dual cell| * eps_r)            [lumped DEC, or Q1-consistent]
  B  : first-order impedance (Sommerfeld) boundary, |dual cell boundary| on boundary nodes
Physics:  -div(mu_r^-1 grad u) - k^2 eps_r u = f ,  d_n u + j k u = g  (e^{+jwt}: outgoing e^{-jkr}).
Passivity of the admittance form  <=>  (A - A^H)/(2j) >= 0  (Im u^H A u >= 0 for all u).

PML: complex coordinate stretching s_x = 1 - j sigma_x(x)/k, s_y likewise.  It only changes the stars:
    x-edge: S *= s_y/s_x,  y-edge: S *= s_x/s_y,  node: M *= s_x s_y .   D0 is untouched.
Learned stars: S_theta = S_std * s_theta (per edge), M_theta = M_std * m_theta (per node).
Blended stars (alpha1, alpha0): S = (1-a1) S_DEC + a1 S_Whitney,  M = (1-a0) M_lumped + a0 M_Q1.
  (a1, a0) = (0,0): DEC/Yee; (1,1): bilinear Q1 FEM; (1/2, 1/2): fourth-order isotropic dispersion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class Grid:
    nx: int
    ny: int
    h: float
    x0: float = 0.0
    y0: float = 0.0

    @property
    def nn(self):
        return (self.nx + 1) * (self.ny + 1)

    def xy(self):
        x = self.x0 + self.h * np.arange(self.nx + 1)
        y = self.y0 + self.h * np.arange(self.ny + 1)
        X, Y = np.meshgrid(x, y)          # shape (ny+1, nx+1), node p = j*(nx+1)+i
        return X.ravel(), Y.ravel()

    def node(self, i, j):
        return j * (self.nx + 1) + i

    def weights(self):
        """1D dual-cell widths along x and y (h/2 at the ends)."""
        wx = np.full(self.nx + 1, self.h); wx[[0, -1]] = self.h / 2
        wy = np.full(self.ny + 1, self.h); wy[[0, -1]] = self.h / 2
        return wx, wy


def incidence(g: Grid):
    """D0 (edges x nodes), first all x-edges (i,j)->(i+1,j), then all y-edges (i,j)->(i,j+1).
    Also returns edge midpoints and the orientation flag (0 = x-edge, 1 = y-edge)."""
    I, J = np.meshgrid(np.arange(g.nx), np.arange(g.ny + 1))
    I, J = I.ravel(), J.ravel()
    tail_x, head_x = g.node(I, J), g.node(I + 1, J)
    mx = (g.x0 + g.h * (I + 0.5), g.y0 + g.h * J, I, J)
    I2, J2 = np.meshgrid(np.arange(g.nx + 1), np.arange(g.ny))
    I2, J2 = I2.ravel(), J2.ravel()
    tail_y, head_y = g.node(I2, J2), g.node(I2, J2 + 1)
    my = (g.x0 + g.h * I2, g.y0 + g.h * (J2 + 0.5), I2, J2)
    ne_x, ne_y = tail_x.size, tail_y.size
    rows = np.concatenate([np.arange(ne_x), ne_x + np.arange(ne_y)])
    rows = np.concatenate([rows, rows])
    cols = np.concatenate([tail_x, tail_y, head_x, head_y])
    vals = np.concatenate([-np.ones(ne_x + ne_y), np.ones(ne_x + ne_y)])
    D = sp.csr_matrix((vals, (rows, cols)), shape=(ne_x + ne_y, g.nn))
    ex = np.concatenate([mx[0], my[0]]); ey = np.concatenate([mx[1], my[1]])
    orient = np.concatenate([np.zeros(ne_x, int), np.ones(ne_y, int)])
    eI = np.concatenate([mx[2], my[2]]); eJ = np.concatenate([mx[3], my[3]])
    return D, ex, ey, orient, eI, eJ


@dataclass
class PML:
    """Quadratic-profile PML: sigma(t) = smax ((t - t_in)/d)^2 beyond |t - c| > half - d."""
    d: float
    smax: float
    xlim: tuple
    ylim: tuple

    def sx(self, x, k):
        return self._s(x, self.xlim, k)

    def sy(self, y, k):
        return self._s(y, self.ylim, k)

    def _s(self, t, lim, k):
        lo, hi = lim
        dist = np.maximum(np.maximum(lo + self.d - t, t - (hi - self.d)), 0.0)
        return 1 - 1j * self.smax * (dist / self.d) ** 2 / k


@dataclass
class Problem:
    grid: Grid
    eps_node: np.ndarray | None = None      # eps_r per node (dual-cell average)
    impedance: np.ndarray | None = None     # bool per node: impedance BC active (default: all boundary)
    pml: PML | None = None
    alpha1: float = 0.0
    alpha0: float = 0.0
    s_theta: np.ndarray | None = None       # learned edge factors (diagonal stars only)
    m_theta: np.ndarray | None = None       # learned node factors
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        g = self.grid
        self.D, self.ex, self.ey, self.orient, self.eI, self.eJ = incidence(g)
        self.X, self.Y = g.xy()
        wx, wy = g.weights()
        WX, WY = np.meshgrid(wx, wy)
        self.area = (WX * WY).ravel()                     # |dual cell|
        # |dual edge| / |edge|: x-edge (i,j): wy[j]/h ; y-edge (i,j): wx[i]/h
        self.ratio = np.where(self.orient == 0, wy[self.eJ], wx[self.eI]) / g.h
        bl = np.zeros(g.nn)                               # |dual cell boundary| on boundary nodes
        I, J = np.meshgrid(np.arange(g.nx + 1), np.arange(g.ny + 1))
        I, J = I.ravel(), J.ravel()
        bl += np.where((J == 0) | (J == g.ny), wx[I], 0.0)
        bl += np.where((I == 0) | (I == g.nx), wy[J], 0.0)
        self.blen = bl
        self.on_bnd = bl > 0
        if self.eps_node is None:
            self.eps_node = np.ones(g.nn, complex)
        if self.impedance is None:
            self.impedance = self.on_bnd.copy()

    # ------------------------------------------------------------------ stars
    def stretch(self, k):
        if self.pml is None:
            return 1.0, 1.0, 1.0
        sxn, syn = self.pml.sx(self.X, k), self.pml.sy(self.Y, k)
        sxe, sye = self.pml.sx(self.ex, k), self.pml.sy(self.ey, k)
        fe = np.where(self.orient == 0, sye / sxe, sxe / sye)
        return fe, sxn * syn, (sxn, syn)

    def S_diag(self, k):
        fe, _, _ = self.stretch(k)
        s = self.ratio * fe
        if self.s_theta is not None:
            s = s * self.s_theta
        return s

    def M_diag(self, k):
        _, fn, _ = self.stretch(k)
        m = self.area * self.eps_node * fn
        if self.m_theta is not None:
            m = m * self.m_theta
        return m

    def whitney_edge_star(self, k):
        """Consistent Whitney (lowest-order rectangular Nedelec) edge mass, cell by cell.
        Per cell: parallel x-edges (bottom, top) get [[1/3, 1/6], [1/6, 1/3]], same for y-edges.
        PML stretch evaluated at the cell centre."""
        g = self.grid
        key = ("W1", k)
        if key in self._cache:
            return self._cache[key]
        nex = g.nx * (g.ny + 1)
        ci, cj = np.meshgrid(np.arange(g.nx), np.arange(g.ny))
        ci, cj = ci.ravel(), cj.ravel()
        xb = cj * g.nx + ci; xt = (cj + 1) * g.nx + ci             # x-edges index i + j*nx
        yl = nex + cj * (g.nx + 1) + ci; yr = nex + cj * (g.nx + 1) + ci + 1
        if self.pml is not None:
            xc = g.x0 + g.h * (ci + 0.5); yc = g.y0 + g.h * (cj + 0.5)
            sx, sy = self.pml.sx(xc, k), self.pml.sy(yc, k)
            fx, fy = sy / sx, sx / sy
        else:
            fx = fy = np.ones(ci.size)
        r, c, v = [], [], []
        for (a, bb, f) in ((xb, xt, fx), (yl, yr, fy)):
            for (p, q, w) in ((a, a, 1 / 3), (bb, bb, 1 / 3), (a, bb, 1 / 6), (bb, a, 1 / 6)):
                r.append(p); c.append(q); v.append(w * f)
        ne = self.D.shape[0]
        W = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(ne, ne))
        self._cache[key] = W
        return W

    def q1_mass(self, k):
        """Consistent bilinear (Q1) node mass with eps_r averaged per cell (from node values)."""
        g = self.grid
        ci, cj = np.meshgrid(np.arange(g.nx), np.arange(g.ny))
        ci, cj = ci.ravel(), cj.ravel()
        n = [g.node(ci, cj), g.node(ci + 1, cj), g.node(ci + 1, cj + 1), g.node(ci, cj + 1)]
        epsc = sum(self.eps_node[q] for q in n) / 4
        if self.pml is not None:
            xc = g.x0 + g.h * (ci + 0.5); yc = g.y0 + g.h * (cj + 0.5)
            epsc = epsc * self.pml.sx(xc, k) * self.pml.sy(yc, k)
        Me = g.h**2 / 36 * np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]])
        r, c, v = [], [], []
        for a in range(4):
            for b in range(4):
                r.append(n[a]); c.append(n[b]); v.append(Me[a, b] * epsc)
        return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(g.nn, g.nn))

    def boundary_mass(self, consistent=False):
        """Lumped: diag(blen * impedance). Consistent: P1 mass along boundary edges."""
        if not consistent:
            return sp.diags(self.blen * self.impedance)
        g = self.grid
        r, c, v = [], [], []
        sides = []
        sides.append([g.node(i, 0) for i in range(g.nx + 1)])
        sides.append([g.node(i, g.ny) for i in range(g.nx + 1)])
        sides.append([g.node(0, j) for j in range(g.ny + 1)])
        sides.append([g.node(g.nx, j) for j in range(g.ny + 1)])
        for s in sides:
            s = np.array(s)
            a, b = s[:-1], s[1:]
            on = self.impedance[a] & self.impedance[b]
            a, b = a[on], b[on]
            for (p, q, w) in ((a, a, 1 / 3), (b, b, 1 / 3), (a, b, 1 / 6), (b, a, 1 / 6)):
                r.append(p); c.append(q); v.append(np.full(p.size, w * g.h))
        return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(g.nn, g.nn))

    def S_mat(self, k):
        S = sp.diags(self.S_diag(k))
        if self.alpha1:
            S = (1 - self.alpha1) * S + self.alpha1 * self.whitney_edge_star(k)
        return S

    def M_mat(self, k):
        M = sp.diags(self.M_diag(k))
        if self.alpha0:
            M = (1 - self.alpha0) * M + self.alpha0 * self.q1_mass(k)
        return M

    def B_mat(self):
        B = self.boundary_mass(False)
        if self.alpha0:
            B = (1 - self.alpha0) * B + self.alpha0 * self.boundary_mass(True)
        return B

    def assemble(self, k):
        K = (self.D.T @ self.S_mat(k) @ self.D)
        return (K - k**2 * self.M_mat(k) + 1j * k * self.B_mat()).tocsc()

    def energy_norm(self, k):
        """N = D^T S0 D + k^2 M0 with the geometric (standard, unstretched, eps=1) stars."""
        return (self.D.T @ sp.diags(self.ratio) @ self.D + k**2 * sp.diags(self.area)).tocsc()

    # ------------------------------------------------------------------ sources
    def robin_rhs_planewave(self, k, d, consistent=False):
        """f = boundary integral of g, g = d_n u + j k u for u = exp(-j k d.x) (exact solution u)."""
        g = self.grid
        u = np.exp(-1j * k * (d[0] * self.X + d[1] * self.Y))
        I = np.rint((self.X - g.x0) / g.h).astype(int); J = np.rint((self.Y - g.y0) / g.h).astype(int)
        wx, wy = g.weights()
        if not consistent:
            f = np.zeros(g.nn, complex)
            for (mask, n, w) in (((J == 0), (0, -1), wx[I]), ((J == g.ny), (0, 1), wx[I]),
                                 ((I == 0), (-1, 0), wy[J]), ((I == g.nx), (1, 0), wy[J])):
                gval = 1j * k * (1 - (d[0] * n[0] + d[1] * n[1])) * u
                f += np.where(mask, w * gval, 0)
            return f
        # consistent: P1 boundary mass per side applied to nodal g of that side
        f = np.zeros(g.nn, complex)
        sides = [([g.node(i, 0) for i in range(g.nx + 1)], (0, -1)),
                 ([g.node(i, g.ny) for i in range(g.nx + 1)], (0, 1)),
                 ([g.node(0, j) for j in range(g.ny + 1)], (-1, 0)),
                 ([g.node(g.nx, j) for j in range(g.ny + 1)], (1, 0))]
        for s, n in sides:
            s = np.array(s)
            gv = 1j * k * (1 - (d[0] * n[0] + d[1] * n[1])) * u[s]
            a, b = s[:-1], s[1:]
            np.add.at(f, a, g.h * (gv[:-1] / 3 + gv[1:] / 6))
            np.add.at(f, b, g.h * (gv[1:] / 3 + gv[:-1] / 6))
        return f


class ComplexLU:
    """LU of a complex sparse matrix through its real 2x2-block form
        R = kron(Re A, I2) + kron(Im A, [[0,-1],[1,0]]).
    (SciPy's complex SuperLU is ~1e3x slower than the real one on these matrices.)
    solve(f) = A^{-1} f;  solve(f, 'H') = A^{-H} f  (real form of A^H is R^T)."""

    def __init__(self, A):
        A = sp.csr_matrix(A)
        J = sp.csr_matrix(np.array([[0.0, -1.0], [1.0, 0.0]]))
        R = (sp.kron(A.real, sp.eye(2)) + sp.kron(A.imag, J)).tocsc()
        self.lu = spla.splu(R, permc_spec="MMD_AT_PLUS_A")
        self.n = A.shape[0]

    def solve(self, f, trans="N"):
        f = np.asarray(f, complex)
        r = np.empty(2 * self.n); r[0::2] = f.real; r[1::2] = f.imag
        x = self.lu.solve(r, trans="T" if trans == "H" else "N")
        return x[0::2] + 1j * x[1::2]


def solve(A, f, check=1e-9):
    """Sparse LU solve, verified by residual (one step of iterative refinement if needed)."""
    lu = ComplexLU(A)
    u = lu.solve(f)
    r = np.linalg.norm(A @ u - f) / max(np.linalg.norm(f), 1e-300)
    if not r < check:
        u = u + lu.solve(f - A @ u)
        r = np.linalg.norm(A @ u - f) / max(np.linalg.norm(f), 1e-300)
        if not r < check:
            raise RuntimeError(f"residual {r:.2e}")
    return u, lu


def l2err(P: Problem, u, uex, mask=None):
    w = P.area if mask is None else P.area * mask
    return np.sqrt(np.sum(w * np.abs(u - uex) ** 2) / np.sum(w * np.abs(uex) ** 2))


def dual_cell_eps(P: Problem, eps_fun, sub=8):
    """eps_r per node = average of eps_fun over the node's dual cell (sub x sub samples)."""
    g = P.grid
    t = (np.arange(sub) + 0.5) / sub - 0.5
    acc = np.zeros(g.nn, complex)
    for a in t:
        for b in t:
            x = np.clip(P.X + a * g.h, g.x0, g.x0 + g.nx * g.h)
            y = np.clip(P.Y + b * g.h, g.y0, g.y0 + g.ny * g.h)
            acc += eps_fun(x, y)
    return acc / sub**2


def dispersion_symbol(theta, kh_disc, a1=0.0, a0=0.0):
    """Discrete wavenumber kappa h along direction theta for the blended stars, solving
    stiffness_symbol(kappa) = (k h)^2 mass_symbol(kappa) by Newton (kappa h initial = k h)."""
    c, s = np.cos(theta), np.sin(theta)

    def F(q):
        sx, sy = np.sin(q * c / 2) ** 2, np.sin(q * s / 2) ** 2
        stiff = 4 * (sx + sy - (4 * a1 / 3) * sx * sy)
        mass = (1 - a0) + a0 * (1 - 2 * sx / 3) * (1 - 2 * sy / 3)
        return stiff - kh_disc**2 * mass

    q = np.full_like(np.asarray(theta, float), kh_disc)
    for _ in range(60):
        dq = 1e-7
        q = q - F(q) / ((F(q + dq) - F(q - dq)) / (2 * dq))
    return q
