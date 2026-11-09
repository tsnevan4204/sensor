"""Gates for src/stability (docs/11_indefinite_stability.md).  Run from docs/:  python3 -m pytest -q tests/"""
import os
import sys
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stability"))
import oned_stability as O    # noqa: E402
import dec2d as D             # noqa: E402
import infsup as IS           # noqa: E402
import cells as C             # noqa: E402
import mie                    # noqa: E402

STARS = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]


# ------------------------------------------------------------------ 2D solver: exactness and convergence
def test_whitney_edge_star_gives_q1_stiffness():
    g = D.Grid(3, 2, 0.5)
    P = D.Problem(g, alpha1=1.0)
    K = (P.D.T @ P.whitney_edge_star(1.0) @ P.D).toarray()
    Ke = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6
    Kq = np.zeros((g.nn, g.nn))
    for j in range(g.ny):
        for i in range(g.nx):
            n = [g.node(i, j), g.node(i + 1, j), g.node(i + 1, j + 1), g.node(i, j + 1)]
            Kq[np.ix_(n, n)] += Ke
    assert np.abs(K - Kq).max() < 1e-14
    assert np.abs(P.D @ np.ones(g.nn)).max() == 0          # D0 kills constants (exact topology)


def test_discrete_plane_wave_is_exact_interior_solution():
    n, k, th = 40, 6.0, 0.4
    for a1, a0 in STARS:
        P = D.Problem(D.Grid(n, n, 1 / n), alpha1=a1, alpha0=a0)
        q = D.dispersion_symbol(np.array([th]), k / n, a1, a0)[0] * n
        u = np.exp(-1j * q * (np.cos(th) * P.X + np.sin(th) * P.Y))
        r = (P.D.T @ P.S_mat(k) @ P.D - k**2 * P.M_mat(k)) @ u
        assert np.abs(r[~P.on_bnd]).max() < 1e-12


def test_dispersion_fourth_order_for_half_half_blend():
    th = np.linspace(0, np.pi / 4, 10)
    e = lambda kh, a1, a0: np.abs(D.dispersion_symbol(th, kh, a1, a0) / kh - 1).max()
    r_dec = np.log2(e(0.4, 0, 0) / e(0.2, 0, 0))
    r_bl = np.log2(e(0.4, .5, .5) / e(0.2, .5, .5))
    assert 1.9 < r_dec < 2.1 and 3.8 < r_bl < 4.2


def test_dec_planewave_convergence_rate_two():
    k, d = 10.0, (np.cos(0.3), np.sin(0.3))
    err = []
    for n in (40, 80, 160):
        P = D.Problem(D.Grid(n, n, 1 / n))
        u, _ = D.solve(P.assemble(k), P.robin_rhs_planewave(k, d))
        err.append(D.l2err(P, u, np.exp(-1j * k * (d[0] * P.X + d[1] * P.Y))))
    rates = np.log2(np.array(err[:-1]) / np.array(err[1:]))
    assert np.all((rates > 1.95) & (rates < 2.05)), rates


def test_mie_reference_and_dec_pml_convergence():
    k, a, er, L, dp = 4 * np.pi, 0.4, 2.25, 1.0, 0.3
    ph = np.linspace(0, 2 * np.pi, 9)
    u1 = mie.total_field(a * (1 - 1e-10) * np.cos(ph), a * (1 - 1e-10) * np.sin(ph), k, a, er)
    u2 = mie.total_field(a * (1 + 1e-10) * np.cos(ph), a * (1 + 1e-10) * np.sin(ph), k, a, er)
    assert np.abs(u1 - u2).max() < 1e-7
    err = []
    for n in (100, 200):
        h = 2 * (L + dp) / n
        g = D.Grid(n, n, h, -(L + dp), -(L + dp))
        P = D.Problem(g, pml=D.PML(dp, 60.0, (-(L + dp), L + dp), (-(L + dp), L + dp)), impedance=np.zeros(g.nn, bool))
        P.eps_node = D.dual_cell_eps(P, lambda x, y: np.where(x**2 + y**2 < a * a, er, 1.0) + 0j)
        uinc = np.exp(-1j * k * P.X)
        us, _ = D.solve(P.assemble(k), k**2 * P.area * (P.eps_node - 1) * uinc)
        mask = (np.abs(P.X) <= L) & (np.abs(P.Y) <= L)
        err.append(D.l2err(P, us + uinc, mie.total_field(P.X, P.Y, k, a, er), mask))
    assert err[1] < 0.02 and np.log2(err[0] / err[1]) > 1.9


def test_2d_adjoint_gradient_matches_central_differences():
    n, k = 24, 7.0
    rng = np.random.default_rng(1)

    def build(eps):
        P = D.Problem(D.Grid(n, n, 1 / n)); P.eps_node = eps.copy(); return P
    P = build(np.ones((n + 1) ** 2, complex))
    P.eps_node = D.dual_cell_eps(P, lambda x, y: np.where((x - .5)**2 + (y - .5)**2 < .05, 3.0 - .3j, 1.0) + 0j)
    eps0 = P.eps_node.copy()
    uinc = np.exp(-1j * k * P.X)
    c = rng.standard_normal(P.grid.nn) * P.on_bnd

    def J(eps):
        Q = build(eps)
        us, lu = D.solve(Q.assemble(k), k**2 * Q.area * (eps - 1) * uinc)
        return c @ us, us, lu
    _, us, lu = J(eps0)
    grad = k**2 * P.area * lu.solve(c.astype(complex)) * (us + uinc)
    for i in rng.choice(np.where(np.abs(eps0 - 1) > 0)[0], 4, replace=False):
        e1, e2 = eps0.copy(), eps0.copy(); e1[i] += 1e-5; e2[i] -= 1e-5
        fd = (J(e1)[0] - J(e2)[0]) / 2e-5
        assert abs(fd - grad[i]) / abs(grad[i]) < 1e-6


def test_complex_lu_and_infsup_against_dense():
    P = D.Problem(D.Grid(14, 14, 1 / 14))
    for k in (3.0, 12.0):
        A = P.assemble(k); N = P.energy_norm(k)
        assert abs(IS.beta(A, N) / IS.beta_dense(A, N) - 1) < 1e-6
        lu = D.ComplexLU(A)
        f = np.arange(P.grid.nn) * (1 + 0.5j)
        assert np.linalg.norm(A @ lu.solve(f) - f) / np.linalg.norm(f) < 1e-12
        assert np.linalg.norm(A.conj().T @ lu.solve(f, "H") - f) / np.linalg.norm(f) < 1e-12


def test_passive_but_singular_counterexample_and_full_side_uniqueness():
    n = 20
    P0 = D.Problem(D.Grid(n, n, 1 / n))
    imp = np.zeros(P0.grid.nn, bool)
    imp[P0.grid.node(n // 2, 0)] = imp[P0.grid.node(n // 2, n)] = True
    P = D.Problem(D.Grid(n, n, 1 / n), impedance=imp)
    k1 = 2 * n * np.sin(np.pi / (2 * n))
    A = P.assemble(k1)
    assert IS.passivity_eig(A) >= -1e-14
    assert np.linalg.svd(A.toarray(), compute_uv=False).min() < 1e-12      # singular although passive
    imp2 = np.isclose(P0.X, 0.0)                                            # impedance on one full side
    P2 = D.Problem(D.Grid(n, n, 1 / n), impedance=imp2)
    assert np.linalg.svd(P2.assemble(k1).toarray(), compute_uv=False).min() > 1e-6


def test_pml_stars_are_not_passive_but_system_invertible():
    g = D.Grid(30, 30, 1 / 30)
    P = D.Problem(g, pml=D.PML(0.2, 40.0, (0, 1), (0, 1)), impedance=np.zeros(g.nn, bool))
    A = P.assemble(10.0)
    assert IS.passivity_eig(A) < -1e-3
    assert IS.beta(A, P.energy_norm(10.0)) > 1e-3


def test_lossy_theorem_bound():
    for k, tau in ((10.0, 0.1), (30.0, 0.05)):
        n = int(np.ceil(10 * k / (2 * np.pi)))
        P = D.Problem(D.Grid(n, n, 1 / n), eps_node=np.full((n + 1) ** 2, 1 - 1j * tau))
        b = IS.beta(P.assemble(k), P.energy_norm(k), Nout=sp.diags(P.area), tol=1e-7)
        Ct = np.sqrt(((1 + abs(1 - 1j * tau) / tau) + 1 / tau) / tau)
        assert 1 / b <= Ct / k


# ------------------------------------------------------------------ cells: Trefftz and Schur
def test_trefftz_rank_1d_vs_2d():
    assert C.trefftz_rank_1d(3.0, 0.7)[:2] == (2, 2)
    r0, r1, _ = C.trefftz_rank_2d(1.0)
    assert r0 == 4 and r1 == 8


def test_schur_complement_passive_symmetric_and_exact():
    P = C.cell_problem(8, 1.0, lambda x, y: 3.0 - 0.5j + 0 * x)
    A = (P.D.T @ sp.diags(P.ratio) @ P.D - 4.0**2 * sp.diags(P.area * P.eps_node)).toarray()
    b = C.boundary_nodes(P)
    S = C.schur(A, b)
    H = (S - S.conj().T) / 2j
    assert np.linalg.eigvalsh((H + H.conj().T) / 2).min() > -1e-12 * np.abs(S).max()
    assert np.abs(S - S.T).max() < 1e-12 * np.abs(S).max()
    G = D.Problem(D.Grid(12, 12, 1 / 12))
    Ad = G.assemble(5.0).toarray()
    sk = np.where(G.on_bnd | np.isclose(G.X, 0.5) | np.isclose(G.Y, 0.5))[0]
    Ssk = C.schur(Ad, sk)
    assert np.abs(np.linalg.inv(Ssk) - np.linalg.inv(Ad)[np.ix_(sk, sk)]).max() < 1e-10 * np.abs(np.linalg.inv(Ad)).max()


# ------------------------------------------------------------------ 1D theorems
def test_1d_exact_star_green_function_and_mesh_independence():
    w = 2 * np.pi * 30e9
    eps, ell = O.wall_cells([(0.1, 1.0)], w, [9])
    b, M = O.exact_stars(eps, ell, w)
    Ai = np.linalg.inv(O.assemble(b, M, 1 / O.ETA0, 1 / O.ETA0))
    z = O.node_positions(ell)
    g = O.ETA0 / 2 * np.exp(-1j * w / O.C0 * np.abs(z[:, None] - z[None]))
    assert np.abs(Ai - g).max() < 1e-12 * O.ETA0
    L = [(0.012, 2.8 - 0.1j), (0.05, 1.0), (0.03, 6.0 - 1j)]
    eps, ell = O.wall_cells(L, w)
    b, M = O.exact_stars(eps, ell, w)
    Ai = np.linalg.inv(O.assemble(b, M, 1 / O.ETA0, 1 / O.ETA0))
    b2, M2 = O.exact_stars(np.repeat(eps, 2), np.repeat(ell / 2, 2), w)
    Ai2 = np.linalg.inv(O.assemble(b2, M2, 1 / O.ETA0, 1 / O.ETA0))
    assert np.abs(Ai2[::2, ::2] - Ai).max() < 1e-10 * np.abs(Ai).max()


def test_1d_green_bound_holds_on_random_walls_and_passive_learned_stars():
    rng = np.random.default_rng(7)
    for t in range(40):
        L = O.random_wall(rng, lossless=t % 2 == 0)
        w = 2 * np.pi * rng.uniform(1e9, 100e9)
        eps, ell = O.wall_cells(L, w)
        b, M = O.exact_stars(eps, ell, w)
        assert O.passivity_margin(b, M) >= -1e-12 * np.abs(M).max()
        r, _, _ = O.green_bound_ratio(b, M, 1 / O.ETA0, 1 / O.ETA0)
        assert r <= 1.0
        if t % 2 == 0:   # lossless: random passive learned factors
            n = eps.size
            s, m = O.exact_factors(w / O.C0 * O.nbranch(eps) * ell)
            s = s * rng.uniform(0.3, 3, n) + 1j * rng.uniform(0, .5, n)
            m = m * rng.uniform(0.3, 3, n) - 1j * rng.uniform(0, .5, n)
            b, M = O.learned_stars(eps, ell, w, s, m)
            assert O.passivity_margin(b, M) >= 0
            r, _, _ = O.green_bound_ratio(b, M, 1 / O.ETA0, 1 / O.ETA0)
            assert r <= 1.0


def test_1d_singular_perturbations_are_active():
    rng = np.random.default_rng(11)
    for t in range(30):
        L = O.random_wall(rng, lossless=True)
        w = 2 * np.pi * rng.uniform(1e9, 100e9)
        eps, ell = O.wall_cells(L, w)
        b, M = O.exact_stars(eps, ell, w)
        Ai = np.linalg.inv(O.assemble(b, M, 1 / O.ETA0, 1 / O.ETA0))
        c = rng.integers(eps.size)
        for d in -1 / np.linalg.eigvals(Ai[np.ix_([c, c + 1], [c, c + 1])]):
            assert d.real < 0          # a shift that makes A singular must inject power


def test_1d_std_star_effective_impedance():
    w = 2 * np.pi * 10e9; k = w / O.C0
    kl = np.array([0.2, 1.0, 1.8])
    b, M = O.std_stars(np.ones(3), kl / k, w)
    assert np.allclose(O.eta_eff(b, M).real, O.ETA0 / np.sqrt(1 - (kl / 2) ** 2), rtol=1e-12)


def test_1d_lossy_exact_cell_transfer_growth_bound():
    # ||N(eta) T N(eta)^-1||^2 <= exp(k' tan_d l),  eta = sqrt(mu0/(eps0 eps')), k' = w sqrt(mu0 eps0 eps')
    rng = np.random.default_rng(5)
    for _ in range(200):
        er, td = rng.uniform(1.5, 9), rng.uniform(0, 1.0)
        w = 2 * np.pi * rng.uniform(1e9, 100e9)
        ell = rng.uniform(0.1, 3.0) / (w / O.C0 * np.sqrt(er))
        b, M = O.exact_stars([er * (1 - 1j * td)], [ell], w)
        eta = O.ETA0 / np.sqrt(er)
        Tn = O.normalizer(eta) @ O.transfer(b[0], M[0]) @ np.linalg.inv(O.normalizer(eta))
        kp = w / O.C0 * np.sqrt(er)
        assert np.linalg.norm(Tn, 2) ** 2 <= np.exp(kp * td * ell) * (1 + 1e-12)


def test_1d_foster_expansions_of_exact_modal_admittances():
    # eta Y+ = j tan(w tau) = sum 2 s/(tau (s^2 + w_n^2)),  w_n = (n-1/2) pi/tau
    # eta Y- = -j cot(w tau) = 1/(s tau) + sum 2 s/(tau (s^2 + (n pi/tau)^2))
    tau = 1.0
    w = np.array([0.3, 1.1, 2.0, 4.4])
    s = 1j * w
    n = np.arange(1, 200001)[:, None]
    yp = (2 * s / (tau * (s**2 + ((n - .5) * np.pi / tau) ** 2))).sum(0)
    ym = 1 / (s * tau) + (2 * s / (tau * (s**2 + (n * np.pi / tau) ** 2))).sum(0)
    assert np.allclose(yp, 1j * np.tan(w * tau), rtol=1e-4)
    assert np.allclose(ym, -1j / np.tan(w * tau), rtol=1e-4)


def test_mass_blend_moves_infsup_by_at_most_the_proved_amount():
    # |beta(0, a0) - beta(0, 0)| <= a0 [ (2/9)(kh)^2 + (1/3) kh ]  (Proposition B7)
    for n, k in ((20, 6.0), (30, 15.0)):
        h = 1 / n
        P0 = D.Problem(D.Grid(n, n, h))
        P1 = D.Problem(D.Grid(n, n, h), alpha0=0.5)
        N = P0.energy_norm(k)
        b0, b1 = IS.beta_dense(P0.assemble(k), N), IS.beta_dense(P1.assemble(k), N)
        assert abs(b1 - b0) <= 0.5 * (2 / 9 * (k * h) ** 2 + 1 / 3 * k * h)
        Dm = (P1.assemble(k) - P0.assemble(k)).toarray()
        w, V = np.linalg.eigh(N.toarray()); Nmh = V @ np.diag(w ** -.5) @ V.T
        assert np.linalg.norm(Nmh @ Dm @ Nmh, 2) <= 0.5 * (2 / 9 * (k * h) ** 2 + 1 / 3 * k * h)
