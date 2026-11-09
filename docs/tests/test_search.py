"""Gates for src/oned/search.py (E009-E012)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "oned"))
import wall_toy as W          # noqa: E402
import wall_inverse as I      # noqa: E402
import scenarios as S         # noqa: E402
import search as SR           # noqa: E402
import benchmark as B         # noqa: E402

F = np.linspace(77e9, 81e9, 64)


def _data(seed=3):
    f, yc, sigma, d0, tv = B.make_case("v2", seed)
    return f, yc, sigma, d0


def test_lm_phys_identical_to_reference_lm():
    f, yc, sigma, d0 = _data()
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=d0)
    m.standoff.value = I.best_standoff(m, f, yc, sigma)
    a, ca = I.levenberg_marquardt(m, f, yc, sigma, max_iter=15)
    b, cb = SR.lm(m, f, yc, sigma, max_iter=15)
    assert ca == cb
    assert np.array_equal(a.get(), b.get())


def test_optical_coords_roundtrip_and_chain_rule():
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30)
    m.layers[0].tan_d.value = 0.02
    C = SR.Coords(m, "optical")
    q = C.q_of(m)
    m2 = C.apply(m, q)
    assert np.allclose(m2.get(), m.get(), rtol=1e-13, atol=0)
    # dGamma/dq by chain rule vs central finite differences in q
    _, Jp = I.forward(m, F)
    Jq = Jp @ C.dpdq(m)
    worst = 0.0
    for j in range(len(q)):
        h = 1e-6 * max(abs(q[j]), 1e-3)
        qa, qb = q.copy(), q.copy()
        qa[j] += h
        qb[j] -= h
        fd = (I.forward(C.apply(m, qa), F, False)[0] - I.forward(C.apply(m, qb), F, False)[0]) / (2 * h)
        worst = max(worst, np.abs(fd - Jq[:, j]).max() / np.abs(Jq[:, j]).max())
    assert worst < 1e-6, worst


def test_optical_lm_reaches_same_optimum():
    f, yc, sigma, d0 = _data()
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=d0)
    m.standoff.value = I.best_standoff(m, f, yc, sigma)
    a, ca = SR.lm(m, f, yc, sigma)
    b, cb = SR.lm(a, f, yc, sigma, coords="optical")    # start at the phys optimum
    assert cb <= ca * (1 + 1e-9)


import ambiguity as AM        # noqa: E402
from math import erf, sqrt    # noqa: E402


def test_quadratic_fit_is_exact_on_a_quadratic():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)); W = A @ A.T + 4 * np.eye(4); b = rng.standard_normal(4)
    pts = [tuple(int(i == k) * s for i in range(4)) for k in range(4) for s in (1, -1)]
    pts += [tuple(int(i in (j, k)) for i in range(4)) for j in range(4) for k in range(j + 1, 4)]
    vals = [np.array(a) @ W @ np.array(a) - 2 * np.array(a) @ b for a in pts]
    W2, b2 = AM.fit_quadratic(pts, vals)
    assert np.allclose(W2, W, atol=1e-12) and np.allclose(b2, b, atol=1e-12)


def test_bootstrap_success_diagonal_and_bound():
    s = np.array([0.2, 0.3, 0.25])
    Q = np.diag(s**2)
    Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    exact = np.prod([2 * Phi(1 / (2 * v)) - 1 for v in s])
    assert abs(AM.bootstrap_success(Q) - exact) < 1e-12
    # correlated case: bootstrap is a lower bound on ILS success (Teunissen 1999)
    R = np.array([[1, .9, .5], [.9, 1, .7], [.5, .7, 1]])
    Qc = 0.09 * R
    pb = AM.bootstrap_success(Qc)
    pm = AM.ils_success_mc(Qc, n=40000)
    assert pb <= pm + 3 * np.sqrt(pm * (1 - pm) / 40000) + 1e-3
    Z = AM.decorrelate(Qc)
    assert round(abs(np.linalg.det(Z))) == 1


def test_optical_box_lm_stays_in_box():
    f, yc, sigma, d0 = _data(15)
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=d0)
    m.standoff.value = I.best_standoff(m, f, yc, sigma)
    mo, c = SR.lm(m, f, yc, sigma, coords="optical_box")
    lo, hi = mo.bounds()
    v = mo.get()
    assert np.all(v > lo) and np.all(v < hi)


def test_gaussian_prior_rows_and_penalty():
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30)
    m.layers[0].t.value = 0.0131
    m.layers[2].t.value = 0.0121
    P = SR.GaussianPrior((("drywall_front.t", 0.0127, 0.0004), ("drywall_back.t", 0.0127, 0.0004)))
    r, J = P.rows(m)
    assert np.isclose(r @ r, P.penalty(m), rtol=1e-14)
    free = m.free_names()
    for k, name in enumerate(("drywall_front.t", "drywall_back.t")):
        assert J[k, free.index(name)] == 1 / 0.0004 and np.count_nonzero(J[k]) == 1
    m.layers[2].t.free = False                    # rows only for free parameters
    r2, J2 = P.rows(m)
    assert r2.size == 1 and J2.shape == (1, len(m.free_names()))
