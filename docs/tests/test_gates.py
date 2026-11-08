"""Correctness gates. Every experiment must pass these before its results are trusted.
Run from the repo root:  pytest -q tests/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "oned"))
import wall_toy as W          # noqa: E402
import wall_inverse as I      # noqa: E402
import exact_hodge as X       # noqa: E402
import scenarios as S         # noqa: E402

F = np.linspace(77e9, 81e9, 64)


def test_reference_tmm_against_analytic_slab():
    n, d = 1.6, 0.0127
    sc = W.Scene([W.Layer("s", d, lambda f: np.full(f.shape, n**2 + 0j))], standoff=0.0)
    k = 2 * np.pi * F / W.C0
    r = (1 - n) / (1 + n)
    ph = np.exp(-2j * k * n * d)
    airy = (r - r * ph) / (1 - r * r * ph)
    assert np.allclose(W.reflection_tmm(F, sc), airy, atol=1e-14)


def test_reference_is_passive():
    f = np.linspace(1e9, 100e9, 500)
    for sc in (W.scene_dry_wall(), W.scene_wet_insulation(0.1), W.scene_wicked_drywall()):
        assert np.abs(W.reflection_tmm(f, sc)).max() <= 1.0


def test_exact_star_forward_matches_reference():
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), theta_ins=(0, 0.3))
    m.layers[1].theta.value = 0.0
    G, _ = I.forward(m, F, jacobian=False)
    assert np.abs(G - W.reflection_tmm(F, W.scene_dry_wall())).max() < 1e-12


def test_exact_star_teaching_version_one_cell_per_layer():
    sc = W.scene_dry_wall()
    err = np.abs(X.dec_reflection(F, sc, 1, exact=True) - W.reflection_tmm(F, sc)).max()
    assert err < 1e-12


def test_wet_scene_forward_matches_reference():
    m = S.stud_wall_model(theta_ins=(0, 0.3))
    m.layers[1].theta.value = 0.05
    m.layers[1].theta.free = True
    G, _ = I.forward(m, F, jacobian=False)
    assert np.abs(G - W.reflection_tmm(F, W.scene_wet_insulation(0.05))).max() < 1e-12


def test_adjoint_matches_finite_differences():
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), theta_ins=(0, 0.3))
    m.layers[1].theta.value = 0.002
    m.sync_ties()
    _, J = I.forward(m, F)
    for j in range(len(m.free_names())):
        h = 1e-6 * max(abs(m.get()[j]), 1e-3)
        a = m.copy(); v = a.get(); v[j] += h; a.set(v)
        b = m.copy(); v = b.get(); v[j] -= h; b.set(v)
        fd = (I.forward(a, F, False)[0] - I.forward(b, F, False)[0]) / (2 * h)
        rel = np.abs(fd - J[:, j]).max() / np.abs(J[:, j]).max()
        assert rel < 1e-5, (m.free_names()[j], rel)


def test_tridiagonal_solver_residual():
    rng = np.random.default_rng(0)
    Fn, n = 8, 50
    diag = rng.standard_normal((Fn, n)) + 1j * rng.standard_normal((Fn, n))
    off = rng.standard_normal((Fn, n - 1)) + 1j * rng.standard_normal((Fn, n - 1))
    rhs = rng.standard_normal((Fn, n, 2)) + 0j
    x = I._tridiag_solve(diag, off, rhs)
    Ax = diag[..., None] * x
    Ax[:, :-1] += off[..., None] * x[:, 1:]
    Ax[:, 1:] += off[..., None] * x[:, :-1]
    assert np.abs(Ax - rhs).max() < 1e-8


def test_calibration_recovers_gamma_noise_free():
    radar = W.Radar(snr_db=300)
    sc = S.truth_dry(0.3)
    f, y, _ = W.measure(sc, radar)
    _, yp, _ = W.measure_metal_plate(radar, 0.25)
    yc, _ = I.calibrate(y, yp, f, 0.25)
    assert np.abs(yc - W.reflection_tmm(f, sc)).max() < 1e-10
