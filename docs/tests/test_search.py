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


def test_band_segments_and_block_gate():
    a = np.linspace(77e9, 81e9, 256)
    b = np.linspace(22e9, 26e9, 256)
    f = np.concatenate([a, b])
    assert I.band_segments(a) == [(0, 256)]
    assert I.band_segments(f) == [(0, 256), (256, 512)]
    assert I.band_segments(f[::8]) == [(0, 32), (32, 64)]
    assert I.carrier(a) == a.mean() and abs(I.carrier(f) - a.mean()) < 1
    g = I.Gate(0.27, 0.36)
    Op, Oa, Ob = g.op(f), g.op(a), g.op(b)
    assert np.array_equal(Op[:Oa.shape[0], :256], Oa) and np.array_equal(Op[Oa.shape[0]:, 256:], Ob)
    assert not np.any(Op[:Oa.shape[0], 256:]) and not np.any(Op[Oa.shape[0]:, :256])


def test_dual_benchmark_reuses_v2_truth_and_high_band():
    f, yc, s, d0, tv = B.make_case("v2dual", 4)
    f1, y1, s1, d1, tv1 = B.make_case("v2", 4)
    assert tv == tv1 and d0 == d1 and np.array_equal(yc[:256], y1) and f.size == 512
    # exact forward model on both bands at the truth fits to noise level
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=d0)
    for n, _, p in m.items():
        if n in tv:
            p.value = tv[n]
    m.sync_ties()
    G, _ = I.forward(m, f, jacobian=False)
    chi2n = np.sum(np.abs(G - yc) ** 2) / s**2 / (2 * f.size)
    assert 0.8 < chi2n < 1.25, chi2n


def test_cq_law_exact_at_fref_jacobian_and_matches_realism():
    import realism as RL
    f = np.concatenate([np.linspace(77e9, 81e9, 32), np.linspace(57e9, 64e9, 32)])
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30)
    m.layers[0].tan_d.value = 0.035
    m.sync_ties()
    mc = m.copy(); mc.law = "cq"
    fr = np.array([79e9])
    assert np.abs(I.forward(m, fr, False)[0] - I.forward(mc, fr, False)[0]).max() < 1e-14
    G, J = I.forward(mc, f)
    worst = 0
    for j in range(len(mc.free_names())):
        h = 1e-7 * max(abs(mc.get()[j]), 1e-3)
        a = mc.copy(); v = a.get(); v[j] += h; a.set(v)
        b = mc.copy(); v = b.get(); v[j] -= h; b.set(v)
        fd = (I.forward(a, f, False)[0] - I.forward(b, f, False)[0]) / (2 * h)
        worst = max(worst, np.abs(fd - J[:, j]).max() / np.abs(J[:, j]).max())
    assert worst < 1e-5, worst
    # the model's cq law equals the realism generator's cq truth (independent TMM)
    tv = {n: p.value for n, _, p in mc.items()}
    L = RL.layers_from_truth(tv, RL.Realism(dispersion="cq"))
    GTE, _ = RL.gamma_layers(f, np.zeros((f.size, 1)), L)
    ref = GTE[:, 0] * np.exp(-2j * 2 * np.pi * f / W.C0 * mc.standoff.value)
    assert np.abs(G - ref).max() < 1e-12


def test_jonscher_law_reduces_to_cq_jacobian_and_matches_causal_itu_truth():
    import realism as RL
    f = np.concatenate([np.linspace(77e9, 81e9, 32), np.linspace(57e9, 64e9, 32)])
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30)
    m.layers[0].tan_d.value = 0.035; m.sync_ties()
    mc, mj = m.copy(), m.copy()
    mc.law, mj.law = "cq", "jonscher"
    assert np.abs(I.forward(mc, f, False)[0] - I.forward(mj, f, False)[0]).max() < 1e-13    # p = 0
    mj.layers[0].pinf = I.P(0.45, 0, 0.9, True)
    mj.layers[2].pinf = I.P(0.45, 0, 0.9, True, tie="drywall_front.pinf"); mj.sync_ties()
    G, J = I.forward(mj, f)
    worst = 0
    for j in range(len(mj.free_names())):
        errs = []                    # best of three steps: truncation vs round-off differ per parameter
        for rel in (1e-5, 1e-6, 1e-7):
            h = rel * max(abs(mj.get()[j]), 1e-3)
            a = mj.copy(); v = a.get(); v[j] += h; a.set(v)
            b = mj.copy(); v = b.get(); v[j] -= h; b.set(v)
            fd = (I.forward(a, f, False)[0] - I.forward(b, f, False)[0]) / (2 * h)
            errs.append(np.abs(fd - J[:, j]).max() / np.abs(J[:, j]).max())
        worst = max(worst, min(errs))
    assert worst < 1e-6, worst
    # causal-ITU truth (drywall a' = 1 - 0.9395) equals the Jonscher model with the matching p
    td = 0.035
    ap = max(1 - RL.ITU_D["drywall"], 2 / np.pi * np.arctan(td))
    pp = 1 - td / np.tan(np.pi * ap / 2)
    e_truth = RL.eps_material(2.6, td, f, "itucausal", "drywall")
    L = I.LayerSpec("x", I.P(0.01, 0, 1), I.P(2.6, 1, 4), I.P(td, 0, 1), pinf=I.P(pp, 0, 0.9))
    e_model, _ = I.layer_eps_and_derivs(L, f, 20.0, "jonscher", 79e9)
    assert np.abs(e_truth - e_model).max() < 1e-12


def test_oblique_stars_match_tmm_and_beam_converges():
    import realism as RL
    f = np.concatenate([np.linspace(77e9, 81e9, 24), np.linspace(57e9, 64e9, 24)])
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30); m.law = "cq"
    m.layers[0].tan_d.value = 0.035; m.sync_ties()
    G0, _ = I.forward(m, f, False)
    GTE, GTM, _ = I.forward_oblique(m, f, np.zeros((f.size, 1)), False)
    ph = np.exp(-2j * 2 * np.pi * f / W.C0 * 0.30)
    assert np.abs(GTE[:, 0] * ph - G0).max() < 1e-12 and np.abs(GTM[:, 0] * ph - G0).max() < 1e-12
    kr = np.outer(2 * np.pi * f / W.C0, [0.0, 0.3, 0.7])
    L = RL.layers_from_truth(m.summary(), RL.Realism(dispersion="cq"))
    RTE, RTM = RL.gamma_layers(f, kr, L)
    GTE, GTM, _ = I.forward_oblique(m, f, kr, False)
    assert np.abs(GTE - RTE).max() < 1e-12 and np.abs(GTM - RTM).max() < 1e-12
    for w0 in (0.02, 0.01):
        truth = np.concatenate([RL.received(f[:24], L, 0.30, RL.Realism(beam_w0=w0))[0],
                                RL.received(f[24:], L, 0.30, RL.Realism(beam_w0=w0))[0]])
        mm = m.copy(); mm.beam = (w0, w0); mm.beam_K = 6
        assert np.abs(I.forward(mm, f, False)[0] - truth).max() < 1e-9


def test_lambot_calibration_recovers_beam():
    import realism as RL
    f, y, s, d0, tv, w0s = RL.make_case_lambot(3, RL.Realism(dispersion="cq", beam_w0=0.015, Hf=0.1, Hi=0.02))
    assert all(abs(w - 0.015) < 0.0008 for w in w0s), w0s


def _facing_model(tp=4e-4, ep=2.0, tdp=0.08, law="cq"):
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30)
    m.law = law
    m.layers[0].tan_d.value = 0.035; m.sync_ties()
    m.facing = I.LayerSpec("facing", I.P(tp, 1e-5, 1.5e-3), I.P(ep, 1.0, 4.0), I.P(tdp, 0.0, 0.3))
    m.facing_on = ("drywall_front", "drywall_back")
    return m


def test_facing_matches_realism_paper_and_is_invisible_when_core_material():
    import realism as RL
    f = np.concatenate([np.linspace(77e9, 81e9, 32), np.linspace(57e9, 64e9, 32)])
    m = _facing_model()
    G, _ = I.forward(m, f, False)
    L = RL.layers_from_truth(m.summary(), RL.Realism(dispersion="cq", paper=(4e-4, 2.0, 0.08)))
    ref = RL.gamma_layers(f, np.zeros((f.size, 1)), L)[0][:, 0] * np.exp(-2j * 2 * np.pi * f / W.C0 * 0.30)
    assert np.abs(G - ref).max() < 1e-12
    m0 = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=0.30); m0.law = "cq"
    m0.layers[0].tan_d.value = 0.035; m0.sync_ties()
    mi = _facing_model(ep=m0.layers[0].eps_r.value, tdp=0.035)
    assert np.abs(I.forward(mi, f, False)[0] - I.forward(m0, f, False)[0]).max() < 1e-12


def test_band_offset_zero_is_identity_and_full_jacobian():
    f = np.concatenate([np.linspace(77e9, 81e9, 32), np.linspace(57e9, 64e9, 32)])
    m0 = _facing_model()
    m = _facing_model()
    m.band_offset = I.P(0.0, -1e-3, 1e-3, True)
    assert np.abs(I.forward(m, f, False)[0] - I.forward(m0, f, False)[0]).max() < 1e-14
    m.band_offset.value = 2e-4
    m.rough = I.P(8e-5, 1e-6, 5e-4, True)
    for beam in (None, (0.02, 0.015)):
        mm = m.copy(); mm.beam = beam; mm.beam_K = 3
        G, J = I.forward(mm, f)
        assert J.shape[1] == len(mm.free_names()) and "facing.t" in mm.free_names() and "rough" in mm.free_names()
        worst = 0
        for j in range(len(mm.free_names())):
            errs = []
            for rel in (1e-5, 1e-6, 1e-7):
                h = rel * max(abs(mm.get()[j]), 1e-4)
                a = mm.copy(); v = a.get(); v[j] += h; a.set(v)
                b = mm.copy(); v = b.get(); v[j] -= h; b.set(v)
                fd = (I.forward(a, f, False)[0] - I.forward(b, f, False)[0]) / (2 * h)
                errs.append(np.abs(fd - J[:, j]).max() / max(np.abs(J[:, j]).max(), 1e-30))
            worst = max(worst, min(errs))
        assert worst < 1e-5, (beam, worst)


def test_rough_taper_matches_kirchhoff_to_second_order():
    """The graded taper reproduces the Kirchhoff coherent factor up to second order: with the
    n_bar-corrected thickness the best sigma scale is ~1 and the residual is a small fraction of the
    roughness effect for realistic sigma (<= 0.1 mm)."""
    import realism as RL
    from scipy.optimize import minimize_scalar
    f = np.concatenate([np.linspace(77e9, 81e9, 32), np.linspace(57e9, 64e9, 32)])
    m = S.stud_wall_model(free=("t",), standoff_guess=0.0); m.standoff.value = 0.0
    L = RL.layers_from_truth(m.summary(), RL.Realism())
    smooth = RL.gamma_layers(f, np.zeros((f.size, 1)), L)[0][:, 0]
    out = []
    for sig in (2e-5, 1e-4):
        kir = RL.gamma_layers(f, np.zeros((f.size, 1)), L, rough_sigma=sig)[0][:, 0]
        def mis(sc):
            mm = m.copy(); mm.rough = I.P(sig * sc, 1e-7, 1e-3, False)
            return np.abs(I.forward(mm, f, False)[0] - kir).max()
        r = minimize_scalar(mis, bounds=(0.3, 1.7), method="bounded")
        out.append((r.x, r.fun / np.abs(kir - smooth).max()))
    for scale, rel in out:
        assert 0.9 < scale < 1.1 and rel < 0.1, out


def test_optical_coords_with_facing_and_extras():
    f = np.concatenate([np.linspace(77e9, 81e9, 32), np.linspace(57e9, 64e9, 32)])
    m = _facing_model()
    m.rough = I.P(5e-5, 1e-6, 4e-4, True); m.band_offset = I.P(1e-4, -1e-3, 1e-3, True)
    C = SR.Coords(m, "optical")
    assert set(C.tidx.values()) == {0, 1, 2}                     # only real layers are reparameterised
    q = C.q_of(m)
    assert np.allclose(C.apply(m, q).get(), m.get(), rtol=1e-13, atol=0)
    G, Jp = I.forward(m, f)
    Jq = Jp @ C.dpdq(m)
    for j in range(len(q)):
        h = 1e-6 * max(abs(q[j]), 1e-4)
        qa, qb = q.copy(), q.copy(); qa[j] += h; qb[j] -= h
        fd = (I.forward(C.apply(m, qa), f, False)[0] - I.forward(C.apply(m, qb), f, False)[0]) / (2 * h)
        assert np.abs(fd - Jq[:, j]).max() / np.abs(Jq[:, j]).max() < 1e-5


def test_compiled_forward_matches_numpy_path():
    f = np.concatenate([np.linspace(77e9, 81e9, 64), np.linspace(57e9, 64e9, 64)])
    m = _facing_model(law="jonscher")
    m.layers[0].pinf = I.P(0.3, 0, 0.9, True); m.layers[2].pinf = I.P(0.3, 0, 0.9, True, tie="drywall_front.pinf")
    m.rough = I.P(5e-5, 1e-6, 4e-4, True); m.sync_ties()
    old = I.USE_COMPILED
    try:
        I.USE_COMPILED = False; G0, J0 = I.forward(m, f)
        I.USE_COMPILED = True; G1, J1 = I.forward(m, f)
    finally:
        I.USE_COMPILED = old
    assert np.abs(G1 - G0).max() < 1e-13 and np.abs(J1 - J0).max() / np.abs(J0).max() < 1e-12


def test_min_cells_is_exact():
    f = np.concatenate([np.linspace(77e9, 81e9, 64), np.linspace(3.1e9, 10.6e9, 64)])
    for td in (0.0, 0.03):
        m = _facing_model(law="jonscher"); m.layers[0].tan_d.value = td; m.sync_ties()
        mm = m.copy(); mm.min_cells = True
        G0, J0 = I.forward(m, f); G1, J1 = I.forward(mm, f)
        assert sum(sb[4] for sb in I._effective(mm, f)[0]) < sum(sb[4] for sb in I._effective(m, f)[0])
        assert np.abs(G1 - G0).max() < 1e-12 and np.abs(J1 - J0).max() / np.abs(J0).max() < 1e-11
