"""Gates for src/oned/realism.py (doc 12 reality-gap generator)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "oned"))
import realism as RL          # noqa: E402
import benchmark as B         # noqa: E402
import wall_toy as W          # noqa: E402


def test_all_off_reproduces_benchmark_v2dual():
    for seed in (1, 7):
        f, y, s, d0, tv = RL.make_case_real(seed, RL.Realism(), dual=True)
        f2, y2, s2, d02, tv2 = B.make_case("v2dual", seed)
        assert np.allclose(f, f2) and np.abs(y - y2).max() < 1e-12 and abs(s - s2) < 1e-15 and d0 == d02


def test_normal_incidence_tmm_matches_reference():
    f = np.linspace(22e9, 81e9, 200)
    _, tv = B.S.truth_random_v2(np.random.default_rng(3), 0.3)
    tv["drywall_back.eps_r"], tv["drywall_back.tan_d"] = tv["drywall_front.eps_r"], tv["drywall_front.tan_d"]
    sc, _ = B.S.truth_random_v2(np.random.default_rng(3), 0.3)
    L = RL.layers_from_truth(tv, RL.Realism())
    GTE, GTM = RL.gamma_layers(f, np.zeros((f.size, 1)), L)
    ref = W.reflection_tmm(f, sc, include_standoff=False)
    assert np.abs(GTE[:, 0] - ref).max() < 1e-12 and np.abs(GTM[:, 0] - ref).max() < 1e-12


def test_wide_beam_tends_to_plane_wave():
    f = np.linspace(77e9, 81e9, 64)
    _, tv = B.S.truth_random_v2(np.random.default_rng(3), 0.3)
    tv["drywall_back.eps_r"], tv["drywall_back.tan_d"] = tv["drywall_front.eps_r"], tv["drywall_front.tan_d"]
    L = RL.layers_from_truth(tv, RL.Realism())
    pw, _, _ = RL.received(f, L, 0.3, RL.Realism())
    errs = [np.abs(RL.received(f, L, 0.3, RL.Realism(beam_w0=w0, n_quad=96))[0] - pw).max() for w0 in (0.2, 1.0)]
    assert errs[1] < errs[0] and errs[1] < 2e-3


def test_constant_q_exact_at_reference_and_passive():
    f = np.array([RL.F_REF, 24e9, 60e9])
    e = RL.eps_material(2.6, 0.03, f, "cq", "drywall")
    assert abs(e[0] - 2.6 * (1 - 0.03j)) < 1e-12
    assert np.all(e.imag < 0) and np.allclose(-e.imag / e.real, 0.03, rtol=1e-12)   # constant loss angle
    assert e[1].real > e[0].real                                                    # causal dispersion


def test_paper_with_core_material_is_invisible():
    f = np.linspace(77e9, 81e9, 32)
    _, tv = B.S.truth_random_v2(np.random.default_rng(2), 0.3)
    tv["drywall_back.eps_r"], tv["drywall_back.tan_d"] = tv["drywall_front.eps_r"], tv["drywall_front.tan_d"]
    R0, R1 = RL.Realism(), RL.Realism(paper=(4e-4, tv["drywall_front.eps_r"], tv["drywall_front.tan_d"]))
    a = RL.gamma_layers(f, np.zeros((f.size, 1)), RL.layers_from_truth(tv, R0))[0]
    b = RL.gamma_layers(f, np.zeros((f.size, 1)), RL.layers_from_truth(tv, R1))[0]
    assert np.abs(a - b).max() < 1e-12
