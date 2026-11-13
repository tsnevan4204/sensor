"""E070 gates: the restricted three-layer problem (tri3.py)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/oned"))
import numpy as np
import tri3 as T
import stack as ST
import polish as PL
import wall_inverse as I
import treesolve as TS
import pytest


@pytest.fixture(autouse=True, scope="module")
def _five_class_world():
    T.install()
    yield
    T.uninstall()


def test_sequence_counts():
    assert len(T.SEQ_ALL) == 80 and len(T.SEQ_IN) == 30 and len(T.SEQ_OUT) == 50
    assert all(s[0] in T.BOARDS and s[2] in T.BOARDS for s in T.SEQ_IN)
    assert all(a != b for s in T.SEQ_ALL for a, b in zip(s[:-1], s[1:]))


def test_material_ranges():
    rng = np.random.default_rng(0)
    rngs = {"gypsum": (2.5, 2.96), "wood_board": (2.35, 2.95), "fibrous": (1.07, 1.17), "foam": (1.029, 1.071),
            "tile": (5.69, 6.31)}
    for nm, (lo, hi) in rngs.items():
        e10 = [T.material_law(nm, rng)[1]["eps10"] for _ in range(300)]
        assert lo <= min(e10) and max(e10) <= hi, nm
    # library and priors carry exactly the five classes
    assert set(ST.library_stats()) == set(T.CLASSES)
    lib = TS.Library(np.array([5e9]))
    assert [n for n in lib.names if n not in ("water", "air", "metal")] == list(T.CLASSES)


def test_kernel_parity():
    """polish kernel (compiled) and exact model (search.lm's forward) == stack.gamma_stack for walls
    whose materials are drawn from the five class laws (dispersion shape included)."""
    rng = np.random.default_rng(1)
    f = np.concatenate([np.linspace(3.1e9, 10.6e9, 64), np.linspace(60e9, 64e9, 32)])
    k0 = 2 * np.pi * f / T.C0
    for _ in range(20):
        seq = T.SEQ_ALL[rng.integers(80)]
        lay, kind, er, td, pp, t = [], np.zeros(12, np.int64), np.ones(12), np.zeros(12), np.zeros(12), np.zeros(12)
        spec = []
        for j, nm in enumerate(seq):
            e, ds = T.material_law(nm, rng)
            th = rng.uniform(*T.bounds(nm, "stress"))
            lay.append((th, e)); er[j], td[j], pp[j], t[j] = ds["eps10"], ds["tan10"], ds["p"], th
            spec.append(dict(t=th, er=ds["eps10"], td=ds["tan10"], water=False, pinf=min(ds["p"], 0.999)))
        d = rng.uniform(0.25, 0.35)
        G0 = ST.gamma_stack(f, lay, "air") * np.exp(-2j * k0 * d)
        G1 = PL.gamma_one(kind, er, td, pp, t, 3, 2, d, 20.0, f, k0)
        assert np.max(np.abs(G1 - G0)) < 1e-10
        ST.SBOX = (d - 0.004, d + 0.004)
        m = ST.build(spec, d, back="air")
        ST.SBOX = None
        G2, _ = I.forward(m, f, jacobian=False)
        if max(s["pinf"] for s in spec) < 0.999:
            assert np.max(np.abs(G2 - G0)) < 1e-10


def test_bounds_enforced_and_labels_fixed():
    f, y, s, tr = T.make3(11)
    r = T.solve3(f, y, s, mode="fast")
    assert r["in_bounds"]
    for nm, t in zip(r["seq"], r["t"]):
        lo, hi = T.bounds(nm, "real")
        assert lo - 1e-9 <= t <= hi + 1e-9
    assert r["d"] >= T.D_BOX[0] and r["d"] <= T.D_BOX[1]
