"""S001 part A: numerical checks of the 1D theorems (docs/11, Part A). Writes to results/2026-11-09_S001_stability/."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from oned_stability import *

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "2026-11-09_S001_stability")
os.makedirs(OUT, exist_ok=True)
R = {}
rng = np.random.default_rng(2026_11_09)
t0 = time.time()

# --- A1/A3: random walls, exact stars --------------------------------------------------------
ratios, PoverPi, mesh_err, nwalls, nlossless = [], [], [], 400, 0
for t in range(nwalls):
    lossless = t % 2 == 0
    L = random_wall(rng, lossless)
    w = 2 * np.pi * rng.uniform(1e9, 100e9)
    # cell rule: default |k|l <= 0.9 pi, or (every 4th wall) one cell per layer, which allows
    # |k|l in (pi, 2 pi) etc. (exact star not positive there, but still exact and passive)
    cpl = [1] * len(L) if t % 4 == 1 else None
    eps, ell = wall_cells(L, w, cpl)
    x = w / C0 * nbranch(eps) * ell
    if np.min(np.abs(np.sin(x))) < 1e-3:
        continue
    b, M = exact_stars(eps, ell, w)
    assert passivity_margin(b, M) >= -1e-12 * np.abs(M).max()
    r, P, Ai = green_bound_ratio(b, M, 1 / ETA0, 1 / ETA0)
    ratios.append(r)
    if lossless:
        nlossless += 1
        etas = [ETA0] + [ETA0 / np.sqrt(e.real) for (_, e) in L] + [ETA0]
        PoverPi.append(P / interface_product(etas))
    # mesh independence: split every cell in two -> coarse inverse is a principal submatrix
    eps2 = np.repeat(eps, 2); ell2 = np.repeat(ell / 2, 2)
    b2, M2 = exact_stars(eps2, ell2, w)
    Ai2 = np.linalg.inv(assemble(b2, M2, 1 / ETA0, 1 / ETA0))
    mesh_err.append(np.abs(Ai2[::2, ::2] - Ai).max() / np.abs(Ai).max())
R["A_random_walls"] = dict(n=len(ratios), n_lossless=nlossless, max_ratio_to_bound=max(ratios),
                           median_ratio=float(np.median(ratios)),
                           P_over_interface_product_range=[min(PoverPi), max(PoverPi)],
                           mesh_independence_max_rel_err=max(mesh_err))

# homogeneous matched line: A^-1 = eta0/2 exp(-jk|z-z'|) exactly
w = 2 * np.pi * 30e9
eps, ell = wall_cells([(0.1, 1.0)], w, [17])
b, M = exact_stars(eps, ell, w)
Ai = np.linalg.inv(assemble(b, M, 1 / ETA0, 1 / ETA0)); z = node_positions(ell)
R["A_homogeneous_green_rel_err"] = float(np.abs(Ai - ETA0 / 2 * np.exp(-1j * w / C0 * np.abs(z[:, None] - z[None]))).max() / ETA0)

# --- k-independence sweep (lossless wall) ----------------------------------------------------
L = [(12.7e-3, 2.8), (40e-3, 1.0), (89e-3, 1.4), (12.7e-3, 2.8)]
fs = np.linspace(0.5e9, 200e9, 800)
gmax, Pex, gstd, Pstd = [], [], [], []
for f in fs:
    w = 2 * np.pi * f
    eps, ell = wall_cells(L, w)
    b, M = exact_stars(eps, ell, w)
    r, P, Ai = green_bound_ratio(b, M, 1 / ETA0, 1 / ETA0)
    gmax.append(np.abs(Ai).max() / ETA0); Pex.append(P)
    eps, ell = wall_cells(L, w, max_kl=0.3)               # standard stars at k l <= 0.3
    b, M = std_stars(eps, ell, w)
    r2, P2, Ai2 = green_bound_ratio(b, M, 1 / ETA0, 1 / ETA0)
    gstd.append(np.abs(Ai2).max() / ETA0); Pstd.append(P2)
etas = [ETA0] + [ETA0 / np.sqrt(e) for (_, e) in L] + [ETA0]
R["A_k_sweep"] = dict(f_GHz=[fs[0] / 1e9, fs[-1] / 1e9], max_g_over_eta0_exact=max(gmax),
                      P_exact_range=[min(Pex), max(Pex)], interface_product=interface_product(etas),
                      max_g_over_eta0_std=max(gstd), P_std_range=[min(Pstd), max(Pstd)])

# --- learned passive factors (random complex, Re Y+- >= 0 by construction) -------------------
worst, smin, cnt = 0, np.inf, 0
for t in range(400):
    L = random_wall(rng, lossless=(t % 2 == 0))
    w = 2 * np.pi * rng.uniform(1e9, 100e9)
    eps, ell = wall_cells(L, w)
    x = w / C0 * nbranch(eps) * ell
    se, me = exact_factors(x)
    n = eps.size
    # perturb the exact factors by random passive amounts: Y+ += jw eps l/2 * mr*(-j a), etc.
    a = rng.uniform(0, 0.5, n) * (rng.random(n) < 0.5)
    c = rng.uniform(0, 0.5, n) * (rng.random(n) < 0.5)
    m = me * rng.uniform(0.3, 3, n) - 1j * a          # Re Y+ += (w eps' l/2) a >= 0 (lossless eps)
    s = se * rng.uniform(0.3, 3, n) + 1j * c          # Re b  += c/(w mu l) >= 0
    b, M = learned_stars(eps, ell, w, s, m)
    if passivity_margin(b, M) < 0:                      # lossy eps can rotate the sign; skip those
        continue
    cnt += 1
    r, P, Ai = green_bound_ratio(b, M, 1 / ETA0, 1 / ETA0)
    worst = max(worst, r)
    smin = min(smin, np.linalg.svd(assemble(b, M, 1 / ETA0, 1 / ETA0), compute_uv=False).min() * ETA0)
R["A_learned_passive"] = dict(n=cnt, max_ratio_to_bound=worst, min_sigma_min_times_eta0=smin)

# lossless passband learned factors: P equals the product of effective-impedance ratios
devs = []
for t in range(200):
    L = random_wall(rng, lossless=True)
    w = 2 * np.pi * rng.uniform(1e9, 100e9)
    eps, ell = wall_cells(L, w)
    n = eps.size
    s = rng.uniform(0.7, 1.4, n)
    m = rng.uniform(0.7, 1.4, n)
    b, M = learned_stars(eps, ell, w, s, m)
    Yp, Ym = modal(b, M)
    if not np.all(Yp.imag * Ym.imag < 0):
        continue
    ee = eta_eff(b, M).real
    Pnum, _, _ = stability_constant(b, M, 1 / ETA0, 1 / ETA0, eta_ref=np.concatenate([[ETA0], ee[1:], [ETA0]]))
    # reference at node c = eta_eff of the cell to its right: every lossless passband cell is
    # unitary in its own impedance, so P <= prod of eta_eff mismatch ratios (Corollary A4)
    etas = np.concatenate([[ETA0], ee, [ETA0]])
    devs.append(Pnum / interface_product(etas))
    Ai = np.linalg.inv(assemble(b, M, 1 / ETA0, 1 / ETA0))
    assert np.abs(Ai).max() <= 2 * max(etas.max(), ETA0) * interface_product(etas) * (1 + 1e-10)
R["A_passband_P_over_eta_eff_ratio_product_range"] = [float(min(devs)), float(max(devs))]
R["A_passband_n"] = len(devs)

# --- active counterexample: one cell shifted by delta (both modes) ----------------------------
max_re_delta, min_gain = -np.inf, np.inf
for t in range(200):
    L = random_wall(rng, lossless=True)
    w = 2 * np.pi * rng.uniform(1e9, 100e9)
    eps, ell = wall_cells(L, w)
    b, M = exact_stars(eps, ell, w)
    A = assemble(b, M, 1 / ETA0, 1 / ETA0)
    Ai = np.linalg.inv(A)
    c = rng.integers(eps.size)
    blk = Ai[np.ix_([c, c + 1], [c, c + 1])]
    deltas = -1 / np.linalg.eigvals(blk)              # det(A + delta E_c) = 0
    for d in deltas:
        M2 = M.copy(); M2[c] += d                       # Y+ and Y- of cell c both shift by d
        assert abs(np.linalg.det(assemble(b, M2, 1 / ETA0, 1 / ETA0)) / np.linalg.det(A)) < 1e-8
        max_re_delta = max(max_re_delta, d.real * ETA0)
        min_gain = min(min_gain, -d.real * ETA0)
R["A_active_roots"] = dict(max_Re_delta_times_eta0=max_re_delta, min_gain_needed_times_eta0=min_gain)

# --- pole sin(k l) = 0: one cell per layer, k l sweeping through pi ---------------------------
xs = np.linspace(0.5, 5.5, 2001)
nA, nAi = [], []
t_layer, er = 0.03, 4.0
for xx in xs:
    w = xx / t_layer / np.sqrt(er) * C0
    b, M = exact_stars([er], [t_layer], w)
    A = assemble(b, M, 1 / ETA0, 1 / ETA0)
    nA.append(np.linalg.norm(A, 2) * ETA0); nAi.append(np.linalg.norm(np.linalg.inv(A), 2) / ETA0)
R["A_pole"] = dict(max_normAinv_over_eta0=max(nAi), max_normA_eta0=max(nA))

# --- Re b of the exact star can be negative while the cell is passive --------------------------
x = np.linspace(0.05, 3.0, 600) * np.sqrt(1 - 0.6j)
x = np.where(x.imag > 0, -x, x)
bexact = -1j * x / np.sin(x)                      # b * (w mu l), mu real
Yp = 1j * x * np.tan(x / 2); Ym = -1j * x / np.tan(x / 2)   # times (w mu l)/... same positive factor
R["A_Re_b_negative_but_passive"] = dict(min_Re_b_rel=float((bexact.real / np.abs(bexact)).min()),
                                       min_Re_Yplus=float(Yp.real.min()), min_Re_Yminus=float(Ym.real.min()))

# --- standard star effective impedance and passband edge k l = 2 -------------------------------
kl = np.array([0.1, 0.5, 1.0, 1.5, 1.9])
w = 2 * np.pi * 10e9; k = w / C0
b, M = std_stars(np.ones(kl.size), kl / k, w)
ee = eta_eff(b, M).real
R["A_std_eta_eff_formula_maxrelerr"] = float(np.abs(ee / (ETA0 / np.sqrt(1 - (kl / 2) ** 2)) - 1).max())
b, M = std_stars([1.0], [2.1 / k], w); Yp, Ym = modal(b, M)
R["A_std_stopband_at_kl_2.1"] = bool(Yp.imag[0] * Ym.imag[0] > 0)
R["A_wall_time_s"] = time.time() - t0

# --- figure -----------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fs / 1e9, gmax, label=r"exact stars: $\max_{ij}|A^{-1}_{ij}|/\eta_0$")
ax[0].plot(fs / 1e9, 2 * np.array(Pex) * max(etas) / ETA0, "--", label=r"bound $2\eta_{max}P/\eta_0$ (exact)")
ax[0].plot(fs / 1e9, gstd, label=r"standard stars ($k\ell\leq0.3$)")
ax[0].set_xlabel("f [GHz]"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[0].set_title("1D: discrete Green's function is bounded uniformly in k")
ax[1].semilogy(xs, nAi, label=r"$\|A^{-1}\|_2/\eta_0$")
ax[1].semilogy(xs, nA, label=r"$\|A\|_2\,\eta_0$")
ax[1].axvline(np.pi, color="k", lw=0.5); ax[1].axvline(2 * np.pi, color="k", lw=0.5)
ax[1].set_xlabel(r"$k\ell$ (one exact cell)"); ax[1].legend(); ax[1].set_title(r"pole at $\sin k\ell=0$: conditioning, not stability")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_A_1d.png"), dpi=130)

with open(os.path.join(OUT, "results_1d.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
