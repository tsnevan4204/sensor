"""
Validation suite for the exact-star inversion. Run:  python3 run_validation.py
Writes validation_log.txt and validation.png next to this file.

  0  correctness: forward == reference to machine precision; adjoint == finite differences
  1  L1 dry wall: Monte Carlo over noise; errors vs Cramer-Rao bound; global ambiguity
  2  identifiability: singular values / correlations for each scenario
  3  L2 wet insulation: recover moisture theta
  4  model mismatch: wrong water temperature, wrong insulation host eps
  5  L3 wicked moisture profile: smoothness vs total-variation regularization,
     lambda chosen by the discrepancy principle
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
import wall_toy as W
import wall_inverse as I
import scenarios as S

LOG = open("validation_log.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


# ----------------------------------------------------------------------------- shared pipeline
def acquire(scene, seed, plate_d=0.25):
    radar = W.Radar(seed=seed)
    f, y, info = W.measure(scene, radar)
    _, yp, _ = W.measure_metal_plate(radar, plate_d, seed=seed + 500)
    yc, _ = I.calibrate(y, yp, f, plate_d)
    sigma = info["noise_sigma"] * np.sqrt(1 + np.mean(np.abs(yc) ** 2))   # wall + plate noise
    r, p = W.range_profile(f, yc)
    sel = (r > 0.2) & (r < 0.4)
    d0 = r[sel][np.argmax(p[sel])]                                       # front-face guess
    return f, yc, sigma, d0


def HALF(a, b):
    """Start offsets every half phase period: every basin gets a start within 1/4 period."""
    return np.arange(2 * a, 2 * b + 1) / 2


def stages_l1(d0):
    return [I.Stage(("standoff", "drywall_front.t", "drywall_front.eps_r"), I.Gate(d0 - 0.03, d0 + 0.06),
                    lattice={"drywall_front.t": HALF(-4, 4), "drywall_front.eps_r": [2.0, 2.6, 3.2]}),
            I.Stage(("insulation.t", "insulation.eps_r", "drywall_back.t"), I.Gate(d0 + 0.06, d0 + 0.25),
                    lattice={"insulation.t": HALF(-6, 6), "insulation.eps_r": [1.05, 1.15, 1.3]})]


def run_l1(seed):
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    scene, tv = S.truth_random(rng, d_true)
    f, yc, sigma, d0 = acquire(scene, seed)
    m0 = S.stud_wall_model(standoff_guess=d0)
    best, cost = I.strip_invert(m0, f, yc, sigma, stages_l1(d0))
    amb, best, cost = I.ambiguity_scan(best, f, yc, sigma)
    fi = I.fisher(best, f, sigma)
    est = best.summary()
    return dict(seed=seed, chi2=cost, n_data=2 * f.size,
                err={n: est[n] - tv[n] for n in fi["names"]},
                crb=dict(zip(fi["names"], fi["crb_std"])),
                amb={n: [(v, c) for v, c in lst] for n, lst in amb.items()},
                truth={n: tv[n] for n in fi["names"]})


def wet_model(d0, T=20.0, host_eps=1.15, front_known=False):
    """front_known=True: the front drywall's t and eps come from a scan of a dry patch of the
    same wall (a differential measurement), so only standoff and moisture are free."""
    m = S.stud_wall_model(standoff_guess=d0, theta_ins=(0.0, 0.3))
    m.water_T = T
    for n, _, p in m.items():           # back of the wall is invisible when wet: fix it
        if n.startswith("insulation.") and n != "insulation.theta":
            p.free = False
        if n.startswith("drywall_back."):
            p.free = False
        if front_known and n in ("drywall_front.t", "drywall_front.eps_r"):
            p.free = False
    m.layers[1].eps_r.value = host_eps
    return m


def stages_l2_known(d0):
    return [I.Stage(("standoff", "insulation.theta"), None,
                    lattice={"standoff": np.arange(-2, 2.01, 0.25), "insulation.theta": [0.0, 0.03, 0.08, 0.15, 0.25]})]


def stages_l2(d0):
    # moisture sets how strongly the drywall/insulation boundary reflects, so it belongs
    # in the same stage as the front-drywall echo pair
    return [I.Stage(("standoff", "drywall_front.t", "drywall_front.eps_r", "insulation.theta"),
                    I.Gate(d0 - 0.03, d0 + 0.06),
                    lattice={"drywall_front.t": HALF(-4, 4), "drywall_front.eps_r": [2.0, 2.6, 3.2],
                             "insulation.theta": [0.0, 0.05, 0.15]})]


def wet_scene(theta, d, T=20.0, host_eps=1.15):
    host = lambda f: W.eps_dielectric(host_eps, 0.01, f)
    dry = W.nominal_materials()["drywall"]
    return W.Scene([W.Layer("drywall", 0.0127, lambda f: W.eps_dielectric(**dry, f=f)),
                    W.Layer("wet ins", 0.0889, lambda f: W.eps_crim(host(f), theta, f, T_C=T)),
                    W.Layer("drywall", 0.0127, lambda f: W.eps_dielectric(**dry, f=f))], standoff=d)


def run_l2(args):
    seed, theta, T_true, host_true, front_known = args
    d_true = 0.30 + 0.002 * np.random.default_rng(seed).random()
    f, yc, sigma, d0 = acquire(wet_scene(theta, d_true, T_true, host_true), seed)
    m0 = wet_model(d0, front_known=front_known)   # always assumes 20 C water, host 1.15
    st = stages_l2_known(d0) if front_known else stages_l2(d0)
    best, cost = I.strip_invert(m0, f, yc, sigma, st, final_hop=not front_known)
    fi = I.fisher(best, f, sigma)
    i = fi["names"].index("insulation.theta")
    return dict(seed=seed, theta_true=theta, theta_est=best.summary()["insulation.theta"],
                crb=fi["crb_std"][i], chi2=cost, n_data=2 * f.size, T_true=T_true, host_true=host_true,
                front_known=front_known)


def theta_profile(seed=1, front_known=False, grid=np.linspace(0.0, 0.3, 16)):
    """Profile likelihood: min chi2 over all other free parameters, with theta held at each grid value."""
    d_true = 0.30 + 0.002 * np.random.default_rng(seed).random()
    f, yc, sigma, d0 = acquire(wet_scene(0.05, d_true), seed)
    m = wet_model(d0, front_known=front_known)
    m.layers[1].theta.free = False
    st = [I.Stage(("standoff", "drywall_front.t", "drywall_front.eps_r"), I.Gate(d0 - 0.03, d0 + 0.06),
                  lattice={"drywall_front.t": HALF(-4, 4), "drywall_front.eps_r": [2.0, 2.6, 3.2]})] \
        if not front_known else [I.Stage(("standoff",), None, lattice={"standoff": np.arange(-2, 2.01, 0.25)})]
    out = []
    for th in grid:
        mm = m.copy(); mm.layers[1].theta.value = th
        b, c = I.strip_invert(mm, f, yc, sigma, st, final_hop=False)
        out.append(c)
    return grid, np.array(out)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    results = {}
    f = W.Radar().freqs

    # 0 -------------------------------------------------------------------- correctness
    log("== 0. correctness")
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), theta_ins=(0, 0.3))
    m.layers[1].theta.value = 0.0
    G, _ = I.forward(m, f, jacobian=False)
    log(f"forward vs reference transfer matrix, dry wall: max |dGamma| = {np.abs(G - W.reflection_tmm(f, W.scene_dry_wall())).max():.1e}")
    m.layers[1].theta.value = 0.002
    G, J = I.forward(m, f)
    worst = 0
    for j, nm in enumerate(m.free_names()):
        h = 1e-6 * max(abs(m.get()[j]), 1e-3)
        a = m.copy(); v = a.get(); v[j] += h; a.set(v)
        b = m.copy(); v = b.get(); v[j] -= h; b.set(v)
        fd = (I.forward(a, f, False)[0] - I.forward(b, f, False)[0]) / (2 * h)
        worst = max(worst, np.abs(fd - J[:, j]).max() / np.abs(J[:, j]).max())
    log(f"adjoint Jacobian vs central finite differences ({len(m.free_names())} params): worst relative diff = {worst:.1e}")

    # 2 -------------------------------------------------------------------- identifiability
    log("\n== 2. identifiability at the true wall (relative singular values of the scaled Jacobian)")
    sig_ref = np.sqrt(1 / 1e4 / 2)
    for label, mm in [("dry, all free (drywall eps untied)", S.stud_wall_model(standoff_guess=0.30, same_drywall=False)),
                      ("dry, drywall eps shared", S.stud_wall_model(standoff_guess=0.30)),
                      ("wet 5%, front + theta only", None)]:
        if mm is None:
            mm = wet_model(0.30); mm.layers[1].theta.value = 0.05
        fi = I.fisher(mm, f, sig_ref)
        sv = fi["singular_values"]
        log(f"{label}: singular values / max = {np.array2string(sv / sv.max(), precision=1)}")
        log("   CRB std: " + ", ".join(f"{n}={s:.1e}" for n, s in zip(fi["names"], fi["crb_std"])))
        c = fi["corr"]; iu = np.triu_indices_from(c, 1)
        k = np.argmax(np.abs(c[iu]))
        log(f"   strongest correlation: {fi['names'][iu[0][k]]} vs {fi['names'][iu[1][k]]} = {c[iu][k]:+.3f}")
    mm = S.stud_wall_model(standoff_guess=0.30, theta_ins=(0, 0.3)); mm.layers[1].theta.value = 0.05
    for n, _, p in mm.items():
        if n.startswith("drywall_front") or n == "standoff":
            p.free = False
    _, Jw = I.forward(mm, f)
    log("   wet 5%: |dGamma/dp| for the hidden parameters: " +
        ", ".join(f"{n}={np.abs(Jw[:, j]).max():.0e}" for j, n in enumerate(mm.free_names())))

    # 1 -------------------------------------------------------------------- L1 Monte Carlo
    log("\n== 1. L1 dry wall, drywall material shared, Monte Carlo over noise")
    seeds = list(range(1, 11))
    with Pool(2) as pool:
        l1 = pool.map(run_l1, seeds)
    names = list(l1[0]["err"])
    for n in names:
        z = np.array([r["err"][n] / r["crb"][n] for r in l1])
        log(f"{n:22s} mean|err| {np.mean([abs(r['err'][n]) for r in l1]):.2e}   CRB {np.mean([r['crb'][n] for r in l1]):.1e}"
            f"   err/CRB: mean {z.mean():+.2f} std {z.std():.2f}  max|.| {np.abs(z).max():.1f}")
    log("chi2 / n_data per run: " + ", ".join(f"{r['chi2'] / r['n_data']:.2f}" for r in l1))
    for n in ("drywall_front.t", "insulation.t", "drywall_back.t"):
        sizes = [len(r["amb"][n]) for r in l1]
        has_truth = [any(abs(v - r["truth"][n]) < 3e-4 for v, _ in r["amb"][n]) for r in l1]
        log(f"{n:22s} plausible basins (dchi2<4) per run: {sizes}   truth inside plausible set: {sum(has_truth)}/{len(l1)}")
    results["l1"] = l1

    # 3, 4 ----------------------------------------------------------------- L2 + mismatch
    log("\n== 3. L2 wet insulation (theta = 0.05), back of wall fixed (invisible)")
    jobs = [(s, 0.05, 20.0, 1.15, False) for s in range(1, 5)]   # front drywall unknown
    jobs += [(s, 0.05, 20.0, 1.15, True) for s in range(1, 9)]   # front known (dry-patch scan)
    jobs += [(s, 0.05, 30.0, 1.15, True) for s in range(1, 5)]   # water warmer than assumed
    jobs += [(s, 0.05, 20.0, 1.30, True) for s in range(1, 5)]   # insulation host eps wrong
    with Pool(2) as pool:
        l2 = pool.map(run_l2, jobs)
    for label, sel in [("front drywall unknown (single scan)", lambda r: not r["front_known"]),
                       ("front drywall known (dry-patch scan first)", lambda r: r["front_known"] and r["T_true"] == 20 and r["host_true"] == 1.15),
                       ("== 4. mismatch: water at 30 C, model assumes 20 C", lambda r: r["T_true"] == 30),
                       ("== 4. mismatch: host eps 1.30, model assumes 1.15", lambda r: r["host_true"] == 1.30)]:
        rr = [r for r in l2 if sel(r)]
        e = np.array([r["theta_est"] - r["theta_true"] for r in rr])
        log(f"   per run theta: " + ", ".join(f"{r['theta_est']:.3f}" for r in rr))
        log(f"{label}: theta est {np.mean([r['theta_est'] for r in rr]):.4f} (true 0.05)  bias {e.mean():+.4f}"
            f"  spread {e.std():.4f}  CRB {np.mean([r['crb'] for r in rr]):.4f}  chi2/n {np.mean([r['chi2'] / r['n_data'] for r in rr]):.2f}")
    results["l2"] = l2
    with Pool(2) as pool:
        prof_free, prof_known = pool.starmap(theta_profile, [(1, False), (1, True)])
    g = prof_free[0]
    log("profile chi2(theta) - min, front unknown: " + ", ".join(f"{t:.2f}:{c:.1f}" for t, c in zip(g, prof_free[1] - prof_free[1].min())))
    log("profile chi2(theta) - min, front known  : " + ", ".join(f"{t:.2f}:{c:.1f}" for t, c in zip(g, prof_known[1] - prof_known[1].min())))
    results["theta_profile"] = dict(theta=g.tolist(), free=prof_free[1].tolist(), known=prof_known[1].tolist())

    # 5 -------------------------------------------------------------------- L3 profile
    log("\n== 5. L3 wicked moisture profile in the front drywall (geometry and dry materials known)")
    K = 8
    truth_scene = W.scene_wicked_drywall(theta_max=0.15, n_sub=40, decay=0.004)
    truth_scene.standoff = 0.3010
    f, yc, sigma, d0 = acquire(truth_scene, seed=7)
    dw = S.DRY
    lay = [I.LayerSpec(f"dw{i}", I.P(0.0127 / K, 0, 1, False), I.P(dw["eps_r"], 1, 4, False),
                       I.P(dw["tan_d"], 0, 0.2, False), I.P(0.02, 0.0, 0.3, True)) for i in range(K)]
    lay += [I.LayerSpec("ins", I.P(0.0889, 0, 1, False), I.P(1.15, 1, 4, False), I.P(0.01, 0, 1, False)),
            I.LayerSpec("dwb", I.P(0.0127, 0, 1, False), I.P(2.6, 1, 4, False), I.P(0.03, 0, 1, False))]
    m3 = I.WallModel(lay, standoff=I.P(d0, d0 - 0.003, d0 + 0.003, True))
    m3, _, _ = I.lattice_invert(m3, f, yc, sigma, {"standoff": [0]}, gate=None)  # quick standoff fit
    names = [f"dw{i}.theta" for i in range(K)]
    zc = (np.arange(K) + 0.5) * 0.0127 / K
    th_true = 0.15 * np.exp(-(0.0127 - zc) / 0.004)
    fi3 = I.fisher(m3, f, sigma)
    sv = fi3["singular_values"]
    log(f"singular values of the {K}-cell theta profile (+standoff), / max: {np.array2string(sv / sv.max(), precision=1)}")
    lams = np.logspace(0, 7, 8)
    prof = {}
    for kind in ("smooth", "tv"):
        mk = lambda lam, kind=kind: [I.Regularizer(names, lam, kind, beta=1e-3)]
        (lam, chi2, mbest), table, target = I.choose_lambda(m3, f, yc, sigma, mk, lams, n_starts=3, rng=0)
        th = np.array([mbest.summary()[n] for n in names])
        prof[kind] = th
        log(f"{kind:6s}: lambda chosen {lam:.0e} (chi2 {chi2:.0f}, target {target})  "
            f"profile rms error {np.sqrt(np.mean((th - th_true) ** 2)):.4f}  mean water true {th_true.mean():.4f} est {th.mean():.4f}")
        log("        lambda -> chi2: " + ", ".join(f"{l:.0e}:{c:.0f}" for l, c, _ in table))
    log(f"true profile: {np.array2string(th_true, precision=3)}")
    for k, v in prof.items():
        log(f"{k:6s} est : {np.array2string(v, precision=3)}")
    results["l3"] = dict(z=zc.tolist(), truth=th_true.tolist(), **{k: v.tolist() for k, v in prof.items()})

    # ---------------------------------------------------------------------- figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    names = list(l1[0]["err"])
    for j, n in enumerate(names):
        z = [r["err"][n] / r["crb"][n] for r in l1]
        ax[0].scatter([j] * len(z), z, s=18)
    ax[0].axhspan(-2, 2, color="0.85", zorder=0)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n.replace("drywall_", "dw_") for n in names], rotation=35, ha="right")
    ax[0].set(ylabel="error / CRB std", title="L1: error in units of the CRB (grey: ±2)")
    ax[0].set_yscale("symlog", linthresh=3)
    r0 = l1[0]
    for n, c in zip(("drywall_front.t", "insulation.t", "drywall_back.t"), ("C0", "C1", "C2")):
        pts = sorted(r0["amb"][n])
        ax[1].plot([v * 1e3 for v, _ in pts], [ch - r0["chi2"] for _, ch in pts], "o-", color=c, label=n)
    ax[1].axhline(4, color="k", ls=":", lw=1)
    ax[1].set(xlabel="thickness (mm)", ylabel="Δχ² vs best", title="L1: plausible basins (Δχ² < 4), seed 1")
    ax[1].legend(fontsize=8)
    zz = np.array(results["l3"]["z"]) * 1e3
    ax[2].plot(zz, th_true, "k-o", label="truth (cell average)")
    ax[2].plot(zz, prof["smooth"], "s--", label="smoothness")
    ax[2].plot(zz, prof["tv"], "^--", label="total variation")
    ax[2].set(xlabel="depth in front drywall (mm)", ylabel="water fraction θ", title="L3: recovered moisture profile")
    ax[2].legend(fontsize=8)
    tp = results["theta_profile"]
    ax[3].plot(tp["theta"], np.array(tp["free"]) - min(tp["free"]), "o-", label="front drywall unknown")
    ax[3].plot(tp["theta"], np.array(tp["known"]) - min(tp["known"]), "s-", label="front known (dry-patch scan)")
    ax[3].axvline(0.05, color="k", lw=1); ax[3].axhline(4, color="k", ls=":", lw=1)
    ax[3].set(xlabel="insulation water fraction θ", ylabel="min χ² − best", title="L2: profile likelihood of θ", ylim=(-2, 80))
    ax[3].legend(fontsize=8)
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("validation.png", dpi=150)
    json.dump(results, open("validation_results.json", "w"), default=float, indent=1)
    log(f"\ntotal time {time.time() - t0:.0f}s")
