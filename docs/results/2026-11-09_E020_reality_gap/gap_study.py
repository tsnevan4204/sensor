"""E020 reality-gap study: run the current best methods on data from realism.py with ONE effect
switched on at a time (and all together), and measure (i) whether chi2 flags it, (ii) the bias of
each estimate in units of its own CRB. Magnitudes are deliberately moderate, real-world-typical."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np
import realism as RL, benchmark as B, wall_inverse as I

PAPER = (0.4e-3, 2.0, 0.08)        # face paper: ~0.4 mm, cellulose eps ~2, tan_d ~0.05-0.1 (assumed)
EFFECTS = {
    "none (today's toy)": RL.Realism(),
    "causal dispersion (constant-Q)": RL.Realism(dispersion="cq"),
    "ITU-style power-law loss": RL.Realism(dispersion="itu"),
    "Gaussian beam w0 = 2 cm": RL.Realism(beam_w0=0.02),
    "Gaussian beam w0 = 1 cm": RL.Realism(beam_w0=0.01),
    "antenna feedback Hf = 0.1 (-20 dB)": RL.Realism(Hf=0.1),
    "plate distance error +0.5 mm": RL.Realism(plate_err=0.5e-3),
    "surface roughness 0.1 mm rms": RL.Realism(rough_sigma=1e-4),
    "drywall face paper 0.4 mm": RL.Realism(paper=PAPER),
    "insulation air gap 5 mm": RL.Realism(air_gap=5e-3),
    "cal-to-scan drift 0.05 dB / 0.01 rad": RL.Realism(drift=(0.05, 0.01)),
    "band phase-centre offset 0.3 mm": RL.Realism(band_offset=0.3e-3),
    "ALL of the above (moderate)": RL.Realism(dispersion="cq", beam_w0=0.02, Hf=0.1, plate_err=0.5e-3,
                                              rough_sigma=1e-4, paper=PAPER, air_gap=3e-3, drift=(0.05, 0.01)),
}
METHODS = {"two-band (e019dual, 2-stage cascade)": ("e019dual_lite", True),
           "one-band + priors (e018b, 2-stage cascade)": ("e018b_lite", False)}
Q = ["standoff", "drywall_front.t", "drywall_front.eps_r", "insulation.t", "drywall_back.t"]


def one(args):
    eff, meth, seed = args
    name, dual = METHODS[meth]
    f, yc, s, d0, tv = RL.make_case_real(seed, EFFECTS[eff], dual=dual)
    t0 = time.perf_counter()
    best, cost, plaus, info = B.METHODS[name](f, yc, s, d0)[:4]
    fi = I.fisher(best, f, s)
    crb = dict(zip(fi["names"], fi["crb_std"]))
    est = best.summary()
    return dict(effect=eff, method=meth, seed=seed, chi2n=cost / (2 * f.size), wall=time.perf_counter() - t0,
                err={q: est[q] - tv[q] for q in Q}, crb={q: crb.get(q, np.nan) for q in Q})


if __name__ == "__main__":
    from multiprocessing import Pool
    seeds = list(range(1, 9))
    jobs = [(e, m, s) for e in EFFECTS for m in METHODS for s in seeds]
    t0 = time.time()
    res = []
    prog = open(sys.argv[1].replace(".json", "_progress.jsonl"), "w")
    with Pool(3) as p:
        for r in p.imap_unordered(one, jobs, chunksize=1):
            res.append(r)
            prog.write(json.dumps(r, default=float) + "\n"); prog.flush()
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
    lines = [f"E020 reality gap: seeds {seeds}; z = |err|/CRB (median over walls); 'silent bias' = chi2/n <= 1.2 and some z > 3"]
    for m in METHODS:
        lines.append(f"\n=== {m}")
        lines.append(f"{'effect':40s} {'flag':>5s} {'silentB':>7s} " + " ".join(f"{q.split('.')[0][:9] + '.' + q.split('.')[-1][:5]:>16s}" for q in Q))
        for e in EFFECTS:
            rr = [r for r in res if r["effect"] == e and r["method"] == m]
            flag = sum(r["chi2n"] > 1.2 for r in rr)
            sil = sum(r["chi2n"] <= 1.2 and max(abs(r["err"][q]) / r["crb"][q] for q in Q if np.isfinite(r["crb"][q])) > 3 for r in rr)
            cells = []
            for q in Q:
                z = np.median([abs(r["err"][q]) / r["crb"][q] for r in rr])
                a = np.median([abs(r["err"][q]) for r in rr]) * (100 / 2.6 if "eps" in q else 1e3)
                cells.append(f"{a:7.3f}{'%' if 'eps' in q else 'mm'} z{z:5.1f}")
            lines.append(f"{e:40s} {flag:3d}/{len(rr)} {sil:3d}/{len(rr)} " + " ".join(f"{c:>16s}" for c in cells))
    lines.append(f"\ntime {time.time() - t0:.0f}s")
    open(sys.argv[1].replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
