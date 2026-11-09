"""
Benchmark harness for the 1D dry-wall inversion (B1-dry). Introduced in E008.

A *benchmark* makes randomized truths and data; a *method* turns calibrated data into a best
model plus a list of plausible basins; the *scorer* is the same for every method. Keeping the
three apart is what lets later experiments (E009-E012) change the method without touching how
it is graded (protocol rule: never change benchmark and method in the same experiment).

Benchmarks
  v1  scenarios.truth_random     (E007): loss tangents equal the model's nominal values
  v2  scenarios.truth_random_v2  (E008): loss tangents randomized as well

Scoring (per run)
  chi2/n                         noise-level fit if <= 1.2 (roadmap threshold)
  truth_in_set                   some plausible basin (dchi2 < 4) has, for EVERY layer, an
                                 optical thickness tau = Re(n) t within lambda_c/4 of the truth,
                                 i.e. it sits in the truth's phase basin in all layers jointly.
                                 (The E007 check used a fixed 0.3 mm window on t per parameter;
                                 for the insulation that window tested a point on a continuous
                                 t/eps ridge, not a basin.)
  outcome                        success   = chi2/n <= 1.2 and truth_in_set
                                 flagged   = chi2/n >  1.2          (the method warns)
                                 silent    = chi2/n <= 1.2 and not truth_in_set   (a bug)
  err/CRB                        at the best model, for every free parameter and for each
                                 layer's optical thickness (delta method on the CRB covariance)
  compute                        wall time and number of forward solves (with and without J)

Run:  python3 benchmark.py <benchmark> <method> <first_seed> <last_seed> <out.json> [workers]
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time
import numpy as np
import wall_toy as W
import wall_inverse as I
import scenarios as S

LAYERS = ("drywall_front", "insulation", "drywall_back")
OWNER = {"drywall_back.eps_r": "drywall_front.eps_r", "drywall_back.tan_d": "drywall_front.tan_d"}


# ----------------------------------------------------------------------------- data
RES_CELL = W.C0 / (2 * 4e9)   # E018: range-resolution cell of the 4 GHz band (37.5 mm)
DUAL_BAND = (22e9, 26e9)     # E017 second band (E014: removes the back-of-wall cycle ambiguity)


BANDS2 = {"dual24": (22e9, 26e9), "dual60": (57e9, 64e9), "dualuwb": (3.1e9, 10.6e9), "proto": (3.1e9, 10.6e9)}
BANDS1 = {"proto": (60e9, 64e9)}   # E035+: the prototype pair (TI IWR6843 60-64 GHz + UWB), doc 13


def real_presets():
    import realism as RL
    PAPER = (0.4e-3, 2.0, 0.08)
    return {"cq": RL.Realism(dispersion="cq"),
            "itu": RL.Realism(dispersion="itu"),
            "itucausal": RL.Realism(dispersion="itucausal"),
            "cqbeam2": RL.Realism(dispersion="cq", beam_w0=0.02),
            "cqbeam1": RL.Realism(dispersion="cq", beam_w0=0.01),
            "cqbeamhf": RL.Realism(dispersion="cq", beam_w0=0.02, Hf=0.1, Hi=0.02),
            "cqbeam1hf": RL.Realism(dispersion="cq", beam_w0=0.01, Hf=0.1, Hi=0.02),
            "cqpaper": RL.Realism(dispersion="cq", paper=PAPER),
            "cqrough": RL.Realism(dispersion="cq", rough_sigma=1e-4),
            "cqoffset": RL.Realism(dispersion="cq", band_offset=0.3e-3),
            "cqgap": RL.Realism(dispersion="cq", air_gap=3e-3),
            "fm100": RL.Realism(dispersion="cq", fmcw={"S": 100e12, "hpf": (175e3, 350e3), "nonlin": 0.0}),
            "fm250": RL.Realism(dispersion="cq", fmcw={"S": 250e12, "hpf": (175e3, 350e3), "nonlin": 0.0}),
            "fm100nl": RL.Realism(dispersion="cq", fmcw={"S": 100e12, "hpf": (175e3, 350e3), "nonlin": 1e6}),
            **{f"cqsnr{v}": RL.Realism(dispersion="cq", snr_db=float(v)) for v in (35, 30, 25, 20)},
            "cqdrift": RL.Realism(dispersion="cq", field_drift=(0.5, 0.3, 0.3, 3.0)),
            "cqdriftref": RL.Realism(dispersion="cq", field_drift=(0.5, 0.3, 0.3, 3.0), leak_ref=True,
                                     untracked=(0.1, 0.02)),
            "allpw": RL.Realism(dispersion="cq", rough_sigma=1e-4, paper=PAPER, air_gap=3e-3, band_offset=0.3e-3),
            "all": RL.Realism(dispersion="cq", beam_w0=0.02, Hf=0.1, plate_err=0.5e-3, rough_sigma=1e-4,
                              paper=PAPER, air_gap=3e-3, drift=(0.05, 0.01), band_offset=0.3e-3)}


_EXTRA = {}          # per-case side information for methods (E023: calibrated beam waists)


def make_case(benchmark, seed, plate_d=0.25):
    if benchmark.startswith("lambot:"):          # E023: multi-distance plate calibration, "lambot:<preset>:<band>"
        import realism as RL
        _, preset, band = benchmark.split(":")
        f, y, s, d0, tv, w0s = RL.make_case_lambot(seed, real_presets()[preset], band2=BANDS2[band])
        _EXTRA["beam"] = w0s
        return f, y, s, d0, tv
    if benchmark.startswith("fmcw:"):            # E031: raw IF data, "fmcw:<preset>:<chain|naive>"
        import realism as RL
        _, preset, mode = benchmark.split(":")
        f, y, s, d0, tv, ops = RL.make_case_fmcw(seed, real_presets()[preset], model_chain=(mode == "chain"))
        _EXTRA["chain"] = ops
        return f, y, s, d0, tv
    if benchmark.startswith("motion:"):          # E038: "motion:<K>:<preset>:<band>", K frames, hand moves +-4 mm
        import realism as RL
        _, K, preset, band = benchmark.split(":")
        rngm = np.random.default_rng(seed + 4242)
        fs, ys, ss = [], [], []
        for k in range(int(K)):
            mv = 0.0 if k == 0 else float(rngm.uniform(-0.004, 0.004))
            f, y, s, d0k, tv = RL.make_case_real(seed, real_presets()[preset], dual=True, plate_d=plate_d,
                                                  band2=BANDS2.get(band), band1=BANDS1.get(band), frame=k, motion=mv)
            if k == 0:
                d0 = d0k
            fs.append(f); ys.append(y); ss.append(s)
            tv0 = tv if k == 0 else tv0
        return np.concatenate(fs), np.concatenate(ys), float(np.sqrt(np.mean(np.square(ss)))), d0, tv0
    if benchmark.startswith("real:"):            # E021+: realistic world (realism.py), "real:<preset>:<band>"
        import realism as RL
        _, preset, band = benchmark.split(":")
        return RL.make_case_real(seed, real_presets()[preset], dual=band != "single",
                                 plate_d=plate_d, band2=BANDS2.get(band), band1=BANDS1.get(band))
    dual = benchmark.endswith("dual")
    benchmark = benchmark[:-4] if dual else benchmark
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    if benchmark == "v1":
        scene, tv = S.truth_random(rng, d_true)
        tv["drywall_front.tan_d"], tv["insulation.tan_d"] = S.DRY["tan_d"], S.FIB["tan_d"]
    elif benchmark == "v2":
        scene, tv = S.truth_random_v2(rng, d_true)
    else:
        raise ValueError(benchmark)
    tv["drywall_back.eps_r"] = tv["drywall_front.eps_r"]
    tv["drywall_back.tan_d"] = tv["drywall_front.tan_d"]
    bands = [W.Radar(seed=seed)]
    if dual:                       # E017: second band, own calibration scan and own noise
        bands.append(W.Radar(f0=DUAL_BAND[0], f1=DUAL_BAND[1], seed=seed + 1000))
    fs, ycs, sig2 = [], [], []
    for radar in bands:
        fb, y, info = W.measure(scene, radar)
        _, yp, _ = W.measure_metal_plate(radar, plate_d, seed=radar.seed + 500)
        ycb, _ = I.calibrate(y, yp, fb, plate_d)
        fs.append(fb); ycs.append(ycb)
        sig2.append((info["noise_sigma"] * np.sqrt(1 + np.mean(np.abs(ycb) ** 2))) ** 2)
    f, yc = np.concatenate(fs), np.concatenate(ycs)
    sigma = float(np.sqrt(np.mean(sig2)))                    # bands have nearly equal noise levels
    r, p = W.range_profile(fs[0], ycs[0])                    # standoff guess from the 77-81 GHz band
    sel = (r > 0.2) & (r < 0.4)
    d0 = r[sel][np.argmax(p[sel])]
    return f, yc, sigma, d0, tv


# ----------------------------------------------------------------------------- optical thickness
def tau_of(vals, layer, fc):
    """Optical thickness Re(n) t of a layer from a parameter dict (eps/tan possibly tied)."""
    er = vals.get(f"{layer}.eps_r", vals.get(OWNER.get(f"{layer}.eps_r", ""), None))
    td = vals.get(f"{layer}.tan_d", vals.get(OWNER.get(f"{layer}.tan_d", ""), None))
    return np.sqrt(er * (1 - 1j * td)).real * vals[f"{layer}.t"]


def tau_crb(model, fi):
    """CRB std of each layer's optical thickness, by the delta method on the Fisher covariance."""
    v = model.summary()
    names, cov = fi["names"], fi["cov"]
    out = {}
    for L in LAYERS:
        er, td, t = v[f"{L}.eps_r"], v[f"{L}.tan_d"], v[f"{L}.t"]
        n = np.sqrt(er * (1 - 1j * td))
        dtau = {f"{L}.t": n.real,
                model.owner(f"{L}.eps_r"): t * ((1 - 1j * td) / (2 * n)).real,
                model.owner(f"{L}.tan_d"): t * ((-1j * er) / (2 * n)).real}
        g = np.zeros(len(names))
        for k, d in dtau.items():
            if k is not None and k in names:
                g[names.index(k)] += d
        out[L] = float(np.sqrt(g @ cov @ g))
    return out


# ----------------------------------------------------------------------------- scoring
def score(f, sigma, tv, best, cost, plaus, yc):
    y_for_truth = (yc,)
    fc = I.carrier(f)                                         # highest band (= f.mean() for one band)
    quarter = W.C0 / fc / 4                                   # lambda_c / 4 (basin half-width in tau)
    est = best.summary()
    fi = I.fisher(best, f, sigma)
    crb = dict(zip(fi["names"], fi["crb_std"]))
    tcrb = tau_crb(best, fi)
    n_data = 2 * f.size
    tau_true = {L: tau_of(tv, L, fc) for L in LAYERS}

    def in_basin(vals):
        return all(abs(tau_of(vals, L, fc) - tau_true[L]) < quarter for L in LAYERS)

    truth_in = any(in_basin(sm) for sm, _ in plaus)
    tm = best.copy()                                          # chi2 at the truth: optimizer vs
    for nm, _, prm in tm.items():                             # identifiability vs model diagnosis
        if nm in tv:
            prm.value = float(tv[nm])
    tm.sync_ties()
    rt, _ = I.residuals(tm, f, y_for_truth[0], sigma)
    chi2_truth = float(rt @ rt)
    chi2n = cost / n_data
    outcome = "flagged" if chi2n > 1.2 else ("success" if truth_in else "silent")
    return dict(chi2=cost, chi2_truth=chi2_truth, n_data=n_data, chi2n=chi2n, truth_in_set=bool(truth_in),
                best_in_truth_basin=bool(in_basin(est)), outcome=outcome, n_plausible=len(plaus),
                err={n: est[n] - tv[n] for n in fi["names"] if n in tv},
                crb=crb,
                tau_err={L: tau_of(est, L, fc) - tau_true[L] for L in LAYERS},
                tau_crb=tcrb, truth={k: float(v) for k, v in tv.items()},
                est={k: float(v) for k, v in est.items()})


# ----------------------------------------------------------------------------- methods
def half(a, b):
    return np.arange(2 * a, 2 * b + 1) / 2


def method_e007(f, yc, sigma, d0, fit_tan=False):
    """The E007 method (time-gated layer stripping with lattice starts, joint LM, cycle hopping,
    ambiguity scan). fit_tan=True also estimates the loss tangents (drywall shared), each in the
    stage whose echoes it shapes; it starts at the nominal value (not on the lattice)."""
    free = ("t", "eps_r", "tan_d") if fit_tan else ("t", "eps_r")
    m0 = S.stud_wall_model(free=free, standoff_guess=d0)
    s0 = ("standoff", "drywall_front.t", "drywall_front.eps_r") + (("drywall_front.tan_d",) if fit_tan else ())
    s1 = ("insulation.t", "insulation.eps_r", "drywall_back.t") + (("insulation.tan_d",) if fit_tan else ())
    stages = [I.Stage(s0, I.Gate(d0 - 0.03, d0 + 0.06),
                      lattice={"drywall_front.t": half(-4, 4), "drywall_front.eps_r": [2.0, 2.6, 3.2]}),
              I.Stage(s1, I.Gate(d0 + 0.06, d0 + 0.25),
                      lattice={"insulation.t": half(-6, 6), "insulation.eps_r": [1.05, 1.15, 1.3]})]
    best, cost = I.strip_invert(m0, f, yc, sigma, stages)
    amb, best, cost, plaus = I.ambiguity_scan(best, f, yc, sigma, return_models=True)
    return best, cost, plaus


def method_search(**kw):
    """search.invert with the given Opts switches (E009-E012 stack one switch each)."""
    import search as SR

    def run(f, yc, s, d0):
        k = dict(kw)
        if k.get("beam") == "auto":
            k["beam"] = _EXTRA["beam"]
        return SR.invert(f, yc, s, d0, SR.Opts(**k))
    return run


E009 = dict(search_step=8, search_atol=0.01)
E010 = dict(E009, init="pencil")
E011 = dict(E010, coords="optical")
E012 = dict(E011, ambiguity="ils")
E011B = dict(E010, coords="optical_box")       # E011 with the physical box enforced
E012B = dict(E011B, ambiguity="ils")
CURRENT = E011B                                  # best accepted method after E012b
# E013: drywall thickness prior, ASTM C36 nominal tolerance +-1/64 in (0.4 mm) used as 1 sd
DRYWALL_PRIOR = (("drywall_front.t", 0.0127, 0.0004), ("drywall_back.t", 0.0127, 0.0004))
# E013b: + stud-depth prior on the cavity (insulation) thickness: 3-1/2 in dressed 2x4, sd 1/16 in
# (1.6 mm) ASSUMED (PS 20 gives the minimum dressed size; no undersize tolerance found)
STUD_PRIOR = (("insulation.t", 0.0889, 0.0016),)
E013 = dict(CURRENT, priors=DRYWALL_PRIOR)
E013B = dict(CURRENT, priors=DRYWALL_PRIOR + STUD_PRIOR)

def method_cascade(*stages, flag=1.2):
    """E015: run the first option set; only if the data chi2/n exceeds `flag` (the method's own
    failure alarm), run the next one, and so on. Keep the result with the lowest posterior cost
    (data chi2 + prior penalty). Only flagged walls pay for the fallbacks."""
    import search as SR

    def run(f, yc, s, d0):
        best, used = None, []
        for k, kw in enumerate(stages):
            kw = dict(kw)
            if kw.get("beam") == "auto":
                kw["beam"] = _EXTRA["beam"]
            m, c, plaus, info = SR.invert(f, yc, s, d0, SR.Opts(**kw))
            post = c + info.get("prior_penalty", 0.0)
            used.append(dict(stage=k, chi2n=c / (2 * f.size), post=post))
            if best is None or post < best[4]:
                best = (m, c, plaus, info, post)
            if best[1] / (2 * f.size) <= flag:
                break
        m, c, plaus, info, _ = best
        info = dict(info, cascade=used)
        return m, c, plaus, info
    return run


# E015 fallbacks keep the same priors as the primary method
E015_FB1 = lambda pri: dict(E009, priors=pri)                    # physical coordinates, coarse search
E015_FB2 = lambda pri: dict(priors=pri)                          # dense full-data search (E007 settings)

def method_e028(hybrid=False, fallback=True):
    """E028: learned starting point (init_net) -> exact LM; hybrid: the network sets standoff, front
    drywall and material starts, a coarse lattice over the insulation / back-drywall cycles picks the
    back basins (compiled min-cell model), then one exact LM on all data. chi2 alarm -> search."""
    import search as SR
    import init_net as N
    fb = METHODS["e027_s4"]

    def run(f, yc, s, d0):
        dh, v = N.predict(f, yc, s)
        opts = SR.Opts(**dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, min_cells=True))
        m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=dh + v[0], standoff_halfwidth=RES_CELL)
        m.law, m.min_cells = "jonscher", True
        m.layers[0].pinf = I.P(0.3, 0.0, 0.9, True)
        m.layers[2].pinf = I.P(0.3, 0.0, 0.9, True, tie="drywall_front.pinf")
        n = lambda er, td: np.sqrt(er * (1 - 1j * td)).real
        vals = {"drywall_front.eps_r": v[2], "drywall_front.tan_d": v[3], "insulation.eps_r": v[5], "insulation.tan_d": v[6]}
        params = {nn: p for nn, _, p in m.items()}
        for k, val in vals.items():
            params[k].value = float(val)
        m.sync_ties()
        m.layers[0].t.value = v[1] / n(v[2], v[3]); m.layers[2].t.value = v[7] / n(v[2], v[3])
        m.layers[1].t.value = v[4] / n(v[5], v[6])
        lo, hi = m.bounds(); m.set(np.clip(m.get(), lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo)))
        if hybrid:
            fs, ys = SR._subset(f, yc, 8)
            per = I._phase_periods(m, f)
            best = None
            for a in range(-6, 7):
                for b in range(-3, 4):
                    t = SR._shift(m, f, [(nm, k) for nm, k in (("insulation.t", a), ("drywall_back.t", b)) if k])
                    if t is None:
                        continue
                    mo, c = SR.lm(t, fs, ys, s, atol=0.05, coords=opts.coords, max_iter=15)
                    if best is None or c < best[1]:
                        best = (mo, c)
            m = best[0]
        mo, c = SR.lm(m, f, yc, s, coords=opts.coords)
        if c / (2 * f.size) > 1.2 and fallback:
            r = fb(f, yc, s, d0)
            return r[0], r[1], r[2], dict(r[3], e028="fallback")
        return mo, c, [(mo.summary(), c)], {"e028": "direct"}
    return run


def chi2_alarm(cost, n_data):
    """Statistically scaled alarm: chi2 exceeds its expectation by more than 3 standard deviations,
    chi2 > n + 3 sqrt(2 n)  (chi2/n ~ 1.13 for n = 1024, 1.19 for n = 512)."""
    return cost > n_data + 3 * np.sqrt(2 * n_data)


def method_e029(K_ins=4, K_back=3, n_polish=3, amb=True, verify=False):
    """E029: phase-aware network -> integer least squares over the insulation / back-drywall cycle
    counts (candidates tau = P_n (N + phase) around the coarse value, chi2 evaluated without
    optimisation on coarse data) -> exact LM polish of the best n_polish -> +-1 ambiguity check ->
    chi2 alarm -> e027_s4 search."""
    import search as SR
    import init_net as N

    def run(f, yc, s, d0):
        dh, v, coarse, phn = N.predict_phase_raw(f, yc, s)
        opts = SR.Opts(**dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, min_cells=True,
                              lat_step=1.0, lat0_eps=(2.6,), lat1_eps=(1.15,), hops=(1, -1), amb_ks=(-1, 1)))
        m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=dh + v[0], standoff_halfwidth=RES_CELL)
        m.law, m.min_cells = "jonscher", True
        m.layers[0].pinf = I.P(0.3, 0.0, 0.9, True)
        m.layers[2].pinf = I.P(0.3, 0.0, 0.9, True, tie="drywall_front.pinf")
        params = {nn: p for nn, _, p in m.items()}
        for k, val in {"drywall_front.eps_r": v[2], "drywall_front.tan_d": v[3],
                       "insulation.eps_r": v[5], "insulation.tan_d": v[6]}.items():
            params[k].value = float(val)
        m.sync_ties()
        nre = lambda er, td: np.sqrt(er * (1 - 1j * td)).real
        nd, ni = nre(v[2], v[3]), nre(v[5], v[6])
        m.layers[0].t.value = v[1] / nd
        lo, hi = m.bounds()
        fs, ys = SR._subset(f, yc, 8)
        cands = []
        Pn = N.P_NARROW
        for a in range(-K_ins, K_ins + 1):
            for b in range(-K_back, K_back + 1):
                ti = Pn * (np.round(coarse[4] / Pn - phn[4]) + a + phn[4]) / ni
                tb = Pn * (np.round(coarse[7] / Pn - phn[7]) + b + phn[7]) / nd
                if not (m.layers[1].t.lo < ti < m.layers[1].t.hi and m.layers[2].t.lo < tb < m.layers[2].t.hi):
                    continue
                c = m.copy(); c.layers[1].t.value = ti; c.layers[2].t.value = tb
                cands.append((SR.cost_of(c, fs, ys, s), c))
        cands.sort(key=lambda t: t[0])
        best = None
        for _, c in cands[:n_polish]:
            mo, cost = SR.lm(c, f, yc, s, coords=opts.coords)
            if best is None or cost < best[1]:
                best = (mo, cost)
        mo, cost = best
        plaus = [(mo.summary(), cost)]
        if verify and chi2_alarm(cost, 2 * f.size):          # E029b: the scan verifies, never rescues
            r = METHODS["e027_s4"](f, yc, s, d0)
            return r[0], r[1], r[2], dict(r[3], e029="fallback-prescan", n_cands=len(cands))
        if amb:
            mo, cost, plaus = SR.ambiguity_scan(mo, f, yc, s, opts)
        if (chi2_alarm(cost, 2 * f.size) if verify else cost / (2 * f.size) > 1.2):
            r = METHODS["e027_s4"](f, yc, s, d0)
            return r[0], r[1], r[2], dict(r[3], e029="fallback", n_cands=len(cands))
        return mo, cost, plaus, {"e029": "direct", "n_cands": len(cands)}
    return run


METHODS = {
    "e007": lambda f, yc, s, d0: method_e007(f, yc, s, d0, fit_tan=False),
    "e007_tanfree": lambda f, yc, s, d0: method_e007(f, yc, s, d0, fit_tan=True),
    "search_base": method_search(),          # search.py with E007 settings (must equal e007_tanfree)
    "e009": method_search(**E009),
    "e010": method_search(**E010),
    "e011": method_search(**E011),
    "e012": method_search(**E012),
    "e011b": method_search(**E011B),
    "e012b": method_search(**E012B),
    "current": method_search(**CURRENT),
    "e013": method_search(**E013),
    "e013b": method_search(**E013B),
    "e017": method_search(**CURRENT),                 # run on benchmark v2dual, no priors
    "e018dual": method_search(**dict(CURRENT, standoff_halfwidth=RES_CELL)),
    "e018b": method_cascade(dict(E013B, standoff_halfwidth=RES_CELL),
                            dict(E015_FB1(DRYWALL_PRIOR + STUD_PRIOR), standoff_halfwidth=RES_CELL),
                            dict(E015_FB2(DRYWALL_PRIOR + STUD_PRIOR), standoff_halfwidth=RES_CELL)),
    "e019dual": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL),
                               dict(E009, standoff_halfwidth=RES_CELL),
                               dict(standoff_halfwidth=RES_CELL)),
    # E020 study variants: same cascades without the expensive dense third stage
    "e019dual_lite": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL), dict(E009, standoff_halfwidth=RES_CELL)),
    "e018b_lite": method_cascade(dict(E013B, standoff_halfwidth=RES_CELL),
                                 dict(E015_FB1(DRYWALL_PRIOR + STUD_PRIOR), standoff_halfwidth=RES_CELL)),
    # E021: causal constant-Q material law in the model (else = e019dual_lite)
    "e021": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="cq"),
                           dict(E009, standoff_halfwidth=RES_CELL, law="cq")),
    # E021b: Jonscher law, drywall eps_inf fraction free (shared by both sheets), insulation constant-Q
    "e021b": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True),
                            dict(E009, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True)),
    # E023: beam forward model (rotated Gauss-Laguerre over exact oblique stars) with calibrated waists
    "e023": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, beam="auto"),
                           dict(E009, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, beam="auto")),
    # E024: + facing paper, front roughness, band phase-centre offset (plane wave / beam)
    "e024": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True,
                                facing=True, rough=True, band_offset=True),
                           dict(E009, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True,
                                facing=True, rough=True, band_offset=True)),
    "e024beam": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True,
                                    facing=True, rough=True, band_offset=True, beam="auto"),
                               dict(E009, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True,
                                    facing=True, rough=True, band_offset=True, beam="auto")),
    # E025 speed variants on the E021b method (two bands): sparser start lattices, fewer hops
    **{name: method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, **kw),
                            dict(E009, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, **kw))
       for name, kw in {
           "e025_base": {},
           "e025_s1": dict(lat_step=1.0),
           "e025_s2": dict(lat_step=1.0, lat0_eps=(2.6,), lat1_eps=(1.15,)),
           "e025_s3": dict(lat_step=2.0, lat0_eps=(2.6,), lat1_eps=(1.15,)),
           "e025_s4": dict(lat_step=1.0, lat0_eps=(2.6,), lat1_eps=(1.15,), hops=(1, -1), amb_ks=(-1, 1)),
           "e025_s5": dict(lat_step=2.0, lat0_eps=(2.6,), lat1_eps=(1.15,), hops=(1, -1), amb_ks=(-1, 1)),
           "e027_s4": dict(lat_step=1.0, lat0_eps=(2.6,), lat1_eps=(1.15,), hops=(1, -1), amb_ks=(-1, 1), min_cells=True),
       }.items()},
    "e024b": method_cascade(dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, band_offset=True),
                            dict(E009, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, band_offset=True)),
    "e028": None, "e028h": None,       # filled below (need METHODS for the fallback)
    # E030: + air-gap layer (model selection: keep the gap only if it lowers chi2 by more than 6 = 3 sd of a 1-dof chi2)
    "e030": None,
    "e016": method_cascade(dict(E013, ambiguity="posterior"),
                           dict(E015_FB1(DRYWALL_PRIOR), ambiguity="posterior"),
                           dict(E015_FB2(DRYWALL_PRIOR), ambiguity="posterior")),
    "e015": method_cascade(E013, E015_FB1(DRYWALL_PRIOR), E015_FB2(DRYWALL_PRIOR)),
    "e015b": method_cascade(E013B, E015_FB1(DRYWALL_PRIOR + STUD_PRIOR), E015_FB2(DRYWALL_PRIOR + STUD_PRIOR)),
}


METHODS["e028"] = method_e028(hybrid=False)
METHODS["e028h"] = method_e028(hybrid=True)
METHODS["e029"] = method_e029()
METHODS["e029_noamb"] = method_e029(amb=False)
METHODS["e029b"] = method_e029(verify=True)


def method_e030():
    """E030: e024b-style search (facing, roughness, band offset) with and without an air-gap layer;
    the gap model is kept only if it lowers chi2 by more than 6 (one extra parameter, ~3 sd)."""
    base = dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, min_cells=True,
                facing=True, rough=True, band_offset=True)
    no_gap = method_cascade(base, dict(E009, **{k: v for k, v in base.items() if k not in CURRENT}))
    with_gap = method_cascade(dict(base, gap=True), dict(E009, gap=True, **{k: v for k, v in base.items() if k not in CURRENT}))

    def run(f, yc, s, d0):
        a = no_gap(f, yc, s, d0)
        b = with_gap(f, yc, s, d0)
        pick = b if b[1] < a[1] - 6.0 else a
        return pick[0], pick[1], pick[2], dict(pick[3], gap_selected=pick is b, chi2_nogap=a[1], chi2_gap=b[1])
    return run


METHODS["e030"] = method_e030()
METHODS["e033"] = method_e030()
# E035: field self-calibration (per-band smooth complex gain fitted with the wall), on the e027_s4 method
_B35 = dict(standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, lat_step=1.0, lat0_eps=(2.6,),
            lat1_eps=(1.15,), hops=(1, -1), amb_ks=(-1, 1), min_cells=True)
for _nm, _cal in (("e035_none", ""), ("e035_gain", "gain"), ("e035_full", "full")):
    METHODS[_nm] = method_cascade(dict(CURRENT, cal=_cal, **_B35), dict(E009, cal=_cal, **_B35))


def method_e038(nb=2):
    """E038: joint inversion of K handheld frames (same wall, each frame its own standoff). Initial
    frame offsets from the front-echo peak of each frame's lowest band (range profile)."""
    import wall_toy as W
    import wall_inverse as I

    def run(f, yc, s, d0):
        segs = I.band_segments(f)
        K = len(segs) // nb
        peaks = []
        for k in range(K):
            a, b = min((segs[k * nb + j] for j in range(nb)), key=lambda ab: f[ab[0]])
            r, p = W.range_profile(f[a:b], yc[a:b])
            sel = (r > d0 - 0.02) & (r < d0 + 0.02)
            i = np.argmax(p[sel]); rr = r[sel]
            peaks.append(rr[i])
        offs = tuple(float(pk - peaks[0]) for pk in peaks[1:])
        return method_cascade(dict(CURRENT, frames=offs, frame_nb=nb, **_B35),
                              dict(E009, frames=offs, frame_nb=nb, **_B35))(f, yc, s, d0)
    return run


METHODS["e038"] = method_e038()


def method_e039(lat_step=1.0, k_amb=(-1, 1), fallback=True):
    """E039: UWB-first cascade for the prototype pair. (1) the stud-wall search on the LOW band only
    (few phase cycles per layer, so a sparse lattice suffices), (2) LM on both bands from that
    solution, (3) +-1 cycle scan on the high-band periods, (4) chi2 alarm -> e035_none search."""
    import search as SR
    import wall_inverse as I
    kw = dict(_B35, lat_step=lat_step, hops=(1, -1), amb_ks=k_amb)

    def run(f, yc, s, d0):
        segs = I.band_segments(f)
        lo = min(segs, key=lambda ab: f[ab[0]])
        fl, yl = f[lo[0]:lo[1]], yc[lo[0]:lo[1]]
        o = SR.Opts(**dict(CURRENT, **dict(kw, ambiguity="scan", amb_ks=())))
        m, c, _, info = SR.invert(fl, yl, s, d0, o)
        oall = SR.Opts(**dict(CURRENT, **kw))
        m, c = SR.lm(m, f, yc, s)
        m, c, plaus = SR.ambiguity_scan(m, f, yc, s, oall)
        if fallback and chi2_alarm(c, 2 * f.size):
            r = METHODS["e035_none"](f, yc, s, d0)
            return r[0], r[1], r[2], dict(r[3], e039="fallback")
        return m, c, plaus, {"e039": "direct"}
    return run


METHODS["e039"] = method_e039()
METHODS["e039_s2"] = method_e039(lat_step=2.0)


def method_e031():
    """E031: the whole search applies the measurement operators (IF high-pass chain x calibrated A)
    to the model on the FULL frequency grid (no coarse subsets, no equalisation: inverting the
    high-pass chain is ill-conditioned, cond ~1e3, error grows like a random walk)."""
    def run(f, y, s, d0):
        kw = dict(CURRENT, standoff_halfwidth=RES_CELL, law="jonscher", pinf_free=True, min_cells=True,
                  lat_step=1.0, lat0_eps=(2.6,), lat1_eps=(1.15,), hops=(1, -1), amb_ks=(-1, 1),
                  search_step=1, search_atol=0.01, chain=_EXTRA["chain"])
        fb = {k: v for k, v in kw.items() if k not in CURRENT}
        return method_cascade(kw, dict(E009, **fb, search_step=1),
                              dict(**{k: v for k, v in fb.items() if k not in ("lat_step", "lat0_eps", "lat1_eps", "hops", "amb_ks")},
                                   search_step=1))(f, y, s, d0)
    return run


METHODS["e031"] = method_e031()


# ----------------------------------------------------------------------------- runner
def run_one(args):
    benchmark, method, seed = args
    counts = {"J": 0, "noJ": 0, "freqJ": 0, "freqnoJ": 0}
    orig = I.forward

    def counted(model, f, jacobian=True):
        counts["J" if jacobian else "noJ"] += 1
        counts["freqJ" if jacobian else "freqnoJ"] += np.size(f)     # cost ~ number of frequencies
        return orig(model, f, jacobian)

    I.forward = counted
    try:
        f, yc, sigma, d0, tv = make_case(benchmark, seed)
        t0 = time.perf_counter()
        res = METHODS[method](f, yc, sigma, d0)
        best, cost, plaus = res[:3]
        info = res[3] if len(res) > 3 else {}
        cpu = time.perf_counter() - t0
    finally:
        I.forward = orig
    out = score(f, sigma, tv, best, cost, plaus, yc)
    out.update(seed=seed, benchmark=benchmark, method=method, wall_s=cpu,
               forward_J=counts["J"], forward_noJ=counts["noJ"],
               work=(counts["freqJ"] + counts["freqnoJ"]) / 256.0, info=info)
    return out


def summarize(runs):
    oc = [r["outcome"] for r in runs]
    lines = [f"runs {len(runs)}: success {oc.count('success')}, flagged {oc.count('flagged')}, silent {oc.count('silent')}",
             f"best model in the truth's basin (all layers): {sum(r['best_in_truth_basin'] for r in runs)}/{len(runs)}",
             "chi2/n: " + ", ".join(f"{r['chi2n']:.2f}" for r in runs),
             "chi2(truth) - chi2(best): " + ", ".join(f"{r['chi2_truth'] - r['chi2']:+.1f}" for r in runs),
             "outcome per seed: " + ", ".join(f"{r['seed']}:{r['outcome'][0]}" for r in runs),
             f"wall time per inversion (1 core): median {np.median([r['wall_s'] for r in runs]):.1f} s; forward solves (with J) median "
             f"{np.median([r['forward_J'] for r in runs]):.0f}, (no J) median {np.median([r['forward_noJ'] for r in runs]):.0f}",
             "work (forward solves in units of one 256-frequency solve): median "
             f"{np.median([r.get('work', np.nan) for r in runs]):.0f}"]
    if runs[0].get("info", {}).get("p_best_basin") is not None:
        pb = np.array([r["info"]["p_best_basin"] for r in runs])
        hit = np.array([r["best_in_truth_basin"] for r in runs])
        lines.append(f"reported P(best basin) mean {pb.mean():.2f} vs observed best-in-truth-basin rate {hit.mean():.2f}; "
                     f"plausible basins median {np.median([r['info']['n_plausible'] for r in runs]):.0f}")
    ok = [r for r in runs if r["outcome"] == "success" and r["best_in_truth_basin"]]
    lines.append(f"err/CRB over runs whose best model is in the truth basin ({len(ok)}):")
    for n in runs[0]["err"]:
        z = np.array([r["err"][n] / r["crb"][n] for r in ok]) if ok else np.array([np.nan])
        a = np.array([abs(r["err"][n]) for r in runs])
        lines.append(f"  {n:22s} mean|err| {a.mean():.2e}  CRB {np.mean([r['crb'][n] for r in runs]):.1e}"
                     f"  err/CRB mean {z.mean():+.2f} std {z.std():.2f} max|.| {np.abs(z).max():.1f}")
    for L in LAYERS:
        z = np.array([r["tau_err"][L] / r["tau_crb"][L] for r in ok]) if ok else np.array([np.nan])
        lines.append(f"  tau[{L:13s}]        mean|err| {np.mean([abs(r['tau_err'][L]) for r in runs]):.2e}"
                     f"  CRB {np.mean([r['tau_crb'][L] for r in runs]):.1e}  err/CRB mean {z.mean():+.2f} std {z.std():.2f}"
                     f" max|.| {np.abs(z).max():.1f}")
    lines.append("plausible basins per run: " + str([r["n_plausible"] for r in runs]))
    return "\n".join(lines)


if __name__ == "__main__":
    from multiprocessing import Pool
    bench, meth, s0, s1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    t0 = time.time()
    with Pool(workers) as pool:
        runs = pool.map(run_one, [(bench, meth, s) for s in range(s0, s1 + 1)])
    txt = f"benchmark {bench}  method {meth}  seeds {s0}-{s1}  wall {time.time() - t0:.0f}s\n" + summarize(runs)
    print(txt)
    json.dump(runs, open(out, "w"), indent=1, default=float)
    open(out.replace(".json", ".txt"), "w").write(txt + "\n")
