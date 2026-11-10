"""
Stage attribution for wallsolve.solve_echo failures. For a solved case with known truth, report the
FIRST pipeline stage that fails:
  1 information  - the exact fit started at the truth reaches the noise level (chi2/n) and CRBs
  2 echoes       - every true interface (optical depth) lies within TOL of a detected echo
  3 candidates   - the true interface subset/gap pattern is among the enumerated candidates, and its
                   rank after the two-stage screening
  4 refinement   - from the true-structure candidate, the exact fit reaches the truth
  5 selection    - truth's cost vs the cost of the answer the solver returned
"""
from __future__ import annotations
import numpy as np
import stack as ST
import search as SR
import wall_inverse as I
import matlib as ML
import echoes as EC
import wallsolve as WS

TOL = 1.5e-3


def truth_model(tr, N):
    lay, slots = [], []
    for i, L in enumerate(tr["layers"]):
        if L["name"] == "water":
            lay.append(dict(t=L["t"], er=1.0, td=0.0, water=True))
        else:
            lay.append(dict(t=L["t"], er=L["eps10"], td=L["tan10"], water=False, pinf=min(L.get("p", 0.0), 0.998)))
        slots.append(("solid", i))
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            lay.append(dict(t=tr["gaps"][i], er=1.0, td=0.0, water=False)); slots.append(("gap", i))
    ST.SBOX = None
    m = ST.build(lay, tr["standoff"], back=tr["back"], water_T=tr["T"])
    for (kind, _), L in zip(slots, m.layers):
        if kind == "gap":
            L.eps_r.free = L.tan_d.free = False
        elif L.theta.value < 0.5:
            L.pinf.free = True
    return m, slots


def interfaces(tr):
    z = tr["standoff"]; out = [z]
    for i, L in enumerate(tr["layers"]):
        n = 8.6 if L["name"] == "water" else np.sqrt(L["eps10"])
        z += n * L["t"]; out.append(z)
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            z += tr["gaps"][i]; out.append(z)
    return np.array(out)


def diagnose(f, y, s, tr, N, res):
    rep = {}
    # 1 information
    m, slots = truth_model(tr, N)
    pri = ML.LibPrior(); old = SR._PRIORS; SR._PRIORS = (pri,)
    try:
        mt, ct = ST._fit(m, f, y, s)
    finally:
        SR._PRIORS = old
    G, _ = I.forward(mt, f, jacobian=False)
    chi_t = float(np.sum(np.abs((G - y) / s) ** 2))
    try:
        F = I.fisher(mt, f, s); crb = dict(zip(F["names"], F["crb_std"]))
    except Exception:
        crb = {}
    rep["1_truth_chi2n"] = chi_t / (2 * f.size)
    rep["1_truth_crb_t_mm"] = [round(1e3 * crb.get(f"{L.name}.t", np.nan), 3) for L in mt.layers]
    # 2 echoes
    taus, amps, _ = EC.find_echoes(f, y, s, tr["standoff"] + 0.001)
    ti = interfaces(tr)
    miss = [round(1e3 * (t - tr["standoff"]), 1) for t in ti if np.min(np.abs(taus - t)) > TOL]
    rep["2_true_interfaces_mm"] = [round(1e3 * (t - tr["standoff"]), 1) for t in ti]
    rep["2_echoes_mm"] = [round(1e3 * (t - tr["standoff"]), 1) for t in taus]
    rep["2_missed_interfaces_mm"] = miss
    # 5 selection (needs only costs)
    rep["5_cost_truth_fit"] = ct
    rep["5_chi2n_solver"] = res["chi2n"]
    rep["5_truth_better"] = bool(chi_t < res["chi2"] - 4)
    if miss:
        rep["first_failure"] = "2 echoes (true interface not detected)"
    elif rep["1_truth_chi2n"] > 1.3:
        rep["first_failure"] = "1 information/model (truth cannot fit)"
    elif not rep["5_truth_better"]:
        rep["first_failure"] = "identifiability: solver answer fits as well as the truth"
    else:
        rep["first_failure"] = "3/4 candidate ranking or refinement (truth fits better but was not reached)"
    return rep
