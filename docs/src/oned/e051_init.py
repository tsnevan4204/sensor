"""
E051: starting-material initialisers, isolated. The TRUE interface geometry (optical depths) is given;
only the starting materials differ. Success = the exact layered fit from that start reaches the
truth (every thickness within max(0.3 mm, 2 %), backing right, chi2/n < 1.3).
  amp    : impedance recursion from echo amplitudes (current pipeline)
  sweep  : coordinate sweeps over the material library at fixed optical thickness (batched exact)
  peel   : per layer, top to bottom: exact downward continuation of the data through the layers
           already initialised, then an exact 'layer over half-space' fit of that layer's (eps', tan d,
           p) with its optical thickness fixed (low band + UWB)
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np
import wallsolve as WS
import stack as ST
import search as SR
import matlib as ML
import wall_inverse as I
import echoes as EC

C0 = WS.C0


def true_geometry(tr):
    """Interfaces (optical, from the true eps' at 10 GHz) and the slot structure."""
    lay, slots = [], []
    for i, L in enumerate(tr["layers"]):
        lay.append(dict(t=L["t"], er=L["eps10"], td=L["tan10"], water=False))
        slots.append(("solid", i))
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            lay.append(dict(t=tr["gaps"][i], er=1.0001, td=1e-5, water=False)); slots.append(("gap", i))
    return lay, slots


def init_amp(tr, f, y, s, N):
    """Amplitude-recursion materials at the true optical thicknesses (what the pipeline would get
    if the echoes were perfect)."""
    lay, slots = true_geometry(tr)
    taus, amps, _ = EC.find_echoes(f, y, s, tr["standoff"] + 0.001)
    z = [tr["standoff"]]
    for L in lay:
        z.append(z[-1] + L["t"] * np.sqrt(L["er"]))
    idx = [int(np.argmin(np.abs(taus - zz))) for zz in z]
    ns = WS._impedance(amps[0], idx)
    out = []
    for L, n, (kind, _) in zip(lay, ns, slots):
        if kind == "gap":
            out.append(L)
        else:
            tau = L["t"] * np.sqrt(L["er"])
            n = max(n, 1.02)
            out.append(dict(t=tau / n, er=n ** 2, td=0.02, water=False))
    return out, slots


def init_sweep(tr, f, y, s, N):
    lay0, slots = init_amp(tr, f, y, s, N)
    x = WS._cand_to_x(tr["standoff"], lay0, slots, N, tr["back"])
    X, _ = WS._class_sweeps(x[None, :], f, y, s, N, tr["back"])
    return WS._x_to_layers(X[0], (lay0, slots), N), slots


def init_peel(tr, f, y, s, N):
    """Top-down: for layer k, fit (eps', tan d, p) by exact fit of the stack [layers 0..k fixed so
    far, layer k free with its optical thickness kept] over a half-space, on f < 11 GHz."""
    lay, slots = true_geometry(tr)
    sel = f < 11e9
    fl, yl, sl = f[sel], y[sel], s[sel]
    cur = []
    for j, ((kind, i), L) in enumerate(zip(slots, lay)):
        if kind == "gap":
            cur.append(L); continue
        tau = L["t"] * np.sqrt(L["er"])
        best = None
        for _, er0, td0, p0 in WS.class_table():
            trial = cur + [dict(t=tau / np.sqrt(er0), er=er0, td=td0, water=False, pinf=min(p0, 0.99))]
            ST.SBOX = None
            m = ST.build(trial, tr["standoff"], back="air")
            m.standoff.free = False
            for Lm in m.layers[:-1]:
                Lm.t.free = Lm.eps_r.free = Lm.tan_d.free = False
            m.layers[-1].t.free = True
            G, _ = I.forward(m, fl, jacobian=False)
            # gate to this layer's echoes: weight only times up to its back face
            c = float(np.sum(np.abs(_gate(fl, yl - G, tr["standoff"] + sum(
                (q["t"] * np.sqrt(q["er"]) if q["er"] > 1.01 else q["t"]) for q in trial) + 0.01) / sl) ** 2))
            if best is None or c < best[0]:
                best = (c, trial[-1])
        cur.append(best[1])
    return cur, slots


def _gate(f, r, rmax):
    out = np.zeros_like(r)
    for a, b in I.band_segments(f):
        Op = I.Gate(-0.01, rmax).op(f[a:b])
        out[a:b] = np.conj(Op).T @ (Op @ r[a:b])
    return out


def refine(lay, slots, tr, f, y, s, N):
    ST.SBOX = (tr["standoff"] - 0.004, tr["standoff"] + 0.004)
    m = ST.build(lay, tr["standoff"], back=tr["back"])
    ST.SBOX = None
    for (kind, _), L in zip(slots, m.layers):
        if kind == "gap":
            L.eps_r.free = L.tan_d.free = False
            L.eps_r.value, L.tan_d.value = 1.0, 0.0
    ST.set_dispersion_free(m, True)
    pri = ML.LibPrior(); old = SR._PRIORS; SR._PRIORS = (pri,)
    try:
        m, c = ST._fit(m, f, y, s)
        for L in m.layers:
            if L.eps_r.free:
                L.pinf.value, L.pinf.free = max(L.pinf.value, 0.02), True
        m, c = ST._fit(m, f, y, s)
    finally:
        SR._PRIORS = old
    G, _ = I.forward(m, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / s) ** 2)) / (2 * f.size)
    ok = chi < 1.3
    j = 0
    for (kind, i), L in zip(slots, m.layers):
        if kind == "solid":
            tt = tr["layers"][i]["t"]
            ok &= abs(L.t.value - tt) <= max(0.3e-3, 0.02 * tt)
    return bool(ok), chi


if __name__ == "__main__":
    N = 3
    res = {"amp": [], "sweep": [], "peel": []}
    for sd in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        f, y, s, tr = WS.make_problem(sd, N)
        row = {}
        for name, fn in (("amp", init_amp), ("sweep", init_sweep), ("peel", init_peel)):
            t0 = time.time()
            try:
                lay, slots = fn(tr, f, y, s, N)
                ok, chi = refine(lay, slots, tr, f, y, s, N)
            except Exception as e:
                ok, chi = False, float("nan")
            res[name].append(ok); row[name] = (ok, round(chi, 2), round(time.time() - t0, 1))
        print(sd, [L["name"][:6] for L in tr["layers"]], row, flush=True)
    print({k: f"{sum(v)}/{len(v)}" for k, v in res.items()})
