"""
E044: inversion with a KNOWN number of layers N (the only user input); thicknesses, materials and the
backing are the outputs.

Why: every many-layer failure so far (E034 growth, E041 peeling, E042/E043 global inversion) came
from deciding the layer COUNT: the data admit many-layer junk explanations that fit to noise. With N
given, that degree of freedom is gone; what remains is a continuous fit with a few discrete choices
(backing, phase cycles).

Pipeline
 1. continuous slice profile (profile.invert_profile, E042: exact recursion, TV, frequency continuation)
 2. OPTIMAL segmentation of the profile into exactly N + 2 pieces by dynamic programming
    (front air = standoff | N layers | back region), minimising the within-piece weighted squared
    deviation of the three channels (log eps', log tan d, water logit): exact, O(K^2 N)
 3. exact layered fit of the N layers (stack.py model, TMM == DEC) under the material-library prior
    (matlib.py), backing tried as air and metal
 4. phase-cycle hops on every thickness (search.cycle_hop) + dispersion release; best MAP cost wins
 5. alarm on the DATA chi2 only
"""
from __future__ import annotations
import time
import numpy as np
import stack as ST
import search as SR
import wall_inverse as I
import profile as PF
import matlib as ML

CH_W = np.array([1.0, 0.3, 1.0])          # channel weights (as the TV weights in profile.py)


def segment_n(m, n_pieces):
    """Optimal piecewise-constant segmentation of m (3, K) into n_pieces contiguous pieces
    (dynamic programming over boundaries; cost = weighted within-piece SSE). Returns cut indices."""
    K = m.shape[1]
    X = (m * np.sqrt(CH_W)[:, None]).T                 # (K, 3)
    c1 = np.vstack([np.zeros(3), np.cumsum(X, 0)])
    c2 = np.vstack([np.zeros(3), np.cumsum(X ** 2, 0)])

    def sse(i, j):                                     # slices i..j-1
        n = j - i
        s1, s2 = c1[j] - c1[i], c2[j] - c2[i]
        return float(np.sum(s2 - s1 ** 2 / n))
    INF = np.inf
    D = np.full((n_pieces + 1, K + 1), INF)
    B = np.zeros((n_pieces + 1, K + 1), int)
    D[0, 0] = 0.0
    for p in range(1, n_pieces + 1):
        for j in range(p, K + 1):
            best, arg = INF, -1
            for i in range(p - 1, j):
                v = D[p - 1, i] + sse(i, j)
                if v < best:
                    best, arg = v, i
            D[p, j], B[p, j] = best, arg
    cuts = [K]
    for p in range(n_pieces, 0, -1):
        cuts.append(B[p, cuts[-1]])
    return cuts[::-1]                                  # [0, c1, ..., K]


def layers_from_cuts(m, z0, dz, cuts):
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        mm = m[:, a:b].mean(axis=1)
        pieces.append(dict(z=z0 + a * dz, t=(b - a) * dz, er=1 + np.exp(mm[0]), td=np.exp(mm[1]),
                           th=1 / (1 + np.exp(-mm[2]))))
    return pieces


def _to_layer(p):
    if p["th"] > 0.6:
        return dict(t=p["t"], er=1.0, td=0.0, water=True)
    return dict(t=max(p["t"], 5e-4), er=float(np.clip(p["er"], 1.0001, 11.9)),
                td=float(np.clip(p["td"], 1e-4, 0.49)), water=False)


def invert(f, y, sigma, N, verbose=False):
    t0 = time.time()
    m, z0, dz, info = PF.invert_profile(f, y, sigma)
    K = m.shape[1]
    cuts = segment_n(m, min(N + 2, K))
    pieces = layers_from_cuts(m, z0, dz, cuts)
    front, mids = pieces[0], pieces[1:-1]
    d = front["z"] + front["t"]                        # standoff = end of the front-air piece
    lay = [_to_layer(p) for p in mids][:N]
    while len(lay) < N:                                # profile too short: split the thickest layer
        i = int(np.argmax([l["t"] for l in lay]))
        a = dict(lay[i], t=lay[i]["t"] / 2)
        lay = lay[:i] + [a, dict(a)] + lay[i + 1:]
    pri = ML.LibPrior()
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    try:
        cands = []
        # candidate A: optimal N-piece segmentation of the profile
        for back in ("air", "metal"):
            ST.SBOX = (d - 0.008, d + 0.008)
            mod = ST.build(lay, d, back=back)
            ST.SBOX = None
            ST.set_dispersion_free(mod, True)          # p = 0 on its bound: constant-Q geometry first
            mod, c = ST._fit(mod, f, y, sigma)
            mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
            cands.append((c, mod, "A"))
        # candidate B: over-segment (threshold change points), exact fit, then reduce to EXACTLY N
        segs = PF.segment(m, z0, dz)
        if len(segs) > N:
            dB = segs[0][0]
            layB = [_to_layer(dict(t=t, er=er, td=td, th=th)) for z, t, er, td, th in segs]
            for back in ("air", "metal"):
                ST.SBOX = (dB - 0.008, dB + 0.008)
                mod = ST.build(layB, dB, back=back)
                ST.SBOX = None
                ST.set_dispersion_free(mod, True)
                mod, c = ST._fit(mod, f, y, sigma)
                mod, c = reduce_to(mod, c, N, f, y, sigma)
                mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
                cands.append((c, mod, "B"))
        c, mod, how = min(cands, key=lambda t: t[0])
        info["candidate"] = how
        rel = []
        for p0 in (0.02, 0.95):                        # release the dispersion shape at the end
            mm = mod.copy()
            for L in mm.layers:
                if L.theta.value < 0.5:
                    L.pinf.value, L.pinf.free = p0, True
            mm, cm = ST._fit(mm, f, y, sigma)
            rel.append((cm, mm))
        cm, mm = min(rel, key=lambda t: t[0])
        if cm < c:
            mod, c = mm, cm
    finally:
        SR._PRIORS = old
    G, _ = I.forward(mod, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
    n_all = 2 * f.size
    info.update(wall_s=time.time() - t0, chi2=chi, map_cost=c, alarm=bool(chi > ST.chi2_alarm(n_all)),
                labels=pri.labels(mod), classes=[{l: 1.0} for l in pri.labels(mod)])
    return mod, chi, info


def reduce_to(model, cost, N, f, y, sigma):
    """Agglomerative reduction of an over-segmented exact fit to EXACTLY N layers: at every step try
    every single-layer drop and every neighbour merge (either neighbour's material), screen each with
    a short exact fit, fully converge the cheapest, repeat. Leading air is folded into the standoff."""
    model = ST._drop_leading_air(model)
    while len(model.layers) > N:
        lay = ST.describe(model)
        trials = []
        for i in range(len(lay)):
            trials.append(lay[:i] + lay[i + 1:])
            if i + 1 < len(lay):
                tt = lay[i]["t"] + lay[i + 1]["t"]
                trials.append(lay[:i] + [dict(lay[i], t=tt)] + lay[i + 2:])
                trials.append(lay[:i] + [dict(lay[i + 1], t=tt)] + lay[i + 2:])
        best = None
        for L2 in trials:
            mm = ST.rebuild(model, layers=L2)
            ST.set_dispersion_free(mm, True)
            mm, cm = ST._fit(mm, f, y, sigma, atol=1e-2)
            if best is None or cm < best[1]:
                best = (mm, cm)
        model, cost = ST._fit(best[0], f, y, sigma)
        model = ST._drop_leading_air(model)
    while len(model.layers) < N:                      # never fewer than N
        lay = ST.describe(model)
        i = int(np.argmax([l["t"] for l in lay]))
        a = dict(lay[i], t=lay[i]["t"] / 2)
        model = ST.rebuild(model, layers=lay[:i] + [a, dict(a)] + lay[i + 1:])
        model, cost = ST._fit(model, f, y, sigma)
    return model, cost


# ============================================================================ E045: material-class search
WATER_N = 8.5        # |n| of liquid water across the UWB band (thickness conversion only)


def _class_options(p, lib, k=3):
    """Top-k library classes for one profile segment, plus water when the segment looks wet or very
    high-permittivity. Each option: (name, layer dict) with the optical thickness preserved."""
    x = np.array([np.log(max(p["er"], 1.0001)), np.log10(max(p["td"], 1e-6))])
    scores = []
    for nm, (mu, C) in lib.items():
        d = x - mu
        scores.append((float(d @ np.linalg.solve(C, d)), nm, mu))
    scores.sort(key=lambda t: t[0])
    tau = np.sqrt(max(p["er"], 1.0)) * p["t"]                 # optical thickness of the segment
    out = []
    for _, nm, mu in scores[:k]:
        er, td = float(np.exp(mu[0])), float(10 ** mu[1])
        out.append((nm, dict(t=max(tau / np.sqrt(er), 5e-4), er=er, td=min(td, 0.49), water=False)))
    if p["th"] > 0.3 or p["er"] > 8.0:
        out.append(("water", dict(t=max(tau / WATER_N, 5e-4), er=1.0, td=0.0, water=True)))
    if p["er"] < 1.3:
        out.append(("air", dict(t=max(tau, 5e-4), er=1.0001, td=1e-4, water=False)))
    return out


def invert_cls(f, y, sigma, N, k=3, beam=6, n_full=3, verbose=False):
    """E045: known N; search over library material sequences (exhaustive for k^N <= 300, else beam
    left to right), short exact fit per sequence, full exact fit + cycle hops for the best n_full."""
    import itertools
    t0 = time.time()
    m, z0, dz, info = PF.invert_profile(f, y, sigma)
    K = m.shape[1]
    cuts = segment_n(m, min(N + 2, K))
    pieces = layers_from_cuts(m, z0, dz, cuts)
    front, mids = pieces[0], pieces[1:-1][:N]
    while len(mids) < N:
        i = int(np.argmax([q["t"] for q in mids]))
        a = dict(mids[i], t=mids[i]["t"] / 2)
        mids = mids[:i] + [a, dict(a)] + mids[i + 1:]
    d = front["z"] + front["t"]
    lib = ST.library_stats()
    opts = [_class_options(q, lib, k) for q in mids]
    pri = ML.LibPrior()
    old = SR._PRIORS
    SR._PRIORS = (pri,)

    def quick(lay, back):
        ST.SBOX = (d - 0.008, d + 0.008)
        mod = ST.build(lay, d, back=back, water_T=20.0)
        ST.SBOX = None
        ST.set_dispersion_free(mod, True)
        return ST._fit(mod, f, y, sigma, atol=1e-2, max_iter=30)
    try:
        scored = []
        n_comb = int(np.prod([len(o) for o in opts]))
        if n_comb <= 300:
            for combo in itertools.product(*opts):
                lay = [c[1] for c in combo]
                for back in ("air", "metal"):
                    mod, c = quick(lay, back)
                    scored.append((c, mod, [c_[0] for c_ in combo], back))
        else:                                          # beam: fix layers left to right
            seqs = [([], [])]
            for i in range(N):
                rest = [opts[j][0][1] for j in range(i + 1, N)]          # best guess for the rest
                cand = []
                for names, lays in seqs:
                    for nm, L in opts[i]:
                        for back in ("air", "metal"):
                            mod, c = quick(lays + [L] + rest, back)
                            cand.append((c, names + [nm], lays + [L], back, mod))
                cand.sort(key=lambda t: t[0])
                seqs = [(c_[1], c_[2]) for c_ in cand[:beam]]
                if i == N - 1:
                    scored = [(c_[0], c_[4], c_[1], c_[3]) for c_ in cand]
        scored.sort(key=lambda t: t[0])
        best = None
        for c, mod, names, back in scored[:n_full]:
            mod, c = ST._fit(mod, f, y, sigma)
            mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
            if best is None or c < best[1]:
                best = (mod, c, names)
        mod, c, names = best
        rel = []
        for p0 in (0.02, 0.95):
            mm = mod.copy()
            for L in mm.layers:
                if L.theta.value < 0.5:
                    L.pinf.value, L.pinf.free = p0, True
            mm, cm = ST._fit(mm, f, y, sigma)
            rel.append((cm, mm))
        cm, mm = min(rel, key=lambda t: t[0])
        if cm < c:
            mod, c = mm, cm
    finally:
        SR._PRIORS = old
    G, _ = I.forward(mod, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
    n_all = 2 * f.size
    info.update(wall_s=time.time() - t0, chi2=chi, map_cost=c, alarm=bool(chi > ST.chi2_alarm(n_all)),
                labels=pri.labels(mod), classes=[{l: 1.0} for l in pri.labels(mod)], n_seq=len(scored))
    return mod, chi, info


def invert_combo(f, y, sigma, N):
    """E045b: material-class search first (fast); only if its chi2 alarm fires, run the E044
    pipeline (DP segmentation + agglomerative reduction) and keep the lower data chi2."""
    a = invert_cls(f, y, sigma, N)
    if not a[2]["alarm"]:
        a[2]["route"] = "classes"
        return a
    b = invert(f, y, sigma, N)
    pick = a if a[1] <= b[1] else b
    pick[2]["route"] = "classes+fallback:" + ("classes" if pick is a else "reduce")
    pick[2]["wall_s"] = a[2]["wall_s"] + b[2]["wall_s"]
    return pick
