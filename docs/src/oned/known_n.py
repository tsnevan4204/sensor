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
        best = None
        for back in ("air", "metal"):
            ST.SBOX = (d - 0.008, d + 0.008)
            mod = ST.build(lay, d, back=back)
            ST.SBOX = None
            ST.set_dispersion_free(mod, True)          # p = 0 on its bound: constant-Q geometry first
            mod, c = ST._fit(mod, f, y, sigma)
            mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
            if best is None or c < best[1]:
                best = (mod, c)
        mod, c = best
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
