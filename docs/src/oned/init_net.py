"""
E028: learned direct initializer for the two-band layered-wall inversion.

The network only proposes a STARTING POINT; the answer is always the exact-star LM optimum, and
a chi2 failure falls back to the full search. So network errors cost time, never correctness
(roadmap 1.1 "amortized initialization").

Pipeline
  1. d_hat: standoff from the matrix pencil (front echo) on the top band, decimated x4.
  2. features: 32 frequencies per band (every 8th), multiplied by exp(+2j k0 d_hat) (removes the
     air path up to the residual delta = d - d_hat), real and imaginary parts -> 128 numbers.
  3. MLP 128 -> 48 -> 48 -> 8 (tanh), 8.9k parameters, outputs scaled to [-1, 1] by the prior box:
        delta, tau_front, eps_d, tan_d, tau_ins, eps_i, tan_i, tau_back
     (optical thickness tau = Re(n) t: the coordinates the data determine best, E011).
  4. Training data: walls drawn from a prior WIDER than the benchmark, causal (constant-Q) materials,
     two bands with noise, rng seeds disjoint from all benchmark seeds.
"""
from __future__ import annotations
import os
import numpy as np
import wall_toy as W
import wall_inverse as I

C0 = W.C0
WEIGHTS = os.path.join(os.path.dirname(__file__), "init_net_weights.npz")
PRIOR = dict(standoff=(0.28, 0.32), t_dw=(0.0095, 0.0165), eps_dw=(2.0, 3.6), tan_dw=(0.005, 0.06),
             t_ins=(0.081, 0.099), eps_ins=(1.0, 1.4), tan_ins=(0.002, 0.03))
DELTA = 6.0e-3            # half-range of d - d_hat covered (d_hat errors: median 0.5 mm, max 5.2 mm on 100 walls)


def n_re(er, td):
    return np.sqrt(er * (1 - 1j * td)).real


OUT_BOX = np.array([[-DELTA, DELTA],
                    [PRIOR["t_dw"][0] * n_re(PRIOR["eps_dw"][0], 0), PRIOR["t_dw"][1] * n_re(PRIOR["eps_dw"][1], 0.06)],
                    list(PRIOR["eps_dw"]), list(PRIOR["tan_dw"]),
                    [PRIOR["t_ins"][0] * n_re(PRIOR["eps_ins"][0], 0), PRIOR["t_ins"][1] * n_re(PRIOR["eps_ins"][1], 0.03)],
                    list(PRIOR["eps_ins"]), list(PRIOR["tan_ins"]),
                    [PRIOR["t_dw"][0] * n_re(PRIOR["eps_dw"][0], 0), PRIOR["t_dw"][1] * n_re(PRIOR["eps_dw"][1], 0.06)]])


def d_hat(f, y, sigma):
    """Standoff estimate: strongest early echo of a matrix pencil on the top band (x4 decimated)."""
    import pencil as PC
    a, b = max(I.band_segments(f), key=lambda ab: f[ab[1] - 1])
    ft, yt = f[a:b][::4], y[a:b][::4]
    r, p = W.range_profile(ft, yt, n_fft=2048)
    sel = (r > 0.15) & (r < 0.6)
    d0 = r[sel][np.argmax(p[sel])]
    rr, c, M = PC.echoes(ft, yt, sigma * 2, d0 - 0.06, d0 + 0.06)
    if rr.size == 0:
        return d0
    big = np.abs(c) > 0.3 * np.abs(c).max()
    return float(rr[big][0])


N_IN = None               # set from the data by train()
FEATURES = "range"      # E028b: windowed range profile ("spectrum" = E028 first version)


def features(f, y, dh):
    """Derotated data (air path removed with d_hat). "range": per band, Hann-windowed, 4x zero-padded
    IFFT, keeping the complex range bins from -2 cm to +22 cm behind the front face (the wall only):
    each echo lands in its own inputs. "spectrum": 32 frequencies per band."""
    k0 = 2 * np.pi * f / C0
    z = []
    for a, b in I.band_segments(f):
        fb = f[a:b]
        zb = y[a:b] * np.exp(2j * k0[a:b] * dh)
        if FEATURES == "spectrum":
            z.append(zb[::8][:32]); continue
        n = 4 * zb.size
        prof = np.fft.ifft(zb * np.hanning(zb.size), n) * np.exp(-2j * np.pi * fb[0] * 0)   # complex profile
        r = np.arange(n) * C0 / (2 * (fb[1] - fb[0]) * n)
        r = np.where(r > r.max() / 2, r - r.max() - r[1], r)                                  # wrap negatives
        sel = (r >= -0.02) & (r <= 0.22)
        z.append(prof[sel][np.argsort(r[sel])] * np.sqrt(zb.size))
    z = np.concatenate(z) / 0.3
    return np.concatenate([z.real, z.imag])


def scale(v):
    return 2 * (v - OUT_BOX[:, 0]) / (OUT_BOX[:, 1] - OUT_BOX[:, 0]) - 1


def unscale(u):
    return OUT_BOX[:, 0] + (np.clip(u, -1, 1) + 1) / 2 * (OUT_BOX[:, 1] - OUT_BOX[:, 0])


# ----------------------------------------------------------------------------- data
def draw_wall(rng):
    U = lambda key: rng.uniform(*PRIOR[key])
    tv = {"drywall_front.t": U("t_dw"), "drywall_back.t": U("t_dw"), "insulation.t": U("t_ins"),
          "drywall_front.eps_r": U("eps_dw"), "drywall_front.tan_d": U("tan_dw"),
          "insulation.eps_r": U("eps_ins"), "insulation.tan_d": U("tan_ins"), "standoff": U("standoff")}
    tv["drywall_back.eps_r"], tv["drywall_back.tan_d"] = tv["drywall_front.eps_r"], tv["drywall_front.tan_d"]
    return tv


def simulate(tv, seed, band2=(57e9, 64e9)):
    import realism as RL
    R = RL.Realism(dispersion="cq")
    fs, ys, s2 = [], [], []
    for radar in (W.Radar(seed=seed), W.Radar(f0=band2[0], f1=band2[1], seed=seed + 1)):
        f, yc, sg = RL.measure_band(tv, radar, R)
        fs.append(f); ys.append(yc); s2.append(sg**2)
    return np.concatenate(fs), np.concatenate(ys), float(np.sqrt(np.mean(s2)))


def target(tv, dh):
    nd = n_re(tv["drywall_front.eps_r"], tv["drywall_front.tan_d"])
    ni = n_re(tv["insulation.eps_r"], tv["insulation.tan_d"])
    return np.array([tv["standoff"] - dh, nd * tv["drywall_front.t"], tv["drywall_front.eps_r"],
                     tv["drywall_front.tan_d"], ni * tv["insulation.t"], tv["insulation.eps_r"],
                     tv["insulation.tan_d"], nd * tv["drywall_back.t"]])


def make_dataset(n, seed0=10_000_000):
    X, Y = [], []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        tv = draw_wall(rng)
        f, y, s = simulate(tv, seed0 + 2 * i + 1)
        dh = d_hat(f, y, s)
        t = target(tv, dh)
        if abs(t[0]) > DELTA:            # pencil missed the front face: outside the network's range
            continue
        X.append(features(f, y, dh)); Y.append(scale(t))
    return np.array(X), np.array(Y)


# ----------------------------------------------------------------------------- MLP (numpy)
def init_params(rng, sizes=(128, 48, 48, 8)):
    P = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        P += [rng.standard_normal((a, b)) * np.sqrt(1 / a), np.zeros(b)]
    return P


def forward_mlp(P, X):
    h = X
    acts = [h]
    for i in range(0, len(P) - 2, 2):
        h = np.tanh(h @ P[i] + P[i + 1]); acts.append(h)
    return h @ P[-2] + P[-1], acts


def train(X, Y, epochs=400, batch=256, lr=2e-3, seed=0, Xv=None, Yv=None, log=print, hidden=48):
    global N_IN
    N_IN = X.shape[1]
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    P = init_params(rng, sizes=(N_IN, hidden, hidden, Y.shape[1]))
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(len(Xn))
        for k in range(0, len(Xn), batch):
            idx = perm[k:k + batch]
            out, acts = forward_mlp(P, Xn[idx])
            g = 2 * (out - Y[idx]) / len(idx)
            grads = [None] * len(P)
            grads[-2] = acts[-1].T @ g; grads[-1] = g.sum(0)
            d = g @ P[-2].T
            for i in range(len(P) - 4, -1, -2):
                d = d * (1 - acts[i // 2 + 1] ** 2)
                grads[i] = acts[i // 2].T @ d; grads[i + 1] = d.sum(0)
                d = d @ P[i].T
            t += 1
            lr_t = lr * (0.1 ** (ep / epochs))
            for j in range(len(P)):
                m[j] = 0.9 * m[j] + 0.1 * grads[j]; v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
                P[j] -= lr_t * (m[j] / (1 - 0.9**t)) / (np.sqrt(v[j] / (1 - 0.999**t)) + 1e-8)
        if Xv is not None and (ep % 50 == 0 or ep == epochs - 1):
            pv, _ = forward_mlp(P, (Xv - mu) / sd)
            log(f"epoch {ep}: val rmse (scaled) {np.sqrt(np.mean((pv - Yv) ** 2, axis=0)).round(3)}")
    return P, mu, sd


def save(P, mu, sd, path=WEIGHTS):
    np.savez(path, mu=mu, sd=sd, **{f"P{i}": p for i, p in enumerate(P)})


_CACHE = {}


def predict(f, y, sigma):
    if "w" not in _CACHE:
        z = np.load(WEIGHTS)
        _CACHE["w"] = ([z[f"P{i}"] for i in range(len([k for k in z.files if k.startswith('P')]))], z["mu"], z["sd"])
    P, mu, sd = _CACHE["w"]
    dh = d_hat(f, y, sigma)
    out, _ = forward_mlp(P, ((features(f, y, dh) - mu) / sd)[None, :])
    v = unscale(out[0])
    return dh, v


def n_params():
    z = np.load(WEIGHTS)
    return int(sum(z[k].size for k in z.files if k.startswith("P")))



# ----------------------------------------------------------------------------- E029: phase-aware outputs
FC1, FC2 = 79e9, 60.5e9                       # band carriers (77-81 GHz, 57-64 GHz)
P_NARROW = C0 / (2 * FC1)                      # 1.90 mm: one cycle of the top band in optical thickness
P_WIDE = C0 / (2 * (FC1 - FC2))                # 8.10 mm: wide lane (band-difference) cycle
TAU_IDX = (1, 4, 7)                            # tau_front, tau_ins, tau_back in the 8-vector
WEIGHTS_PHASE = os.path.join(os.path.dirname(__file__), "init_net_phase_weights.npz")


def target_phase(tv, dh):
    """8 coarse targets (scaled) + for each tau: cos, sin on the wide and narrow lanes (12)."""
    t = target(tv, dh)
    ph = []
    for j in TAU_IDX:
        for P in (P_WIDE, P_NARROW):
            ph += [np.cos(2 * np.pi * t[j] / P), np.sin(2 * np.pi * t[j] / P)]
    return np.concatenate([scale(t), ph])


def make_dataset_phase(n, seed0=10_000_000):
    X, Y = [], []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        tv = draw_wall(rng)
        f, y, s = simulate(tv, seed0 + 2 * i + 1)
        dh = d_hat(f, y, s)
        if abs(tv["standoff"] - dh) > DELTA:
            continue
        X.append(features(f, y, dh)); Y.append(target_phase(tv, dh))
    return np.array(X), np.array(Y)


def decode(out):
    """Cascade integer resolution: coarse -> wide lane -> narrow lane for each optical thickness."""
    v = unscale(out[:8])
    k = 8
    for j in TAU_IDX:
        phw = np.arctan2(out[k + 1], out[k]) / (2 * np.pi)
        phn = np.arctan2(out[k + 3], out[k + 2]) / (2 * np.pi)
        k += 4
        tau_w = P_WIDE * (np.round(v[j] / P_WIDE - phw) + phw)
        v[j] = P_NARROW * (np.round(tau_w / P_NARROW - phn) + phn)
    return v


def predict_phase(f, y, sigma):
    if "wp" not in _CACHE:
        z = np.load(WEIGHTS_PHASE)
        _CACHE["wp"] = ([z[f"P{i}"] for i in range(len([q for q in z.files if q.startswith('P')]))], z["mu"], z["sd"])
    P, mu, sd = _CACHE["wp"]
    dh = d_hat(f, y, sigma)
    out, _ = forward_mlp(P, ((features(f, y, dh) - mu) / sd)[None, :])
    return dh, decode(out[0])



def predict_phase_raw(f, y, sigma):
    """Network outputs (decoded front, coarse values, and the narrow-lane phases of insulation and
    back drywall) for the integer search of E029."""
    if "wp" not in _CACHE:
        predict_phase(f, y, sigma)
    P, mu, sd = _CACHE["wp"]
    dh = d_hat(f, y, sigma)
    out, _ = forward_mlp(P, ((features(f, y, dh) - mu) / sd)[None, :])
    out = out[0]
    v = decode(out)
    coarse = unscale(out[:8])
    phn = {j: np.arctan2(out[8 + 4 * i + 3], out[8 + 4 * i + 2]) / (2 * np.pi) for i, j in enumerate(TAU_IDX)}
    return dh, v, coarse, phn
