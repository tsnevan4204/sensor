"""
E055: tiny simulation-trained initialiser (numpy MLP, < 10k parameters) for the known-N (N = 3) wall.

Input : calibrated reflection with the KNOWN approximate standoff phase removed, y * exp(+2j k0 d_guess),
        on 32 low-band + 64 UWB frequencies (real, imag) -> 192 numbers.
Output: one parameter vector in wallsolve.layout coordinates: per layer (log optical thickness,
        log(eps'-1), log tan d, u), gaps (<= 2.5 mm means none), standoff offset d - d_guess, backing logit.
Training: walls from the same physics generator, seeds 100000+ (disjoint from dev 400-449 and the
held-out 600+), d_guess = d + U(-2.5, 2.5) mm. MSE on standardised targets + BCE for the backing.
The network only PROPOSES; the exact solver (LS polish + cycle scan, or PT seeded with it) decides.
python3 e055_nn.py gen|train|evalA|evalB
"""
from __future__ import annotations
import sys
import time
import json
import numpy as np
import wallsolve as WS

C0 = WS.C0
N = 3
NG = N                     # metal layout has N gaps; air uses the first N-1
D_OUT = 4 * N + NG + 2
DATA = "/home/user/sensor/docs/results/2026-11-10_E055_nn/train.npz"
WTS = "/home/user/sensor/docs/results/2026-11-10_E055_nn/weights.npz"


def features(f, y, d_guess):
    idx = []
    import wall_inverse as I
    segs = I.band_segments(f)
    for (a, b), n in zip(segs[:2], (32, 64)):
        idx += list(np.linspace(a, b - 1, n).astype(int))
    idx = np.array(idx)
    z = y[idx] * np.exp(2j * 2 * np.pi * f[idx] / C0 * d_guess)
    return np.concatenate([z.real, z.imag])


def target(tr, d_guess):
    t = []
    for L in tr["layers"]:
        er, td, p = L["eps10"], L["tan10"], min(L.get("p", 0.0), 0.999)
        t += [np.log(L["t"] * np.sqrt(er)), np.log(max(er - 1, 1e-4)), np.log(max(td, 1e-4)), -np.log(max(1 - p, 1e-3))]
    g = list(tr["gaps"]) + [0.0] * (NG - len(tr["gaps"]))
    t += [gi if gi > 0 else -5e-3 for gi in g[:NG]]
    t += [tr["standoff"] - d_guess, 1.0 if tr["back"] == "metal" else 0.0]
    return np.array(t)


def gen(n=20000, seed0=100000):
    rng = np.random.default_rng(7)
    X, Y = [], []
    t0 = time.time()
    for k in range(n):
        f, y, s, tr = WS.make_problem(seed0 + k, N)
        dg = tr["standoff"] + rng.uniform(-2.5e-3, 2.5e-3)
        X.append(features(f, y, dg)); Y.append(target(tr, dg))
        if k % 2000 == 0:
            print(k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(DATA, X=np.array(X), Y=np.array(Y))


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def forward(self, x):
        hs = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            hs.append(x)
        return x, hs

    def backward(self, hs, g):
        gW, gb = [], []
        for i in range(len(self.W) - 1, -1, -1):
            gW.insert(0, hs[i].T @ g); gb.insert(0, g.sum(0))
            if i > 0:
                g = (g @ self.W[i].T) * (1 - hs[i] ** 2)
        return gW, gb


def train(epochs=400, bs=256, lr=2e-3, hidden=(32, 32)):
    d = np.load(DATA)
    X, Y = d["X"], d["Y"]
    n = X.shape[0]
    nv = n // 10
    xm, xs = X[nv:].mean(0), X[nv:].std(0) + 1e-9
    reg = Y[:, :-1]
    ym, ys = reg[nv:].mean(0), reg[nv:].std(0) + 1e-9
    Xn = (X - xm) / xs
    Yn = (reg - ym) / ys
    B = Y[:, -1]
    rng = np.random.default_rng(0)
    net = MLP([X.shape[1], *hidden, D_OUT], rng)
    print("parameters:", net.n_params(), flush=True)
    m = [np.zeros_like(w) for w in net.W + net.b]; v = [np.zeros_like(w) for w in net.W + net.b]
    step = 0
    for ep in range(epochs):
        perm = nv + rng.permutation(n - nv)
        for a in range(0, perm.size, bs):
            j = perm[a:a + bs]
            out, hs = net.forward(Xn[j])
            g = np.zeros_like(out)
            g[:, :-1] = 2 * (out[:, :-1] - Yn[j]) / (j.size * (D_OUT - 1))
            pb = 1 / (1 + np.exp(-out[:, -1]))
            g[:, -1] = (pb - B[j]) / j.size
            gW, gb = net.backward(hs, g)
            step += 1
            for k, (p_, g_) in enumerate(zip(net.W + net.b, gW + gb)):
                m[k] = 0.9 * m[k] + 0.1 * g_; v[k] = 0.999 * v[k] + 0.001 * g_ ** 2
                mh, vh = m[k] / (1 - 0.9 ** step), v[k] / (1 - 0.999 ** step)
                p_ -= lr * mh / (np.sqrt(vh) + 1e-8)
        if ep % 50 == 0 or ep == epochs - 1:
            out, _ = net.forward(Xn[:nv])
            mse = np.mean((out[:, :-1] - Yn[:nv]) ** 2, axis=0)
            acc = np.mean((out[:, -1] > 0) == (B[:nv] > 0.5))
            print(f"ep {ep}  val MSE (std units) mean {mse.mean():.3f}  per-output {np.round(mse, 2)}  back acc {acc:.3f}", flush=True)
    np.savez(WTS, *net.W, *net.b, xm=xm, xs=xs, ym=ym, ys=ys, nl=len(net.W))


def load():
    d = np.load(WTS)
    nl = int(d["nl"])
    W = [d[f"arr_{i}"] for i in range(nl)]; b = [d[f"arr_{nl + i}"] for i in range(nl)]
    net = MLP([1], np.random.default_rng(0)); net.W, net.b = W, b
    return net, d["xm"], d["xs"], d["ym"], d["ys"]


def predict(f, y, d_guess, model=None):
    """-> (x for the air layout, x for the metal layout, P(metal))"""
    net, xm, xs, ym, ys = model or load()
    out, _ = net.forward(((features(f, y, d_guess) - xm) / xs)[None, :])
    r = out[0, :-1] * ys + ym
    pm = float(1 / (1 + np.exp(-out[0, -1])))
    xs_ = {}
    for back in ("air", "metal"):
        lo, hi, ng = WS.layout(N, back)
        x = np.zeros(lo.size)
        x[0] = d_guess + r[-1]
        for i in range(N):
            x[1 + 5 * i: 5 + 5 * i] = r[4 * i: 4 * i + 4]
            x[5 + 5 * i] = -6.0
        for g in range(ng):
            gv = r[4 * N + g]
            x[1 + 5 * N + g] = gv if gv > 2.5e-3 else -5e-3
        x[-1] = 20.0
        xs_[back] = np.clip(x, lo, hi)
    return xs_["air"], xs_["metal"], pm


def evalA(seeds, out):
    """NN guess -> exact LS polish (both backings) + mmWave cycle scan; no global search."""
    import xsolve as XS
    model = load()
    rec = []
    for sd in seeds:
        views, tr = XS.make_views(sd, N, (0.0,))
        f, y, s, _ = views[0]
        dg = tr["standoff"] + 0.001
        t = time.time()
        xa, xm_, pm = predict(f, y, dg, model)
        best = None
        for back, x0 in (("air", xa), ("metal", xm_)):
            pol = XS.Polisher(views, N, back, dg)
            xp, c, _ = pol.polish(x0)
            xp, c = XS.cycle_scan(pol, xp)
            if best is None or c < best[0]:
                best = (c, xp, back, pol)
        c, x, back, pol = best
        import pt as PT
        lay, slots = PT.x_to_lay(x, N, back)
        res = dict(lay=lay, slots=slots, back=back, labels=XS.class_evidence(pol, x))
        sc = XS.score(res, tr, N)
        n_all = 2 * f.size
        r = dict(seed=sd, ok=sc["ok"], geo_ok=sc["geo_ok"], chi2n=c / n_all, s=time.time() - t, pm=pm,
                 t_err_mm=sc["t_err_mm"])
        rec.append(r)
        open(out, "a").write(json.dumps(r, default=float) + "\n")
        print(sd, r["ok"], r["geo_ok"], round(r["chi2n"], 2), f"{r['s']:.1f}s", flush=True)
    return rec


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    elif cmd == "train":
        train()
    elif cmd == "evalA":
        evalA(range(int(sys.argv[2]), int(sys.argv[3]) + 1), sys.argv[4])
