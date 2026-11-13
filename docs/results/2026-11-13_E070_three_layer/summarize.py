"""E070 summary: python3 summarize.py run.jsonl [--map]   (--map: noise-free confusion-map view)"""
import sys, json
import numpy as np
from collections import Counter, defaultdict

CL = ("gypsum", "wood_board", "fibrous", "foam", "tile")
AB = {"gypsum": "gyp", "wood_board": "wood", "fibrous": "fib", "foam": "foam", "tile": "tile"}
R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
crash = [r for r in R if "crash" in r]
R = [r for r in R if "crash" not in r]
n = len(R)
print(f"{sys.argv[1]}: {n} walls ({len(crash)} crashed)")
if not n:
    sys.exit()
st = Counter(r["status"] for r in R)
joint = sum(r["joint"] for r in R)
seq_ok = sum(r["seq_ok"] for r in R)
silent = [r for r in R if r["silent"]]
ans = [r for r in R if r["status"] == "answer"]
print(f"  exact ordered sequence (best answer, before abstention): {seq_ok}/{n}")
print(f"  joint success (labels + all 3 thicknesses within max(0.5 mm, 2 %)): {joint}/{n}")
print(f"  status: {dict(st)}")
print(f"  answered: {len(ans)}  of which joint-correct {sum(r['joint'] for r in ans)};  CONFIDENTLY WRONG (silent): {len(silent)}")
ab = [r for r in R if r["status"] != "answer"]
print(f"  abstained/alarmed: {len(ab)}  (their best answer was joint-correct in {sum(r['joint'] for r in ab)})")
for r in silent:
    o = r.get("oracle") or {}
    print(f"    SILENT {r['seed']}: {'|'.join(r['truth'])} {r['t_true_mm']} -> {'|'.join(r['fit'])} {r['t_fit_mm']} chi2n {r['chi2n']:.3f}"
          f"  oracle: search_miss={o.get('search_miss')} true-seq cost vs answer {o.get('dcost_vs_best', float('nan')):+.1f}")
orc = [r["oracle"] for r in R if r.get("oracle")]
if orc:
    miss = [r for r in R if r.get("oracle") and r["oracle"]["search_miss"]]
    print(f"  ORACLE (true sequence fitted from the truth): search missed the true sequence's best fit on {len(miss)}/{len(orc)} walls"
          + (": " + ", ".join(f"{r['seed']} (own {r['oracle']['own_cost'] - r['oracle']['cost']:+.1f})" for r in miss) if miss else ""))
    wrong = [r for r in R if r.get("oracle") and not r["seq_ok"]]
    info = [r for r in wrong if not r["oracle"]["search_miss"]]
    print(f"  wrong best sequence: {len(wrong)}; of these the search found the true sequence's optimum on {len(info)} "
          f"(= data/prior preferred a wrong label: identifiability), missed it on {len(wrong) - len(info)}")
# per-layer confusion
M = np.zeros((5, 5), int)
for r in R:
    for a, b in zip(r["truth"], r["fit"]):
        M[CL.index(a), CL.index(b)] += 1
print("  per-layer confusion (rows truth, cols fit):  " + " ".join(f"{AB[c]:>5}" for c in CL))
for i, c in enumerate(CL):
    print(f"    {AB[c]:>5}  " + " ".join(f"{M[i, j]:5d}" for j in range(5)))
# thickness
E = defaultdict(list)
for r in R:
    if r["seq_ok"]:
        for c, e in zip(r["truth"], r["t_err_mm"]):
            E[c].append(abs(e))
allE = [e for v in E.values() for e in v]
if allE:
    print(f"  |thickness error| on right-label walls: median {np.median(allE):.3f} mm, 90% {np.percentile(allE, 90):.3f}, max {max(allE):.3f}")
    for c in CL:
        if E[c]:
            print(f"    {c:10s} n={len(E[c]):3d} median {np.median(E[c]):.3f} 90% {np.percentile(E[c], 90):.3f} max {max(E[c]):.3f} mm")
d = [abs(r["d_err_mm"]) for r in R]
print(f"  |standoff error|: median {np.median(d):.3f} mm, max {max(d):.3f} mm")
ws = [r["wall_s"] for r in R]
print(f"  runtime: median {np.median(ws):.0f} s, max {max(ws):.0f} s (1 core per wall); exact fits/wall median {np.median([r['n_exact'] for r in R]):.0f},"
      f" hops/wall median {np.median([r['n_hop'] for r in R]):.0f}, grid walls/wall median {np.median([r['n_grid'] for r in R]):.2e}")
tm = defaultdict(list)
for r in R:
    for k, v in r["timing"].items():
        tm[k].append(v)
print("  stage time medians: " + ", ".join(f"{k} {np.median(v):.1f}s" for k, v in tm.items()))
oob = [r["seed"] for r in R if not r["in_bounds"]]
print(f"  answers outside the thickness bounds: {len(oob)} {oob}")
hg = [h for r in R for h in r.get("hop_gains", [])]
if hg:
    big = [h for h in hg if h["gain"] > 5]
    print(f"  cycle hops that lowered a sequence's cost by > 5: {len(big)}; largest dcost-before of such a sequence: "
          f"{max([h['dcost_before'] for h in big], default=0):.0f}")
# by sequence class of difficulty
grp = defaultdict(lambda: [0, 0, 0])
for r in R:
    k = "|".join(AB[c] for c in r["truth"])
    grp[k][0] += 1; grp[k][1] += r["joint"]; grp[k][2] += r["silent"]
worst = sorted(grp.items(), key=lambda kv: kv[1][1] / kv[1][0])[:12]
print("  hardest truth sequences (n, joint, silent): " + "; ".join(f"{k} {v}" for k, v in worst))

if "--map" in sys.argv:
    # noise-free confusion map: Delta(a) = min cost of every other sequence on the noise-free data of
    # the truth (cost of the true sequence ~ prior only); a wrong sequence within the noise scale cannot
    # be told apart by any optimiser
    print("\n  CONFUSION MAP (noise-free data; Delta = best MAP cost of a wrong sequence - that of the true one)")
    pairs = Counter()
    dmins = []
    for r in R:
        ps = r["per_seq"]
        kt = "|".join(r["truth"])
        if kt not in ps:
            print(f"    {r['seed']}: true sequence not refitted"); continue
        c0 = ps[kt]["cost"]
        wrong = sorted((v["cost"] - c0, k) for k, v in ps.items() if k != kt)
        dmins.append((wrong[0][0], r["seed"], kt, wrong[0][1]))
        for dlt, k in wrong:
            if dlt < 25:
                a, b = kt.split("|"), k.split("|")
                for i in range(3):
                    if a[i] != b[i]:
                        pairs[(i, AB[a[i]], AB[b[i]])] += 1
    dm = np.array([x[0] for x in dmins])
    for thr in (0, 10, 25, 100):
        print(f"    walls whose closest wrong sequence has Delta < {thr:>3}: {int(np.sum(dm < thr))}/{len(dm)}")
    print("    closest wrong sequence per wall (Delta, seed, truth -> rival):")
    for x in sorted(dmins)[:25]:
        print(f"      {x[0]:9.1f}  {x[1]:4d}  {x[2]} -> {x[3]}")
    print("    layer-class substitutions among rivals with Delta < 25 (layer index, truth -> rival): "
          + ", ".join(f"L{i + 1} {a}->{b} x{c}" for (i, a, b), c in pairs.most_common()))
