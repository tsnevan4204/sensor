"""E070 figures: python3 make_figures.py   (re-run when the chain finishes; uses final runs when present)
Writes figures/*.png. Simulation results only (no measured data)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../src/oned"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
# reference palette (dataviz skill, light mode) - validated: all checks pass, aqua needs direct labels
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 13,
})
NICE = {"gypsum": "gypsum", "wood_board": "wood board", "fibrous": "fibrous insul.", "foam": "rigid foam", "tile": "ceramic tile"}


def load(name):
    for fn in (f"{name}.jsonl", f"v4_{name}.jsonl"):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            R = [json.loads(l) for l in open(p) if l.strip()]
            R = [r for r in R if "crash" not in r]
            if R:
                return R, fn
    return [], None


def foot(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


# ---------------------------------------------------------------- 1. outcomes
def outcomes():
    main, fm = load("main_ref")
    outv4 = [json.loads(l) for l in open(os.path.join(HERE, "v4_outscope_ref.jsonl"))] if os.path.exists(os.path.join(HERE, "v4_outscope_ref.jsonl")) else []
    outf, fo = load("outscope_ref")
    outf = outf if fo == "outscope_ref.jsonl" else []
    rows = []

    def cats_in(R):
        c = dict(ok=0, abst=0, alarm=0, wrong=0)
        for r in R:
            if r["status"] == "answer":
                c["ok" if r["joint"] else "wrong"] += 1
            elif r["status"] == "alarm":
                c["alarm"] += 1
            else:
                c["abst"] += 1
        return c

    def cats_out(R):   # out-of-scope: any committed answer is wrong by construction
        c = dict(ok=0, abst=0, alarm=0, wrong=0)
        for r in R:
            c["wrong" if r["status"] == "answer" else ("alarm" if r["status"] == "alarm" else "abst")] += 1
        return c
    rows.append((f"In-scope walls\n(board | any | board), n={len(main)}", cats_in(main)))
    if outv4:
        rows.append((f"Insulation on an outer face\n(solver doesn't know), n={len(outv4)}\nfit check only", cats_out(outv4)))
    if outf:
        rows.append((f"Same, + material-out-of-class\nalarm, n={len(outf)}", cats_out(outf)))
    a80, fa = load("all80_ref")
    if fa == "all80_ref.jsonl":
        rows.append((f"All 80 sequences allowed\n(incl. outer insulation), n={len(a80)}", cats_in(a80)))
    fig, ax = plt.subplots(figsize=(10, 1.25 + 1.15 * len(rows)))
    keys = [("ok", "correct answer", GOOD, "white"), ("abst", "abstained (\"can't tell\")", NEUTRAL, INK),
            ("alarm", "alarm (\"doesn't fit\")", WARN, INK), ("wrong", "confidently wrong", CRIT, "white")]
    for i, (lab, c) in enumerate(rows[::-1]):
        n = sum(c.values()); x = 0
        for k, nm, col, tc in keys:
            v = c[k] / n
            if v > 0:
                ax.barh(i, v, left=x, height=0.62, color=col, edgecolor=SURF, linewidth=2)
                if v > 0.035:
                    ax.text(x + v / 2, i, f"{c[k]}", ha="center", va="center", color=tc, fontsize=10.5, fontweight="bold")
                x += v
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=10, color=INK2)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=col, label=nm) for _, nm, col, _ in keys]
    ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9.5)
    ax.set_title("Three-layer wall inversion: what the solver says on each wall", pad=30)
    foot(fig, f"Simulation, 3.1-10.6 + 60-64 GHz, 40 dB SNR. Reference solver over all allowed sequences. Data: {fm}"
              + (", v4_outscope_ref.jsonl" if outv4 else "") + (", outscope_ref.jsonl" if outf else "") + ", all80_ref.jsonl")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "1_outcomes.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 2. thickness accuracy
def thickness():
    R, fn = load("main_ref")
    cls = ["tile", "gypsum", "wood_board", "foam", "fibrous"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rng = np.random.default_rng(0)
    for i, c in enumerate(cls):
        e = [abs(x) for r in R if r["seq_ok"] for cc, x in zip(r["truth"], r["t_err_mm"]) if cc == c]
        e = np.maximum(np.array(e) * 1e3, 0.5)          # um, floor for log axis
        ax.scatter(e, i + rng.uniform(-0.18, 0.18, e.size), s=26, color=S1, alpha=0.75, edgecolor=SURF, linewidth=0.8, zorder=3)
        ax.plot([np.median(e)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
    ax.axvline(500, color=CRIT, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.text(520, len(cls) - 0.55, "tolerance floor 0.5 mm\n(max(0.5 mm, 2 %))", color=INK2, fontsize=9, va="top")
    ax.set_xscale("log"); ax.set_xlim(0.4, 3000)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1 µm", "10 µm", "0.1 mm", "1 mm"])
    ax.set_yticks(range(len(cls))); ax.set_yticklabels([NICE[c] for c in cls], color=INK2)
    ax.grid(axis="y", visible=False)
    allE = [abs(x) for r in R if r["seq_ok"] for x in r["t_err_mm"]]
    ax.set_title(f"Thickness error per layer (median {1e3 * np.median(allE):.0f} µm, worst {max(allE):.2f} mm)")
    ax.set_xlabel("|estimated - true thickness|   (black bar = median)")
    foot(fig, f"Simulation: {len(R)} in-scope walls, layers whose sequence was right. Data: {fn}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "2_thickness_accuracy.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 3. why labels are ambiguous
def ambiguity():
    import tri3 as T
    T.install()
    rng = np.random.default_rng(5)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for ax, pair, title in ((axs[0], ("gypsum", "wood_board"), "Boards: gypsum vs wood board"),
                            (axs[1], ("foam", "fibrous"), "Insulation: rigid foam vs fibrous")):
        for c, col in zip(pair, (S1, S2)):
            X = np.array([[d["eps10"], d["tan10"]] for d in (T.material_law(c, rng)[1] for _ in range(700))])
            ax.scatter(X[:, 0], X[:, 1], s=9, color=col, alpha=0.55, edgecolor="none", label=NICE[c])
        ax.set_yscale("log"); ax.set_xlabel("ε′ at 10 GHz"); ax.set_ylabel("loss tangent at 10 GHz")
        ax.set_title(title, fontsize=12)
        ax.legend(frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=9.5, loc="upper left" if pair[0] == "foam" else "lower right", markerscale=2.5)
    R, fn = load("main_ref")
    gaps = []
    for r in R:
        ps = r["per_seq"]; kt = "|".join(r["truth"])
        if kt in ps:
            c0 = ps[kt]["cost"]
            w = min(v["cost"] for k, v in ps.items() if k != kt)
            gaps.append(max(w - c0, 0.05))
    gaps = np.array(gaps)
    ax = axs[2]
    bins = np.logspace(np.log10(0.05), 5, 29)
    ax.hist(gaps, bins=bins, color=S1, edgecolor=SURF, linewidth=1.5)
    ax.axvline(10, color=CRIT, lw=1.6, ls=(0, (4, 3)))
    k = int(np.sum(gaps < 10))
    ax.text(12, ax.get_ylim()[1] * 0.92, f"tie window (Δ < 10):\n{k} of {gaps.size} walls have a wrong\nsequence that fits as well", fontsize=9, color=INK2, va="top")
    ax.set_xscale("log"); ax.set_xlabel("misfit of closest WRONG sequence − true sequence")
    ax.set_ylabel("walls"); ax.set_title("How close is the nearest impostor?", fontsize=12)
    fig.suptitle("Why some labels can't be told apart: class material distributions overlap", x=0.01, ha="left", fontweight="bold", fontsize=13)
    foot(fig, f"Left/middle: 700 samples per class from the simulation's material models. Right: noisy in-scope walls, data {fn}.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, "3_label_ambiguity.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 4. example fit
def example(seed=2024):
    import tri3 as T, wall_inverse as I, wall_toy as W
    T.install()
    cache = os.path.join(OUT, f"example_{seed}.npz")
    f, y, s, tr = T.make3(seed, seq=T.SEQ_IN[seed % 30])
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); G = z["G"]; info = z["info"].item()
    else:
        r = T.solve3(f, y, s, mode="reference")
        lay = [dict(t=t, er=e, td=d, water=False, pinf=0.5) for t, e, d in zip(r["t"], r["er"], r["td"])]
        # rebuild the exact model of the answer from its own per-layer parameters
        import stack as ST
        best = r
        G = None
        info = dict(seq=r["seq"], t=r["t"], d=r["d"], status=r["status"], chi2n=r["chi2n"], crb=r["crb_t"])
        # recompute response by re-running the exact refit of the answer (cheap) to get the model
        ctx = T.Ctx(f, y, s, [tuple(r["seq"])], "real")
        cand = (0, 0, tuple(r["seq"]), np.array(r["t"]), r["d"], r["er"], r["td"], r["pinf"])
        m = T.exact_refit(ctx, cand)["model"]
        G, _ = I.forward(m, f, jacobian=False)
        np.savez(cache, G=G, info=np.array(info, dtype=object))
    segs = I.band_segments(f)
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1.25])
    for k, (a, b) in enumerate(segs):
        ax = fig.add_subplot(gs[0, k])
        fg = f[a:b] / 1e9
        ax.plot(fg, np.abs(y[a:b]), "o", ms=3.2, color=S1, alpha=0.8, label="simulated measurement", zorder=3)
        ax.plot(fg, np.abs(G[a:b]), "-", lw=2, color=S2, label="fitted model", zorder=4)
        ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("|reflection|" if k == 0 else "")
        ax.set_title("UWB 3.1-10.6 GHz" if k == 0 else "mmWave 60-64 GHz", fontsize=12)
        if k == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper right")
    # wall cartoon: true vs estimated
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    cols = {"gypsum": "#e1e0d9", "wood_board": "#eda100", "fibrous": "#e87ba4", "foam": "#1baf7a", "tile": "#2a78d6"}
    tt = [L["t"] * 1e3 for L in tr["layers"]]
    disp = np.maximum(np.array(tt) / sum(tt), 0.16); disp = 0.84 * disp / disp.sum()     # drawn widths (min width, not to scale)
    x0 = 0.08
    for nm, w in zip(tr["seq"], disp):
        ax.add_patch(matplotlib.patches.Rectangle((x0, 0.47), w, 0.27, color=cols[nm], ec=SURF, lw=2))
        ax.text(x0 + w / 2, 0.44, NICE[nm], ha="center", va="top", fontsize=8.5, color=INK2)
        x0 += w
    ax.text(0.08, 0.76, "radar →  front", fontsize=8, color=MUTED)
    ax.text(0.92, 0.76, "back (air)", fontsize=8, color=MUTED, ha="right")
    lines = ["layer          true       estimated (±1σ)"]
    for nm, t, te, e in zip(tr["seq"], tt, info["t"], info["crb"]):
        lines.append(f"{NICE[nm]:<14} {t:6.2f} mm  {1e3 * te:6.2f} ± {1e3 * e:.2f} mm")
    lines.append(f"standoff       {1e3 * tr['standoff']:6.2f} mm  {1e3 * info['d']:6.2f} mm")
    ax.text(0.02, 0.33, "\n".join(lines) + "\n(cartoon not to scale)", family="DejaVu Sans Mono", fontsize=8.6, va="top", color=INK2)
    lab = "labels: " + " | ".join(NICE[x] for x in info["seq"]) + ("  ✓" if list(info["seq"]) == list(tr["seq"]) else "")
    ax.text(0.02, 0.92, lab, fontsize=9.5, color=INK, fontweight="bold")
    ax.text(0.02, 0.84, f"χ²/n = {info['chi2n']:.2f}  ·  decision: {info['status']}", fontsize=9, color=INK2)
    fig.suptitle("One wall, end to end: recovered sequence and thicknesses from radar reflection data",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    foot(fig, f"Simulated wall (seed {seed}), 40 dB SNR, random material samples within each class; exact multilayer forward model.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(os.path.join(OUT, "4_example_wall.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 5. confusion map (noise-free, all 80 sequences)
def confmap():
    R, fn = load("map_noisefree")
    cls = ["gypsum", "wood_board", "tile", "fibrous", "foam"]
    M = np.zeros((5, 5)); N = np.zeros(5)
    for r in R:
        ps = r["per_seq"]; kt = "|".join(r["truth"])
        if kt not in ps:
            continue
        c0 = ps[kt]["cost"]
        o = r.get("oracle")
        if o and o["cost"] < c0:
            c0 = o["cost"]
        for c in r["truth"]:
            N[cls.index(c)] += 1
        hit = set()
        for k, v in ps.items():
            if k == kt or v["cost"] - c0 >= 10:
                continue
            for i, (a, b) in enumerate(zip(r["truth"], k.split("|"))):
                if a != b:
                    hit.add((i, a, b))
        for i, a, b in hit:
            M[cls.index(a), cls.index(b)] += 1
    F = M / np.maximum(N[:, None], 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cfe0f6", "#7fb0ea", "#2a78d6", "#123d73"])
    im = ax.imshow(F, cmap=cmap, vmin=0, vmax=max(0.5, F.max()))
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
            else:
                ax.text(j, i, f"{100 * F[i, j]:.0f}%", ha="center", va="center", fontsize=10.5,
                        color="white" if F[i, j] > 0.3 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels([NICE[c] for c in cls], rotation=20, ha="right", color=INK2)
    ax.set_yticks(range(5)); ax.set_yticklabels([NICE[c] for c in cls], color=INK2)
    ax.set_xlabel("impostor label that fits as well"); ax.set_ylabel("true layer class")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("share of layers with a tie", color=INK2)
    ax.set_title("Which labels can radar NOT tell apart?\n(noise-free data, all 80 sequences)", fontsize=12.5)
    foot(fig, f"Share of true layers with a wrong-label fit within Δ<10.\n{len(R)} simulated walls (1 per sequence). Data: {fn}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "5_confusion_map.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["outcomes", "thickness", "ambiguity", "example", "confmap"]
    for w in which:
        globals()[w]() if w != "example" else example()
        print("done", w, flush=True)
