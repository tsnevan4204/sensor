"""E070 runner: tri3.solve3 on tri3.make3 walls, one wall per worker process.
python3 run.py --out X.jsonl --seeds 1 40 [--truth in|out|all] [--solver in|all] [--thick real|stress]
               [--bands 2|3] [--mode reference|fast] [--noisefree] [--balanced] [--hop-all] [--workers 4]
--balanced: the truth sequence of seed k is POOL[k % len(POOL)] (every sequence covered in order)."""
import os, sys, json, time, argparse, warnings
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/oned"))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--seeds", nargs=2, type=int, required=True)
ap.add_argument("--truth", default="in"); ap.add_argument("--solver", default="in"); ap.add_argument("--thick", default="real")
ap.add_argument("--bands", type=int, default=2); ap.add_argument("--mode", default="reference")
ap.add_argument("--noisefree", action="store_true"); ap.add_argument("--balanced", action="store_true")
ap.add_argument("--hop-all", action="store_true"); ap.add_argument("--workers", type=int, default=4)
A = ap.parse_args()


def one(seed):
    warnings.filterwarnings("ignore")
    import tri3 as T
    T.install()
    pool = {"in": T.SEQ_IN, "out": T.SEQ_OUT, "all": T.SEQ_ALL}[A.truth]
    seq = pool[seed % len(pool)] if A.balanced else None
    bands, snr = (T.BANDS2, T.SNR2) if A.bands == 2 else (T.BANDS3, T.SNR3)
    f, y, s, tr = T.make3(seed, scope=A.truth, thick=A.thick, bands=bands, snr=snr, seq=seq, noise=not A.noisefree)
    t0 = time.time()
    try:
        r = T.solve3(f, y, s, scope=A.solver, thick=A.thick, mode=A.mode, hop_all=A.hop_all)
        sc = T.score(r, tr)
    except Exception as e:
        import traceback; traceback.print_exc()
        return dict(seed=seed, crash=repr(e), truth=tr["seq"], wall_s=time.time() - t0)
    return dict(seed=seed, truth=tr["seq"], t_true_mm=[round(1e3 * L["t"], 3) for L in tr["layers"]],
                d_true_mm=round(1e3 * tr["standoff"], 3), loss_db={k: round(v, 1) for k, v in tr["loss_db"].items()},
                mat=[{k: (round(v, 5) if isinstance(v, float) else v) for k, v in L.items() if k != "t"} for L in tr["layers"]],
                fit=r["seq"], t_fit_mm=[round(1e3 * x, 3) for x in r["t"]], d_fit_mm=round(1e3 * r["d"], 3),
                d_est_mm=round(1e3 * r["d_est"], 2), chi2n=r["chi2n"], cost=r["cost"], status=r["status"], alarm=r["alarm"],
                crb_mm=[round(1e3 * e, 4) for e in r["crb_t"]], crb_d_mm=round(1e3 * r["crb_d"], 4), weak_t=r["weak_t"],
                rivals=r["rivals"], same_seq_alt=r["same_seq_alt"], in_bounds=r["in_bounds"],
                per_seq=r["per_seq"], hop_gains=r["hop_gains"], timing={k: round(v, 2) for k, v in r["timing"].items()},
                wall_s=round(r["wall_s"], 1), n_grid=r["n_grid"], n_polish=r["n_polish"], n_exact=r["n_exact"], n_hop=r["n_hop"],
                **sc)


if __name__ == "__main__":
    done = set()
    if os.path.exists(A.out):
        done = {json.loads(l)["seed"] for l in open(A.out) if l.strip()}
    seeds = [s for s in range(A.seeds[0], A.seeds[1] + 1) if s not in done]
    from multiprocessing import Pool
    with Pool(A.workers) as P:
        for r in P.imap_unordered(one, seeds):
            open(A.out, "a").write(json.dumps(r, default=float) + "\n")
            if "crash" in r:
                print(r["seed"], "CRASH", r["crash"], flush=True); continue
            tag = "JOINT" if r["joint"] else ("SILENT" if r["silent"] else r["status"])
            print(r["seed"], tag, r["status"], f"{r['wall_s']:.0f}s", "|".join(r["truth"]), r["t_true_mm"], "->", "|".join(r["fit"]),
                  r["t_fit_mm"], f"chi2n {r['chi2n']:.3f}", "rivals", [("|".join(x["seq"]), round(x["dcost"], 1)) for x in r["rivals"]][:3],
                  flush=True)
