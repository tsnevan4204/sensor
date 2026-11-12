"""Summary of a run.py output file: python3 summarize.py file.jsonl [file2.jsonl ...] (later files override seeds)."""
import sys, json
import numpy as np
rows = {}
for fn in sys.argv[1:]:
    for l in open(fn):
        if l.strip():
            r = json.loads(l); rows[r["seed"]] = r
R = [rows[k] for k in sorted(rows)]
ok = [r for r in R if r.get("ok")]
crash = [r for r in R if "crash" in r]
flag = [r for r in R if not r.get("ok") and r.get("alarm")]
ok3 = [r for r in R if not r.get("ok") and not r.get("alarm") and r.get("ok3")]
silent = [r for r in R if not r.get("ok") and not r.get("alarm") and not r.get("ok3") and "crash" not in r]
print(f"walls {len(R)}: exact {len(ok)}, within 3 CRB sigma (same structure) {len(ok3)}, flagged {len(flag)}, silent {len(silent)}, crash {len(crash)}")
t = np.array([r["wall_s"] for r in R if "wall_s" in r])
print(f"time per wall: median {np.median(t):.0f} s, mean {t.mean():.0f} s, max {t.max():.0f} s")
errs = np.array([abs(e) for r in ok for e in r["t_err_mm"]])
print(f"solid-thickness |error| on exact walls: median {np.median(errs):.3f} mm, 90% {np.percentile(errs, 90):.3f} mm, max {errs.max():.3f} mm")
gp = [(a, b) for r in ok for a, b in zip(r["gaps_true"], r["gaps_fit"]) if a > 0]
if gp:
    ge = np.array([abs(a - b) for a, b in gp])
    print(f"gaps on exact walls: {len(gp)} gaps of {min(a for a, _ in gp):.1f}-{max(a for a, _ in gp):.1f} mm, |error| median {np.median(ge):.3f} mm, max {ge.max():.3f} mm")
lab = [m for r in ok for m in r["mat_ok"]]
print(f"material label exactly right (nearest class) on exact walls: {sum(lab)}/{len(lab)}")
esc = sum(1 for r in R if len(r.get("attempts", [])) > 1)
print(f"walls that escalated on their own alarm: {esc}")
for r in R:
    tag = "CRASH" if "crash" in r else ("exact" if r.get("ok") else ("flag" if r.get("alarm") else ("3sigma" if r.get("ok3") else "SILENT")))
    print(f"| {r['seed']} | {' / '.join(f'{n} {t}' for n, t in r.get('truth', []))} | {r.get('gaps_true')} | {r.get('back')} | {' / '.join(f'{n} {t}' for n, t in r.get('fit', []))} | {r.get('gaps_fit')} | {r.get('back_fit')} | {r.get('chi2n', 0):.2f} | {r.get('wall_s', 0):.0f} | {tag} |")
