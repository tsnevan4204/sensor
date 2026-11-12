"""
E051: known-N wall solver = exhaustive class-discrete structure search (wallsearch.py, compiled
exact forward) + exact continuous refinement on all bands (adjoint LM, cycle hops, dispersion).

  solve(f, y, sigma, N, d_guess) -> result dict compatible with wallsolve.report / wallsolve.score
"""
from __future__ import annotations
import time
import numpy as np
import stack as ST
import search as SR
import wall_inverse as I
import matlib as ML
import treesolve as TS
import wallsearch as LS
import wallsolve as WS


def _to_model(ctx, S):
    lay, slots, k = [], [], 0
    for j in range(S.nl):
        c = int(S.cls[j])
        if c == ctx.AIR:
            lay.append(dict(t=float(S.t[j]), er=1.0001, td=1e-5, water=False))
            slots.append(("gap", k - 1))
        elif c == ctx.WATER:
            lay.append(dict(t=float(S.t[j]), er=1.0, td=0.0, water=True))
            slots.append(("solid", k)); k += 1
        else:
            lay.append(dict(t=float(S.t[j]), er=float(S.er[j]), td=float(S.td[j]), water=False,
                            pinf=float(min(ctx.pp0[c], 0.998))))
            slots.append(("solid", k)); k += 1
    back = "metal" if S.back == ctx.METAL else "air"
    ST.SBOX = (S.d - 0.004, S.d + 0.004)
    mod = ST.build(lay, S.d, back=back, water_T=float(S.Tw))
    ST.SBOX = None
    for (kind, _), L in zip(slots, mod.layers):
        if kind == "gap":
            L.eps_r.free = L.tan_d.free = False
            L.eps_r.value, L.tan_d.value = 1.0, 0.0
    return mod, slots, back


def refine(ctx, S, f, y, sigma):
    """Exact refinement of one search state on all bands, under the library prior."""
    mod, slots, back = _to_model(ctx, S)
    mod, c = ST._fit(mod, f, y, sigma)                       # dispersion shape at its class value
    mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
    for L in mod.layers:                                     # release the dispersion shape
        if L.theta.value < 0.5 and L.eps_r.free:
            L.pinf.value, L.pinf.free = float(np.clip(L.pinf.value, 0.02, 0.998)), True
    mod, c = ST._fit(mod, f, y, sigma)
    mod, c = SR.cycle_hop(mod, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
    return mod, c, slots, back


def _without_gap(ctx, S, j):
    """State S with gap layer j closed (the layers behind move forward)."""
    keep = [i for i in range(S.nl) if i != j]
    back = S.back
    if j == S.nl - 1 and S.back == ctx.METAL:
        back = ctx.METAL                                   # metal directly behind the last solid
    return LS.State(S.cls[keep], S.t[keep], S.er[keep], S.td[keep], back, S.d, S.Tw, S.cost, S.chi)


def n_gaps(slots):
    return sum(1 for k, _ in slots if k == "gap")


def solve_once(f, y, sigma, N, d_guess, n_refine=3, verbose=False, trace=None, extra_extent=0.0, rho=1.5, n_keep=24):
    """Structure search on the low bands, exact refinement on all bands, and model-order selection
    over the number of gaps by the Schwarz criterion (each gap is one extra parameter: a candidate
    with one more gap must lower the MAP cost by more than ln n)."""
    t0 = time.time()
    fs, ys, ss, _ = TS.search_subset(f, y, sigma)
    states, ctx = LS.search(fs, ys, ss, N, d_guess, verbose=verbose, trace=trace, extra_extent=extra_extent,
                            rho=rho, n_keep=n_keep)
    t_search = time.time() - t0
    pri = ML.LibPrior()
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    n_all = 2 * f.size
    bic = np.log(n_all)
    cand = list(states[:n_refine])
    for S in states[:2]:                                   # Occam variants: every gap closed in turn
        for j in range(S.nl):
            if S.cls[j] == ctx.AIR:
                cand.append(_without_gap(ctx, S, j))
    refined, seen = [], set()
    try:
        for S in cand:
            if S.key() in seen:
                continue
            seen.add(S.key())
            mod, c, slots, back = refine(ctx, S, f, y, sigma)
            G, _ = I.forward(mod, f, jacobian=False)
            chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
            refined.append((c + bic * n_gaps(slots), chi, mod, slots, back, S, c))
            if verbose:
                print(f"  refine: search cost {S.cost:.1f} -> MAP {c:.1f} (+{bic * n_gaps(slots):.0f} BIC), chi2/n {chi / n_all:.3f} {S.describe(ctx)}", flush=True)
    finally:
        SR._PRIORS = old
    refined.sort(key=lambda t: t[0])
    cb, chi, mod, slots, back, S, c = refined[0]
    crb = {}
    try:
        F = I.fisher(mod, f, sigma)
        crb = dict(zip(F["names"], F["crb_std"]))
    except Exception:
        pass
    t_err = [crb.get(f"{L.name}.t", np.nan) for L in mod.layers]
    alts = []
    for cb2, chi2, m2, sl2, bk2, S2, c2 in refined[1:]:
        if cb2 <= cb + WS.AMBIG_DCHI2:
            alts.append(dict(dcost=cb2 - cb, chi2=chi2, wall=S2.describe(ctx), labels=pri.labels(m2),
                             t_mm=[round(1e3 * L.t.value, 2) for L in m2.layers]))
    return dict(model=mod, slots=slots, back=back, chi2=chi, chi2n=chi / n_all,
                alarm=bool(chi > ST.chi2_alarm(n_all)), wall_s=time.time() - t0, search_s=t_search,
                labels=pri.labels(mod), search_labels=[ctx.lib.base[c] for c in S.cls],
                t_err=t_err, weak=[bool(np.isfinite(e) and e > 0.2 * L.t.value) for e, L in zip(t_err, mod.layers)],
                ambiguous=bool(alts), alternatives=alts, n_eval=ctx.n_eval, n_polish=ctx.n_polish,
                search_chi2n=states[0].chi / (2 * fs.size), n_states=len(states), bnb=ctx.bnb)


# (extra search depth m, rho, sequences kept) tried in turn while the chi2 alarm is raised
ESCALATE = ((0.0, 1.5, 24), (0.15, 4.0, 64))


def solve(f, y, sigma, N, d_guess, verbose=False, **kw):
    """Self-checked solve: if the answer raises the chi2 alarm, search deeper (interfaces that only
    the sub-GHz band sees, e.g. behind a water layer, lie beyond the UWB data extent). The alarm
    decides; the best attempt (lowest data chi2) is returned with the list of attempts."""
    best, attempts, t0 = None, [], time.time()
    for extra, rho, n_keep in ESCALATE:
        res = solve_once(f, y, sigma, N, d_guess, verbose=verbose, extra_extent=extra, rho=rho, n_keep=n_keep, **kw)
        attempts.append(dict(extra_extent=extra, rho=rho, chi2n=res["chi2n"], alarm=res["alarm"], wall_s=res["wall_s"]))
        if best is None or res["chi2"] < best["chi2"]:
            best = res
        if not res["alarm"]:
            break
        if verbose:
            print(f"  ALARM (chi2/n {res['chi2n']:.2f}): escalating", flush=True)
    best["attempts"] = attempts
    best["wall_s"] = time.time() - t0
    return best
