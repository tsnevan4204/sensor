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


CRB_WEAK = 1.5e-3         # absolute error-bar limit (well-resolved layers have CRB < 0.8 mm)
UNRESOLVED_DCOST = 6.0    # a thickness whose +-50 % change costs less than this is not resolved
T_PLAUSIBLE = 2.0e-3      # a dry solid layer thinner than this in the final fit is not accepted


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


def refine(ctx, S, f, y, sigma, fast=False):
    """Exact refinement of one search state on all bands, under the library prior.
    fast (E053): one cycle-hop pass, after the dispersion shape is released, instead of two."""
    mod, slots, back = _to_model(ctx, S)
    mod, c = ST._fit(mod, f, y, sigma)                       # dispersion shape at its class value
    if not fast:
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


def _slots_of(kinds):
    out, k = [], 0
    for kd in kinds:
        if kd == "gap":
            out.append(("gap", k - 1))
        else:
            out.append(("solid", k)); k += 1
    return out


def close_gap(mod, slots, back, j, f, y, sigma):
    """Refit the refined model with gap layer j removed (model level: materials, dispersion and
    all other thicknesses start from the refined values)."""
    lay = ST.describe(mod)
    pin = [L.pinf.value for L in mod.layers]
    keep = [i for i in range(len(lay)) if i != j]
    lay2 = [dict(t=lay[i]["t"], er=max(lay[i]["er"], 1.0001), td=max(lay[i]["td"], 1e-5), water=lay[i]["water"],
                 pinf=float(min(pin[i], 0.998))) for i in keep]
    kinds = [slots[i][0] for i in keep]
    d = mod.standoff.value
    ST.SBOX = (d - 0.004, d + 0.004)
    m2 = ST.build(lay2, d, back=back, water_T=mod.water_Tp.value if mod.water_Tp is not None else 20.0)
    ST.SBOX = None
    for kd, L in zip(kinds, m2.layers):
        if kd == "gap":
            L.eps_r.free = L.tan_d.free = False
            L.eps_r.value, L.tan_d.value = 1.0, 0.0
        elif L.theta.value < 0.5:
            L.pinf.value, L.pinf.free = float(np.clip(L.pinf.value, 0.02, 0.998)), True
    m2, c = ST._fit(m2, f, y, sigma)
    m2, c = SR.cycle_hop(m2, f, y, sigma, SR.Opts(search_step=1, search_atol=0.0))
    return m2, c, _slots_of(kinds)


def n_gaps(slots):
    return sum(1 for k, _ in slots if k == "gap")


def solve_once(f, y, sigma, N, d_guess, n_refine=3, verbose=False, trace=None, extra_extent=0.0, rho=1.5, n_keep=24,
               fast=False):
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
    if not fast:
        for S in states[:2]:                               # Occam variants: every gap closed in turn
            for j in range(S.nl):
                if S.cls[j] == ctx.AIR:
                    cand.append(_without_gap(ctx, S, j))

    def _geom(S):
        """Geometry of a state up to labels: gap pattern, backing, optical interface depths (mm)."""
        nopt = np.where(S.cls == ctx.AIR, 1.0, np.where(S.cls == ctx.WATER, 8.6, np.sqrt(np.maximum(S.er, 1.0))))
        return (tuple(S.cls == ctx.AIR), S.back, tuple(np.round(np.cumsum(S.t * nopt) * 1e3).astype(int)))
    refined, seen, geoms = [], set(), set()
    try:
        for S in cand:
            if S.key() in seen:
                continue
            seen.add(S.key())
            if fast:
                # E053: candidates that differ only in labels refit to the same wall; and once a
                # refit is at the noise level, the remaining candidates are skipped
                if _geom(S) in geoms:
                    continue
                geoms.add(_geom(S))
                # stop only at a fit within one sigma of the noise level (held-out 312: stopping at
                # the alarm level, chi2/n 1.04, skipped the true wall at 1.01)
                if refined and min(r[1] for r in refined) <= n_all + np.sqrt(2 * n_all):
                    break
            mod, c, slots, back = refine(ctx, S, f, y, sigma, fast)
            G, _ = I.forward(mod, f, jacobian=False)
            chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
            refined.append((c + bic * n_gaps(slots), chi, mod, slots, back, S, c))
            if verbose:
                print(f"  refine: search cost {S.cost:.1f} -> MAP {c:.1f} (+{bic * n_gaps(slots):.0f} BIC), chi2/n {chi / n_all:.3f} {S.describe(ctx)}", flush=True)
        # Schwarz criterion on the REFINED answer: close each gap in turn; keep the simpler wall
        # unless the gap lowers the MAP cost by more than ln n
        refined.sort(key=lambda t: t[0])
        cb, chi, mod, slots, back, S, c = refined[0]
        changed = True
        while changed:
            changed = False
            for j, (kd, _) in enumerate(slots):
                if kd != "gap":
                    continue
                m2, c2, sl2 = close_gap(mod, slots, back, j, f, y, sigma)
                if c2 < c + bic:
                    G, _ = I.forward(m2, f, jacobian=False)
                    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
                    if verbose:
                        print(f"  gap {j} ({1e3 * mod.layers[j].t.value:.2f} mm) closed: MAP {c:.1f} -> {c2:.1f} (< +ln n)", flush=True)
                    mod, c, slots = m2, c2, sl2
                    refined[0] = (c + bic * n_gaps(slots), chi, mod, slots, back, S, c)
                    changed = True
                    break
        # E053b: which thicknesses do the data actually determine? Local error bars (CRB) are
        # meaningless at a spurious optimum (held-out 316, 317: a layer behind 8-12 mm of water
        # answered 200 mm too thick with a 2-14 mm error bar). Profile test: halve / one-and-a-half
        # each thickness, refit everything else; if the MAP cost rises by less than UNRESOLVED_DCOST
        # the layer is reported as NOT RESOLVED.
        cb, chi, mod, slots, back, S, c = refined[0]
        unresolved = []
        for j, L in enumerate(mod.layers):
            dmin = np.inf
            for fac in (0.5, 1.5):
                m2 = mod.copy()
                L2 = m2.layers[j]
                L2.t.value = float(np.clip(L.t.value * fac, L.t.lo * 1.01, L.t.hi * 0.99))
                L2.t.free = False
                m2, c2 = ST._fit(m2, f, y, sigma, max_iter=60)
                dmin = min(dmin, c2 - c)
            unresolved.append(bool(dmin < UNRESOLVED_DCOST))
    finally:
        SR._PRIORS = old
    # plausibility: the exact refit may shrink a solid layer into a sliver that the search space
    # excludes (dry solids >= bnb.T_SOLID); such fits can reach the noise level with a wrong
    # structure (wall 211), so they rank behind every plausible candidate and raise the alarm
    def _sliver(m_, sl_):
        lab = pri.labels(m_)
        thin = any(k == "solid" and L.theta.value < 0.5 and L.t.value < T_PLAUSIBLE for (k, _), L in zip(sl_, m_.layers))
        # two adjacent solid layers of the same class = one layer counted twice, so one of the N
        # layers was not found (wall 212: a batt layer behind marble that the data barely see)
        twice = any(a == b and ka == kb == "solid" for a, b, (ka, _), (kb, _) in zip(lab[:-1], lab[1:], sl_[:-1], sl_[1:]))
        # a 'solid' layer that the refit turned into air is a gap: one of the N solids is missing (212)
        void = any(k == "solid" and a == "air" for a, (k, _) in zip(lab, sl_))
        return thin or twice or void
    refined.sort(key=lambda t: (_sliver(t[2], t[3]), t[0]))
    cb, chi, mod, slots, back, S, c = refined[0]
    implausible = _sliver(mod, slots)
    crb = {}
    try:
        F = I.fisher(mod, f, sigma)
        crb = dict(zip(F["names"], F["crb_std"]))
    except Exception:
        pass
    t_err = [crb.get(f"{L.name}.t", np.nan) for L in mod.layers]
    # a thickness with an error bar above 20 % or above CRB_WEAK is reported as not resolved
    # (held-out 316: the chipboard behind 11.6 mm of water has CRB 10.5 mm even at the true wall)
    weak = [bool(np.isfinite(e) and (e > 0.2 * L.t.value or e > CRB_WEAK)) for e, L in zip(t_err, mod.layers)]
    alts = []
    for cb2, chi2, m2, sl2, bk2, S2, c2 in refined[1:]:
        if cb2 <= cb + WS.AMBIG_DCHI2:
            alts.append(dict(dcost=cb2 - cb, chi2=chi2, wall=S2.describe(ctx), labels=pri.labels(m2),
                             t_mm=[round(1e3 * L.t.value, 2) for L in m2.layers]))
    return dict(model=mod, slots=slots, back=back, chi2=chi, chi2n=chi / n_all,
                alarm=bool(chi > ST.chi2_alarm(n_all) or implausible), implausible=bool(implausible),
                wall_s=time.time() - t0, search_s=t_search,
                labels=pri.labels(mod), search_labels=[ctx.lib.base[c] for c in S.cls],
                t_err=t_err, weak=weak,
                unresolved=unresolved, partial=bool(any(unresolved) or any(weak)),
                ambiguous=bool(alts), alternatives=alts, n_eval=ctx.n_eval, n_polish=ctx.n_polish,
                search_chi2n=states[0].chi / (2 * fs.size), n_states=len(states), bnb=ctx.bnb)


FAST = False     # E053 fast refit path (opt-in until held-out-validated: protocol of E029b)
# (extra search depth m, rho, sequences kept) tried in turn while the chi2 alarm is raised
ESCALATE = ((0.0, 1.5, 24), (0.15, 4.0, 64))


def solve(f, y, sigma, N, d_guess, verbose=False, fast=None, **kw):
    """Self-checked solve: if the answer raises the chi2 alarm, search deeper (interfaces that only
    the sub-GHz band sees, e.g. behind a water layer, lie beyond the UWB data extent). The alarm
    decides; the best attempt (lowest data chi2) is returned with the list of attempts."""
    best, attempts, t0 = None, [], time.time()
    kw["fast"] = FAST if fast is None else fast
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


def solve_unknown_n(f, y, sigma, d_guess, n_max=4, verbose=False, **kw):
    """E055: the number of solid layers is NOT given. Occam rule: solve for N = 1, 2, ... and
    return the first (smallest) N whose answer passes the self-check (chi2 alarm, no sliver, no
    class twice in a row). A wall with fewer layers than the truth cannot reach the noise level;
    a wall with more layers must split a true layer, which the plausibility checks reject and the
    Schwarz criterion would penalise by ~4 ln n per extra layer. If no N passes, the answer with
    the lowest chi2 is returned with its alarm set."""
    tried, best = [], None
    t0 = time.time()
    for N in range(1, n_max + 1):
        res = solve(f, y, sigma, N, d_guess, verbose=verbose, **kw)
        tried.append(dict(N=N, chi2n=res["chi2n"], alarm=res["alarm"], wall_s=res["wall_s"]))
        res["N"] = N
        if not res["alarm"]:
            best = res
            break
        if best is None or res["chi2"] < best["chi2"]:
            best = res
    best["tried"] = tried
    best["wall_s"] = time.time() - t0
    return best
