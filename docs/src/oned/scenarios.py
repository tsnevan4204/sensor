"""Scenario definitions: the truth used to make data, and the starting model + which
parameters are free. Edit or add scenarios here; nothing else needs to change."""
import numpy as np
import wall_toy as W
from wall_inverse import WallModel, LayerSpec, P

DRY = dict(eps_r=2.6, tan_d=0.03)      # drywall nominal
FIB = dict(eps_r=1.15, tan_d=0.01)     # fiberglass nominal


def stud_wall_model(free=("t", "eps_r"), theta_ins=None, standoff_guess=0.30, standoff_halfwidth=0.010,
                    same_drywall=True):
    """Three-layer stud wall with generous bounds. `free` picks which per-layer quantities
    are unknown; theta_ins=(lo, hi) frees the insulation moisture."""
    def lay(name, t, t_lo, t_hi, mat):
        return LayerSpec(name,
                         P(t, t_lo, t_hi, "t" in free),
                         P(mat["eps_r"], 1.0, 4.0, "eps_r" in free),
                         P(mat["tan_d"], 0.0, 0.15, "tan_d" in free),
                         P(0.0, 0.0, 0.3, False))
    m = WallModel([lay("drywall_front", 0.0127, 0.009, 0.017, DRY),
                   lay("insulation", 0.0889, 0.080, 0.100, FIB),
                   lay("drywall_back", 0.0127, 0.009, 0.017, DRY)],
                  standoff=P(standoff_guess, standoff_guess - standoff_halfwidth,
                             standoff_guess + standoff_halfwidth, True))
    if same_drywall:   # both sheets are the same product: share eps_r and tan_d
        m.layers[2].eps_r.tie = "drywall_front.eps_r"
        m.layers[2].tan_d.tie = "drywall_front.tan_d"
        m.sync_ties()
    if theta_ins is not None:
        m.layers[1].theta = P(0.02, theta_ins[0], theta_ins[1], True)
    return m


def truth_dry(standoff=0.30):
    return W.Scene(W.scene_dry_wall().layers, standoff=standoff)


def truth_wet(theta=0.05, standoff=0.30):
    return W.Scene(W.scene_wet_insulation(theta).layers, standoff=standoff)


def truth_values_dry():
    return {"drywall_front.t": 0.0127, "drywall_front.eps_r": 2.6, "insulation.t": 0.0889,
            "insulation.eps_r": 1.15, "drywall_back.t": 0.0127, "drywall_back.eps_r": 2.6}


def truth_random(rng, standoff):
    """A random but realistic dry stud wall (both drywall sheets the same product)."""
    v = {"drywall_front.t": 0.0127 + rng.uniform(-0.6e-3, 0.6e-3),
         "drywall_back.t": 0.0127 + rng.uniform(-0.6e-3, 0.6e-3),
         "insulation.t": 0.0889 + rng.uniform(-3e-3, 3e-3),
         "drywall_front.eps_r": 2.6 * rng.uniform(0.92, 1.08),
         "insulation.eps_r": 1.15 * rng.uniform(0.95, 1.05),
         "standoff": standoff}
    dw = lambda f: W.eps_dielectric(v["drywall_front.eps_r"], DRY["tan_d"], f)
    ins = lambda f: W.eps_dielectric(v["insulation.eps_r"], FIB["tan_d"], f)
    scene = W.Scene([W.Layer("drywall", v["drywall_front.t"], dw),
                     W.Layer("insulation", v["insulation.t"], ins),
                     W.Layer("drywall", v["drywall_back.t"], dw)], standoff=standoff)
    return scene, v
