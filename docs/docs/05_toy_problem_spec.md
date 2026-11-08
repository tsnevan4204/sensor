# 05 · The 1D layered-wall toy problem

## Physical setup

- **Geometry:** radar at z = 0, normal incidence. Air standoff ≈ 300 mm (randomized ±1 mm in the benchmarks).
- **Wall:** drywall 12.7 mm, fiberglass stud cavity 88.9 mm, drywall 12.7 mm. Air behind.
- **Nominal materials** (check against literature before trusting):
  - drywall εr 2.6, tan δ 0.03
  - fiberglass εr 1.15, tan δ 0.01
  - wood stud εr 2.0, tan δ 0.08 (not yet used)
- **Moisture:** CRIM mixing with Debye water (code in `src/oned/wall_toy.py`):
  - √ε = (1−θ)√ε_host + θ√ε_w(ω)
  - water: ε_s(T), ε_∞ = 5.2, τ(T) ≈ 9.4 ps at 20 °C
- **Radar:** FMCW 77–81 GHz treated as 256 stepped frequencies.
  - Range resolution c/2B = 3.75 cm. Drywall's optical thickness is about 2.05 cm, so its front and back faces are unresolved.
  - Noise is absolute: `snr_db` = 40 dB is the SNR of a perfect reflector (|Γ| = 1), so weak wall echoes have lower SNR.
- **System response:** an unknown smooth gain and phase ripple A(f) (0.5 dB, 0.1 rad), removed by a metal-plate calibration scan (Γ = −1) at a known distance (0.25 m).

## Measurement model

$y(f_m)=A(f_m)\,\Gamma(f_m;p)+n_m$, m = 1…256. Here `radar.freqs` is real with shape (256,), and y is complex with shape (256,). The magnitude carries loss; the phase carries the depths of the interfaces.

## Unknowns p

Per layer:
- thickness t
- εr
- tan δ
- moisture θ (CRIM)

Plus the standoff. Every parameter can be free or fixed, bounded, or **tied** to another (for example, both drywall sheets are the same product: shared εr and tan δ).

## Scenes (`wall_toy.py`, `scenarios.py`)

| Scene | Truth |
|---|---|
| dry | nominal three-layer wall |
| wet insulation | θ = 0.05 in the insulation (wet insulation **hides everything behind it**: |∂Γ/∂p| ≈ 1e-14) |
| wicked drywall | graded θ(z) = 0.15·exp(−(t−z)/4 mm) in the front drywall (40 sub-layers) |
| random dry (benchmark) | thicknesses ±0.6 mm (drywall) and ±3 mm (insulation); drywall εr ±8%; insulation εr ±5%; standoff 300–302 mm |

## Levels

- **L0:** forward model equals the reference (done; 1e-14).
- **L1:** dry wall — thicknesses and εr (done).
- **L2:** moisture θ in insulation (done; needs a dry-patch scan first).
- **L3:** continuous moisture profile (done; data supports about one number).
- **L4:** add a second band; coarse learned DDEC star (open; see roadmap).
