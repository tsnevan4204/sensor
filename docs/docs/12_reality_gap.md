# 12 · Reality gap: every assumption of the 1D toy, how big it is, and what to do about it

**Purpose.** The 1D algorithm solves the simulated benchmark almost perfectly: E019 gets 100/100 held-out walls right, with errors calibrated to the CRB. This document lists every assumption the simulation makes that a real measurement will break. For each one it gives the physics and literature basis, and either a measured effect on the current algorithm (E020) or an estimate.

The aim is that moving to real data produces no surprises we could have predicted.

**How effects were measured.**
- `src/oned/realism.py` is a more physical data generator with each effect as a switch. Its gates are in `tests/test_realism.py`; with all switches off it reproduces the benchmark data to < 1e-12.
- The **inversion model is not changed**. E020 runs the best methods on these data, one effect at a time, and records two things:
  - (i) whether χ² flags the mismatch;
  - (ii) the bias in units of each estimate's own CRB, z = |err|/CRB.
- **Silent bias** (χ²/n ≤ 1.2 but z > 3) is the dangerous outcome: the answer looks fine and is wrong. It sets the priorities.

Notation: the radar is at z = 0, normal incidence, time dependence e^{+jωt}. "Band" means a calibrated frequency sweep (77–81 GHz; second band 22–26 GHz in E017/E019).

---

## A. Wave physics and geometry

### A1. Plane wave at normal incidence (infinite beam)
- **Toy:** Γ(f) of an infinite plane wave.
- **Reality:**
  - A finite antenna radiates a beam. At 30 cm the setup is **not in the far field**: the Fraunhofer distance is 2D²/λ ≈ 2·(3 cm)²/3.8 mm ≈ 0.47 m for a 3 cm aperture at 79 GHz.
  - Each echo's amplitude depends on its path length (spreading), and its phase picks up a Gouy/Fresnel term.
  - Later (deeper) echoes are weaker than the plane-wave model predicts.
  - The metal-plate calibration at a different distance (0.25 m) does not cancel this.
- **Physics:**
  - A beam is a superposition of plane waves (the angular spectrum / Sommerfeld integral).
  - For planar layers, each transverse wavenumber k_ρ is an independent 1D problem with k_z = √(εk₀² − k_ρ²) and TE/TM impedances.
  - So the received signal is S(f) = ∫ W(k_ρ) Γ_co(f, k_ρ) e^{−2jk_z d} k_ρ dk_ρ.
- **Key point for this project:** each k_ρ slice is still 1D and planar, so **the exact Hodge stars remain exact** with k → k_z and the TE/TM impedances. The beam is not a reason to abandon exactness; it adds a quadrature over k_ρ (~30–60 points).
- **Literature:**
  - Lambot et al. 2004 (IEEE TGRS) model a GPR antenna by three transfer functions (H_i, H_t, H_f) plus the layered Green's function.
  - Free-space permittivity measurement papers list "Gaussian beam ≠ plane wave" as a main error source (Ghodgaonkar et al. 1990; IEICE 2022 Gaussian-beam correction).
- **Measured:** E020 rows "Gaussian beam".
- **Fix:**
  - (i) Add the angular-spectrum integral to the forward model, with the antenna described by a few parameters (beam waist, or H_i/H_t/H_f per band).
  - (ii) Calibrate those parameters with metal-plate scans at **several** distances (Lambot's procedure), not one.

### A2. Laterally infinite, homogeneous layers
- **Reality:**
  - Studs every 406 mm (16 in), drywall screws every ~200–300 mm, taped and mudded joints, electrical boxes, pipes and cables.
  - Fiberglass batts are non-uniform, compressed, with gaps at the sides.
  - The beam footprint on the wall is several cm, so one scan can see a stud edge or a screw.
- **Effect:** 2D/3D scattering that a 1D model cannot fit. It usually shows up as χ² failures, not silent bias, as long as χ² is trusted (see C5).
- **Fix (1D-compatible):**
  - Scan several positions and reject outliers by χ².
  - Average only consistent scans.
  - Use the 1D model for "between studs".
  - Full treatment is Phase 2.

### A3. Isotropic layers, single polarization
- **Reality:** wood grain and oriented fibre mats are anisotropic, and the antenna has a polarization. At normal incidence only the in-plane permittivity along the E-field direction matters.
- **Fix:** record the polarization; use dual polarization to measure in-plane anisotropy (a uniaxial ε).

### A4. Perfectly smooth interfaces
- **Reality:** paint texture, paper and the gypsum surface.
- **Rayleigh criterion:** smooth if h < λ/(8 cos θ) ≈ 0.47 mm at 79 GHz.
- **Kirchhoff/Ament:** the specular reflection is reduced by exp(−2(k₀σ_h)²). For σ_h = 0.1 mm that is 5.3% at 79 GHz and 0.5% at 24 GHz, which is frequency dependent and so mimics dispersion.
- **Measured:** E020 row "surface roughness".
- **Fix:** one roughness parameter per exposed face (it enters only the front-face reflection), or a roughness prior.

### A5. Antenna exactly normal to a flat wall
- **Reality:** a handheld tilt of a few degrees moves the specular return off the antenna axis, causing amplitude loss and a path change. Walls also bow.
- **Fix:**
  - Tilt is one parameter in the beam model (the beam integral with a tilted reflector).
  - An IMU or several antennas can measure it.

### A6. Air half-space behind the wall
- **Reality:**
  - Interior partitions have a room behind them.
  - Exterior walls have sheathing (OSB/plywood), housewrap, cladding, and sometimes foil-faced insulation (metal: a total reflector).
- **Fix:** unknown layer count and back-termination type (model selection over a layer library including PEC/impedance backing). This is the generalization roadmap item.

### A7. Each "layer" is homogeneous
- **Reality:**
  - Drywall is **paper + gypsum core + paper**, with 0.3–0.6 mm facings.
  - Paint adds 0.05–0.15 mm with εr ≈ 3–4.
  - Fiberglass batts often have a **kraft facing** (asphalt-coated paper) or a polyethylene vapour barrier.
  - Batts may not fill the cavity, leaving an **air gap**.
- **Measured:** E020 rows "drywall face paper", "insulation air gap".
- **Fix:** model selection (does a thin facing layer improve χ² more than its complexity cost?), or an effective-layer model for thin facings. A layer much thinner than the resolution is identifiable only through its effective impedance step.

---

## B. Materials

### B1. Frequency-independent εr and tan δ: violates causality
- **Toy:** ε(f) = εr(1 − j tan δ), constant across 22–81 GHz.
- **Physics:**
  - ε(ω) − ε_∞ is the Fourier transform of a causal response, so its real and imaginary parts are a Hilbert-transform pair (Kramers–Kronig).
  - A constant loss angle δ over a band forces the real part to drift.
  - The exactly causal constant-loss model is **Kjartansson's constant-Q law**: ε(f) = ε_ref (jf/f_ref)^{−α}, with α = (2/π) arctan(tan δ). It is a fractional integrator, causal and passive, with **two parameters, the same as today's model**.
  - For drywall (tan δ ≈ 0.03), ε′ is **2.3% higher at 24 GHz than at 79 GHz**, about 20× the two-band εr CRB.
- **Literature:**
  - ITU-R P.2040-3 §2.1.3: "the value of ε′ is constant from DC up to around 5–10 GHz after which it begins to fall with frequency".
  - ITU-R P.2040-3 Table 3 nevertheless fits ε′ = const and σ = c f^d. For plasterboard that is a = 2.73, c = 0.0085, d = 0.9395 (1–100 GHz), i.e. tan δ ≈ 0.046 at 24 GHz and 0.043 at 79 GHz.
  - Kjartansson 1979 (JGR) for the constant-Q model.
- **Measured:** E020 rows "causal dispersion", "ITU-style power-law loss".
- **Fix:** replace the material law in the *model* with a causal law that has the same parameter count:
  - constant-Q (εr at f_ref, tan δ); or
  - a power-law/Jonscher form if the data demand a frequency-dependent loss.
  - The Hodge stars stay exact, because only ε(f) changes.
  - This is the "relationship across frequency" that lets two bands be combined honestly.

### B2. Benchmark material ranges too narrow
- **Toy:** drywall εr ∈ [2.39, 2.81].
- **Literature:**
  - ITU plasterboard 2.73.
  - A 60 GHz measurement of indoor drywall gives n = 1.88, i.e. **εr ≈ 3.5** (cited in Pramanik et al. 2026, arXiv 2609.04429).
  - Products differ: regular, Type X (glass fibre), moisture-resistant, and gypsum cores with additives.
- **Fix:** widen the benchmark ranges to the literature spread (drywall εr 2.4–3.6, tan δ 0.01–0.06) and add the ITU material classes (wood, plywood, chipboard, concrete, brick, glass, metal) for the generalization benchmark.

### B3. Moisture law (CRIM + single-Debye water)
- **Reality:**
  - A single Debye term for free water is adequate to ~20 GHz. At 77 GHz double-Debye fits (Liebe 1991; Ellison 2007) differ by a few percent.
  - Water bound to pore surfaces relaxes at lower frequencies than free water.
  - CRIM's exponent ½ is empirical.
  - εr depends on temperature and salinity (conductivity matters at the lower band).
- **Fix:** a causal multi-pole mixture law with a few parameters, calibrated on lab coupons. This is the natural first *learned* (DDEC closure) component; passivity and causality hold by construction (doc 11 §A6).

---

## C. Sensor and measurement chain

### C1. Calibration with one ideal metal plate at an exactly known distance
- **Reality:**
  - Plate distance error (sub-mm).
  - Plate tilt.
  - Finite plate (edge diffraction; Ghodgaonkar et al. 1990 name it the main free-space error source).
  - Plate scanned at a different distance than the wall, so beam effects (A1) do not cancel.
- **Measured:** E020 row "plate distance error".
- **Fix:** several plate distances (which also give the antenna functions, Lambot); TRL/GRL-type free-space calibration; fit the plate distance as a nuisance parameter.

### C2. System response stationary between calibration and measurement
- **Reality:** the LO/PA gain and phase drift with temperature; the time between the plate scan and the wall scan matters.
- **Measured:** E020 row "cal-to-scan drift".
- **Fix:** frequent re-calibration; an internal reference path; a low-order A(f) correction as nuisance parameters (variable projection).

### C3. No antenna–wall multiple reflections, no leakage
- **Reality:**
  - The wall echo re-reflects from the antenna and returns again (H_f term).
  - Direct Tx→Rx leakage and the antenna's own reflection (H_i).
  - This affects the plate scan (|Γ| = 1) even more strongly.
- **Measured:** E020 row "antenna feedback Hf".
- **Fix:** the Lambot model y = H_i + H_t Γ/(1 − H_f Γ), with H's from multi-distance plate scans; background subtraction for H_i.

### C4. FMCW radar is not an ideal stepped-frequency VNA
- **Reality:**
  - Chirp nonlinearity smears range.
  - Phase noise is correlated and range-dependent.
  - The **IF high-pass filter attenuates near ranges**, i.e. a range-dependent gain that a single A(f) cannot represent.
  - ADC quantization, I/Q imbalance (image echoes), and residual video phase.
- **Literature:** IEEE work on FMCW accuracy (ramp nonlinearity, phase noise, SNR); radar datasheets.
- **Fix:** model the actual IF signal of the chosen radar chip (beat frequency = range) including the IF filter response vs range; measure the chirp linearity.
- **Not yet quantified** (radar-specific; needs the chip choice).

### C5. Noise is white, Gaussian and of known σ
- **Reality:** σ must be estimated; phase noise and clutter are not white. The χ² test (our failure detector) depends on the right σ.
- **Fix:**
  - Estimate σ from repeated sweeps or from the residual of an over-parameterized fit.
  - Use a robust/whiteness test as well as χ²/n.

### C6. Two bands share one standoff and one phase centre
- **Reality:** two bands means two antennas or chips with different phase centres, separate calibrations, and possibly sweeps at different times (handheld motion).
- **Physics:** a phase-centre offset δ adds 2k₀δ of phase to one band only. That is the same signature as the cross-band phase of a wrong cycle (E014), so it attacks exactly the information that makes two bands work.
- **Measured:** E020 row "band phase-centre offset".
- **Fix:** a mechanical design with a common phase centre; calibrate the offset with the plate in both bands; or fit a per-band standoff (costs some cross-band information; test).

### C7. Frequency bands are not legal for this product
- **Toy:** second band 22–26 GHz (E017/E019).
- **Reality (US; to verify with counsel):**
  - 22–29 GHz UWB is for vehicular radar.
  - 24.0–24.25 GHz is a narrow ISM band.
  - 57–71 GHz is unlicensed under Part 15.255 (radar permitted, with conditions).
  - **3.1–10.6 GHz UWB wall-imaging systems** are permitted under Part 15.509 for construction-related use, with Part 90 eligibility.
  - 77–81 GHz is primarily vehicular radar.
- **Implication:** a building-diagnostics product would likely use **57–64 GHz** (E014: worst-wall gap Δ = 62, nearly resolving) or **3.1–10.6 GHz** (E014: no wrong minima at all, and better penetration). E017/E019 must be repeated with a legal second band.

---

## D. Algorithm and statistics

| # | Assumption | Status / fix |
|---|---|---|
| D1 | σ known | estimate it (C5) |
| D2 | number and order of layers known | model selection (roadmap 1.2) |
| D3 | priors from standards (single band) | only valid for conforming walls; monitor the prior penalty |
| D4 | the same physics generates and inverts the data ("inverse crime") | the realism generator partly breaks this; for real independence, validate against a full-wave 3D solver (openEMS/FDTD) with an actual antenna model, then against lab coupons of known thickness and εr |

---

## E. Measured effect on the current algorithm (E020)

*(filled in from `results/2026-11-09_E020_reality_gap/`)*

---

## F. Priority list
*(after E020)*
