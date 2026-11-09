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

Seeds 1–8, each effect alone at a moderate, real-world magnitude, then all together. Cells show the median |error| and the median z = |error|/CRB. **flag** = χ²/n > 1.2 (the algorithm notices). **silentB** = not flagged but some z > 3 (the algorithm is wrong and does not know it).

```
E020 reality gap: seeds [1, 2, 3, 4, 5, 6, 7, 8]; z = |err|/CRB (median over walls); 'silent bias' = chi2/n <= 1.2 and some z > 3

=== two-band (e019dual, 2-stage cascade)
effect                                    flag silentB   standoff.stand      drywall_f.t  drywall_f.eps_r      insulatio.t      drywall_b.t
none (today's toy)                         0/8   0/8   0.000mm z  0.6   0.006mm z  0.7    0.093% z  0.7   0.097mm z  0.6   0.006mm z  0.5
causal dispersion (constant-Q)             8/8   0/8   0.008mm z 10.8   0.131mm z 12.9    2.834% z 16.2   1.352mm z  7.7   0.206mm z 17.9
ITU-style power-law loss                   0/8   4/8   0.001mm z  1.7   0.011mm z  1.1    0.192% z  1.3   0.364mm z  2.2   0.021mm z  2.0
Gaussian beam w0 = 2 cm                    8/8   0/8   0.032mm z 44.3   0.771mm z 75.7   11.185% z 88.5   0.509mm z  3.0   0.805mm z 59.7
Gaussian beam w0 = 1 cm                    8/8   0/8   0.046mm z 61.7   0.839mm z 76.4   12.461% z 93.6   6.403mm z 45.4   0.571mm z 51.7
antenna feedback Hf = 0.1 (-20 dB)         8/8   0/8   0.001mm z  1.5   0.009mm z  0.9    0.111% z  0.7   0.166mm z  1.0   0.032mm z  2.7
plate distance error +0.5 mm               0/8   8/8   0.500mm z796.0   0.007mm z  0.7    0.102% z  0.7   0.109mm z  0.6   0.007mm z  0.6
surface roughness 0.1 mm rms               5/8   3/8   0.002mm z  2.3   0.182mm z 18.8    2.609% z 19.0   0.752mm z  4.6   0.183mm z 16.2
drywall face paper 0.4 mm                  8/8   0/8   0.110mm z162.7   0.442mm z 41.7   10.203% z 76.6   3.482mm z 18.3   0.338mm z 30.5
insulation air gap 5 mm                    7/8   1/8   0.001mm z  1.7   0.023mm z  2.3    0.432% z  2.7   1.229mm z  6.5   0.081mm z  7.9
cal-to-scan drift 0.05 dB / 0.01 rad       0/8   1/8   0.000mm z  0.6   0.006mm z  0.6    0.080% z  0.6   0.205mm z  1.3   0.009mm z  0.7
band phase-centre offset 0.3 mm            8/8   0/8   0.035mm z 61.2   0.631mm z 58.1    7.266% z 43.7   6.634mm z 35.2   0.502mm z 36.3
ALL of the above (moderate)                8/8   0/8   0.423mm z560.7   1.120mm z106.8   18.130% z153.4   1.926mm z  9.6   1.415mm z 97.7

=== one-band + priors (e018b, 2-stage cascade)
effect                                    flag silentB   standoff.stand      drywall_f.t  drywall_f.eps_r      insulatio.t      drywall_b.t
none (today's toy)                         0/8   0/8   0.001mm z  0.6   0.013mm z  0.8    0.213% z  0.9   0.633mm z  0.3   0.012mm z  0.6
causal dispersion (constant-Q)             0/8   1/8   0.001mm z  0.6   0.027mm z  1.8    0.433% z  1.7   1.459mm z  0.5   0.019mm z  0.8
ITU-style power-law loss                   0/8   0/8   0.001mm z  0.6   0.012mm z  0.8    0.196% z  0.8   0.640mm z  0.3   0.012mm z  0.5
Gaussian beam w0 = 2 cm                    8/8   0/8   0.028mm z 29.4   0.488mm z 32.0    7.194% z 33.2   1.536mm z  0.7   0.592mm z 19.4
Gaussian beam w0 = 1 cm                    8/8   0/8   0.016mm z 14.9   1.030mm z 63.3   13.885% z 70.6   1.841mm z  0.8   0.249mm z  8.3
antenna feedback Hf = 0.1 (-20 dB)         8/8   0/8   0.005mm z  5.1   0.610mm z 39.9    0.780% z  3.5   3.059mm z  3.3   0.390mm z 13.5
plate distance error +0.5 mm               0/8   8/8   0.500mm z578.4   0.008mm z  0.6    0.153% z  0.7   0.797mm z  0.5   0.005mm z  0.2
surface roughness 0.1 mm rms               0/8   8/8   0.001mm z  0.8   0.355mm z 22.7    5.208% z 23.8   1.567mm z  0.8   0.602mm z 20.6
drywall face paper 0.4 mm                  0/8   8/8   0.117mm z 84.6   0.377mm z 23.6   20.773% z104.7   1.521mm z  0.9   0.213mm z  7.7
insulation air gap 5 mm                    0/8   1/8   0.001mm z  0.8   0.009mm z  0.6    0.158% z  0.7   1.735mm z  1.1   0.029mm z  1.5
cal-to-scan drift 0.05 dB / 0.01 rad       0/8   3/8   0.000mm z  0.4   0.021mm z  1.4    0.347% z  1.6   1.122mm z  0.7   0.023mm z  1.0
band phase-centre offset 0.3 mm            0/8   0/8   0.001mm z  0.6   0.013mm z  0.8    0.213% z  0.9   0.633mm z  0.3   0.012mm z  0.6
ALL of the above (moderate)                8/8   0/8   0.423mm z306.6   0.375mm z 22.9   27.397% z151.4   0.837mm z  0.6   0.232mm z  5.4

time 1839s
```

**Reading the table.**

1. **The antenna beam is the largest single effect.**
   - Gaussian beam (w₀ = 1–2 cm): drywall εr is biased by 7–14% and thicknesses by 0.5–1 mm (z = 30–90) in both methods.
   - It is always flagged, so χ² sees it.
   - Cause: a plane-wave model cannot represent the faster decay of deeper echoes in a spreading beam, nor the different spreading of the plate scan. Amplitudes carry the εr information, so εr absorbs the error.
2. **Two bands are more sensitive to model error, but also more self-aware.**
   - Causal dispersion, face paper, an air gap, the band phase-centre offset and antenna feedback are all **flagged** with two bands. With one band they are absorbed into biased parameters.
   - Single band: roughness, face paper and plate error are **8/8 silent**; drift 3/8.
   - Richer data exposes model error instead of hiding it. That is a strong argument for two bands even beyond the cycle ambiguity.
3. **Silent biases (the dangerous ones):**
   - **Plate distance error:** silent in both methods, but it moves only the standoff, by exactly the plate error. Layer properties are unaffected (z < 1). The standoff should be reported as relative to the calibration plane; it is then harmless.
   - **Surface roughness 0.1 mm rms:** single band silently biases drywall εr by 5% and the front thickness by 0.36 mm (z ≈ 23). Two bands flag 5/8, with 3 silent.
   - **Drywall face paper 0.4 mm:** single band silently biases εr by **21%** (it fits an effective sheet). Two bands flag it.
   - ITU-type loss law (two bands, z ≈ 2), drift (one band, z ≈ 1.5) and the air gap (one band) are small.
4. **Prediction (a) confirmed:** causal dispersion breaks the two-band fit (8/8 flagged, z 8–18) and barely touches the single band.
5. **Prediction (c) confirmed:** a 0.5 mm plate error is exactly a 0.5 mm standoff error.
6. **All effects together:** both methods are flagged on every wall. Run on real data today, the algorithm would refuse to give an answer (good: not silent) but would be useless. The fixes below are needed before real data.

## F. Priority list

| priority | gap | evidence | fix (keeps exactness) | cost |
|---|---|---|---|---|
| 1 | antenna beam / near field (A1) + antenna functions (C3) | z 30–90 both methods; Hf flags | forward model = angular-spectrum integral of exact 1D stars over k_ρ (TE/TM), antenna parameters (beam waist or H_i, H_t, H_f per band) calibrated from plate scans at several distances | ~30–60 k_ρ solves per frequency: needs speed work |
| 2 | causal material dispersion (B1) | two-band 8/8 flagged, z 8–18 | constant-Q (Kjartansson) law in the model: same 2 parameters; optional Jonscher exponent | free |
| 3 | thin facings (paper, paint) + roughness (A4, A7) | single-band **silent** 21% / 5% εr bias | two bands (flags them) + thin-sheet model (impedance sheet, 1–2 params) or a facing layer chosen by model selection; roughness parameter on the front face | small |
| 4 | band phase-centre offset (C6) | two-band 8/8 flagged, z 35–60 | common phase centre by design, plate calibration in both bands, or a fitted per-band offset | small |
| 5 | legal second band (C7) | 22–26 GHz unlikely legal | redo E017/E019 with 57–64 GHz or 3.1–10.6 GHz | small |
| 6 | material ranges (B2) | drywall εr 3.5 measured vs 2.4–2.8 benchmark | widen benchmark ranges; ITU material classes | free |
| 7 | calibration plane (C1) | standoff silently = plate error | report the standoff relative to the calibration plane; multi-distance plate | free |
| 8 | FMCW chip effects (C4), noise σ (C5) | not quantified | model the chosen radar's IF signal; estimate σ | needs hardware choice |
| 9 | lateral structure (A2), back termination (A6), layer count (D2) | — | multi-position scans with χ² outlier rejection; model selection over a layer library incl. metal | generalization work |
| 10 | independent validation (D4) | — | full-wave 3D (openEMS/FDTD) with a real antenna, then lab coupons | outside 1D |
