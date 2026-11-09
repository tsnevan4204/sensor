# 14 · Current limitations of the algorithm (living list)

Last updated 2026-11-09, after E033 was started. Each entry says what the limitation is, why it exists, how much it matters, and what would fix it. Evidence references point to `experiments/EXPERIMENT_LOG.md`.

## A. What it does not do at all (scope)

| # | Limitation | Why | Impact | Fix / status |
|---|---|---|---|---|
| A1 | **1D only**: flat, parallel, laterally uniform layers seen head-on | the whole forward model is 1D (planar layers, plane-wave spectrum for the beam) | studs, screws, pipes, edges, wires and boxes are not modelled; a scan over a stud gives wrong numbers (usually flagged by χ²) | Phase 2 (2D); in practice, scan between studs and reject scans flagged by χ² |
| A2 | **Fixed wall recipe**: drywall / insulation / drywall (plus optional paper facing and air gap) | the layer sequence is given, not inferred | other walls (plywood, OSB, brick, concrete, tile, metal backing, exterior sheathing) and an unknown layer count are not supported | roadmap 1.2: model selection over a material library (ITU-R P.2040 classes incl. metal) |
| A3 | **Dry walls only** in the realistic tests | the moisture model (CRIM mixing + Debye water) exists but has not been run through the realistic chain or two bands | moisture estimates are unvalidated; earlier single-band work showed a 20% bias if the host material is wrong | next 1D item: wet benchmark with two bands (water's dispersion separates host from water) |
| A4 | **Simulation only** | no real measurement yet; the "realistic" data come from our own simulator | unknown effects may exist that neither simulator nor model contains | an independent full-wave 3D check (openEMS/FDTD with a real antenna), then lab coupons (acrylic, glass, drywall of measured thickness) with the IWR6843 |

## B. Works only under these conditions

| # | Condition | Evidence | If violated |
|---|---|---|---|
| B1 | **Two frequency bands**; the second must be **UWB 3.1–10.6 GHz** for full reliability | E032: 100/100 held-out with UWB; 95/100 (5 silent) with 57–64 GHz; one band: back-of-wall cycle counts ambiguous (E009–E014) | with one band, the back drywall and insulation are off by whole cycles (1–5 mm) unless thickness priors are used (E013) |
| B2 | **Antenna beam waist ≥ ~2 cm** at the working distance (lens or aperture), or a shorter standoff | E023: 9/10 at 2 cm, 2/10 at 1 cm (echoes 11–14 dB weaker) | back-of-wall ambiguity returns |
| B3 | **Calibration with metal-plate scans at several staggered distances** (λ/6 steps at two base distances) | E023: equal 25 mm spacings made the calibration singular | antenna model wrong → biased εr |
| B4 | The **radar's IF filter settings are known** and in the model; **chirp slope ≥ ~250 MHz/µs** | E031: ignoring the filters fails 10/10; 250 MHz/µs gives 10/10, 100 MHz/µs 9/10 | everything flagged as a bad fit |
| B5 | **SNR ~40 dB per sweep** (perfect reflector) | all benchmarks assume it; never tested lower | below that, cycle ambiguities and search failures grow (expected; untested). Chirp averaging (100 chirps ≈ 2 ms ≈ +20 dB) is the cheap remedy |
| B6 | The wall is **still** during the measurement, and both bands see the same standoff up to a fitted offset | band offset is fitted (E024b); motion during or between chirps is not modelled | handheld motion would bias the standoff and the cross-band phase |

## C. Known weaknesses inside its scope

| # | Weakness | Size | Fix / status |
|---|---|---|---|
| C1 | **Speed**: ~1–4 s per wall (robust search, plane wave); **minutes** with the antenna-beam model; the 16–54 ms fast path is not yet reliable | E026–E029b, E032 | retrain the fast path for the UWB band pair and validate on 100 held-out walls; beam: K = 1–2 nodes during the search |
| C2 | **Thin, low-contrast air gap** behind insulation | E030 estimated 0.6–0.8 mm for a 3 mm gap. **Not an information limit** (Cramér–Rao bound 0.04 mm); it was a search limitation: the gap's phase cycles were not searched | E033 running: gap in the cycle lattice, hops and ambiguity scan |
| C3 | **All effects at once with the beam model**: only 5 walls tested; 4/5 correct but εr/thickness biased ~2–3 error bars | E024b; likely the air gap (C2) | rerun after E033 on the prototype band pair (60–64 GHz + UWB) |
| C4 | **Uncertainty reporting**: error bars are calibrated *within* the right cycle, but the "plausible alternatives" list (Δχ² < 4) can miss the truth when many cycle options are near-equal | E016 | with UWB the alternatives mostly vanish; otherwise posterior-mass sets (E016) |
| C5 | **Search failures still occur** in hard settings; most are caught by χ², but the fallback search is slow | E029b, E031 | the χ² alarm is now statistically scaled (χ² > n + 3√(2n)); the 3-stage cascade |

## D. Assumptions not yet tested (could surprise us on real data)

| # | Assumption | Why it might matter |
|---|---|---|
| D1 | Materials follow a smooth causal law (constant-Q / Jonscher) across 3–81 GHz | wet or composite materials can have relaxation peaks; the ITU table itself is acausal (E021b) |
| D2 | Noise is white, Gaussian, with known σ | real radars have phase noise, clutter, room multipath and residual Tx–Rx leakage; σ must be estimated |
| D3 | No antenna tilt; isotropic layers; one polarisation | handheld tilt loses signal and shifts paths; wood and fibre mats are anisotropic |
| D4 | Plate calibration is close in time to the wall scan (drift small) | temperature drift was tested only at 0.05 dB / 0.01 rad |
| D5 | The UWB front end behaves like the mmWave one (calibratable linear chain) | a UWB antenna is not a Gaussian beam; its pattern must be measured (E023 note) |
