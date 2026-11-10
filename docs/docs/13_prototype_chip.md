# 13 · First prototype radar chip: TI IWR6843 (60–64 GHz FMCW)

**Pick:** Texas Instruments **IWR6843**, bought as the **IWR6843ISK** evaluation module plus the **DCA1000EVM** raw-ADC capture board. A dielectric lens (or horn) in front of the antennas is needed to meet the beam requirement from E023.

## Requirements from our results

| need (evidence) | IWR6843 |
|---|---|
| a band that is **legal unlicensed** for a building-diagnostics product (doc 12 §C7) | 60–64 GHz lies inside 57–64 GHz, FCC §15.255, field-disturbance sensors. The 2023 amendment (Report & Order FCC 23-35, ET Docket 21-264) expanded radar use with peak EIRP up to 20 dBm indoors. To verify with counsel. |
| ≥ 4 GHz sweep (range resolution, layer separation) | 4 GHz (60–64 GHz) |
| **raw complex IF samples**, not point clouds (the whole method fits the frequency response) | DCA1000EVM streams raw ADC data over Ethernet; complex ADC up to 12.5 Msps (1×), 12-bit |
| a **known, configurable IF chain** (doc 12 §C4: the IF high-pass filter is a range-dependent gain the model must include) | two independent first-order HPFs, HPF1 ∈ {175, 235, 350, 700} kHz and HPF2 ∈ {350, 700, 1400, 2800} kHz; IF bandwidth up to 10 MHz; pass-band ripple < ±0.5 dB; anti-alias > 60 dB |
| configurable chirp (slope, samples, start frequency) | ramp up to 250 MHz/µs, fully programmable |
| several Tx/Rx | 3 Tx / 4 Rx: Tx–Rx leakage and background can be estimated, antenna phase centres measured, MIMO later |
| precedent for material measurement | the same TI family (IWR1443, 79 GHz) was used for complex-permittivity measurement with metal-plate calibration and small distance increments (arXiv 2403.12726), the procedure E023 formalizes. A 60 GHz IWR6843AOP thesis measured transmittance and absorption of building materials (Illinois IDEALS 131913). |
| a path to the **second band** (E017–E022: two bands make the back of the wall identifiable) | same family, same tooling (DCA1000, mmWave Studio): IWR1843 at 76–81 GHz. E022 gives 10/10 with 77–81 + 57–64 GHz. Its US status for a non-vehicular building device needs checking. The alternative second band is 3.1–10.6 GHz UWB under §15.509 (construction eligibility), e.g. a UWB impulse/FMCW module. |

## Why not the others
- **Infineon BGT60TR13C** (58–63.5 GHz, **5.5 GHz** bandwidth, 1 Tx / 3 Rx, antennas in package). It has more bandwidth and costs less. But its tiny in-package antennas (2–5 dBi) give a very wide beam, which E023 shows costs 10+ dB at 30 cm unless refocused, and there is no simple way to add a lens-matched aperture. It has 1 Tx only, and its baseband chain is less documented. **A strong second choice** if bandwidth turns out to limit thin-layer resolution.
- **TI IWR1843 / AWR1843** (76–81 GHz): the band we modelled first, but the regulatory path for a non-vehicular building device in the US is unclear (Part 95 Subpart M covers vehicular radar). Keep it as the second band once that is clarified.
- **Acconeer A121** (60 GHz pulsed coherent): excellent for 1D distance, but it is not FMCW and exposes no swept-frequency response, so it does not fit our frequency-domain forward model.
- **Vayyar** (3–81 GHz in one chip) would be ideal for multi-band, but it is not openly available for prototyping.

## What this choice changes in the model (next steps)
1. **Primary band moves to 60–64 GHz** (4 GHz). Rerun the single-band and two-band benchmarks on 60–64 GHz: `real:<preset>:dual60`-style configurations with the low band as the primary.
2. **FMCW chain model for this chip (E026):** the received IF signal of a stationary multilayer target. With chirp slope S, the beat frequency of a path of delay τ is S·τ, so the **IF high-pass filters act as a range-dependent gain and phase**, H_HPF(Sτ). Our "frequency samples" are time samples of the chirp, so the sweep nonlinearity, residual video phase and ADC sampling must be modelled.
   - Design rule: at 30 cm, S·τ = 2SR/c. For S = 100 MHz/µs that is 200 kHz, right at the HPF1 corner (175 kHz), which would distort the wall echoes. Use S ≳ 200 MHz/µs and the lowest HPF corners, and model the known HPF response exactly (two first-order sections).
3. **Antenna:** the ISK patch antennas have a wide beam. Put a dielectric lens at the aperture to reach an effective waist ≥ 2 cm at the working distance (E023), and calibrate with the multi-distance staggered-plate procedure (antenna.py; plate steps λc/6 ≈ 0.8 mm at 62 GHz, two base distances).
4. **First lab measurements** (known samples before walls): metal plate at staggered distances (calibration), then acrylic, glass and a drywall coupon of measured thickness. These test εr and tan δ recovery against literature and the causal material law.

## Sources
- TI IWR6843 datasheet (SWRS219F): RF 60–64 GHz, ramp 250 MHz/µs, IF bandwidth 10 MHz, complex ADC 12.5 Msps, HPF corners, ripple.
- FCC §15.255 2023 amendment (FCC 23-35; FCC TCB slides Oct 2023; DA 23-698).
- Infineon BGT60TR13C product page and datasheet (58–63.5 GHz, 5.5 GHz bandwidth, 1 Tx / 3 Rx antennas in package).
- TI IWR1843 product page (76–81 GHz, 3 Tx / 4 Rx).
- Small-distance-increment permittivity with a COTS TI radar: arXiv 2403.12726.

## Update after E031/E032
- **IF chain:** modelled exactly as a linear operator (never inverted). With it, 10/10 at **S = 250 MHz/µs** (9/10 at 100 MHz/µs); ignoring it fails on every wall. A chirp nonlinearity of 1 MHz is negligible.
- **Second band decision: UWB 3.1–10.6 GHz** (E032: 100/100 held-out, 0 silent), not 76–81 GHz IWR1843 / 57–64 GHz alone. So the prototype pair is IWR6843 (60–64 GHz) + a UWB wall-imaging front end. Next: rerun the two-band benchmarks with 60–64 GHz as the TOP band paired with UWB, with the IF chain of each chip.
