# 01 · Vision and business context

## The idea

A new kind of camera that sees through walls and other occlusions. It takes in radio signals (mainly RF and mmWave) and uses physics-structured ML to work out **what** hidden objects and scenes are there. That means not only that something moves, and not only shape, but also **material properties**: permittivity, conductivity and moisture. Later it may extend across bands (UWB, mmWave, possibly THz), staying within limits that are safe for humans.

## Why it could be a business

Seeing through walls already exists, but nothing owns the middle of the market:
- **Low end:** shallow, cheap scanners such as Walabot (Vayyar's consumer product) find studs and pipes a few inches deep.
- **High end:** expensive military and law-enforcement systems (Camero-Tech Xaver, Lumineye) mostly show blobs for people and motion.
- **Research:** systems at MIT and Penn reconstruct real geometry, but they are not products.

The bet: a fast, physics-exact model gives
- **higher fidelity**, because it inverts the wave physics (multiple reflections, layers, loss, dispersion) instead of hallucinating or just backprojecting;
- **speed**, because learned and exact reduced models are far cheaper than full simulation;
- **lower cost**, because better software allows simpler hardware.

## The owner's edge

- Works in **Prof. Nathaniel Trask's PIMI lab at UPenn** on physics-respecting ML: data-driven Whitney forms, data-driven exterior calculus (DDEC), graph surrogates and other physics simulation work.
- The plan is to bring fast learned models that still obey physics exactly to RF sensing.
- The exact technique was deliberately left open, to learn from the people already working in this space.

## What existing systems miss (and our angle)

- **They treat sensing as a geometry problem.** MIT and Penn use ray, mirror and specular models plus generative AI fill-in. Their published failure modes are wave effects: penetration and metal occlusion, hollow or two-surface objects, edges (diffraction), rough surfaces, and material behaviour. See `04_literature_and_competitors.md`.
- **No one recovers materials.** Our model's unknown is ε(x), σ(x) (in DEC terms, the Hodge stars), so shape, material and moisture come from a single field. In particular, water has very large ε and loss.
- **Physics-based super-resolution at penetrating frequencies.** The 1D toy already resolves 12.7 mm drywall layers with 3.75 cm radar range resolution.

## Competitive landscape (as of Nov 2026; verify before relying on it)

- **Cartesian (Cartesian Systems).** Co-founded 2023 by Fadel Adib (MIT Signal Kinetics PI) and Isaac Perper, licensing MIT patents.
  - Today: RFID item localization for retail, 700+ stores, including Inditex brands.
  - Stated next markets: warehouses, manufacturing, robotics.
  - Cartesian is an affiliation on mmNorm, Wave-Former and RISE, so it is the likely vehicle for MIT's imaging work: a probable future competitor in logistics and package inspection.
  - Check MIT patents (Adib as inventor) for freedom-to-operate before raising money.
- **Applied Electrodynamics (WaveSight).** YC company that launched 2026-07-30, founded by Apple and Humane alumni. Maps pipes and wiring in walls with a claimed ~1 mm accuracy; preorders open. It targets in-wall *geometry*.
- **Vayyar.** 4D imaging radar; Walabot is its consumer product.
- **Lumineye** (first responders, US Army) and **Camero-Tech Xaver** (military): high end, focused on people and motion.
- **Penn WAVES Lab (Mingmin Zhao):** no startup found.

## First-market candidates (where the physics edge matters and Cartesian is not)

1. **Hidden moisture and building diagnostics.** Buyers: water-damage restoration, insurance adjusters, home inspectors. Material sensing, not people sensing, likely avoids the FCC through-wall-imaging restrictions (verify).
2. **Industrial inspection of layered structures:** corrosion under insulation, composite delamination, concrete voids. High value, but long sales cycles and certification.
3. Food and agriculture (moisture, foreign objects): crowded, because X-ray dominates.

**Avoid:** warehouses and retail (Cartesian), security screening (incumbents), and people-through-walls (regulated, military).

**Regulatory note to verify:** US FCC ultra-wideband rules (Part 15 Subpart F) are understood to restrict through-wall imaging devices to law enforcement, fire and rescue. Get regulatory advice early.

## Contacts identified

- **Penn WAVES Lab (PI Mingmin Zhao):** HoloRadar students Haowen Lai and Zitong Lan.
- **MIT Signal Kinetics (PI Fadel Adib):** Laura Dodds (mmNorm, Wave-Former, RFlect, MITO) and Maisy Lam (simulation and training data).
  - They are close to Cartesian. Talk technology with them; keep go-to-market thinking for Penn and potential customers.
- **Kaibo Hu:** finite element exterior calculus (FEEC), DEC stability theory. Potential collaborator on the stability theory for 2D wave problems.
- **Oliver Gross** (geometry and physics simulation; works with Albert Chern).
