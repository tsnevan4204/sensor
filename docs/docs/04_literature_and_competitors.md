# 04 · Literature: who is doing what

## MIT Media Lab, Signal Kinetics (PI Fadel Adib; also Cartesian)

They treat imaging mostly as a **geometry problem** (specular rays + generative AI). Papers in `papers/`.

| Work | What it does | Physics model | Stated limits |
|---|---|---|---|
| **mmNorm** (Dodds et al. 2025) | Non-line-of-sight 3D object reconstruction from mmWave via surface normals | (1) Normal estimation by coherent voting across synthetic-aperture positions (specularity). (2) Normal field inverted to a relative SDF (RSDF), defined only up to an offset, so every isosurface is a candidate. (3) Isosurface chosen by ray tracing + free-space path loss, compared on FFT magnitudes; ignores noise and reflectivity | Metal/thick occluders; hollow objects (two surfaces); surfaces outside aperture coverage; sharp edges |
| **Wave-Former** (Dodds et al., CVPR 2026) | Through-occlusion shape completion | Samples many candidate surfaces from mmNorm's RSDF, completes each with a transformer trained on vision datasets turned into mmWave-like partials (specularity masks with angle thresholds, plus noise), then selects by lowest local "entropy" (PCA eigenvalue ratio) | Recall about 72%, precision about 85% |
| **RISE** (Zhou et al., CVPR 2026) | Room-scale scene from one static radar | Simulator built on about 35k real floor plans; 2D skeletons; first-hit rays; one reflective segment per object; diffusion model | Layout error about 16 cm |
| **MITO** (NeurIPS 2026 datasets track, arXiv 2502.10259) | mmWave dataset + simulator | 76 objects, 24M+ frames, 550 SAR images, two radars; simulator for any triangle mesh | n/a |
| **RFlect** (Dodds et al. 2024) | Around-the-corner imaging with complex reflectors | Specular reflection only | Diffuse scattering, polarization, materials left as future work |
| Super-resolution mmWave normal estimation (MobiCom 2026 work-in-progress) | Improves mmNorm step 1 | n/a | n/a |
| Detection of cardiac implants (JAMA Cardiology, Sept 2026) | Medical use case | n/a | n/a |

Also in `papers/`'s source set: RFID localization (handheld, RL2), MiFly, MiNav, AR non-line-of-sight perception. These are background on Cartesian's RFID roots.

**Where we differ:**
- We model wave physics (penetration, layers, loss, dispersion, multiple reflections) exactly in the forward model.
- We recover **materials**, not only shape.
- We get guarantees (energy, passivity, causality) and calibrated uncertainty, including global ambiguity.

## Penn WAVES Lab (PI Mingmin Zhao)

- **HoloRadar** (NeurIPS 2025): sees around corners with a single mmWave radar on a robot.
  - Physics-guided learning that treats walls and floors as mirrors and maps mirrored observations back to true locations, using ray interactions.
  - Data from 32 corners in 5 buildings (real-world). Whether a simulator is used is unknown; ask.
  - Contacts: Haowen Lai, Zitong Lan.
- Zhao started this line of work as a PhD student at MIT.

## Other groups and tools

- **Haitham Hassanieh (EPFL), Saurabh Gupta (UIUC):** co-authors on RFlect.
- **Moeness Amin (Villanova, Philadelphia):** through-the-wall radar imaging, layered Green's functions. Strongest classical baseline.
- **Computational EM / inverse scattering community:** ML for inverse scattering, e.g. single-transmitter permittivity reconstruction (arXiv 2506.21349), subspace-optimization methods.
- **NVIDIA Sionna RT:** differentiable radio ray tracing (geometric).

## Companies

See `01_vision_and_business.md`: Cartesian, Applied Electrodynamics/WaveSight, Vayyar/Walabot, Lumineye, Camero-Tech.

## Source links collected during research

- MIT News, Wave-Former: https://news.mit.edu/2026/generative-ai-improves-wireless-vision-system-sees-through-obstructions-0319
- Wave-Former: https://arxiv.org/abs/2511.14152
- RISE: https://arxiv.org/pdf/2511.14019
- MITO: https://arxiv.org/abs/2502.10259v3
- HoloRadar: https://proceedings.neurips.cc/paper_files/paper/2025/hash/6994a52f9499b865b7c9e167f5495ef3-Abstract-Conference.html and https://haowenlai.github.io/HoloRadar-Website/
- mmNorm project: https://www.media.mit.edu/projects/reconstructing-hidden-objects-with-wireless-signals-1/overview/
- Cartesian (MIT News): https://news.mit.edu/index%2ephp/2026/cartesian-helps-retailers-track-their-products-in-real-time-0605
- WaveSight: https://runtimewire.com/article/applied-electrodynamics-wavesight-wall-camera
- Laura Dodds publications: https://laura-dodds.github.io/
- Fadel Adib publications: https://www.mit.edu/~fadel/publications.html
- Chern & Gross, conformal geodesics: https://arxiv.org/abs/2312.05252v2
- Electromagnetic inverse scattering from a single transmitter: https://arxiv.org/abs/2506.21349v1
- Back-Projection Diffusion: https://arxiv.org/abs/2408.02866v3
- Plug-and-play latent diffusion for EM inverse scattering: https://arxiv.org/html/2509.04860v1
- Sionna RT: https://research.nvidia.com/publication/2023-12_sionna-rt-differentiable-ray-tracing-radio-propagation-modeling
