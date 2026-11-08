# radar-ddec

**Physics-exact, low-compute inversion of radar data to recover hidden structure and materials behind walls**, built on data-driven exterior calculus (DDEC) and Whitney forms.

- **Status:** 1D layered-wall inversion is working and validated:
  - exact forward model (1e-14 vs reference);
  - adjoint gradients;
  - calibration;
  - Levenberg–Marquardt with layer stripping;
  - Cramér–Rao error bars plus global ambiguity sets.
- **Next:** make 1D fast and general, then 2D. See `docs/07_roadmap.md`.

Start here:
1. `CLAUDE.md`: mission, constraints and working rules (for Claude Code and for humans).
2. `docs/00_INDEX.md`: all context.
3. `docs/physics_explainer.html`: illustrated derivation.

```bash
pip install -r requirements.txt
pytest -q tests/
cd src/oned && python3 run_validation.py   # ~26 min on 2 cores
```
