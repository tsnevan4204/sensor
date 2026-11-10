# Version history: known-good states and how to return to them

Branch: `claude/funny-cannon-mrihcy`. Every experiment is its own commit; this table lists the states worth returning to.
To restore one file from a state: `git checkout <commit> -- docs/src/oned/<file>.py` (then commit).
To look at a whole state without changing anything: `git switch --detach <commit>` (return with `git switch claude/funny-cannon-mrihcy`).

| commit | state | what it is good for | known limits |
|---|---|---|---|
| `0d147cb` | stud-wall solver + limitations doc 14 | 3-layer drywall/insulation/drywall walls (E032: 100/100 held-out with UWB) | fixed wall recipe |
| `18a3b13` | E033 air-gap fix, E034 world (`stack.py`: material library, LHM water + fitted T, metal) | gap measured to 0.1 mm; general-stack truth generator | greedy growth search fails past ~3 layers |
| `9efc383` | E035 field self-calibration, E039 UWB-first fast path | no plate in the field; stud wall 100/100 held-out, 0.7–1.1 s | stud-wall recipe only |
| `8a3194d` | E041 exact peeling + exact TMM forward/adjoint (== DEC to 1e-14) | exact building blocks (reused by later solvers) | |
| `bc7151e` | E041 peeling solver, best configuration | 4/6 development 3-layer walls | **1/8 held-out**; minutes per wall; brittle |
| `538409c` | E041 held-out reality check logged | decision point before the E042 rebuild | |

## E042 (in progress): global blocky inversion
Replaces the per-depth search of E034/E041 with one continuous inversion of all depths (thin slices, total-variation regularisation, frequency continuation), then segmentation and the exact polish. Lives in a new file, `profile.py`; `stack.py` (world, TMM) and `peel.py` stay untouched, so reverting = not calling `profile.py`.
