---
name: doublebee-eval-variance
description: eval_climb reach numbers swing 15-25 points between identical runs; pool 10 checkpoints and report across-process spread, never a single run
metadata:
  type: project
---

Measured 2026-09-09. A single `eval_climb.py` process gives a reach number with
a huge, undocumented error bar. Two independent demonstrations, same checkpoint
and same flags both times:

- hE4/model_5899, play terrain: reach-6cm **17%** (pass 1) vs **34%** (pass 2).
- hE4/model_5899, `--step-height 0.06`: reach-6cm **22%** (sweep.sh) vs **47%**
  (night.sh). z = 5.5 against the binomial SE, so NOT episode sampling noise.

Cause: the terrain generator runs once per process (seed 42, `use_cache=False`)
and all 64 envs draw spawn/target patches from that single instance. One process
is ONE draw of the terrain-and-spawn configuration, sampled 200 times. The 200
episodes are correlated, so effective n is far below 200 and a Wilson interval
on them is much too narrow.

WHICH STATISTICS ARE SAFE:
- **Power (W) is stable**: 237 vs 242 W across the same two passes, under 2%. It
  averages over every timestep of every episode.
- **Termination mix is stable**: tilt 85% vs 88%.
- **Reach at a threshold is NOT stable**: it is a crossing on the upper tail of
  the gain distribution and inherits the whole draw-to-draw variance.
- Hard zeros are safe: wo/ws/po give mean gain 0.0001-0.0000 m, which no draw
  can manufacture.

HOW TO MEASURE: pool 10 checkpoints, one process each, as `night.sh` does for
the energy sweep. That is why the energy figure's pooled numbers (59/52/13%)
are trustworthy while any single-run ablation number is not. Report the spread
ACROSS processes as the error bar.

Do not re-run an arm and keep the pass where it looks better. See
[[doublebee-energy-ablation]] and [[doublebee-paper-status]].
