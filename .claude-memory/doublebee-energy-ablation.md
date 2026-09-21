---
name: doublebee-energy-ablation
description: The pre-2026-09-04 energy sweep is void; the redone sweep is warm-started and reports penalty share, not raw weight
metadata:
  type: project
---

Every energy-weight result before 2026-09-04 is unusable. Two independent
defects: the `goal_reached` gate needed arrival within 0.25 m while the goal
command carried only a bearing and no range, so it almost never fired (median
closest approach 0.972 m) and `terrain_levels` measured the broken gate rather
than climbing ability; and with one seed per setting, run-to-run variance
exceeded the entire effect (baseline_4000 and its byte-identical config twin
finished at 1.18 and 0.04).

The redone sweep (2026-09-04) warm-starts all five arms from one policy via
`scripts/paper/transplant_obs.py`, so differences are attributable to w_E rather
than to initialization luck. Two consequences to state in the paper:

- The shared checkpoint was trained at w_E=0.25, so effects are UNDERSTATED, not
  inflated. Safe direction.
- The "energy penalty helps escape the flat-ground local optimum" story is no
  longer testable, because the warm start hands the policy the climbing
  behaviour. Do not write that claim.

Report w_E as the energy penalty's share of task reward (0%, 5%, 13%, 27%, 53%),
never the raw weight, and say the shares were measured at initialization: they
drift during training as the curriculum hardens and the task reward falls.

Framing is SELECTION ("we swept five values and deployed the best"), not a
measured effect size. With n=1 per setting that is the only claim the data
supports, and it needs no error bars.

2026-09-08, measured: CHECKPOINT variance dominates the clearance ranking. On a
fixed 6 cm staircase, hE6's median max_gain swung 0.047 -> 0.082 m across three
consecutive checkpoints (74%), which exceeds the entire spread between all five
weights at any single checkpoint. hE4 beats hE6 at two of those three
iterations. The single-checkpoint figure therefore ranked snapshots, not
weights, and the "wE=4 degrades into a do-nothing optimum" story does not
survive its own data (it fails to explain why wE=6 is fine) -- do not write it.

Energy is the stable measurement, clearance is not. Report the claim as
"clearance unchanged, energy down ~25%" with error bars that are spread across
the final 10 CHECKPOINTS of one run, and say in the caption that they are
checkpoint spread and not seed spread. Single seed per weight remains a stated
limitation; seeding the full sweep does not fit before 2026-09-15.

Success criteria are NOT measured by goal_reached in any reported number:
eval_climb measures height, hold and displacement kinematically. goal_reached
carries a 0.15 m height tolerance on a 0.12 m play staircase, so it is vacuous
there and a robot at the foot of the stairs satisfies it. Gated behind
DOUBLEBEE_GOAL_DZ (default 0.15, unchanged); not retrained.

See [[doublebee-icra-plan]], [[doublebee-paper-status]], [[doublebee-hardware-climbs]].
