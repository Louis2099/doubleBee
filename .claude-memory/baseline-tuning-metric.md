---
name: baseline-tuning-metric
description: "Score baseline/controller sweeps on progress toward the goal, never on episode length or survival"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-13T02:24:26.490Z
---

When sweeping signs, gains or thresholds for a hand-written baseline, score each
configuration by **signed progress toward the target**, never by mean episode
length, survival steps, or "clears" alone.

**Why:** on 2026-09-12 the decoupled PID sign sweep picked
`DOUBLEBEE_WHEEL_SIGN=-1` because it gave 87.8 mean steps against ~34 for the
alternatives. A robot driving away from the staircase never meets a step, so it
never falls, so it survives longest. The sweep was selecting for running away.
Every downstream number was void: max gain exactly 0.0000 m across 32 episodes,
reported to the user as "the decoupled controller cannot climb". It had never
been evaluated at all. The same class of error hit the switched-thrust step
detector, which scanned along body +X while sim forward is body +Y, and passed
its unit test because the test built terrain on the same wrong axis.

**How to apply:** before trusting any sweep result, check that the winning
configuration actually moves toward the goal. Require the output CSV to carry
displacement and a signed forward component, not just `cleared` and `steps`. If
a metric can be maximised by doing nothing or by leaving the scene, it is the
wrong metric. Related: [[doublebee-eval-variance]].
