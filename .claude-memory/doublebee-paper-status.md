---
name: doublebee-paper-status
description: DoubleBee paper venue status and where the ICRA follow-up list lives
metadata: 
  node_type: memory
  type: project
  originSessionId: cf051c47-3a28-4ce2-addc-104ec95c93da
  modified: 2026-08-13T04:16:58.594Z
---

The IROS Sim2Real workshop version of the DoubleBee energy-aware RL paper was submitted
on 2026-08-13 **knowingly unchanged**, despite a review that found paper-vs-code
discrepancies. The user's explicit call: workshop is low-stakes, fix everything for ICRA.
Do not re-raise those findings as blockers.

All findings are recorded in `ICRA_TODOS.md` at the repo root — read that before doing any
new paper or training work rather than re-deriving from the configs.

Key non-obvious fact captured there: the paper's simulation results come from
`ckpts/g1_38/model_1700.pt` (TQC, 38-dim obs), while the hardware results come from
`doubleBee_isaac_og/logs/co_rl/2026-03-01_1650-doram_thr_servo_hybrid_stair/model_800.pt`
(PPO, 22-dim obs, no height scan — the deployed policy is blind to terrain).
