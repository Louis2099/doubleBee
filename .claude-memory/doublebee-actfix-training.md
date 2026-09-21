---
name: doublebee-actfix-training
description: "2026-09-14 from-scratch TQC runs with hardware-matched actuators (props ~560 rad/s, servo 10 rad/s); side project, not in the paper"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-14T23:09:00.773Z
---

Launched 2026-09-14 23:03 UTC on the box, user request ("just for the sake of it,
see how it does in sim and if good, irl"). NOT for the ICRA paper.

Driver `train_matched_actuators.sh` (repo root on box), logs `sweep_logs/actfix/`,
runs `logs/co_rl/doublebee_velocity/tqc/*_actfix_wE40` and `*_actfix_wE025`.
Recipe: REWARD_V2, NO_DR, EPISODE_S 12, no resume, 1024 envs, 4000 it, seed 42;
W_E 4.0 (paper weight) plus 0.25 hedge (from-scratch at 4.0 may never learn thrust).
Actuators: `DOUBLEBEE_PROP_EFFORT=25 DOUBLEBEE_PROP_DAMPING=0.16
DOUBLEBEE_PROP_ARMATURE=0.0016` (probe: props reach 470-562 rad/s, stable; base
caps at 237), `DOUBLEBEE_WHEEL_ARMATURE=0.0085`, `DOUBLEBEE_SERVO_VEL_LIMIT=10`.
Verified in dumped env.yaml.

Hardware deployment of these policies should NOT need prop_scale 2/5 or the servo
slew hacks; the db_inference.py interface will need re-tuning.
Code pushed first: branch ish/hybrid_mode, tag `icra2027-submission` = f766976.
Related: [[doublebee-prop-thrust-ceiling]], [[doublebee-hw-terrain-commands]].
