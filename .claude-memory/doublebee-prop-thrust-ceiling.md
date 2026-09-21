---
name: doublebee-prop-thrust-ceiling
description: Sim propellers cap at ~237 rad/s (T/W 0.5) because the 5 N·m effort limit balances a quadratic drag; hardware needs ~500 rad/s. Env overrides added 2026-09-14
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-14T22:54:11.745Z
---

Measured 2026-09-14 across every training log (wE0-wE4, swA3, swB3): propeller
target 640 rad/s, achieved **234-241 rad/s**, PWM ~1308, ~8.0 N per rotor, total
~16 N, **T/W ~0.5**. Hardware PWM cap 1650 = ~500 rad/s in the sim map
(`pwm = 1000 + ω/500·650`, aerodynamics.py) = 36.6 N, T/W 1.16.

Root cause (the 2026-09-08 version of this note blamed a 0.2 N·m effort limit;
that was WRONG, hE4's dumped config has `effort_limit 5.0`): the propeller
velocity loop's torque `damping·(target-ω)` is clipped at 5 N·m, and the prop
settles where that equals a drag that grows as k·ω². 5 N·m at 237 rad/s gives
k ≈ 8.9e-5. Reaching 500 rad/s needs ~22-25 N·m. Raising damping to deliver it
makes the loop unstable at the 5 ms physics step unless the effective inertia
(armature) is raised too: stability needs roughly damping·dt/inertia < 2.

Overrides added in doublebee_v1.py (defaults unchanged):
`DOUBLEBEE_PROP_EFFORT` (5.0), `DOUBLEBEE_PROP_DAMPING` (0.015),
`DOUBLEBEE_PROP_ARMATURE` (1e-5), `DOUBLEBEE_WHEEL_ARMATURE` (0.01), plus the
existing `DOUBLEBEE_SERVO_VEL_LIMIT` (2.0; hardware 10). Wheel: 0.51 N·m over
~0.01 kg·m² ≈ 51 rad/s² vs 58 measured, so armature ~0.0085 should match while
keeping the measured 0.51 N·m torque.

Consequence for deployed policies trained at the cap: hardware needs
`--prop_scale 2` (×5 near steps). A policy trained with the fix should not.
Fitted model still has a 2.3 N per-rotor floor at zero command.
Related: [[doublebee-actuation-envelope]], [[doublebee-sim-renders]].
