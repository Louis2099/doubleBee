---
name: doublebee-icra-plan
description: "DoubleBee ICRA plan as of 2026-08-21 — TQC only, stability-before-climbing gate, and the pendulum/delay analysis"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-08-21T22:59:10.900Z
---

ICRA submission **2026-09-15**; user needs a hardware climb by ~Aug 31.

**Hard constraint: the paper's policy is TQC.** Do NOT propose falling back to the
PPO checkpoint (`model_800`) — the user's words: "ppo was a joke and a lie it
didnt really climb clean." They are first author and TQC has to be the system. I
suggested the PPO fallback once and it was rejected; don't raise it again.

**Method that matters: separate stability from climbing.** Trying for both at once
produced neither and cost hardware (a broken propeller and a broken bumper on
2026-08-20/21). The gate test is flat ground, no step, one question: does it stay
upright 30 s? Nothing about climbing gets tuned until that passes.

**Why it was diverging** (transfer3.csv, 2026-08-21): the policy recovered 69° of
lean under thrust, overshot through vertical, then oscillated apart. Not lack of
authority — excess loop gain against unmodelled delay. The numbers:
- propellers sit ~443 mm above the wheel axle, CoM ~139 mm above it
- inverted-pendulum time constant τ = √(L/g) ≈ **119 ms**
- real actuator delay 40–100 ms → barely stabilisable
- upward thrust offsetting fraction f of weight stretches τ by 1/√(1−f):
  f=0.7 → 217 ms. **Sustained vertical thrust is what makes the plant
  controllable at these delays** — this was the user's own insight ("the props
  just gotta make the whole thing upright, point up, that's all").

**Changes staged 2026-08-21** (in `doubleBee_isaac`, needs syncing to the training
box): tied servos via `TiedJointPositionAction` (5 actions / 37 obs, scalar scale
— a mirrored dict would lock the arms opposed); wheel `scale=47`; propeller range
halved to 0–250 rad/s so training authority matches the deployed `--prop_scale`;
propeller actuator delay 2–5 steps; `reward_props_upright` 1.5→5.0; new
`reward_vertical_thrust_support` (w=3.0, target_frac=0.7).

Expect energy to get worse — stability was deliberately traded for it.

Still unmeasured and load-bearing: **end-to-end latency**. The 40–100 ms is my
estimate; a timestamped loopback `/Jai_command` → `/mavros/rc/out` would pin it.
See [[doublebee-actuation-envelope]] and [[doublebee-no-rewiring]].
