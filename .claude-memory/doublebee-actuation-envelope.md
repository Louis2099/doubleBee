---
name: doublebee-actuation-envelope
description: Measured DoubleBee wheel/propeller actuation limits (2026-08-20) and the sim params they contradict
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-08-21T00:37:04.220Z
---

Measured on hardware 2026-08-20 with `db_wheels.py`. None of it is derivable from
the code — the configs carry datasheet assumptions that are wrong.

**Wheels** (RoboClaw 2x7a v4.2.8, encoders on the RoboClaw, not the Cube):
- `counts_per_rev = 1920` **verified** by hand: 1926 (M1) / 1934 (M2) over 10 turns.
- No-load ceiling **23.6 rad/s** at full duty (duty→speed dead linear). The
  configs said 35.0, from an assumed 330 RPM — 47% too high.
- Loaded (robot's weight) sustained **~14–15 rad/s**; with the frame support
  dragging, ~10.8.
- Acceleration under load **~43 rad/s²**. Sim relies on **74 median / 104 p90 /
  270 p99** — this 1.7–6× gap is the real sim2real blocker, and it is governed by
  `effort_limit` (1.0 N·m, never validated; estimate 0.2–0.6, start 0.35).
- QPPS was 10000 (claims 32.7 rad/s); **set to 7212 and committed to NVM**
  via `SetM1/M2VelocityPID` + `WriteNVM` — Motion Studio changes were not
  persisting. P/I/D left at defaults 1.0/0.5/0.25, which measured fine.

**Propellers**: sim caps PWM at **1650 µs** (`aerodynamics.py:189`,
`1000 + 1.3·|ω|`). Thrust from `pwm2thrust_params.json`: T/W = 0.62 at 1325 µs
(sim's mid-throttle), 1.34 at 1650. The robot runs at a **median 42° tilt** in
sim — it is a leaning machine, so a 60° hardware reading is not by itself a fall.

**Physics note**: accelerating the wheels forward from rest tips this robot
backwards (reaction torque). Any open-loop constant-velocity wheel test will
tip it — that is not a fault.

**Geometry gap the sim does not have**: the real robot carries a protective
support extending outwards from the Cube, so when it tips it comes to rest on
that instead of crashing. Sim has two wheels and nothing else, so past ~60° it
is unrecoverable there while on hardware it simply sits down. This also
contaminates loaded wheel measurements — with the support dragging, wheel speed
reads ~10.8 rad/s versus ~15 upright. Worth adding as a collision body.

Deployment fixes made the same day, all in `db_inference.py`: servo2 takes
`+action[3]` not `−action[3]`; no action duplication (the `ActionsCfg4D`
docstring claims it, the code does not); `--prop_map sim`; wheels gated on
ARMED **and** CH6 > 1500 µs. See [[doublebee-no-rewiring]] and
[[doublebee-paper-status]].
