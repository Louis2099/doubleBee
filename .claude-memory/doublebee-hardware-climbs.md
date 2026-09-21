---
name: doublebee-hardware-climbs
description: "2026-08-27 milestone — DoubleBee hardware climbs two risers (11 cm), and RUN 1 beat its old sim peak at 2.6x terrain difficulty"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-08-28T02:58:09.485Z
---

**2026-08-27: the robot climbs on hardware.** Measured (hw_v33): 1.87 m traversed,
**11 cm climbed over two risers** (6 cm then 5 cm, 0.4 m apart), net yaw 10.9°,
mean 0.31 m/s, zero wheel saturation. Checkpoint: `ckpts/SAFE_run1_peak/model_3500.pt`
(backed up out of `ckpts/latest/` so training cannot overwrite it).

The reproducing command, with the rationale for every non-obvious flag, is
appended to `doubleBee_isaac/scripts/paper/cmds`. Generate the terrain map with
`modified_mav_ros_src/.../srv/make_step_map.py` — never hand-convert.

**Sim RUN 1 beat its own old peak.** Old peak was success 0.3419 / terrain_levels
0.5840 at iteration ~3500, and it was decaying by 3665. After the
`terminal_reward_goal_reached` forfeited-income compensation plus halving the three
position-independent prop rewards, RUN 1 hit **success 0.3563 at terrain_levels
1.51** (iteration 4871) — same success, 2.6× the terrain difficulty, still rising.
Task reward now outbids posture reward 1.50 : 1.20; it was 1.94 : 1 the other way.

**Two calibrations that go stale silently and break everything:**
- `--base_z_offset` — the mocap rigid body definition changed mid-project. The old
  `-0.0315` assumed a mocap z of 0.1315; the robot now reads ~0.034 upright. A stale
  value puts all 16 `height_scan` dims (40% of the observation) out of distribution
  and the same command that worked before silently fails. Recalibrate from live pose.
- Forward is **body +Y** = `(-sin yaw, cos yaw)`, not `(cos yaw, sin yaw)`. Confirmed
  empirically: motion projects onto it at +0.89/+0.91 across two logs.

**Still open:** it pendulums when it leans — the attitude hold is statically stable
at *any* lean by design, and `--servo_hold_damping` is the only damper. Measured
pumping from the wheels, `corr(pitch_rate, a0) = +0.276` with the wheel command
lagging pitch by 160 ms, traced to `--roboclaw_accel 13139` (43 rad/s² against sim's
74 median). Untested: `--roboclaw_accel 0` with `--servo_hold_damping 0.30`.

Related: [[doublebee-icra-plan]], [[doublebee-actuation-envelope]], [[doublebee-no-rewiring]]
