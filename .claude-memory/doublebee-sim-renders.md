---
name: doublebee-sim-renders
description: "How to make clean 4K sim figure renders on the box (scan dots, no arrows) without overwriting existing run videos"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-13T07:20:22.461Z
---

Added 2026-09-13 (both default OFF, training/eval unaffected):
- `DOUBLEBEE_SCAN_VIS=1` draws the 4x4 height-scan ray hits
  (velocity_env_cfg.py height_scanner debug_vis).
- `DOUBLEBEE_NO_ARROWS=1` hides the blue/green velocity arrows in play.py.
- `DOUBLEBEE_SCAN_RGB="r,g,b"` / `DOUBLEBEE_SCAN_R` recolour/resize the scan dots
  (default red 0.02 m, which looks like the red target; cyan 0.0,0.85,1.0 reads well).
- `DOUBLEBEE_PLAY_TARGET_Y="lo,hi"` / `DOUBLEBEE_PLAY_TARGET_Z="lo,hi"` narrow the Play
  terrain's target sampling (default 1.5,3.2 / 0.03,0.20) for a farther, higher goal.
- `DOUBLEBEE_POSE_LOG=1` prints `[POSE] tilt heading fL fR z` every step in play.py, for
  picking upright both-wheels-down frames. Wheel contact must be indexed with the
  CONTACT SENSOR's own body order (`sensor.find_bodies`); the articulation's
  `find_bodies` index read 0 N forever (fixed 2026-09-13, not yet re-verified on a run).
  Proxy that works without contact: tilt < 4 deg and base height steady within 4 mm.

Render: `play.py --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo tqc
--num_envs 1 --headless --video --video_length 450 --render_res 3840 2160
--cam_follow --cam_eye X Y Z --cam_lookat X Y Z --checkpoint <path>`.
Follow-camera offsets are relative to the robot, X = right, Y = forward, Z = up.
The Play terrain IS the training inverted pyramid (0.40 m tread, 3-8 cm); the
"2 gentle stairs" print in the cfg is stale.

**Overwrite trap:** play.py writes the video to `dirname(checkpoint)/videos/play/`
and names it `rl-video-epoch-<first digits in the checkpoint path>`. Rendering
straight from a run dir overwrites that run's existing video; two renders from one
folder overwrite each other. Use `renders/hE4_<shot>/` folders with a symlink to a
checksum-verified copy (`renders/hE4_fig/model_5899.pt`).

Dark "paper look" used by the generalisation renders (navy sky, dark step faces):
`DOUBLEBEE_SKY_RGB=0.05,0.06,0.12 DOUBLEBEE_SKY_INTENSITY=300 DOUBLEBEE_SUN_INTENSITY=6000`.
Without them renders come out pale grey. To show sky, keep the camera low (eye z
~0.85) and look slightly up. To see the scan split across a step edge, film from
BEHIND the robot (negative Y eye offset); a side camera looks along the edges and
the step faces vanish. Pick mid-climb frames from the `[TARGET] robot_xyz` lines
play.py prints every step (z rising between resets).

Figure assembly: `doubleBee_isaac/scripts/paper/fig_terrain_scan.py` (main render +
height-scan inset). Existing hE4 4K play still: `figs/sim_climb_still.png`.
Related: [[doublebee-clearance-metric]].
