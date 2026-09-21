---
name: doublebee-hw-terrain-commands
description: Exact hardware commands per terrain live in hw_final/terrain_commands/; only the trained geometry used FROZEN_COMMAND.sh unchanged
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-13T06:17:00.900Z
---

Hardware trials of 2026-09-11/12 (hE4/model_5899) did NOT all use the frozen
command. Recovered 2026-09-13 by matching each scored log's start time to
~/.zsh_history (db_inference.py does not save its own arguments):

- trained 2x6 cm, 40 cm tread: `hw_final/FROZEN_COMMAND.sh`, all 10 trials.
- narrow 6/7/8 cm, 20 cm tread: `terrain_commands/narrow_678.sh` —
  --prop_boost_dist 0.14, --servo_bias_dist 0.10, three-step map.
- ramp: `terrain_commands/ramp.sh` — --servo_hold_blend 0.8,
  --servo_hold_blend_step 0.5, target y -0.66, and the TRAINED two-step map
  (no ramp map exists; db_inference has no ramp flag).
- foam 8/9 cm: three variants (`foam_89_v1..v3`) — --prop_scale_step 7/8/9,
  --servo_step_bias 0.7 on the last, which produced the best segment.

**Why:** the paper's hardware section claims one frozen command with nothing
tuned between trials or terrains; that is only true for the trained geometry.

**How to apply:** use these files for any reshoot. Before submission the text
must disclose per-terrain deployment tuning. If adding hardware runs, save the
command next to the log (db_inference.py still does not). Related:
[[doublebee-hardware-climbs]], [[doublebee-iros-reviews]].
