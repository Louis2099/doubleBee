---
name: doublebee-clearance-metric
description: "Paper's \"clears a step\" % is peak gain >= step height from eval_climb CSVs, NOT the CSV's cleared column"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-13T05:00:03.366Z
---

The actuator-ablation clearance numbers in the paper (Figure 4, hE4 67.9 / 56.9 /
53.9 / 43.4 / 33.2 % at 3-7 cm; fixed arms ct10 27.6, ct050 8.3, ctm05 1.3,
ctm45 1.1 at 6 cm) are computed from `abl_h/climb_<tag>_h<HH>_<ckpt>.csv` as the
per-checkpoint fraction of episodes with `max_gain_m >= step height`, then mean
+- sd across the ten checkpoints. Power is total `energy_J` / total `steps` /
0.02 s. Both reproduce every published cell exactly (verified 2026-09-13).

The CSV's own `cleared` column is stricter (hold + displacement) and gives only
7.7 % for hE4 at 6 cm. Using it silently makes every arm look ~5x worse.

Box script that does it right: `/data/doubleBee/doubleBee_terr_spawn/summ_arms.py
abl_h <tags...>`.

Open question for the paper: Section III-D says success requires the strict
four-condition conjunction, but Figure 4's clearance is peak gain only. Needs one
sentence reconciling them. Related: [[doublebee-eval-variance]],
[[doublebee-energy-ablation]].
