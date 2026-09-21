---
name: doublebee-eval-launch-noise
description: eval_climb goal patches are resampled per launch (unseeded) -> one cell swung 20-83 %; use --seed for paired comparisons (Option B)
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-14T02:33:06.072Z
---

Found 2026-09-14. `eval_climb.py` never seeded anything, so each launch builds the
terrain with a fresh random set of 5 target flat patches per tile (3-20 cm up,
1.5-3.2 m ahead) and picks goals per episode with unseeded `torch.randint`. Low,
close goals let the robot reach the goal having climbed < 1 step, which the
paper's clearance metric (peak gain >= step height) counts as a miss.
Identical checkpoint (swA3 5700), switch setting and 6 cm pin gave 83 %, 20 %,
45 % across three launches. Pooling 10 launches (as Figs. 4/5 do) is unbiased but
noisy.

Fix: `eval_climb.py --seed N` (added 2026-09-14; default None keeps every old
result reproducible; backup `.bak_eval_climb_seed_*`). Seeds python, numpy,
torch CPU/CUDA and env_cfg.seed before gym.make.

Option B (user-approved 2026-09-14, final box task for the paper): matched seeded
eval of hE4, swA3, swB3, ct10 (T/W 0.55) — last 10 ckpts each, 3-7 cm, 200 eps,
seed = 1000*H + k identical across arms. Driver `optionB_driver.sh` -> abl_seeded/,
summary `summ_optionB.py` (paired learned-minus-baseline diffs, power, equal-work
energy by steps climbed). Figs. 4/5 stay untouched; B results go in a separate
baseline table. User wants learned to beat the switch on both climbing and energy;
earlier unseeded data: learned wins first-step climbing, low-idle switch matches
it on 2+ steps with less energy there.

**How to apply:** verify two same-seed runs agree before trusting B; report
paired diffs with SE; never report single-launch cells.
Related: [[doublebee-switched-baseline-results]], [[doublebee-eval-variance]], [[doublebee-clearance-metric]].
