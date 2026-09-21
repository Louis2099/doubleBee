---
name: doublebee-switched-baseline-results
description: "Matched switched-thrust baseline (swA3/swB3, hE4 recipe) numbers and the open sweep outlier, 2026-09-14"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4a1ea8f7-460d-4372-9a50-1f738c3051e4
  modified: 2026-09-14T00:05:25.337Z
---

Matched switched-thrust arms trained 2026-09-13 on hE4's exact recipe
(DOUBLEBEE_REWARD_V2=1, NO_DR=1, EPISODE_S=12, W_E=4.0, warm from
gE0/model_1900 to 5899; saved configs diff to hE4 only in the propeller block).
swA3 low hold -0.05 (T/W 0.31), swB3 low 0.50 (T/W 0.46); high 1.0, thresh 0.02,
latch 3.0 s. Eval in abl_h/ (paper metric, 10 ckpts x 200 eps; +- below is SD
across ckpts, SE = SD/sqrt(10)):

| clears % | 3 | 4 | 5 | 6 | 7 cm | power W |
|---|---|---|---|---|---|---|
| hE4 learned | 67.9 | 56.9 | 53.9 | 43.4 | 33.2 | 281-297 |
| swA3 | 67.4 | 50.9 | 38.5 | 29.6 | 16.2 | 263-271 |
| swB3 | 64.3 | 51.3 | 37.0 | 33.1 | 13.9 | 320-326 |
| ct10 best fixed | 65.1 | 42.7 | 37.5 | 27.6 | 8.7 | 383-392 |

Test-time sweep (abl_sweep/, single best checkpoint, 200 eps, 18 configs) showed
swA3 at thresh 0.04 m / latch 3.0 clearing 83 % at 6 cm on ckpt 5700 alone.
POOLED OVER 10 CKPTS (abl_confirm/climb_swA3t04_*, 2026-09-14) IT CLEARS 24.1 %
(SD 13.9), below the trained setting's 29.6 %. The 83 % was single-cell noise;
learned hE4 (43.4 %) still leads at 6 cm. Lesson re-confirmed: never pick or
report from single-checkpoint sweep cells. swB3's best sweep cell (45.5 %,
thresh 0.01, latch 1.5) is being pooled as swB3best; remaining heights of both
continue in confirm_best.sh.

**Why:** this is the Reviewer 1 mode-switching baseline; the abstract's
"x % saved against a controller" depends on it.
**How to apply:** report pooled numbers only; the thresh-0.04 setting was chosen
on 6 cm, so if it wins at 6 cm check the other heights before generalising.
Fixed-thrust ct* arms remain on the old recipe (user declined retrain) -> disclose.
Related: [[doublebee-iros-reviews]], [[doublebee-clearance-metric]], [[doublebee-eval-variance]].
