# Mode-switching baseline: what was built, and how to run it

> **STATUS 2026-09-11 06:00 UTC.** Two training arms are RUNNING on the box.
> Nothing needs launching. See "Status" at the bottom for what to check first.

Built 2026-09-11. Answers IROS R1: *"how much better is it than a well-designed
mode-switching controller?"*. See `REVIEWER_RESPONSE_PLAN.md` item 2.1.

## What it is

Thrust switched between two fixed levels by a hand-written threshold on the
height scan, with a latch:

```
if a step is detected within `lookahead`:   propellers := hold_high
else (and latch expired):                   propellers := hold_low
```

Wheels and servos stay under policy control and are trained by the same
algorithm on the same reward, exactly as in the constant-thrust arms. The only
difference from the full policy is that **continuous** thrust modulation is
replaced by **discrete** modulation on a threshold. That is the architecture the
introduction argues against, built and measured instead of asserted.

## Why this beats the constant-thrust arms as an answer to R1

The constant arms remove modulation entirely, and a reviewer can fairly call a
controller nobody would build a strawman. This one is the controller people
actually build. Every design choice goes the baseline's way:

- **Same information.** The trigger reads the same 4x4 height scanner the policy
  gets in its observation. (The `baseline` notes are explicit that "sees the
  terrain vs doesn't" must not ride along as "learned vs classical".)
- **Better information, actually.** The trigger reads the sensor directly, while
  the policy's observation is delayed one control step. The baseline sees the
  step first.
- **Tunable, with a stated budget.** Five knobs, all environment variables, so
  "you under-tuned it" has a number for an answer.

## Files changed

| File | Change |
|---|---|
| `mdp/actions.py` | `SwitchedPropellerAction` + `Cfg`, `ActionsCfg4DSwitchedThrust` |
| `hybrid_stair/hybrid_stair_cfg.py` | `...SwitchedThrustCfg` and `..._PLAY` |
| `doublebee_env/__init__.py` | registers `SwitchThrust-v1-ppo` and `SwitchThrust-Play-v1-ppo` |
| `scripts/paper/tests/test_switch_trigger.py` | trigger geometry unit test |

Nothing existing changed behaviour. The new task ids are additive; every
previous arm is byte-identical.

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `DOUBLEBEE_SWITCH_LOW` | -0.45 | thrust away from a step. 7.3 N, T/W 0.23, just above the static-stability threshold |
| `DOUBLEBEE_SWITCH_HIGH` | 1.00 | thrust at a step. 17.3 N, T/W 0.55, the published controller's hold point |
| `DOUBLEBEE_SWITCH_THRESH` | 0.02 | rise counting as a step, m. Below the smallest trained step (0.03) so none is missed |
| `DOUBLEBEE_SWITCH_LOOKAHEAD` | 0.105 | forward reach, m. The 4x4 / 0.07 m grid tops out here |
| `DOUBLEBEE_SWITCH_LATCH` | 0.50 | seconds the high mode persists after the last detection |

The default pair spans the full swept fixed-allocation range and brackets the
46 % of body weight that Section II-A says a 6 cm step needs, so the switch has
genuine authority on both sides.

## What I verified, and what I did not

**Verified here.** All three edited files parse. The gym imports resolve. The
trigger geometry passes a unit test covering yaw invariance, translation
invariance, threshold and lookahead bounds, rejection of terrain behind the
robot, and latch behaviour through a climb:

```
cd ~/doublebee_PID_JAI/doubleBee_isaac && python3 scripts/paper/tests/test_switch_trigger.py
```

**Verified on the box, 2026-09-11.** The task instantiates and trains. A
30-iteration smoke run stepped cleanly with no exception.

**One real bug was caught by that smoke run and fixed.** The latch mask was
built as a 1-D `(N,)` tensor and `torch.where` broadcast it against the `(N, 1)`
action into an `(N, N)` tensor, which raised one line later. Had the shapes
happened to line up it would have silently mixed environments. There is now a
shape regression test in `tests/test_switch_trigger.py` that asserts the 1-D
form really does produce the wrong shape and the `(N, 1)` form does not.

One design point the unit test surfaced and settled: while the robot straddles a
step edge there is no higher terrain within reach, so the detector correctly
goes quiet mid-climb. Thrust is held up by the **latch**, not the detector. If
the latch is set too short the baseline will drop thrust halfway up a step. Do
not lower `DOUBLEBEE_SWITCH_LATCH` below about 0.3 s without checking the duty
cycle.

## Running it

**This arm needs a TRAINING run.** I told you earlier it was evaluation only and
that was wrong. Wheels and servos have to learn under the switched thrust
regime, exactly as the `ct*` arms did, or the "you crippled the baseline"
objection lands.

Step 1, train. **Protocol recovered from the `ct*` runs' own `params/agent.yaml`
on the box, 2026-09-11**, so this matches them exactly:

| Setting | Value | Source |
|---|---|---|
| warm start | **none, cold** | `resume: false` in `abl_ct10/params/agent.yaml` |
| envs | 1024 | same file |
| iterations | 4000 | same file |
| seed | 42 | same file |
| energy weight | 4.0 | `energy_consumption.weight` in `params/env.yaml` |

The earlier note in this runbook said to mirror a warm start. That was wrong:
the warm start applies to the **energy-weight** sweep (`hE*`), not to the
actuator ablation arms. These were all trained cold.

On the energy weight: it must be 4.0 to match `hE4` and the `ct*` arms, or the
energy axis is not shared and the comparison stops being matched. Note that the
penalty has little leverage on this arm, since the policy cannot modulate thrust
and can only reduce energy through the wheels, through finishing sooner, or by
triggering the switch less often. That is the same bind the constant arms are
in, so consistency is still right.

```
ssh -i ~/.airlabcloud/ishaan-key.pem ubuntu@172.19.220.34
source /data/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd /data/doubleBee/doubleBee_terr_spawn && \
../../isaaclab/IsaacLab/isaaclab.sh -p scripts/co_rl/train.py \
  --task Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-v1-ppo \
  --algo tqc --num_envs 1024 --max_iterations 4000 --headless --seed 42
```

Conda must be activated explicitly; a non-interactive `ssh` does not source
`.zshrc`, and `isaaclab.sh` then fails with "python: command not found".

Step 2, evaluate, same protocol as every other arm: ten checkpoints, 200
episodes, staircases pinned at 3/4/5/6/7 cm.

Staged on the box as `eval_switch.sh`. It takes the last ten checkpoints
automatically, skips any output that already exists, and writes
`abl_h/climb_sw_h<HH>_<ckpt>.csv`, matching every other arm.

```
ssh -i ~/.airlabcloud/ishaan-key.pem ubuntu@172.19.220.34
cd /data/doubleBee/doubleBee_terr_spawn
./eval_switch.sh logs/co_rl/doublebee_velocity/tqc/<the switch run dir> sw
```

Step 3, sweep the switch at **evaluation** time on that one trained checkpoint
and report the best configuration. Training under every setting is unaffordable;
sweeping at test time and giving the baseline its best result is generous to the
baseline and cheap. Say exactly that in the paper, and state the count.

```
for HI in 0.4 0.7 1.0; do for LO in -1.0 -0.45 0.0; do
  DOUBLEBEE_SWITCH_HIGH=$HI DOUBLEBEE_SWITCH_LOW=$LO \
  python3 scripts/paper/eval_climb.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-SwitchThrust-Play-v1-ppo \
    --checkpoint <best ckpt> --step-height 0.06 --episodes 200 --num_envs 64 \
    --out "abl_h/sweep_sw_hi${HI}_lo${LO}.csv"
done; done
```

## Plotting

`fig_step_height.py` keys arms off a filename tag, so adding the row is one
line in `ARMS`:

```python
("Switched (hand-tuned)", "sw", None),
```

Put it directly under `Learned modulation` so the two modulation strategies sit
together, above the constant arms.

## How to report it

Three outcomes, all publishable, and worth deciding the wording in advance so
the result is not written to fit a hope.

1. **Learned wins clearly.** The introduction's claim becomes measured. State
   the margin at each step height alongside the power, exactly as the constant
   arms are reported.
2. **They tie.** Then the win is in *timing*, not *thresholding*, and the
   open-loop replay experiment (plan item 2.3) is what separates them. Say so
   plainly; a tie against a hand-built controller that needed five knobs and a
   terrain-specific threshold is still an argument for learning.
3. **Switched wins.** Report it. It would mean the continuous-allocation claim
   does not survive on this task, and the paper's contribution narrows to the
   energy objective and the actuator characterisation. Better found now than by
   a reviewer.

Also report the **duty cycle** (`SwitchedPropellerAction.duty`, fraction of
control steps in the high mode). It turns the switch from an unexamined knob
into a described controller, and it is the natural quantity to compare against
the learned policy's continuous thrust trace.


---

## Status, 2026-09-11

**Two arms launched, 4000 iterations each, cold, seed 42, 1024 envs, wE 4.0.**
They differ only in the latch, which is the one parameter that decides whether
the switch behaves like a real mode switch or like a twitch.

| Tag | Latch | Run directory | Log |
|---|---|---|---|
| `sw` | 0.5 s | `tqc/2026-09-11_06-03-40_abl_sw` | `sweep_logs/switch/train_sw.log` |
| `swL` | 1.5 s | `tqc/2026-09-11_06-04-03_abl_swL` | `sweep_logs/switch/train_swL.log` |

Both are **warm started from `energy_abl/2026-09-06_19-25-46_gE0/model_1900.pt`
and run 4000 further iterations, ending at 5899. That is hE4's exact
initialisation and exact budget.** See the section below for why that matters.

**Timing, measured rather than guessed.** The per-iteration time printed in the
log is about 1.15 s, but that counts only the learning step. Measured wall-clock
rate with both arms sharing the GPU is **11 to 12 s per iteration**, which is
consistent with the historical 33 hours when four arms shared the card.

Measured over a 300 s window with both arms running:

| | |
|---|---|
| rate, two arms in parallel | 13.0 s / iteration |
| interim 6 cm result | ~13:40 UTC, 2026-09-11 |
| training ends | ~20:30 UTC |
| full evaluation ends | ~23:00 UTC |

Do not trust the in-log `ETA:` line. It is computed from the learning time only
and reads roughly ten times too fast.

Independent confirmation that the low hold reaches the simulator: the propeller
target velocity printed during startup is 176 rad/s, which is exactly
`320 x (-0.45) + 320`, the configured `hold_low`.

### Check this first

The duty cycle. Grep either log:

```
ssh -i ~/.airlabcloud/ishaan-key.pem ubuntu@172.19.220.34 \
  'grep -a "\[switch\]" /data/doubleBee/doubleBee_terr_spawn/sweep_logs/switch/train_sw.log | tail -20'
```

Measured so far, and the trend is the encouraging part:

| tick | `sw`, 0.5 s latch | `swL`, 1.5 s latch |
|---|---|---|
| 200 | 0.017 | 0.021 |
| 800 | 0.019 | 0.034 |
| 1000 | 0.022 | 0.035 |

Both are rising, and the longer latch sits meaningfully higher, which is what it
was added to do. A cold run reached only 0.008 at tick 200, so the warm start
also helps here: the policy drives toward steps sooner, so the trigger sees
more of them. Absolute values are still low, but tick 1000 is barely 1 % of the
run. That is plausible for an untrained policy that
barely reaches a step, but **if it is still near 0.01 late in training the arm
has silently degenerated into a constant-thrust arm at `hold_low`**, and the
comparison would be meaningless. That is the single failure mode to rule out
before trusting any number from this experiment. The 1.5 s latch arm exists
precisely as the hedge against it.

If both arms degenerate, the fix is a larger `DOUBLEBEE_SWITCH_THRESH` window or
a longer latch, and a retrain. The lookahead cannot go much past 0.105 m because
the height scan is only 0.21 m across.

### Interim result, ready before the full run finishes

A third detached job, `interim_eval.sh`, waits for both arms to pass iteration
4000 and then evaluates their ten most recent checkpoints **at 6 cm only**, into
`abl_h_interim/`. 6 cm is the paper's headline cell, the geometric rolling limit
where the learned policy is reported at 292 W against 391 W for the best fixed
allocation. Expect it to land around 13:00 UTC.

```
ssh -i ~/.airlabcloud/ishaan-key.pem ubuntu@172.19.220.34 \
  'cat /data/doubleBee/doubleBee_terr_spawn/sweep_logs/switch/interim.log'
```

Those checkpoints are **not** the final ones, and the directory is separate from
`abl_h/` precisely so they cannot be mistaken for the real numbers or block the
final evaluation from writing.

### Then

```
cd /data/doubleBee/doubleBee_terr_spawn
./eval_switch.sh logs/co_rl/doublebee_velocity/tqc/2026-09-11_06-03-40_abl_sw sw
./eval_switch.sh logs/co_rl/doublebee_velocity/tqc/2026-09-11_06-04-03_abl_swL swL
```

Then add to `fig_step_height.py`'s `ARMS`, directly under `Learned modulation`:

```python
("Switched (hand-tuned)", "sw", None),
```

### Backup

The three files this touched on the box are backed up unmodified at
`/data/doubleBee/doubleBee_terr_spawn/.bak_switch_20260911/`. The changes were
verified purely additive first: zero lines present on the box were absent
locally.


---

## The warm-start asymmetry in Figure 4 (found 2026-09-11, needs a decision)

You were right to ask. Reconstructed from each run's own dumped
`params/agent.yaml` on the box:

**hE4, the learned arm:**

```
baseline_4000/warm_start.pt          (the balancing policy)
  -> wE0   resume=true   -> model_1400
  -> gE0   resume=true   -> model_1900
  -> hE4   resume=true   -> model_5899      energy weight 4.0
```

**ct10 / ct050 / ctm05 / ctm45, the fixed-allocation arms, and the wheels-only,
wheels-and-servos and propeller-only arms:**

```
resume: false            -> model_3999      cold, from scratch
```

So the learned policy in the actuator ablation has a **balancing warm start plus
about 5899 iterations**, against **cold starts and 4000 iterations** for every
arm it is compared with. That is a 47 % larger budget on top of a better
initialisation. It is the exact shape of the objection R1 already raised once
("the baseline was poorly tuned"), and a reviewer who asks how each arm was
initialised will find it.

**What I did about it for this baseline.** Both switched arms are warm started
from `gE0/model_1900` and given 4000 further iterations, landing on 5899.
Identical initialisation, identical budget, identical reward, identical terrain.
The only difference from hE4 is who decides thrust. That makes
learned-versus-switched a clean comparison whatever the fixed arms do.

**What still needs your decision, for the rest of Figure 4.** Three options:

1. **Retrain the four `ct*` arms warm started from `gE0/model_1900` to 5899.**
   Cleanest, and makes the whole figure matched. Four runs; the GPU has room to
   run them in parallel and it is idle.
2. **Disclose it.** One sentence in the ablation saying the learned arm was warm
   started from a balancing policy and the fixed arms were trained from scratch,
   and that the comparison therefore bounds the fixed allocation from below.
   Honest, cheap, and a reviewer may still object.
3. **Report the switched arms as the primary comparison** and demote the
   constant arms to a supporting sweep, since the switched arms *are* matched.

I would do 1 if there is GPU time, since it is unattended, and 2 regardless.

**Also noted:** the servo rate limit of 2 rad/s applies to every arm equally,
since it lives in the robot asset config rather than in any task config. It
handicaps the baseline and the learned policy identically, so it does not bias
this comparison. It is already reported as a known limitation in Section IV-A.

**Tooling change this exposed.** There was no way to set a warm start except by
editing `co_rl_tqc_cfg.py` before a launch and editing it back after, which is
why the repo reads `resume: false` while the runs that matter were warm started.
`scripts/co_rl/train.py` now accepts `DOUBLEBEE_RESUME_PATH=/abs/path/model_N.pt`,
which takes a file path rather than a regex, because `get_checkpoint_path()`
only matches run directories directly under the log root and the checkpoints
that matter now live one level down in `energy_abl/`.

---

## Interim result, and the correction it forced (2026-09-11 17:00 UTC)

The first two arms were **badly configured by me and have been stopped.**
Interim 6 cm numbers, ten checkpoints each at 200 episodes:

| arm | clears 6 cm | power |
|---|---|---|
| `sw`, low T/W 0.23, latch 0.5 s | 1.7 % +/- 0.7 | 229 W |
| `swL`, low T/W 0.23, latch 1.5 s | 1.6 % +/- 0.7 | 228 W |
| learned hE4 (paper) | 43.4 % | 292 W |
| best fixed T/W 0.55 (paper) | 27.6 % | 391 W |

**Diagnosis.** `hold_low` was set to action -0.45, i.e. T/W 0.23. The paper's own
fixed-allocation sweep puts that level at 1.1 % clearance on its own. With the
switch firing only 3 to 6 % of the time, the arm spends nearly its whole life
there and reproduces that arm almost exactly. The latch was not the binding
constraint: 1.7 against 1.6 across a threefold change in latch says so.

**Why -0.45 was chosen, which matters because it is a repeat offence.** The
`ConstantPropellerAction` docstring calls -0.45 "the 7.17 N static-stability
threshold". Section II-A of the paper derives that threshold as **10.0 N**, 32 %
of weight. The two disagree and the code comment is the lower. This is the same
class of error as the 14.6 N against 17.7 N discrepancy in `DCTRL_VERIFICATION.md`.
**Trust the paper's derivations over the code comments until they are reconciled.**

### Corrected arms, running

Both warm started from `gE0/model_1900`, 4000 iterations to 5899, seed 42,
wE 4.0, latch 3.0 s, high level T/W 0.55 (the best fixed allocation).

| Tag | low level | run dir | rationale |
|---|---|---|---|
| `swA` | -0.05, T/W 0.31 | `2026-09-11_17-11-26_abl_swA` | just above the 32 % static threshold: the lowest a sane engineer would idle |
| `swB` | +0.50, T/W 0.46 | `*_abl_swB` | clears 8.3 % alone: a conservative, strictly stronger switch |

The high level equals the best fixed allocation, so a working switch should land
**between its own low level and 27.6 %**, while drawing less power than the
constant 0.55 arm's 391 W. If it lands below its low level, the switch is
actively harmful and that is worth reporting too.

`switch_driver2.sh` evaluates each arm when its run dir reaches `model_5899.pt`.
Expect training to end near 07:30 UTC on 2026-09-12 and evaluation near 10:30.
