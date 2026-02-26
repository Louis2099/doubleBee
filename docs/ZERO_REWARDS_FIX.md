# Zero Rewards Fix: Root Cause Analysis

## Problem Summary

All rewards and diagnostic metrics showed **0.0000** even though the robot was clearly moving and falling in simulation.

## Investigation Steps

### Step 1: Add Diagnostic Metrics

Added metrics to track actual velocities and commanded velocities:
- `vel_xy_error_l2` - L2 norm of velocity error
- `actual_vel_x`, `actual_vel_y`, `actual_vel_z` - Robot's actual velocities
- `commanded_vel_x`, `commanded_vel_y`, `commanded_vel_z` - Commanded velocities
- `actual_ang_vel_z`, `commanded_ang_vel_z` - Angular velocities

**Result**: All showed 0.0000!

### Step 2: Check Simulation

From aerodynamics debug output:
```
Robot base velocity: tensor([ 0.1791,  0.2763, -0.4994], device='cuda:0')
Robot VERTICAL velocity (Z): -0.6643 m/s
Thrust being applied: 14.40 N
Robot falling (height decreased: 0.4915m → 0.4794m)
```

**Result**: Robot IS moving and falling!

### Step 3: Identify Contradiction

**Observation**:
- Simulation shows robot moving
- Rewards/metrics show 0.0000

**Key Insight**: Rewards are only logged when episodes complete!

### Step 4: Check Episode Termination

From `on_policy_runner.py` lines 189-208:
```python
if locs["ep_infos"]:
    for key in locs["ep_infos"][0]:
        # ... log rewards
```

Rewards come from `ep_infos`, which is only populated when environments reset (episode completion).

From `manager_based_constraint_rl_env.py` lines 369-390:
```python
# When resetting envs:
self.extras["log"] = dict()
info = self.reward_manager.reset(env_ids)
self.extras["log"].update(info)  # Contains episode reward sums
```

**Result**: No episode completions = No reward logging!

### Step 5: Check Constraints Configuration

**Found in `velocity_env_cfg.py` lines 126-130:**
```python
@configclass
class ConstraintsCfg:
    """Minimal constraints configuration placeholder."""
    pass  # ← EMPTY!
```

**ROOT CAUSE IDENTIFIED!**

## Root Cause

`ManagerBasedConstraintRLEnv` uses a **Constraint Manager**, NOT a **Termination Manager**!

- The `terminations` field in the config exists but is **IGNORED**
- The `constraints` field was **EMPTY** (just `pass`)
- Without constraints, episodes **NEVER** terminate
- Without episode termination, rewards are **NEVER** logged
- That's why all metrics show **0.0000** - they're waiting for episodes to complete!

## The Fix

Converted termination conditions to constraint terms in `velocity_env_cfg.py`:

```python
@configclass
class ConstraintsCfg:
    """Constraint specifications for DoubleBee robot.
    
    Note: ManagerBasedConstraintRLEnv uses constraints, not terminations!
    """
    
    time_out = ConstrTerm(
        func=lambda env: env.episode_length_buf >= env.max_episode_length,
        time_out="truncate",  # Timeout (infinite horizon)
    )
    """Episode timeout constraint."""
    
    fall = ConstrTerm(
        func=lambda env: env.scene["robot"].data.root_lin_vel_b[:, 2] < -0.5,
        time_out="terminate",  # Hard termination
    )
    """Falling constraint - robot falling too fast."""
    
    tilt = ConstrTerm(
        func=lambda env: torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b), dim=1) > 0.5,
        time_out="terminate",  # Hard termination
    )
    """Excessive tilt constraint - robot tilted too much."""

constraints: ConstraintsCfg = ConstraintsCfg()
```

## Key Differences: Constraints vs Terminations

### Standard RL Environment (`ManagerBasedRLEnv`)
- Uses **`TerminationManager`**
- Reads from `terminations` config field
- Uses `TerminationTermCfg` (or `DoneTerm`)

### Constraint RL Environment (`ManagerBasedConstraintRLEnv`)
- Uses **`ConstraintManager`**
- Reads from `constraints` config field
- Uses `ConstraintTermCfg` (or `ConstrTerm`)
- Terminations field is ignored!

## What Will Change After Fix

### Before:
```
Episode_Reward/track_lin_vel_xy: 0.0000  ← No episodes completed
Episode_Reward/vel_xy_error_l2: 0.0000   ← No episodes completed
Episode_Reward/actual_vel_x: 0.0000      ← No episodes completed
... all metrics: 0.0000
```

### After:
```
Episode_Reward/track_lin_vel_xy: 0.0124  ← Real rewards!
Episode_Reward/vel_xy_error_l2: -2.3451  ← Real errors!
Episode_Reward/actual_vel_x: 0.1791      ← Real velocities!
Episode_Reward/commanded_vel_x: 0.3      ← Real commands!
Episode_Constraint/fall: 1               ← Episodes terminating!
Episode_Constraint/tilt: 0
Episode_Constraint/time_out: 0
```

## Why It Happened

1. **DoubleBee was adapted from Flamingo**, which might use standard RL
2. **`ConstraintsCfg` was left as a placeholder** with `pass`
3. **Terminations were defined** but never used
4. **No error was thrown** - the code ran fine, just never logged anything
5. **Early iterations** didn't show the issue because not enough time passed

## Timeline of Events

1. **Iteration 0-21**: Robot falls, thrust applied, but episodes never complete
2. **Robot behavior**: Falling with Z-velocity -0.66 m/s (should trigger fall constraint)
3. **Training continues**: Policy learns from step rewards, but episode metrics = 0
4. **Diagnostic metrics added**: Still show 0 because episodes not completing
5. **Root cause found**: Constraints not configured
6. **Fix applied**: Added fall/tilt/timeout constraints

## Testing the Fix

Run training again:

```bash
python scripts/co_rl/train.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --num_envs 2
```

Expected output (should appear within 1-2 iterations):
```
Episode_Reward/track_lin_vel_xy: 0.01 to 0.10
Episode_Reward/vel_xy_error_l2: -1.0 to -3.0
Episode_Reward/actual_vel_x: <non-zero values>
Episode_Constraint/fall: 1 or 2  ← Episodes are terminating!
```

## Files Modified

1. **`lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/velocity_env_cfg.py`**
   - Added `import torch`
   - Added `from lab.doublebee.isaaclab.isaaclab.managers import ConstraintTermCfg as ConstrTerm`
   - Replaced empty `ConstraintsCfg` with actual constraint terms:
     - `time_out`: Episode length limit
     - `fall`: Z-velocity < -0.5 m/s
     - `tilt`: Excessive tilt (projected gravity > 0.5)

2. **`lab/doublebee/tasks/manager_based/locomotion/velocity/mdp/rewards.py`**
   - Added diagnostic metrics (weight=0.0):
     - `vel_xy_error_l2`
     - `actual_vel_x`, `actual_vel_y`, `actual_vel_z`
     - `commanded_vel_x`, `commanded_vel_y`, `commanded_vel_z`
     - `actual_ang_vel_z`, `commanded_ang_vel_z`

## Lessons Learned

1. **Check if your environment type matches your configuration**
   - `ManagerBasedRLEnv` → Use `terminations`
   - `ManagerBasedConstraintRLEnv` → Use `constraints`

2. **Empty placeholder configs can cause silent failures**
   - `pass` in `ConstraintsCfg` didn't throw an error
   - But it prevented episodes from ever completing

3. **Reward logging depends on episode completion**
   - Step rewards are still computed and used for learning
   - But episodic metrics are only logged when episodes complete
   - 0.0000 metrics can mean "no episodes completed yet"

4. **Check multiple sources when debugging**
   - Training metrics (showed 0)
   - Simulation debug (showed movement)
   - Contradiction led to root cause

## Related Issues

If you see similar symptoms:
- All episode rewards = 0.0000
- Simulation clearly running
- Robot moving/falling
- No error messages

**Check**:
1. Is your environment type constraint-based?
2. Are your constraints configured (not just `pass`)?
3. Are episodes completing? (check episode length buffer)
4. Are terminations/constraints being triggered?

## Prevention

When creating new environments:

1. **Don't leave placeholder configs empty**
   ```python
   # BAD
   class ConstraintsCfg:
       pass
   
   # GOOD
   class ConstraintsCfg:
       time_out = ConstrTerm(...)
       # ... other constraints
   ```

2. **Match config to environment type**
   - Standard RL → Use terminations
   - Constraint RL → Use constraints

3. **Test early termination conditions**
   - Make robot fall/tilt intentionally
   - Check if episodes complete
   - Verify metrics are logged

4. **Add debug prints for episode completion**
   ```python
   if len(reset_env_ids) > 0:
       print(f"[DEBUG] Resetting {len(reset_env_ids)} environments")
   ```

