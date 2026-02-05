# Diagnostic Metrics Guide

## Overview

Added diagnostic metrics to help debug why rewards are showing as 0.0 during training. These metrics have `weight=0.0`, meaning they don't affect the reward calculation but are logged for monitoring.

## Added Metrics

### Velocity Error Metric

**`Episode_Reward/vel_xy_error_l2`**
- **Formula**: `-sqrt(||v_xy - v_cmd_xy||²)`
- **What it measures**: L2 norm (Euclidean distance) of the velocity error
- **Range**: Negative values (e.g., -0.5 means 0.5 m/s error)
- **Purpose**: Shows the actual mismatch between commanded and actual velocity

**Relationship to reward**:
```
reward = exp(-error²)
error = sqrt(sum((actual - commanded)²))

If error = 0.0 → reward = exp(0) = 1.0 (perfect)
If error = 1.0 → reward = exp(-1) = 0.368
If error = 2.0 → reward = exp(-4) = 0.018
If error = 5.0 → reward = exp(-25) ≈ 0.0000 (shown as 0.0000)
```

### Individual Velocity Components

**XY Velocities:**
- `Episode_Reward/actual_vel_x` - Robot's actual X velocity (body frame)
- `Episode_Reward/actual_vel_y` - Robot's actual Y velocity (body frame)
- `Episode_Reward/commanded_vel_x` - Commanded X velocity
- `Episode_Reward/commanded_vel_y` - Commanded Y velocity

**Z Velocity:**
- `Episode_Reward/actual_vel_z` - Robot's actual Z velocity (vertical)
- `Episode_Reward/commanded_vel_z` - Commanded Z velocity

**Angular Velocity:**
- `Episode_Reward/actual_ang_vel_z` - Robot's actual yaw rate
- `Episode_Reward/commanded_ang_vel_z` - Commanded yaw rate

## Diagnostic Scenarios

### Scenario 1: Nothing is Moving (Simulation Issue)

**Expected metrics:**
```
Episode_Reward/vel_xy_error_l2: -0.3 to -0.5  (moderate error)
Episode_Reward/actual_vel_x: 0.0000  ← Robot not moving
Episode_Reward/actual_vel_y: 0.0000  ← Robot not moving
Episode_Reward/commanded_vel_x: 0.3  ← Commands exist
Episode_Reward/commanded_vel_y: 0.2  ← Commands exist
Episode_Reward/track_lin_vel_xy: 0.9 to 0.7  (moderate reward)
```

**Diagnosis**: If actual velocities are all 0 but commands are non-zero, the robot is not responding to actions. This could mean:
- Propeller aerodynamics not working
- Actions not being applied correctly
- Physics simulation issue
- Robot stuck/constrained

### Scenario 2: Large Mismatch (Learning Issue)

**Expected metrics:**
```
Episode_Reward/vel_xy_error_l2: -5.0 to -10.0  ← HUGE error!
Episode_Reward/actual_vel_x: -8.0  ← Moving fast in wrong direction
Episode_Reward/actual_vel_y: 3.0   ← Uncontrolled motion
Episode_Reward/commanded_vel_x: 0.5  ← Small command
Episode_Reward/commanded_vel_y: 0.2  ← Small command
Episode_Reward/track_lin_vel_xy: 0.0000  ← Reward is zero due to exp(-25+)
```

**Diagnosis**: Robot is moving but with very large errors. This could mean:
- Policy is poorly initialized (random actions causing chaos)
- Propeller forces too strong (robot flying out of control)
- Need more training time for policy to learn
- Action scaling issue

### Scenario 3: Commands Are Zero (Command Issue)

**Expected metrics:**
```
Episode_Reward/vel_xy_error_l2: -0.01  (small error)
Episode_Reward/actual_vel_x: 0.01  (small drift)
Episode_Reward/actual_vel_y: -0.02  (small drift)
Episode_Reward/commanded_vel_x: 0.0000  ← No commands!
Episode_Reward/commanded_vel_y: 0.0000  ← No commands!
Episode_Reward/track_lin_vel_xy: 0.9999  (near-perfect)
```

**Diagnosis**: If both actual and commanded velocities are near zero, the command generator might not be working. Check:
- `rel_standing_envs` setting (should be < 1.0 for moving commands)
- Command resampling (is it working?)
- Command ranges (are they too small?)

### Scenario 4: Normal Early Training

**Expected metrics:**
```
Episode_Reward/vel_xy_error_l2: -1.5 to -2.5  (learning)
Episode_Reward/actual_vel_x: -0.8  ← Some motion
Episode_Reward/actual_vel_y: 1.2   ← Some motion
Episode_Reward/commanded_vel_x: 0.3
Episode_Reward/commanded_vel_y: 0.2
Episode_Reward/track_lin_vel_xy: 0.05 to 0.10  (low but improving)
```

**Diagnosis**: This is normal for early training. The policy is exploring and hasn't learned good control yet.

## How to Use These Metrics

### 1. Check if Robot is Moving

Look at `actual_vel_x`, `actual_vel_y`, `actual_vel_z`:
- If all are exactly 0.0000 → Robot is NOT moving (Problem!)
- If they vary → Robot IS moving (Good!)

### 2. Check if Commands Exist

Look at `commanded_vel_x`, `commanded_vel_y`, `commanded_vel_z`:
- If all are exactly 0.0000 → Command generator might be broken
- If they vary → Commands are working (Good!)

### 3. Check Error Magnitude

Look at `vel_xy_error_l2`:
- `-0.1 to -0.5`: Small error, reward should be > 0.5 (good tracking)
- `-0.5 to -2.0`: Moderate error, reward should be 0.01 to 0.5 (learning)
- `-2.0 to -5.0`: Large error, reward should be 0.0001 to 0.01 (poor tracking)
- `< -5.0`: Huge error, reward will show as 0.0000 (needs investigation)

### 4. Calculate Expected Reward Manually

```python
error = abs(vel_xy_error_l2)  # Remove negative sign
expected_reward = exp(-error²)

Example:
vel_xy_error_l2 = -3.0
error = 3.0
expected_reward = exp(-9.0) = 0.0001  (will show as 0.0000 in logs)
```

## Example Log Analysis

**Your current output:**
```
Episode_Reward/track_lin_vel_xy: 0.0000
Episode_Reward/track_lin_vel_z: 0.0000
Episode_Reward/track_ang_vel_z: 0.0000
```

**After adding diagnostics, you should see:**
```
Episode_Reward/track_lin_vel_xy: 0.0000
Episode_Reward/vel_xy_error_l2: -???.??  ← CHECK THIS!
Episode_Reward/actual_vel_x: ???.??     ← CHECK THIS!
Episode_Reward/actual_vel_y: ???.??     ← CHECK THIS!
Episode_Reward/commanded_vel_x: ???.??  ← CHECK THIS!
Episode_Reward/commanded_vel_y: ???.??  ← CHECK THIS!
```

**Possible outcomes:**

1. **All velocities are 0.0000** → Robot not moving (Scenario 1)
2. **Commands are 0.0000, actual near 0** → Command generator issue (Scenario 3)
3. **Error is -5.0 or worse** → Huge mismatch (Scenario 2)
4. **Error is -1.5 to -2.5** → Normal early training (Scenario 4)

## Action Items Based on Diagnosis

### If Robot Not Moving:
1. Check propeller thrust coefficient (currently 1e-4)
2. Verify aerodynamics function is being called
3. Check if actions are being applied to joints
4. Verify physics simulation is running

### If Commands Not Generated:
1. Check `velocity_env_cfg.py` → `CommandsCfg`
2. Verify `rel_standing_envs=0.0` (not forcing standing)
3. Check command ranges (currently: x: ±0.5, y: ±0.3, z: ±0.5)
4. Add debug prints in command generator

### If Error Too Large:
1. Reduce propeller thrust (try 5e-5 instead of 1e-4)
2. Increase action smoothness penalty weight
3. Reduce initial noise std (currently 1.0 → try 0.5)
4. Check if robot is spawning in stable position

### If Normal Early Training:
1. Be patient, let training continue
2. Rewards should gradually improve over hundreds of iterations
3. Monitor if error decreases over time
4. Check tensorboard for trends

## Monitoring During Training

Run training and watch for these patterns:

```bash
# Look for these in the log output every 10 iterations
Episode_Reward/vel_xy_error_l2: -X.XX  # Should decrease over time
Episode_Reward/actual_vel_x: X.XX      # Should vary and approach commanded
Episode_Reward/commanded_vel_x: X.XX   # Should vary (not always 0)
```

**Healthy training progression:**
```
Iteration 10:  vel_xy_error_l2: -3.5, track_lin_vel_xy: 0.0001
Iteration 50:  vel_xy_error_l2: -2.8, track_lin_vel_xy: 0.001
Iteration 100: vel_xy_error_l2: -2.0, track_lin_vel_xy: 0.02
Iteration 500: vel_xy_error_l2: -1.0, track_lin_vel_xy: 0.37
```

## Files Modified

- `lab/doublebee/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - Added 9 diagnostic reward terms (all with `weight=0.0`)

## Usage

Simply run training as normal:

```bash
python scripts/co_rl/train.py --task Isaac-Velocity-Flat-DoubleBee-v1-ppo --num_envs 2
```

The new metrics will automatically appear in:
- Terminal output (every `log_interval` iterations)
- TensorBoard logs
- WandB logs (if configured)

No configuration changes needed!

