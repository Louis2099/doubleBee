# Reward System Explanation - Why Rewards Are Zero

## Overview

The DoubleBee environment uses a **penalty-based reward system** where all rewards are **negative** (or zero). In early episodes, rewards appear as **0.0** because:

1. **Episodes haven't completed yet** - Rewards are only logged when episodes finish
2. **Commands are zero at episode start** - Velocity commands reset to `[0, 0, 0, 0]`
3. **All reward terms are penalties** - They're all negative values
4. **Robot is falling/not tracking** - Large tracking errors → large negative rewards

## 📍 Where Rewards Are Defined

### Primary Reward Configuration

**File**: `lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/flat_env/hybrid_stair/hybrid_stair_cfg.py`

**Lines 22-57**: `DoubleBeeRewardsCfg` class

```python
@configclass
class DoubleBeeRewardsCfg:
    """Reward configuration for DoubleBee stand and drive task."""

    # Tracking rewards
    tracking_lin_vel = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_b[:, :2] - env.command_manager.get_command("base_velocity")[:, :2]), dim=1),
        weight=1.0,
    )
    
    tracking_ang_vel = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.root_ang_vel_b[:, 2:3] - env.command_manager.get_command("base_velocity")[:, 2:3]), dim=1),
        weight=0.5,
    )
    
    # Stability rewards
    upright = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]), dim=1),
        weight=0.5,
    )
    
    # Energy efficiency
    energy = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1),
        weight=0.0001,
    )
    
    # Action smoothness
    action_rate = RewTerm(
        func=lambda env: -torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1),
        weight=0.01,
    )
```

## 🔍 Step-by-Step: How Rewards Are Computed

### Step 1: Reward Computation (Every Step)

**File**: `lab/doublebee/isaaclab/isaaclab/envs/manager_based_constraint_rl_env.py`

**Line 211**: Rewards computed after each environment step:
```python
self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
```

**What happens**:
1. Reward manager calls each reward term's `func(env)`
2. Each term returns a tensor of shape `[num_envs]` (one value per environment)
3. Terms are weighted and summed: `total_reward = Σ(term_value * weight)`

### Step 2: Individual Reward Terms

#### **tracking_lin_vel** (Weight: 1.0)
```python
func=lambda env: -torch.sum(torch.square(
    env.scene["robot"].data.root_lin_vel_b[:, :2]  # Actual XY velocity
    - env.command_manager.get_command("base_velocity")[:, :2]  # Commanded XY velocity
), dim=1)
```

**Formula**: `-||v_actual_xy - v_cmd_xy||²`

**Range**: `(-∞, 0]`
- **Best case** (perfect tracking): `0.0`
- **Worst case** (large error): Large negative value (e.g., `-10.0`)

**Example**:
- Command: `[0.5, 0.0]` m/s
- Actual: `[0.0, 0.0]` m/s (robot stationary)
- Error: `[0.5, 0.0]`
- Reward: `-0.5² = -0.25`

#### **tracking_ang_vel** (Weight: 0.5)
```python
func=lambda env: -torch.sum(torch.square(
    env.scene["robot"].data.root_ang_vel_b[:, 2:3]  # Actual yaw angular velocity
    - env.command_manager.get_command("base_velocity")[:, 2:3]  # Commanded yaw velocity
), dim=1)
```

**Formula**: `-||ω_actual_z - ω_cmd_z||²`

**Range**: `(-∞, 0]`
- **Best case**: `0.0`
- **Worst case**: Large negative value

#### **upright** (Weight: 0.5)
```python
func=lambda env: -torch.sum(torch.square(
    env.scene["robot"].data.projected_gravity_b[:, :2]  # Gravity projection in XY
), dim=1)
```

**Formula**: `-||gravity_xy||²`

**Range**: `(-∞, 0]`
- **Best case** (perfectly upright): `0.0` (gravity only in Z)
- **Worst case** (tipped over): `-1.0` (gravity fully in XY plane)

**Note**: `projected_gravity_b` is a unit vector, so max magnitude is 1.0

#### **energy** (Weight: 0.0001)
```python
func=lambda env: -torch.sum(torch.square(
    env.scene["robot"].data.applied_torque  # All joint torques
), dim=1)
```

**Formula**: `-||torque||²`

**Range**: `(-∞, 0]`
- **Best case** (no torque): `0.0`
- **Worst case** (high torque): Large negative value

**Note**: Very small weight (0.0001) means this has minimal impact

#### **action_rate** (Weight: 0.01)
```python
func=lambda env: -torch.sum(torch.square(
    env.action_manager.action - env.action_manager.prev_action  # Action change
), dim=1)
```

**Formula**: `-||action - prev_action||²`

**Range**: `(-∞, 0]`
- **Best case** (smooth actions): `0.0`
- **Worst case** (jerky actions): Large negative value

### Step 3: Total Reward Calculation

**Formula**:
```python
total_reward = (
    1.0 * tracking_lin_vel +
    0.5 * tracking_ang_vel +
    0.5 * upright +
    0.0001 * energy +
    0.01 * action_rate
)
```

**Example Calculation** (Early Episode):
- `tracking_lin_vel = -2.5` (large tracking error)
- `tracking_ang_vel = -1.0` (angular error)
- `upright = -0.3` (slightly tilted)
- `energy = -100.0` (high torque usage)
- `action_rate = -0.5` (jerky actions)

**Total**:
```
total = 1.0*(-2.5) + 0.5*(-1.0) + 0.5*(-0.3) + 0.0001*(-100.0) + 0.01*(-0.5)
     = -2.5 - 0.5 - 0.15 - 0.01 - 0.005
     = -3.165
```

## 🚨 Why Rewards Are Zero in Early Episodes

### Reason 1: Episodes Haven't Completed Yet

**File**: `scripts/co_rl/core/runners/on_policy_runner.py`

**Lines 138-155**: Episode rewards are only logged when episodes complete:

```python
if self.log_dir is not None:
    # ...
    if not self.cfg["use_constraint_rl"]:
        new_ids = (dones > 0).nonzero(as_tuple=False)  # Episodes that just finished
    else:
        new_ids = (dones == 1.0).nonzero(as_tuple=False)
    
    # Only log rewards for completed episodes
    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
```

**Lines 219-221**: Rewards only displayed if `rewbuffer` has entries:
```python
if len(locs["rewbuffer"]) > 0:
    self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
```

**What this means**:
- If no episodes have completed yet → `rewbuffer` is empty → **no reward logged** → displays as `0.0`
- Once episodes start completing → rewards are logged

### Reason 2: Commands Are Zero at Episode Start

**File**: `lab/doublebee/tasks/manager_based/locomotion/velocity/mdp/velocity_command.py`

**Lines 36-39**: Commands reset to zero when environments reset:
```python
def _update_command(self):
    reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        self.vel_command_b[reset_env_ids] = 0.0  # ← Commands set to zero!
```

**What this means**:
- At episode start: Command = `[0, 0, 0, 0]`
- If robot is stationary: `tracking_lin_vel = -0² = 0.0` ✅
- If robot is falling: `tracking_lin_vel = -large_value² = large_negative` ❌

**Command Resampling**:
- Commands are resampled every 6-8 seconds (see `resampling_time_range=(6.0, 8.0)`)
- Until first resample, command stays at `[0, 0, 0, 0]`

### Reason 3: Robot Is Falling/Not Tracking

In early episodes with a random policy:
- Robot spawns in mid-air (height ~0.5m)
- Random actions → robot falls
- Robot velocity ≠ command velocity → **large tracking errors**
- Robot tilts → **upright penalty**
- High torque usage → **energy penalty**

**Result**: All reward terms are large negative values → total reward is very negative

### Reason 4: Episode Length

**File**: `lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/flat_env/hybrid_stair/hybrid_stair_cfg.py`

**Line 123**: Episode length is 20 seconds:
```python
self.episode_length_s = 20.0
```

**Calculation**:
- Physics dt: `0.005s`
- Decimation: `4`
- Step dt: `0.005 * 4 = 0.02s`
- Episode steps: `20.0 / 0.02 = 1000 steps`

**What this means**:
- Episodes are long (1000 steps)
- With only 2 environments, episodes take time to complete
- Early iterations may have **zero completed episodes** → no rewards logged

## 📊 Reward Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Environment Step                                │
│   - Robot executes action                                │
│   - Physics simulation runs                              │
│   - Robot state updated                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Reward Manager Compute                         │
│   reward_buf = reward_manager.compute(dt)                │
│                                                          │
│   For each reward term:                                 │
│     term_value = term.func(env)  # [num_envs]          │
│     weighted = term_value * term.weight                 │
│                                                          │
│   total_reward = sum(all weighted terms)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Accumulate Episode Reward                       │
│   cur_reward_sum += rewards  # Per environment          │
│                                                          │
│   Check if episode done:                                │
│     if dones > 0:                                       │
│       rewbuffer.append(cur_reward_sum)  # Log episode  │
│       cur_reward_sum = 0  # Reset                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Logging (Only if episodes completed)            │
│   if len(rewbuffer) > 0:                                │
│     mean_reward = mean(rewbuffer)                       │
│     writer.add_scalar("Train/mean_reward", ...)         │
│   else:                                                 │
│     # No episodes completed → no reward logged          │
│     # Terminal shows: "Mean reward: 0.0000"            │
└─────────────────────────────────────────────────────────┘
```

## 🔧 How to Debug Reward Issues

### Check 1: Are Episodes Completing?

Look at terminal output:
```
Mean episode length: 0.0000  ← Episodes not completing!
```

If episode length is 0, no episodes have finished → no rewards logged.

### Check 2: Check Individual Reward Terms

**File**: `scripts/co_rl/core/runners/on_policy_runner.py`

**Lines 189-208**: Environment-specific metrics are logged:
```python
if locs["ep_infos"]:
    for key in locs["ep_infos"][0]:
        # Logs: Episode_Reward/tracking_lin_vel, etc.
        self.writer.add_scalar(key, value, locs["it"])
```

**Check TensorBoard**:
- `Episode_Reward/tracking_lin_vel`
- `Episode_Reward/tracking_ang_vel`
- `Episode_Reward/upright`
- `Episode_Reward/energy`
- `Episode_Reward/action_rate`

If these are all `0.0`, episodes haven't completed yet.

### Check 3: Verify Commands Are Non-Zero

Add debug print in reward function:
```python
tracking_lin_vel = RewTerm(
    func=lambda env: (
        cmd = env.command_manager.get_command("base_velocity")[:, :2]
        print(f"Command: {cmd[0]}")  # Debug
        -torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_b[:, :2] - cmd), dim=1)
    ),
    weight=1.0,
)
```

### Check 4: Check Robot State

Verify robot is actually moving:
- Check `root_lin_vel_b` - should be non-zero if robot is moving
- Check `projected_gravity_b` - should be small if robot is upright

## 💡 Solutions

### Solution 1: Wait for Episodes to Complete

**Normal behavior**: Rewards will appear once episodes start completing (after ~1000 steps per episode).

**Check**: Look for `Mean episode length` to become non-zero.

### Solution 2: Reduce Episode Length (For Testing)

**File**: `hybrid_stair_cfg.py`, line 123:
```python
self.episode_length_s = 5.0  # Reduced from 20.0
```

This makes episodes complete faster → rewards appear sooner.

### Solution 3: Increase Number of Environments

More environments → more episodes completing per iteration → rewards appear faster.

```bash
python scripts/co_rl/train.py --task ... --num_envs 1024
```

### Solution 4: Check Command Resampling

**File**: `velocity_command.py`, line 61:
```python
resampling_time_range=(6.0, 8.0)  # Commands resample every 6-8 seconds
```

Commands start at zero, then resample after 6-8 seconds. This is **normal behavior**.

## 📝 Summary

| Issue | Cause | Solution |
|-------|-------|----------|
| **Rewards show 0.0** | Episodes haven't completed yet | Wait for episodes to finish |
| **Rewards are negative** | All terms are penalties (normal) | This is expected behavior |
| **Rewards stay 0.0** | Episodes not completing (too long) | Reduce episode length or increase num_envs |
| **Commands are zero** | Reset behavior (normal) | Commands resample after 6-8 seconds |

## 🎯 Expected Reward Values

### Early Training (Random Policy)
- **Total reward**: `-5.0` to `-20.0` (very negative)
- **tracking_lin_vel**: `-2.0` to `-10.0`
- **tracking_ang_vel**: `-1.0` to `-5.0`
- **upright**: `-0.1` to `-1.0`
- **energy**: `-0.01` to `-0.1` (small due to weight)
- **action_rate**: `-0.1` to `-1.0`

### After Training (Good Policy)
- **Total reward**: `-0.5` to `-2.0` (less negative)
- **tracking_lin_vel**: `-0.1` to `-0.5` (better tracking)
- **tracking_ang_vel**: `-0.05` to `-0.2`
- **upright**: `-0.01` to `-0.1` (more stable)
- **energy**: `-0.001` to `-0.01`
- **action_rate**: `-0.01` to `-0.1` (smoother)

**Note**: Rewards are **always negative or zero** - this is by design! The policy learns to **maximize** (make less negative) the total reward.

