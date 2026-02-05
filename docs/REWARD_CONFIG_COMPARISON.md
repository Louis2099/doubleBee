# Reward Configuration Comparison

## Overview

There are **two different reward configurations** in the codebase:

1. **`RewardsCfg`** in `mdp/rewards.py` - **NOT USED** (base/template)
2. **`DoubleBeeRewardsCfg`** in `flat_env_stand_drive_cfg.py` - **ACTUALLY USED** (task-specific)

## Side-by-Side Comparison

| Feature | `RewardsCfg` (mdp/rewards.py) | `DoubleBeeRewardsCfg` (flat_env_stand_drive_cfg.py) |
|---------|-------------------------------|-----------------------------------------------------|
| **Status** | ❌ Not used (base class) | ✅ **Used in training** |
| **Reward Shape** | Exponential (exp(-error²)) | Quadratic (-error²) |
| **Reward Range** | (0, 1] (always positive) | (-∞, 0] (always negative) |
| **Vertical Velocity** | ✅ Separate `track_lin_vel_z` | ❌ Not tracked separately |
| **Upright Reward** | ❌ Missing | ✅ **Included** |
| **Energy Metric** | Propeller speed only | All applied torques |
| **Action Penalty** | Action magnitude | Action rate (change) |

## Detailed Differences

### 1. **Velocity Tracking Rewards**

#### `RewardsCfg` (mdp/rewards.py):
```python
# Separate XY and Z tracking
track_lin_vel_xy = RewTerm(
    func=lambda env: torch.exp(-torch.sum(torch.square(...), dim=1)),
    weight=1.0,
)
track_lin_vel_z = RewTerm(
    func=lambda env: torch.exp(-torch.square(...)),
    weight=1.0,
)
track_ang_vel_z = RewTerm(
    func=lambda env: torch.exp(-torch.square(...)),
    weight=0.5,
)
```

**Formula**: `exp(-error²)` → Range: `(0, 1]`
- **Best case** (perfect tracking): `1.0`
- **Worst case** (large error): `~0.0` (but always positive)

#### `DoubleBeeRewardsCfg` (flat_env_stand_drive_cfg.py):
```python
# Combined XY tracking (no Z)
tracking_lin_vel = RewTerm(
    func=lambda env: -torch.sum(torch.square(...), dim=1),
    weight=1.0,
)
tracking_ang_vel = RewTerm(
    func=lambda env: -torch.sum(torch.square(...), dim=1),
    weight=0.5,
)
```

**Formula**: `-error²` → Range: `(-∞, 0]`
- **Best case** (perfect tracking): `0.0`
- **Worst case** (large error): Large negative value (e.g., `-10.0`)

**Key Differences**:
- ✅ `RewardsCfg`: **Exponential** (bounded, always positive)
- ✅ `DoubleBeeRewardsCfg`: **Quadratic** (unbounded, always negative)
- ✅ `RewardsCfg`: **Separate Z tracking** (`track_lin_vel_z`)
- ❌ `DoubleBeeRewardsCfg`: **No Z tracking** (only XY)

### 2. **Upright/Stability Reward**

#### `RewardsCfg` (mdp/rewards.py):
```python
# ❌ MISSING - No upright reward!
```

#### `DoubleBeeRewardsCfg` (flat_env_stand_drive_cfg.py):
```python
upright = RewTerm(
    func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]), dim=1),
    weight=0.5,
)
```

**Formula**: `-||gravity_xy||²`
- **Best case** (perfectly upright): `0.0`
- **Worst case** (tipped over): `-1.0`

**Purpose**: Encourages robot to stay upright (penalizes tilting)

### 3. **Energy Efficiency Rewards**

#### `RewardsCfg` (mdp/rewards.py):
```python
propeller_efficiency = RewTerm(
    func=lambda env: -torch.sum(torch.square(
        env.scene["robot"].data.joint_vel[:, 
            [env.scene["robot"].joint_names.index("leftPropeller"),
             env.scene["robot"].joint_names.index("rightPropeller")]
        ]
    ), dim=1),
    weight=0.0001,
)
```

**What it penalizes**: Only propeller joint velocities
- Focuses on propeller efficiency
- Requires joint name lookup (dynamic)

#### `DoubleBeeRewardsCfg` (flat_env_stand_drive_cfg.py):
```python
energy = RewTerm(
    func=lambda env: -torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1),
    weight=0.0001,
)
```

**What it penalizes**: All applied torques (all joints)
- More general energy consumption metric
- Simpler (no joint name lookup)

**Key Differences**:
- `RewardsCfg`: **Propeller-specific** efficiency
- `DoubleBeeRewardsCfg`: **Total energy** (all joints)

### 4. **Action Smoothness Rewards**

#### `RewardsCfg` (mdp/rewards.py):
```python
action_smoothness = RewTerm(
    func=lambda env: -torch.sum(torch.square(env.action_manager.action), dim=1),
    weight=0.001,
)
```

**Formula**: `-||action||²`
- Penalizes **large action magnitudes**
- Encourages small actions

#### `DoubleBeeRewardsCfg` (flat_env_stand_drive_cfg.py):
```python
action_rate = RewTerm(
    func=lambda env: -torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1),
    weight=0.01,
)
```

**Formula**: `-||action - prev_action||²`
- Penalizes **action changes** (jerky control)
- Encourages smooth control (small changes)

**Key Differences**:
- `RewardsCfg`: Penalizes **action magnitude** (absolute)
- `DoubleBeeRewardsCfg`: Penalizes **action rate** (relative change)

## Summary Table

| Reward Term | `RewardsCfg` | `DoubleBeeRewardsCfg` | Difference |
|-------------|--------------|----------------------|------------|
| **XY Velocity Tracking** | `exp(-error²)` | `-error²` | Shape: exponential vs quadratic |
| **Z Velocity Tracking** | ✅ `exp(-error²)` | ❌ Missing | Separate Z tracking |
| **Angular Velocity** | `exp(-error²)` | `-error²` | Shape: exponential vs quadratic |
| **Upright/Stability** | ❌ Missing | ✅ `-||gravity_xy||²` | **Only in DoubleBeeRewardsCfg** |
| **Energy Efficiency** | Propeller speeds | All torques | Scope: specific vs general |
| **Action Penalty** | Action magnitude | Action rate | Type: absolute vs relative |

## Mathematical Comparison

### Reward Shape Functions

**Exponential (RewardsCfg)**:
```
R = exp(-error²)
Range: (0, 1]
- error = 0 → R = 1.0 (best)
- error = 1 → R = 0.37
- error = 2 → R = 0.018
- error → ∞ → R → 0
```

**Quadratic (DoubleBeeRewardsCfg)**:
```
R = -error²
Range: (-∞, 0]
- error = 0 → R = 0.0 (best)
- error = 1 → R = -1.0
- error = 2 → R = -4.0
- error → ∞ → R → -∞
```

### Implications

**Exponential Rewards** (`RewardsCfg`):
- ✅ **Bounded**: Always in (0, 1]
- ✅ **Saturates**: Large errors don't dominate
- ✅ **Smooth gradient**: Good for learning
- ❌ **Less sensitive**: Small differences in large errors

**Quadratic Rewards** (`DoubleBeeRewardsCfg`):
- ✅ **Unbounded**: Can scale with error
- ✅ **More sensitive**: Large errors strongly penalized
- ✅ **Linear gradient**: Easier to optimize
- ❌ **Can dominate**: Large errors can overwhelm other terms

## Which One Is Used?

**Answer**: `DoubleBeeRewardsCfg` from `flat_env_stand_drive_cfg.py`

**Evidence**:
1. Task registration uses `DoubleBeeFlatStandDriveCfg`
2. `DoubleBeeFlatStandDriveCfg` overrides rewards with local `DoubleBeeRewardsCfg`
3. Metrics show `upright` reward (only in `DoubleBeeRewardsCfg`)

## Why Two Configurations?

**`RewardsCfg` (mdp/rewards.py)**:
- Base/template configuration
- More sophisticated (exponential rewards, separate Z tracking)
- **Not currently used** - may be for future use or alternative tasks

**`DoubleBeeRewardsCfg` (flat_env_stand_drive_cfg.py)**:
- Task-specific configuration
- Simpler (quadratic rewards, combined tracking)
- **Currently used** - optimized for the stand-and-drive task
- Includes stability reward (`upright`) for better control

## Recommendation

If you want to use the exponential rewards from `RewardsCfg`, you would need to:

1. **Modify** `flat_env_stand_drive_cfg.py`:
```python
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.rewards import RewardsCfg

class DoubleBeeFlatStandDriveCfg(DoubleBeeVelocityEnvCfg):
    rewards: RewardsCfg = RewardsCfg()  # Use base config instead
```

2. **Or** add the exponential shape to `DoubleBeeRewardsCfg`:
```python
tracking_lin_vel = RewTerm(
    func=lambda env: torch.exp(-torch.sum(torch.square(...), dim=1)),
    weight=1.0,
)
```

**Note**: The current quadratic rewards are working fine - exponential rewards are just an alternative design choice.

