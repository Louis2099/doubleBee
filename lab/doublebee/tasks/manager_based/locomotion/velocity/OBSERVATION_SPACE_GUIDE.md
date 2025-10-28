# DoubleBee Observation Space Implementation Guide

## Overview

This document explains the complete observation space implementation for the DoubleBee robot, including all sensors, observation functions, and how they work together.

---

## Observation Space Summary

The DoubleBee robot observation space includes **~57+ dimensions** (exact count depends on action space size):

| **Component** | **Dimensions** | **Description** |
|---------------|----------------|-----------------|
| Wheel velocities | 2 | Left and right wheel rotation speeds |
| Servo positions | 2 | Left and right propeller tilt angles |
| Propeller velocities | 2 | Left and right propeller rotation speeds |
| Base linear velocity | 3 | Robot velocity in body frame [vx, vy, vz] |
| Base angular velocity | 3 | Robot rotation rates [wx, wy, wz] |
| Base orientation | 3 | Projected gravity (encodes roll/pitch) |
| Height scan | 36 | 6x6 elevation map (terrain heights) |
| Velocity commands | 3 | Desired velocities [vx, vy, wz] |
| Last actions | N | Previous control actions (N = action dim) |

---

## Implementation Details

### 1. Height Scanner Sensor (6x6 Elevation Map)

**Location:** `velocity_env_cfg.py` in `SceneCfg`

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Doublebee/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.07,  # 7cm spacing between rays
        size=[0.35, 0.35]  # 35cm x 35cm square
    ),
    debug_vis=True,
    mesh_prim_paths=["/World/ground"],
)
```

**How it works:**
- Casts 36 rays downward in a 6x6 grid pattern
- Rays start 20m above the robot and hit the terrain
- Only follows yaw rotation (not affected by robot tilt)
- Returns Z-coordinates of terrain hit points
- Grid covers 35cm x 35cm area centered on robot

**Grid Calculation:**
- Number of rays = `(size / resolution) + 1` per dimension
- For 6x6: `(0.35 / 0.07) + 1 = 5 + 1 = 6` rays per side
- Total rays: 6 × 6 = 36

---

### 2. Custom Observation Functions

**Location:** `mdp/observations.py`

#### a) Base Linear Velocity Functions

```python
def base_lin_vel(env, asset_cfg) -> torch.Tensor:
    """Returns 3D linear velocity [vx, vy, vz] in body frame.
    Shape: (num_envs, 3)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b
```

Variants for individual components:
- `base_lin_vel_x()` - Forward/backward velocity (shape: N×1)
- `base_lin_vel_y()` - Lateral velocity (shape: N×1)
- `base_lin_vel_z()` - Vertical velocity (shape: N×1)

#### b) Height Scan Function

```python
def height_scan(env, sensor_cfg, offset=0.5) -> torch.Tensor:
    """Returns height differences for 6x6 grid.
    Shape: (num_envs, 36)
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2] - offset
```

**Calculation:** `height = sensor_z_position - terrain_z_position - offset`

#### c) DoubleBee-Specific Joint Functions

```python
def wheel_velocities(env, asset_cfg) -> torch.Tensor:
    """Returns [leftWheel_vel, rightWheel_vel].
    Shape: (num_envs, 2)
    """

def servo_positions(env, asset_cfg) -> torch.Tensor:
    """Returns [leftServo_pos, rightServo_pos].
    Shape: (num_envs, 2)
    Range: [-1.57, 1.57] radians (±90°)
    """

def propeller_velocities(env, asset_cfg) -> torch.Tensor:
    """Returns [leftPropeller_vel, rightPropeller_vel].
    Shape: (num_envs, 2)
    """
```

---

### 3. Observation Configuration

**Location:** `mdp/observations.py` in `ObservationsCfg`

```python
@configclass
class PolicyCfg(ObsGroup):
    # Joint states
    wheel_vel = ObsTerm(func=wheel_velocities, scale=0.05)
    servo_pos = ObsTerm(func=servo_positions)
    propeller_vel = ObsTerm(func=propeller_velocities, scale=0.01)
    
    # Base state
    base_lin_vel = ObsTerm(func=base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    base_projected_gravity = ObsTerm(func=mdp.projected_gravity)
    
    # Terrain perception
    height_scan = ObsTerm(
        func=height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
        clip=(-1.0, 1.0),
    )
    
    # Command
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    
    # Action history
    actions = ObsTerm(func=mdp.last_action)
```

---

## Observation Scaling

Proper scaling is crucial for RL training. Here's why each observation is scaled:

| **Observation** | **Scale** | **Reason** |
|-----------------|-----------|------------|
| `wheel_vel` | 0.05 | Wheel speeds can be 0-200 rad/s → scaled to 0-10 |
| `servo_pos` | 1.0 | Already in good range [-1.57, 1.57] |
| `propeller_vel` | 0.01 | Propeller speeds can be 0-600 rad/s → scaled to 0-6 |
| `base_lin_vel` | 2.0 | Emphasize velocity for tracking (typ. -1 to 1 m/s → -2 to 2) |
| `base_ang_vel` | 0.25 | Reduce magnitude (typ. -4 to 4 rad/s → -1 to 1) |
| `height_scan` | 1.0 | Clipped to [-1, 1] range |

---

## Comparison with Flamingo

| **Feature** | **Flamingo** | **DoubleBee** | **Status** |
|-------------|--------------|---------------|------------|
| Joint positions | ✅ Hip/shoulder/leg | ✅ Servos | Adapted |
| Joint velocities | ✅ All legs | ✅ Wheels/propellers | Adapted |
| Base linear velocity | ✅ Separate x/y/z | ✅ Combined 3D | Implemented |
| Base angular velocity | ✅ Yes | ✅ Yes | Reused |
| Base orientation | ✅ Projected gravity | ✅ Projected gravity | Reused |
| Elevation map | ✅ GridPattern | ✅ 6x6 grid | Adapted |
| Lift mask | ✅ For legs | ❌ Not needed | N/A |
| Contact sensor | ✅ For wheels | ❌ Not added yet | Optional |

---

## DoubleBee Joint Structure

The DoubleBee robot has 6 actuated joints:

```
DoubleBee
├── Wheels (2 joints)
│   ├── leftWheel    → Ground locomotion (Z-axis rotation)
│   └── rightWheel   → Ground locomotion (Z-axis rotation)
├── Servos (2 joints)
│   ├── leftPropellerServo  → Propeller tilt (Z-axis, ±90°)
│   └── rightPropellerServo → Propeller tilt (Z-axis, ±90°)
└── Propellers (2 joints)
    ├── leftPropeller  → Thrust generation (Y-axis rotation)
    └── rightPropeller → Thrust generation (Y-axis rotation)
```

**Joint Ranges:**
- Wheels: Continuous rotation (no limits)
- Servos: [-1.57, 1.57] radians (±90°)
- Propellers: Continuous rotation (no limits)

---

## Data Flow

```
Simulation Step
    ↓
Sensor Updates (height_scanner)
    ↓
Observation Functions Called
    ├── wheel_velocities()
    ├── servo_positions()
    ├── propeller_velocities()
    ├── base_lin_vel()
    ├── base_ang_vel()
    ├── projected_gravity()
    ├── height_scan()
    ├── generated_commands()
    └── last_action()
    ↓
Observations Scaled & Concatenated
    ↓
Flattened Vector → Policy Network
```

---

## Observation Shapes

For a single environment (batch size = 1):

```python
wheel_vel:            torch.Size([1, 2])    # 2 wheels
servo_pos:            torch.Size([1, 2])    # 2 servos
propeller_vel:        torch.Size([1, 2])    # 2 propellers
base_lin_vel:         torch.Size([1, 3])    # vx, vy, vz
base_ang_vel:         torch.Size([1, 3])    # wx, wy, wz
base_projected_gravity: torch.Size([1, 3])  # gx, gy, gz
height_scan:          torch.Size([1, 36])   # 6x6 grid
velocity_commands:    torch.Size([1, 3])    # cmd_vx, cmd_vy, cmd_wz
actions:              torch.Size([1, N])    # N = action space size

# Total concatenated: [1, 54 + N]
```

For 4096 parallel environments:
```python
observation_vector: torch.Size([4096, 54 + N])
```

---

## Testing Observations

To verify observations are working correctly:

```python
# In your training script or debugging session
env = gym.make("Isaac-Velocity-Flat-DoubleBee-v1-ppo")
obs, _ = env.reset()

print("Observation keys:", obs.keys())
print("Policy observation shape:", obs["policy"].shape)

# Check individual observation shapes (before concatenation)
# Add this to PolicyCfg.__post_init__ for debugging:
# self.concatenate_terms = False

# Then you can inspect:
# obs["wheel_vel"], obs["servo_pos"], etc.
```

---

## Key Differences from Standard IsaacLab Observations

1. **Custom joint observations**: DoubleBee uses specific joint names (wheels, servos, propellers) instead of generic joint queries

2. **Height scan configuration**: Adapted grid pattern to 6x6 (35cm square) suitable for DoubleBee's size

3. **Velocity scaling**: Tuned for DoubleBee's expected movement ranges (different from legged robots)

4. **No leg-specific observations**: DoubleBee is wheeled+aerial, so no foot contact or leg phase observations

---

## Future Enhancements

Potential observations to add:

1. **Contact sensors**: Detect wheel-ground contact
   ```python
   from isaaclab.sensors import ContactSensorCfg
   
   contact_forces = ContactSensorCfg(
       prim_path="{ENV_REGEX_NS}/Doublebee/.*",
       history_length=3,
   )
   ```

2. **Propeller thrust estimation**: Derive from propeller velocity and servo angle

3. **Base height**: Explicit Z-position observation
   ```python
   base_height = ObsTerm(
       func=mdp.root_pos_w,
       params={"asset_cfg": SceneEntityCfg("robot")},
   )
   ```

4. **IMU-like observations**: Linear acceleration from velocity derivatives

---

## Troubleshooting

### Issue: Height scan returns all zeros
**Solution:** Check that `height_scanner` sensor is added to scene and prim_path matches robot structure

### Issue: Joint velocities are wrong shape
**Solution:** Verify joint names match USD file exactly (case-sensitive)

### Issue: Observations are too large/small
**Solution:** Adjust scale factors in ObsTerm definitions

### Issue: Sensor not found error
**Solution:** Ensure sensor name in ObsTerm matches SceneCfg sensor name exactly

---

## References

- **Flamingo observations**: `/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/observations.py`
- **IsaacLab MDP functions**: `isaaclab.envs.mdp`
- **RayCaster documentation**: IsaacLab sensors module
- **DoubleBee config**: `/lab/doublebee/assets/doublebee/doublebee_v1.py`

---

## Summary

✅ **All required observations are now implemented:**
- ✅ Joint speeds (wheels, propellers)
- ✅ Joint positions (servos)
- ✅ Robot base speed (3D velocity)
- ✅ Robot base orientation (projected gravity)
- ✅ 6x6 elevation map (height scan)

The implementation follows IsaacLab best practices and is adapted from the proven Flamingo observation system.

