# DoubleBee Propeller Aerodynamics Implementation

## Overview

This implementation adds realistic aerodynamic forces to DoubleBee's propellers, inspired by IsaacLab's quadcopter example. The thrust and drag forces are calculated based on propeller angular velocity and applied during each physics step.

## Physics Model

### Thrust Force
```
F_thrust = k_t * ω²
```
- `k_t`: Thrust coefficient (tunable parameter)
- `ω`: Propeller angular velocity (rad/s)
- Thrust direction: Along propeller's Y-axis (local frame)

### Drag Torque
```
τ_drag = -k_d * sign(ω) * ω²
```
- `k_d`: Drag coefficient (tunable parameter)
- Opposes propeller rotation

## Implementation Structure

### 1. Aerodynamics Module
**File**: `mdp/aerodynamics.py`

Contains three functions:
- **`apply_propeller_aerodynamics()`**: Full implementation with per-propeller forces
- **`apply_simple_propeller_thrust()`**: Simplified version applying to base link
- **`apply_thrust_with_tilt_control()`**: Advanced version accounting for servo tilt

### 2. Event Configuration
**File**: `doublebee_env/flat_env/hybrid_stair/hybrid_stair_cfg.py`

Aerodynamics is added as an event term:
```python
propeller_aerodynamics = EventTerm(
    func=aerodynamics.apply_propeller_aerodynamics,
    mode="interval",
    interval_range_s=(0.0, 0.0),  # Every step
    params={
        "thrust_coefficient": 0.1,  # TUNE THIS!
        "drag_coefficient": 0.01,
        "max_thrust_per_propeller": 50.0,
    },
)
```

## How Aerodynamics Works

### Execution Flow
1. **Action Phase**: RL policy outputs propeller velocities
2. **Action Manager**: Applies actions to propeller joints
3. **Event Manager**: Calls `apply_propeller_aerodynamics()` every step
4. **Aerodynamics Function**:
   - Reads propeller angular velocities from robot state
   - Calculates thrust: `F = k_t * ω²`
   - Gets propeller body orientations (quaternions)
   - Rotates thrust vector from local to world frame
   - Applies forces using `robot.set_external_force_and_torque()`
5. **Physics Step**: PhysX simulates with external forces
6. **State Update**: Robot state updated with aerodynamic effects

### Key Concepts

#### 1. Propeller Body Orientation
The propellers rotate around the Y-axis. Their orientation in world frame determines thrust direction:
```python
propeller_quat_w = robot.data.body_quat_w[:, propeller_body_ids, :]
```

#### 2. Force Application
Forces are applied to propeller bodies (not joints):
```python
robot.set_external_force_and_torque(external_forces, external_torques, body_ids=None)
```

#### 3. World Frame Transformation
Thrust is defined in propeller local frame, then rotated to world frame:
```python
thrust_world = quat_rotate(propeller_quat_w, thrust_local)
```

## Tuning Parameters

### Thrust Coefficient (`thrust_coefficient`)

**Determines how much thrust is generated per unit of propeller speed squared.**

#### Guidelines:
- **Start with**: `0.1`
- **Too low**: Robot doesn't lift, stays on ground
- **Too high**: Robot flies uncontrollably, hard to stabilize
- **Good value**: Robot can hover when propellers at 50-70% max speed

#### How to tune:
1. Set propellers to constant velocity (e.g., 10 rad/s)
2. Check if robot lifts:
   - No lift → Increase by 2-5x
   - Lifts too fast → Decrease by 0.5x
3. Calculate hover thrust needed:
   ```
   F_hover = robot_mass * gravity = 5.0 kg * 9.81 m/s² ≈ 49 N
   F_per_propeller = F_hover / 2 ≈ 25 N
   
   If ω_hover = 10 rad/s:
   k_t = F / ω² = 25 / 100 = 0.25
   ```

**Typical ranges**:
- Light robot (< 3 kg): `0.01 - 0.1`
- Medium robot (3-10 kg): `0.1 - 1.0`
- Heavy robot (> 10 kg): `1.0 - 10.0`

### Drag Coefficient (`drag_coefficient`)

**Determines torque opposing rotation.**

- Usually 10-100x smaller than thrust coefficient
- Default: `0.01` (if `k_t = 0.1`)
- Affects: Energy efficiency, propeller spin-up time

### Max Thrust (`max_thrust_per_propeller`)

**Maximum thrust force per propeller (safety clamp).**

- Set to 2-3x hover thrust per propeller
- Example: For 5 kg robot, `25 N * 3 = 75 N`
- Prevents unrealistic forces at very high speeds

## Testing and Debugging

### 1. Check Propeller Names
```python
robot = env.scene["robot"]
print(f"Joints: {robot.joint_names}")
print(f"Bodies: {robot.body_names}")
```
Verify `leftPropeller` and `rightPropeller` exist in both lists.

### 2. Monitor Propeller Velocities
```python
propeller_ids = [robot.joint_names.index("leftPropeller"), 
                 robot.joint_names.index("rightPropeller")]
vel = robot.data.joint_vel[:, propeller_ids]
print(f"Propeller velocities (rad/s): {vel[0]}")
```

### 3. Check Forces Being Applied
```python
# In aerodynamics.py, add debug output:
print(f"Thrust magnitude: {thrust_magnitude[0]}")
print(f"Thrust world: {thrust_world[0]}")
print(f"Robot Z velocity: {robot.data.root_lin_vel_w[0, 2]}")
```

### 4. Visualize Robot Motion
- **Falling**: `k_t` too low
- **Shooting up**: `k_t` too high  
- **Spinning**: Unbalanced forces, check propeller positions
- **Tilting**: Check propeller orientations, servo angles

## Common Issues

### Issue 1: Robot Doesn't Lift

**Symptoms**: Propellers spin but robot stays on ground

**Solutions**:
1. Increase `thrust_coefficient` by 5-10x
2. Check propeller rotation direction (should be positive)
3. Verify forces are applied to correct bodies
4. Check if thrust direction is correct (should be upward)

**Debug**:
```python
# Add to aerodynamics.py
print(f"Thrust direction (local): {thrust_local[0, 0]}")
print(f"Thrust direction (world): {thrust_world[0, 0]}")
print(f"Expected: Z-component should be positive for upward thrust")
```

### Issue 2: Robot Flies Too Fast

**Symptoms**: Robot immediately shoots up and leaves screen

**Solutions**:
1. Reduce `thrust_coefficient` by 0.5x
2. Reduce `max_thrust_per_propeller`
3. Add reward penalty for high altitude
4. Increase action smoothness penalty

### Issue 3: Unstable Physics

**Symptoms**: Robot jitters, explodes, or simulation crashes

**Solutions**:
1. Reduce `sim.dt` (e.g., from 0.005 to 0.001)
2. Reduce `thrust_coefficient`
3. Increase propeller damping in `doublebee_v1.py`
4. Enable contact processing: `disable_contact_processing=False`

### Issue 4: Propeller Servo Tilt Not Working

**Symptoms**: Tilting servos but thrust direction doesn't change

**Solution**: Use `apply_thrust_with_tilt_control()` instead:
```python
# In hybrid_stair_cfg.py
propeller_aerodynamics = EventTerm(
    func=aerodynamics.apply_thrust_with_tilt_control,  # Changed
    ...
)
```

## Advanced Features

### 1. Ground Effect

Add height-dependent thrust scaling (thrust increases near ground):
```python
# In aerodynamics.py
robot_height = robot.data.root_pos_w[:, 2]
ground_effect = 1.0 + 0.2 * torch.exp(-robot_height / 0.5)
thrust_magnitude = thrust_coefficient * torch.square(propeller_vel) * ground_effect
```

### 2. Air Density Effects

Model altitude-dependent air density:
```python
def get_air_density(altitude):
    """Air density as function of altitude (simplified)."""
    rho_0 = 1.225  # kg/m³ at sea level
    return rho_0 * torch.exp(-altitude / 8500.0)  # Scale height ~8.5km

air_density = get_air_density(robot.data.root_pos_w[:, 2])
thrust_magnitude *= air_density / 1.225  # Normalize to sea level
```

### 3. Propeller Interference

Model thrust reduction when propellers are close:
```python
# Distance between propellers
prop_pos_left = robot.data.body_pos_w[:, left_body_id]
prop_pos_right = robot.data.body_pos_w[:, right_body_id]
distance = torch.norm(prop_pos_right - prop_pos_left, dim=1)

# Interference factor (reduces thrust when close)
interference = 1.0 - 0.2 * torch.exp(-distance / 0.5)
thrust_magnitude *= interference.unsqueeze(1)
```

## Switching Between Implementations

### Use Simple Thrust (Start Here)
```python
propeller_aerodynamics = EventTerm(
    func=aerodynamics.apply_simple_propeller_thrust,
    params={
        "thrust_coefficient": 0.1,
        "max_total_thrust": 100.0,
    },
)
```
**Pros**: Easier to tune, more stable  
**Cons**: Less realistic, no differential thrust control

### Use Full Aerodynamics (Advanced)
```python
propeller_aerodynamics = EventTerm(
    func=aerodynamics.apply_propeller_aerodynamics,
    params={
        "thrust_coefficient": 0.1,
        "drag_coefficient": 0.01,
        "max_thrust_per_propeller": 50.0,
    },
)
```
**Pros**: More realistic, independent propeller control  
**Cons**: Harder to tune, can be unstable

### Use With Tilt Control (Most Advanced)
```python
propeller_aerodynamics = EventTerm(
    func=aerodynamics.apply_thrust_with_tilt_control,
    params={
        "thrust_coefficient": 0.1,
        "max_thrust_per_propeller": 50.0,
    },
)
```
**Pros**: Vectored thrust, full DoubleBee capabilities  
**Cons**: Most complex, requires careful tuning

## References

1. **IsaacLab Quadcopter**: `/home/yuanliu/Louis_Project/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py`
2. **Propeller Aerodynamics**: "Momentum Theory" and "Blade Element Theory"
3. **IsaacLab Documentation**: https://isaac-sim.github.io/IsaacLab/

## Next Steps

1. **Test basic functionality**: Run environment and check if propellers generate thrust
2. **Tune thrust coefficient**: Adjust until robot can hover
3. **Train RL policy**: Teach robot to control altitude and orientation
4. **Add complexity**: Ground effect, tilt control, advanced aerodynamics
5. **Sim-to-real**: Transfer learned policy to physical robot

