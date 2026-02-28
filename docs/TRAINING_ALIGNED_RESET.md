# Update: Training Environment Now Uses Aligned Reset

## Change Summary

Updated `DoubleBeeEventsCfg` (training configuration) to use the same aligned reset method as `DoubleBeeEventsCfg_PLAY` (play configuration).

## What Changed

### Before
```python
# DoubleBeeEventsCfg (Training)
reset_base = EventTerm(
    func=mdp.reset_root_state_from_terrain,  # Random, non-aligned reset
    mode="reset",
    params={...},
)
```

### After
```python
# DoubleBeeEventsCfg (Training)
reset_base = EventTerm(
    func=mdp.reset_root_state_from_terrain_aligned,  # Aligned reset (same as Play)
    mode="reset",
    params={
        "pose_range": {
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw_noise": (0.0, 0.0),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        "align_axis": "x",
    },
)
```

## Benefits

### 1. **Consistent Training and Evaluation**
- Both training and play use the same reset method
- No sim-to-sim gap between training and evaluation environments
- Policy trained with aligned initialization is evaluated with aligned initialization

### 2. **Improved Learning Efficiency**
- Robot always spawns facing the target
- Reduces exploration needed to find target direction
- More consistent initial conditions → more stable learning
- Each episode starts with clear objective (target is aligned and visible)

### 3. **Better Generalization**
- Policy learns to navigate toward aligned targets
- Transferable to various aligned configurations
- Removes randomness from spawn orientation and position alignment

### 4. **Easier Debugging**
- Predictable initial states make debugging easier
- Can compare training and play behaviors directly
- Visualization shows clear alignment during training

## Three Types of Alignment (Training & Play)

Both environments now enforce:

1. **SPATIAL ALIGNMENT**: Start and target share same X coordinate
   - Robot moves along Y axis toward target
   - Configurable via `align_axis` parameter

2. **ORIENTATION ALIGNMENT**: Robot's yaw faces target
   - Computed as `yaw = atan2(direction_x, direction_y)`
   - No random yaw offset (`yaw_noise = 0.0`)

3. **YAW RATE ALIGNMENT**: Zero angular velocity at spawn
   - Robot not spinning when episode starts
   - Configured via `velocity_range["yaw"] = (0.0, 0.0)`

## Training Configuration Options

If you want to add domain randomization during training while maintaining alignment:

```python
# Example: Add controlled randomization
reset_base = EventTerm(
    func=mdp.reset_root_state_from_terrain_aligned,
    mode="reset",
    params={
        "pose_range": {
            "roll": (-0.05, 0.05),     # ±3° roll variation
            "pitch": (-0.05, 0.05),    # ±3° pitch variation
            "yaw_noise": (-0.1, 0.1),  # ±6° yaw noise around target direction
        },
        "velocity_range": {
            "x": (-0.2, 0.2),          # Small initial velocity
            "y": (-0.2, 0.2),
            "z": (0.0, 0.0),
            "yaw": (-0.1, 0.1),        # Small initial yaw rate
        },
        "align_axis": "x",
    },
)
```

## Task IDs (Both Use Aligned Reset Now)

### Training Task
```bash
python scripts/co_rl/train.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo \
    --algo ppo --num_envs 128 --max_iterations 5000
```
- Uses: `DoubleBeeEventsCfg` ✅ Now with aligned reset
- Purpose: Train policy with consistent aligned initialization

### Play/Evaluation Task
```bash
python scripts/co_rl/play.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo \
    --video --video_length 1000 \
    --load_run <run_name> --checkpoint <model_file>
```
- Uses: `DoubleBeeEventsCfg_PLAY` ✅ Already had aligned reset
- Purpose: Evaluate policy with consistent aligned initialization

## Migration Notes

### Existing Policies
If you have policies trained with the old random reset method:
- They may perform differently with aligned reset during evaluation
- Consider retraining with the new aligned reset for best results
- Or revert to old reset method for evaluation of old policies

### Revert to Random Reset (if needed)
If you need to revert the training environment to random reset:

```python
# DoubleBeeEventsCfg
reset_base = EventTerm(
    func=mdp.reset_root_state_from_terrain,  # Back to random reset
    mode="reset",
    params={...},
)
```

## Files Modified

- `/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/flat_env/hybrid_stair/hybrid_stair_cfg.py`
  - Changed `DoubleBeeEventsCfg.reset_base` to use `reset_root_state_from_terrain_aligned`
  - Updated parameters to match `DoubleBeeEventsCfg_PLAY`
  - Updated comments to reflect alignment enforcement

## Related Documentation

- `ALIGNMENT_SYSTEM.md` - Complete alignment system documentation
- `FIX_RESET_ORDER_ALIGNMENT.md` - Details on the reset order fix
- `BUG_FIX_ALIGNMENT.md` - Yaw calculation fix documentation

## Testing

After this change, verify training behaves as expected:

```bash
# Start training with aligned reset
python scripts/co_rl/train.py \
    --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo \
    --algo ppo --num_envs 128 --max_iterations 5000

# Expected: All environments spawn with robot facing target
# Expected: Robot X coordinate matches target X coordinate
# Expected: Learning should be more efficient with consistent initialization
```

## Commit Message

```
feat(config): use aligned reset for both training and play environments

- Training env now uses reset_root_state_from_terrain_aligned (same as play)
- Ensures consistency between training and evaluation
- Improves learning efficiency with predictable initial states
- Both envs enforce spatial, orientation, and yaw rate alignment

Benefits:
- No sim-to-sim gap between train and eval
- More stable learning with consistent initialization
- Easier debugging with predictable spawns
- Robot always faces target at episode start
```
