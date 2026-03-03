# Quick Reference: Energy & Success Metrics

## 📊 Three New TensorBoard Metrics

| Metric | Key | What it measures |
|--------|-----|------------------|
| **Average Energy** | `Metrics/energy/average_consumption` | Mean energy (J) per episode, all trajectories |
| **Success Rate** | `Metrics/success/rate` | Fraction of episodes that reached target (0.0-1.0) |
| **Success Energy** | `Metrics/energy/successful_trajectories` | Mean energy (J) for successful episodes only |

## 🔧 How It Works

### Energy Tracking
1. Every step: Calculate power from joint velocities
   - Propellers: `rad/s → PWM → Power (W)`
   - Wheels: `rad/s → RPM → Power (W)`
2. Accumulate: `Energy += Power × dt`
3. On reset: Log average and reset buffer

### Success Tracking  
1. Every step: Check `goal_reached` constraint value
2. If `goal_reached == 1.0`: Mark episode successful
3. On reset: Calculate success rate and log

## 📍 Key Files

| File | Purpose |
|------|---------|
| `isaaclab/envs/manager_based_constraint_rl_env.py` | Main implementation (lines 88-90, 246-250, 374-520, 597-601) |
| `mdp/constraints.py` | `goal_reached()` constraint (lines 102-173) |
| `mdp/thrust_energy_model.py` | Power conversion functions |
| `velocity_env_cfg.py` | Success threshold config (line 157: `distance_threshold: 0.2`) |

## ⚙️ Configuration

### Change Success Threshold
Edit `velocity_env_cfg.py`, line 157:
```python
params={"distance_threshold": 0.2},  # Change from 0.2m
```

### Verify Joint Ordering
Check that your robot joints are ordered as:
- Joint 0: Left propeller
- Joint 1: Right propeller
- Joint 2: Left wheel
- Joint 3: Right wheel

If not, edit `_update_energy_tracking()` in `manager_based_constraint_rl_env.py` (around line 405-408).

## 🚀 Usage

### Start Training
```bash
python scripts/co_rl/train.py --task DoubleBee-Velocity-Flat-StandDrive-v0
```

### View Metrics
```bash
tensorboard --logdir logs/co_rl
# Open http://localhost:6006
# Go to SCALARS tab
# Look for Metrics/* group
```

## 📈 Expected Trends

| Metric | Start | Training | Well-Trained |
|--------|-------|----------|--------------|
| Energy | High, variable | Decreasing | Low, stable |
| Success Rate | ~0% | Increasing | 80-100% |
| Success Energy | N/A | Variable | Low, stable |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Metrics not appearing | Check constraint is named `goal_reached` in config |
| Energy always zero | Verify `pwm2power_params.json` and `rpm2power_params.json` exist |
| Success rate always zero | Check `distance_threshold` isn't too small (default 0.2m) |
| Training crashes | Check error logs; tracking has try-except to prevent crashes |

## 📝 Documentation

- **Main Guide:** `docs/ENERGY_SUCCESS_METRICS.md` - Full implementation details
- **Update Summary:** `docs/SUCCESS_TRACKING_UPDATE.md` - Why we use constraint
- **Complete Overview:** `docs/ENERGY_SUCCESS_IMPLEMENTATION.md` - Everything in one place

## ✅ Verification Checklist

Before training:
- [ ] Check joint ordering matches assumption
- [ ] Verify regression models exist (`pwm2power_params.json`, `rpm2power_params.json`)
- [ ] Confirm `goal_reached` constraint is configured in env config
- [ ] Check success threshold (0.2m default) is appropriate for your task

During training:
- [ ] Metrics appear in TensorBoard after first episode completes
- [ ] Energy values are non-zero and reasonable (10-1000+ J range expected)
- [ ] Success rate starts near 0% and increases
- [ ] No crash/error messages related to metric tracking

## 🎯 Key Points

1. **Success = `goal_reached` constraint fires** (distance < 0.2m to target)
2. **Energy = Power × Time**, accumulated over episode
3. **Metrics logged when episodes reset**, aggregated per batch
4. **No duplicate logic** - reuses existing constraint for success
5. **Error-safe** - training continues even if tracking fails
