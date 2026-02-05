# Episode Timeout Debug Investigation

## Problem

Episodes are never completing despite having a `time_out` constraint enabled. Evidence:
- `[DEBUG _reset_idx]` never prints
- All `Episode_*` metrics show 0.0000
- `Episode_Constraint/time_out: 0.0000` (never triggers)

## Expected Behavior

With config:
```python
episode_length_s = 20.0
decimation = 4
sim.dt = 0.005
```

**Calculated max_episode_length**:
```python
max_episode_length = ceil(20.0 / (4 * 0.005)) = ceil(20.0 / 0.02) = 1000 steps
```

**Expected**: After 1000 steps, `time_out` constraint should trigger and reset the episode.

## Debug Logging Added

### 1. Environment Initialization
```python
[DEBUG Environment Init] max_episode_length = ??? steps
[DEBUG Environment Init] episode_length_s = ??? s
[DEBUG Environment Init] step_dt = ??? s
[DEBUG Environment Init] episode_length_buf initialized: ???
```

### 2. Constraint Manager Initialization
```python
[DEBUG ConstraintManager] Initialized with ??? constraint terms:
  - time_out (or empty if not registered)
```

### 3. Constraint Computation (Every 500 Steps)
```python
[DEBUG Constraints] Step ???:
  episode_length_buf[0] = ???  # Should increment each step
  max_episode_length = ???     # Should be 1000
  reset_buf[0] = ???          # Should be True when episode_length_buf >= 1000
  reset_time_outs[0] = ???    # Should be 1.0 when time_out triggers
  reset_delta[0] = ???        # Should be 0.0 (no early termination)
```

### 4. Constraint Manager Details
```python
[DEBUG ConstraintManager.compute] Constraint 'time_out':
  value[0] = ???              # Should be True when episode_length_buf[0] >= max_episode_length
  time_out type = truncate

[DEBUG ConstraintManager.compute] reset_buf calculation:
  _truncated_buf[0] = ???     # Should be 1.0 when time_out triggers
  _delta_buf[0] = ???         # Should be 0.0 (no terminate constraints)
  reset_buf[0] = ???          # max(_truncated_buf, _delta_buf)
  (reset_buf == 1.0)[0] = ??? # Boolean result returned to environment
```

## Possible Issues to Check

### Issue 1: Constraint Not Registered
**Symptom**: `[DEBUG ConstraintManager] Initialized with 0 constraint terms`

**Cause**: `ConstraintsCfg` not properly passed to environment

**Fix**: Verify `constraints: ConstraintsCfg = ConstraintsCfg()` in config

### Issue 2: max_episode_length Incorrectly Calculated
**Symptom**: `max_episode_length` is not 1000

**Cause**: Wrong `episode_length_s`, `decimation`, or `step_dt` values

**Fix**: Verify configuration values match expectations

### Issue 3: episode_length_buf Not Incrementing
**Symptom**: `episode_length_buf[0]` stays at 0 or doesn't reach 1000

**Cause**: Buffer not being incremented in `step()`

**Fix**: Verify `self.episode_length_buf += 1` is executed

### Issue 4: Constraint Function Returns Wrong Type
**Symptom**: Error or `value[0]` is not Boolean True/False

**Cause**: Lambda function doesn't return Boolean tensor

**Fix**: Ensure `lambda env: env.episode_length_buf >= env.max_episode_length` returns Boolean

### Issue 5: _truncated_buf Not Set Correctly
**Symptom**: `_truncated_buf[0]` is 0.0 even when constraint triggers

**Cause**: Constraint value not properly converted or stored

**Fix**: Check conversion from Boolean to float in ConstraintManager.compute()

### Issue 6: reset_buf Comparison Issue
**Symptom**: `(reset_buf == 1.0)[0]` is False even when `reset_buf[0]` is 1.0

**Cause**: Floating point precision issue

**Fix**: Use `>= 0.99` instead of `== 1.0`

## Diagnostic Metrics Weight Issue (SOLVED)

**Problem**: All diagnostic metrics with `weight=0.0` showed 0.0000

**Root Cause**: IsaacLab's `reward_manager.py` line 145-146:
```python
if term_cfg.weight == 0.0:
    continue  # Skip terms with zero weight
```

**Fix**: Changed diagnostic metric weights from `0.0` to `1e-10` (negligible impact on training, but prevents skipping)

## Next Steps

1. **Run training with debug logging**
2. **Check which issue manifests** from the debug output
3. **Apply corresponding fix**
4. **Verify episodes complete at 1000 steps**

