# VecEnv Wrapper Compatibility Fix

## Problem

The `CoRlVecEnvWrapper` was originally designed for the Flamingo environment, which has dedicated observation groups for frame stacking:
- `stack_policy` - Observations that should be stacked for the policy
- `none_stack_policy` - Observations that should NOT be stacked for the policy
- `stack_critic` - Observations that should be stacked for the critic
- `none_stack_critic` - Observations that should NOT be stacked for the critic

However, the DoubleBee environment uses a simpler observation structure with just:
- `policy` - All policy observations (no separate stacking groups)
- `critic` - Privileged critic observations (optional)

This caused a `KeyError` when trying to initialize the wrapper with DoubleBee:
```
KeyError: '"stack_policy" key is missing in observation_manager.group_obs_dim'
```

## Solution

Modified `CoRlVecEnvWrapper` to support **both** observation structures:

1. **Flamingo-style** (with dedicated stacking groups) → Frame stacking enabled
2. **DoubleBee-style** (standard groups) → Frame stacking disabled

The wrapper now automatically detects which type of environment it's wrapping and adapts accordingly.

## Changes Made

### 1. Import Path Fix

Fixed incorrect import paths that were pointing to `lab.flamingo` instead of `lab.doublebee`:

**Files affected:**
- `scripts/co_rl/core/wrapper/vecenv_wrapper.py` (line 11)
- `scripts/co_rl/train.py` (line 60)
- `scripts/co_rl/play_dep.py` (line 81)
- `scripts/co_rl/play.py` (line 89)

**Change:**
```python
# Before (wrong)
from lab.flamingo.isaaclab.isaaclab.envs.manager_based_constraint_rl_env import ManagerBasedConstraintRLEnv

# After (correct)
from lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env import ManagerBasedConstraintRLEnv
```

### 2. Observation Group Detection

Added automatic detection of observation group structure (lines 64-74):

```python
# Check if environment has stacking-specific observation groups
self.has_stacking_groups = False
if hasattr(self.unwrapped, "observation_manager"):
    group_obs_dim = self.unwrapped.observation_manager.group_obs_dim
    # Check if this environment has separate stacking groups (like Flamingo)
    if all(key in group_obs_dim for key in ["stack_policy", "none_stack_policy", "stack_critic", "none_stack_critic"]):
        self.has_stacking_groups = True
        print("[INFO] Environment has dedicated stacking observation groups")
    else:
        print(f"[INFO] Environment uses standard observation groups (available: {list(group_obs_dim.keys())})")
        print(f"[INFO] Frame stacking (num_policy_stacks={self.num_policy_stacks}, num_critic_stacks={self.num_critic_stacks}) will be disabled")
```

### 3. Conditional State Handler Creation

Modified policy and critic observation initialization to conditionally create state handlers (lines 82-121):

**For Policy:**
```python
if hasattr(self.unwrapped, "observation_manager"):
    if self.has_stacking_groups:
        # -- Policy observations with stacking (Flamingo-style)
        stack_policy_dim = self.unwrapped.observation_manager.group_obs_dim["stack_policy"][0]
        nonstack_policy_dim = self.unwrapped.observation_manager.group_obs_dim["none_stack_policy"][0]
        self.policy_state_handler = StateHandler(self.num_policy_stacks + 1, stack_policy_dim, nonstack_policy_dim)
        self.unwrapped.observation_manager.group_obs_dim["policy"] = (self.policy_state_handler.num_obs,)
        self.num_obs = self.policy_state_handler.num_obs
    else:
        # -- Standard policy observations without stacking (DoubleBee-style)
        self.policy_state_handler = None
        self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
```

**For Critic:**
```python
if hasattr(self.unwrapped, "observation_manager"):
    if self.has_stacking_groups:
        # -- Critic observations with stacking (Flamingo-style)
        stack_critic_dim = self.unwrapped.observation_manager.group_obs_dim["stack_critic"][0]
        nonstack_critic_dim = self.unwrapped.observation_manager.group_obs_dim["none_stack_critic"][0]
        self.critic_state_handler = StateHandler(self.num_critic_stacks + 1, stack_critic_dim, nonstack_critic_dim)
        self.unwrapped.observation_manager.group_obs_dim["critic"] = (self.critic_state_handler.num_obs,)
        self.num_privileged_obs = self.critic_state_handler.num_obs
    else:
        # -- Standard critic observations without stacking (DoubleBee-style)
        self.critic_state_handler = None
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        else:
            # If no critic group, use policy observations
            self.num_privileged_obs = self.num_obs
```

### 4. Updated Observation Methods

Modified `get_observations()`, `reset()`, and `step()` methods to handle both cases:

**Pattern used in all three methods:**
```python
# Policy observations
if self.policy_state_handler is not None:
    # Use state handler for stacking (Flamingo-style)
    policy_obs = self.policy_state_handler.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
    obs_dict["policy"] = policy_obs
else:
    # Use standard policy observations (DoubleBee-style)
    policy_obs = obs_dict["policy"]

# Critic observations
if self.critic_state_handler is not None:
    # Use state handler for stacking (Flamingo-style)
    critic_obs = self.critic_state_handler.update(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
    obs_dict["critic"] = critic_obs
elif "critic" not in obs_dict:
    # If no separate critic observations, use policy observations
    obs_dict["critic"] = policy_obs
```

## Behavior Summary

### For Flamingo Environment
- **Detection**: Has `stack_policy`, `none_stack_policy`, `stack_critic`, `none_stack_critic` groups
- **Behavior**: Frame stacking enabled via `StateHandler`
- **Observation Dimension**: `(stack_dim * (num_stacks + 1)) + nonstack_dim`
- **Log Message**: `"Environment has dedicated stacking observation groups"`

### For DoubleBee Environment
- **Detection**: Only has `policy` group (and optionally `critic`)
- **Behavior**: Frame stacking disabled, observations used as-is
- **Observation Dimension**: Unchanged from environment definition
- **Log Message**: `"Environment uses standard observation groups (available: ['policy'])"`
- **Additional Message**: `"Frame stacking (num_policy_stacks=2, num_critic_stacks=2) will be disabled"`

## Testing

Run training to verify the fix:

```bash
cd /home/yuanliu/Louis_Project/doubleBee
python scripts/co_rl/train.py --task Isaac-Velocity-Flat-DoubleBee-v1-ppo --num_envs 2
```

Expected output:
```
[INFO] Environment uses standard observation groups (available: ['policy'])
[INFO] Frame stacking (num_policy_stacks=2, num_critic_stacks=2) will be disabled
```

## Backward Compatibility

The changes are **fully backward compatible**:
- Flamingo environments will continue to use frame stacking as before
- DoubleBee environments will work without stacking
- No changes needed to existing environment configurations

## Future Work

If you want to enable frame stacking for DoubleBee, you would need to:

1. Split observations in `mdp/observations.py` into stacking/non-stacking groups:
   ```python
   @configclass
   class StackPolicyCfg(ObsGroup):
       # Observations that should be stacked
       wheel_vel = ObsTerm(...)
       servo_pos = ObsTerm(...)
       # ... other stackable observations
   
   @configclass
   class NoneStackPolicyCfg(ObsGroup):
       # Observations that should NOT be stacked
       velocity_commands = ObsTerm(...)
       # ... other non-stackable observations
   ```

2. Similarly split critic observations into `StackCriticCfg` and `NoneStackCriticCfg`

3. Update the `ObservationsCfg` to use these new groups

However, this is **optional** - the current implementation works fine without stacking.

## Related Documents

- `docs/OBSERVATION_STACKING_EXPLANATION.md` - Detailed explanation of frame stacking
- `docs/TRAINING_GUI_FIX.md` - Previous environment initialization fixes
- `docs/REWARD_CONFIG_COMPARISON.md` - Reward configuration differences

