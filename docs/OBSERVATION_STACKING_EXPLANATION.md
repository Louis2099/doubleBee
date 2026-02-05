# Observation Stacking: `num_policy_stacks` and `num_critic_stacks`

## Overview

`num_policy_stacks` and `num_critic_stacks` control **temporal observation stacking** for the policy (actor) and critic networks respectively. They allow the networks to see a history of observations, not just the current frame.

## What is Observation Stacking?

**Observation stacking** concatenates multiple consecutive frames of observations into a single vector. This gives the network temporal information, allowing it to infer velocity, acceleration, and other time-dependent features from the observation history.

### Example

If `num_policy_stacks = 2`:
- The policy sees: `[obs(t-2), obs(t-1), obs(t)]` concatenated together
- This is **3 frames total** (2 previous + 1 current)

If `num_policy_stacks = 0`:
- The policy sees: `[obs(t)]` only (no stacking)

## How It Works

### 1. **StateHandler Class**

The `StateHandler` class (in `state_handler.py`) manages the observation buffer:

```python
class StateHandler:
    def __init__(self, total_frames: int, stack_dim: int, nonstack_dim: int):
        self.total_frames = total_frames  # num_stacks + 1
        self.stack_dim = stack_dim       # Dimension of stackable observations
        self.nonstack_dim = nonstack_dim  # Dimension of non-stackable observations
        self.num_obs = stack_dim * total_frames + nonstack_dim
        self.stack_buffer = None  # Circular buffer for storing frames
```

**Key Formula**:
```python
total_frames = num_stacks + 1  # +1 for current frame
final_obs_dim = (stack_dim * total_frames) + nonstack_dim
```

### 2. **Observation Groups**

IsaacLab environments split observations into groups:

- **`stack_policy`**: Observations that **can be stacked** for the policy (e.g., joint positions, velocities)
- **`none_stack_policy`**: Observations that **should NOT be stacked** for the policy (e.g., command, time-to-go)
- **`stack_critic`**: Observations that **can be stacked** for the critic (privileged info)
- **`none_stack_critic`**: Observations that **should NOT be stacked** for the critic

**Why split?**
- Some observations (like commands) don't benefit from stacking
- Some observations (like velocities) are more informative when stacked (shows acceleration)

### 3. **Initialization in vecenv_wrapper.py**

```python
# Policy stacking
self.policy_state_handler = StateHandler(
    self.num_policy_stacks + 1,  # total_frames = num_stacks + 1
    stack_policy_dim,             # Dimension of stackable policy obs
    nonstack_policy_dim          # Dimension of non-stackable policy obs
)

# Critic stacking
self.critic_state_handler = StateHandler(
    self.num_critic_stacks + 1,  # total_frames = num_stacks + 1
    stack_critic_dim,             # Dimension of stackable critic obs
    nonstack_critic_dim           # Dimension of non-stackable critic obs
)
```

### 4. **Observation Processing**

In `get_observations()`:

```python
# Policy observations
if self.policy_state_handler.stack_buffer is None:
    # First call: Initialize buffer with current observation
    policy_obs = self.policy_state_handler.reset(
        obs_dict["stack_policy"], 
        obs_dict["none_stack_policy"]
    )
else:
    # Subsequent calls: Update buffer (FIFO queue)
    policy_obs = self.policy_state_handler.update(
        obs_dict["stack_policy"], 
        obs_dict["none_stack_policy"]
    )

# Critic observations (same process)
if self.critic_state_handler.stack_buffer is None:
    critic_obs = self.critic_state_handler.reset(...)
else:
    critic_obs = self.critic_state_handler.update(...)
```

### 5. **Buffer Update Mechanism**

The `StateHandler.update()` method maintains a **FIFO (First-In-First-Out) queue**:

```python
def update(self, stack_obs, nonstack_obs):
    # Add new observation to front, remove oldest from back
    self.stack_buffer = [stack_obs.clone()] + self.stack_buffer[:-1]
    
    # Concatenate all stacked observations
    stacked = torch.cat(self.stack_buffer, dim=-1)
    
    # Concatenate with non-stackable observations
    return torch.cat([stacked, nonstack_obs], dim=-1)
```

**Example with `num_policy_stacks = 2`**:
```
Time t=0: buffer = [obs(0), obs(0), obs(0)]  # Initialized with first obs
Time t=1: buffer = [obs(1), obs(0), obs(0)]  # New obs added, oldest removed
Time t=2: buffer = [obs(2), obs(1), obs(0)]  # Now has 3 different frames
Time t=3: buffer = [obs(3), obs(2), obs(1)]  # Continues sliding window
```

## Mathematical Details

### Observation Dimension Calculation

**Without stacking** (`num_stacks = 0`):
```
final_obs_dim = stack_dim + nonstack_dim
```

**With stacking** (`num_stacks = 2`):
```
total_frames = num_stacks + 1 = 3
stacked_dim = stack_dim * total_frames = stack_dim * 3
final_obs_dim = stacked_dim + nonstack_dim
```

### Example Calculation

Assume:
- `stack_policy_dim = 20` (e.g., joint positions + velocities)
- `nonstack_policy_dim = 5` (e.g., command + time-to-go)
- `num_policy_stacks = 2`

Then:
```
total_frames = 2 + 1 = 3
stacked_dim = 20 * 3 = 60
final_policy_obs_dim = 60 + 5 = 65
```

## Why Use Different Stacking for Policy and Critic?

**Policy (Actor)**:
- Needs to make decisions based on **current and recent history**
- Typically uses fewer stacks (e.g., `num_policy_stacks = 2`)
- Focuses on immediate control

**Critic (Value Function)**:
- Estimates **long-term value**, may benefit from more history
- Can use more stacks (e.g., `num_critic_stacks = 4`)
- Focuses on value estimation

**Note**: In practice, they're often set to the same value for simplicity.

## Benefits of Observation Stacking

1. **Velocity Inference**: From position history, the network can infer velocity
2. **Acceleration Inference**: From velocity history, the network can infer acceleration
3. **Temporal Patterns**: Helps detect patterns that span multiple timesteps
4. **Smoothing**: Reduces sensitivity to noisy single-frame observations

## Trade-offs

**Pros**:
- ✅ Provides temporal information without recurrent networks
- ✅ Simple to implement (just concatenation)
- ✅ Works well for short-term dependencies

**Cons**:
- ❌ Increases observation dimension (more parameters in network)
- ❌ Limited to fixed history length
- ❌ Not as flexible as recurrent networks (LSTM/GRU) for long-term dependencies

## Configuration

### Default Values

In `rl_cfg.py`:
```python
num_policy_stacks: int = 0  # Default: no stacking
num_critic_stacks: int = 0  # Default: no stacking
```

### Command Line Arguments

In `train.py`:
```python
parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")
```

**Note**: The CLI defaults (2) override the config defaults (0) when provided.

### Usage Example

```bash
# Train with 2 frames of stacking for both policy and critic
python train.py --num_policy_stacks 2 --num_critic_stacks 2

# Train with different stacking for policy and critic
python train.py --num_policy_stacks 1 --num_critic_stacks 3

# Train without stacking
python train.py --num_policy_stacks 0 --num_critic_stacks 0
```

## Relationship to Recurrent Networks

**Observation Stacking** vs **Recurrent Networks (LSTM/GRU)**:

| Feature | Observation Stacking | Recurrent Networks |
|---------|---------------------|-------------------|
| **History Length** | Fixed (num_stacks) | Variable (unlimited) |
| **Memory** | Explicit (buffer) | Implicit (hidden state) |
| **Complexity** | Simple (concat) | Complex (RNN cells) |
| **Gradient Flow** | Direct | Through time |
| **Use Case** | Short-term dependencies | Long-term dependencies |

**Note**: If `is_recurrent=True` in the actor-critic network, stacking may be disabled (see `srmppo.py` line 141).

## Code Flow Summary

1. **Configuration**: `num_policy_stacks` and `num_critic_stacks` set in `train.py` or config
2. **Wrapper Initialization**: `CoRlVecEnvWrapper` creates `StateHandler` instances
3. **Observation Collection**: `get_observations()` calls `StateHandler.update()` each step
4. **Buffer Management**: `StateHandler` maintains FIFO queue of observations
5. **Concatenation**: Stacked observations are concatenated with non-stacked observations
6. **Network Input**: Final observation vector is passed to policy/critic networks

## Visual Example

**Without Stacking** (`num_stacks = 0`):
```
Time t:  [obs(t)]
         └─> Policy Network
```

**With Stacking** (`num_stacks = 2`):
```
Time t-2: [obs(t-2)]
Time t-1: [obs(t-1)] ─┐
Time t:   [obs(t)]    ├─> Concatenate ─> [obs(t-2), obs(t-1), obs(t)] ─> Policy Network
         └────────────┘
```

## Summary

- **`num_policy_stacks`**: Number of **previous frames** to stack for the policy (actor) network
- **`num_critic_stacks`**: Number of **previous frames** to stack for the critic (value) network
- **Total frames**: `num_stacks + 1` (includes current frame)
- **Purpose**: Provides temporal information (velocity, acceleration) without recurrent networks
- **Effect**: Increases observation dimension: `final_dim = (stack_dim * (num_stacks + 1)) + nonstack_dim`
- **Default**: Usually set to `2` (meaning 3 frames total: t-2, t-1, t)

