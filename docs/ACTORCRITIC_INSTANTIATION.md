# ActorCritic Class Instantiation Flow

## Overview

This document traces how the `ActorCritic` class is created during training and playing, and how the `activation` parameter flows through the system.

## Complete Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│  1. TASK REGISTRATION (Environment __init__.py)                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  gym.register(                                                    │
│      id="Isaac-Velocity-HybridStair-DoubleBee-v1-ppo",          │
│      kwargs={                                                     │
│          "env_cfg_entry_point": DoubleBeeHybridStairCfg,        │
│          "co_rl_cfg_entry_point": DoubleBeeCoRlCfg,  ← CONFIG   │
│      }                                                            │
│  )                                                                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────────┐
│  2. CONFIGURATION DEFINITION (co_rl_cfg.py)                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  @configclass                                                     │
│  class DoubleBeeCoRlCfg(CoRlPolicyRunnerCfg):                   │
│      policy: CoRlPpoActorCriticCfg = CoRlPpoActorCriticCfg(      │
│          init_noise_std=1.0,                                      │
│          actor_hidden_dims=[512, 256, 128],                       │
│          critic_hidden_dims=[512, 256, 128],                      │
│          activation="tanh",  ← ACTIVATION DEFINED HERE            │
│      )                                                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────────┐
│  3. CONFIG LOADING (train.py / play.py)                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  # In train.py line 93 / play.py line 277                        │
│  agent_cfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)  │
│      ↓                                                            │
│  # In cli_args.py line 53                                        │
│  corl_cfg = load_cfg_from_registry(                             │
│      task_name,                                                  │
│      "co_rl_cfg_entry_point"  ← Points to DoubleBeeCoRlCfg      │
│  )                                                                │
│      ↓                                                            │
│  Returns: DoubleBeeCoRlCfg instance with activation="tanh"       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────────┐
│  4. RUNNER INITIALIZATION (on_policy_runner.py)                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  class OnPolicyRunner:                                            │
│      def __init__(self, env, cfg, log_dir, device):              │
│          # Line 30: Extract policy config                        │
│          self.policy_cfg = cfg["policy"]                         │
│          # policy_cfg = {                                        │
│          #     "class_name": "ActorCritic",                      │
│          #     "init_noise_std": 1.0,                            │
│          #     "actor_hidden_dims": [512, 256, 128],             │
│          #     "critic_hidden_dims": [512, 256, 128],            │
│          #     "activation": "tanh"  ← STILL HERE                │
│          # }                                                      │
│                                                                    │
│          # Line 42: Get ActorCritic class                        │
│          actor_critic_class = eval(                              │
│              self.policy_cfg.pop("class_name")                   │
│          )  # Returns ActorCritic class                          │
│                                                                    │
│          # Line 43-45: Instantiate ActorCritic                   │
│          actor_critic = actor_critic_class(                      │
│              num_obs,                                             │
│              num_critic_obs,                                      │
│              self.env.num_actions,                                │
│              **self.policy_cfg  ← UNPACKS ALL CONFIG             │
│          )                                                        │
│          # Unpacks to:                                           │
│          # ActorCritic(                                          │
│          #     num_actor_obs=num_obs,                            │
│          #     num_critic_obs=num_critic_obs,                    │
│          #     num_actions=self.env.num_actions,                 │
│          #     init_noise_std=1.0,                                │
│          #     actor_hidden_dims=[512, 256, 128],                 │
│          #     critic_hidden_dims=[512, 256, 128],                │
│          #     activation="tanh"  ← PASSED HERE!                 │
│          # )                                                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────────┐
│  5. ACTORCRITIC CONSTRUCTION (actor_critic.py)                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  class ActorCritic(nn.Module):                                    │
│      def __init__(                                                │
│          self,                                                    │
│          num_actor_obs,                                           │
│          num_critic_obs,                                          │
│          num_actions,                                             │
│          actor_hidden_dims=[256, 256, 256],                       │
│          critic_hidden_dims=[256, 256, 256],                      │
│          activation="elu",  ← DEFAULT (overridden by "tanh")     │
│          init_noise_std=1.0,                                      │
│          **kwargs                                                 │
│      ):                                                           │
│          # Line 32: Get activation function                      │
│          activation = get_activation(activation)                 │
│          # With "tanh", returns nn.Tanh()                        │
│                                                                    │
│          # Build actor network                                   │
│          actor_layers = []                                        │
│          actor_layers.append(Linear(...))                         │
│          actor_layers.append(activation)  ← nn.Tanh()            │
│          # ... more layers with activation ...                   │
│          actor_layers.append(Linear(...))                         │
│          actor_layers.append(nn.Tanh())  ← Final tanh bounding   │
│          self.actor = nn.Sequential(*actor_layers)                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## File Locations

### 1. Configuration Definition
**File:** `/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/agents/co_rl_cfg.py`

```python
@configclass
class DoubleBeeCoRlCfg(CoRlPolicyRunnerCfg):
    """Configuration for DoubleBee CO-RL agent."""
    
    policy: CoRlPpoActorCriticCfg = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="tanh",  # ← DEFINED HERE
    )
```

**Line 31:** `activation="tanh"` is where the activation type is specified.

### 2. Task Registration
**File:** `/home/yuanliu/Louis_Project/doubleBee/lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/__init__.py`

```python
gym.register(
    id="Isaac-Velocity-HybridStair-DoubleBee-v1-ppo",
    entry_point="...",
    kwargs={
        "env_cfg_entry_point": DoubleBeeHybridStairCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.DoubleBeeCoRlCfg,  # ← Links to config
    },
)
```

**Lines 27, 37, 48, 58:** The `co_rl_cfg_entry_point` links each task to `DoubleBeeCoRlCfg`.

### 3. Config Loading (Training)
**File:** `/home/yuanliu/Louis_Project/doubleBee/scripts/co_rl/train.py`

```python
# Line 93
agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
```

**File:** `/home/yuanliu/Louis_Project/doubleBee/scripts/co_rl/cli_args.py`

```python
# Line 53
corl_cfg: CoRlPolicyRunnerCfg = load_cfg_from_registry(task_name, "co_rl_cfg_entry_point")
```

This loads `DoubleBeeCoRlCfg` from the registry using the task name.

### 4. Config Loading (Playing)
**File:** `/home/yuanliu/Louis_Project/doubleBee/scripts/co_rl/play.py`

```python
# Line 277
agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
```

Same mechanism as training - loads config from registry.

### 5. ActorCritic Instantiation
**File:** `/home/yuanliu/Louis_Project/doubleBee/scripts/co_rl/core/runners/on_policy_runner.py`

```python
# Line 30
self.policy_cfg = cfg["policy"]

# Line 42
actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic

# Line 43-45
actor_critic: ActorCritic = actor_critic_class(
    num_obs,
    num_critic_obs,
    self.env.num_actions,
    **self.policy_cfg  # ← Unpacks all config including activation="tanh"
)
```

The `**self.policy_cfg` unpacks the dictionary and passes all parameters as keyword arguments.

### 6. ActorCritic Constructor
**File:** `/home/yuanliu/Louis_Project/doubleBee/scripts/co_rl/core/modules/actor_critic.py`

```python
# Line 15-24
def __init__(
    self,
    num_actor_obs,
    num_critic_obs,
    num_actions,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",  # Default value
    init_noise_std=1.0,
    **kwargs,
):
    # Line 32
    activation = get_activation(activation)  # Converts "tanh" → nn.Tanh()
```

## Parameter Override Chain

```
Default in ActorCritic.__init__:
    activation="elu"

Overridden by DoubleBeeCoRlCfg:
    activation="tanh"

Passed through:
    train.py → cli_args.parse_co_rl_cfg() → load_cfg_from_registry() →
    DoubleBeeCoRlCfg → OnPolicyRunner.__init__() → actor_critic_class() →
    ActorCritic.__init__()

Final value:
    activation="tanh"
```

## Activation Function Mapping

**File:** `/home/yuanliu/Louis_Project/doubleBee/scripts/co_rl/core/modules/actor_critic.py`

```python
# Line 119-137
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()  # ← Used for DoubleBee
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
```

## Summary

### Training Flow
1. **Command:** `python scripts/co_rl/train.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo ...`
2. **Task lookup:** Finds task in gym registry
3. **Config loading:** `co_rl_cfg_entry_point` → `DoubleBeeCoRlCfg`
4. **Config parsing:** Extracts `policy` section with `activation="tanh"`
5. **Runner init:** `OnPolicyRunner` unpacks config to `ActorCritic(**policy_cfg)`
6. **Network construction:** `activation="tanh"` → `nn.Tanh()` used in layers

### Playing Flow
1. **Command:** `python scripts/co_rl/play.py --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo ...`
2. **Same as training:** Uses same config loading mechanism
3. **Network loaded:** Checkpoint contains network trained with `activation="tanh"`

### Key Points

1. **Single source of truth:** `DoubleBeeCoRlCfg` (line 31) defines `activation="tanh"`
2. **Both training and playing** use the same config entry point
3. **Activation is used twice:**
   - As activation function between hidden layers (`nn.Tanh()`)
   - As final output bounding layer (added in our recent change)

## Changing Activation

To change the activation function, edit:

```python
# File: lab/doublebee/tasks/.../agents/co_rl_cfg.py
policy: CoRlPpoActorCriticCfg = CoRlPpoActorCriticCfg(
    ...
    activation="relu",  # Change from "tanh" to "relu", "elu", etc.
)
```

Options: `"elu"`, `"selu"`, `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`
