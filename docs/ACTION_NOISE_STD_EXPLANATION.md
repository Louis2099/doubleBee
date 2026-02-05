# Mean Action Noise Std - Complete Explanation

## Overview
The **"Mean action noise std"** metric represents the average standard deviation of the action noise distribution used for exploration in PPO (Proximal Policy Optimization). This is a learnable parameter that controls how much the policy explores vs exploits.

## Where It's Calculated and Printed

### 1. **Calculation Location**
**File**: `scripts/co_rl/core/runners/on_policy_runner.py`

**Line 209**: The metric is calculated as:
```python
mean_std = self.alg.actor_critic.std.mean()
```

This takes the mean of the `std` parameter across all action dimensions.

### 2. **Logging Location**
**File**: `scripts/co_rl/core/runners/on_policy_runner.py`

**Line 215**: Logged to TensorBoard/WandB:
```python
self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
```

**Lines 238 & 252**: Printed to terminal:
```python
f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
```

## Step-by-Step: How Action Noise Std Works

### Step 1: **Initialization** 
**File**: `scripts/co_rl/core/modules/actor_critic.py`, **Line 64**

```python
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
```

- `std` is a **learnable PyTorch parameter** (not a fixed value!)
- Initialized to `init_noise_std` (default: 1.0) for each action dimension
- Shape: `[num_actions]` - one std value per action
- Example: For 6 actions → `std = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

**Why learnable?** The policy can adapt exploration during training - high std early (explore), low std later (exploit).

### Step 2: **Action Mean Calculation**
**File**: `scripts/co_rl/core/modules/actor_critic.py`, **Line 100**

```python
def update_distribution(self, observations):
    mean = self.actor(observations)  # Neural network outputs action means
    self.distribution = Normal(mean, mean * 0.0 + self.std)
```

- Actor network (MLP) processes observations → outputs **action means**
- Shape: `[batch_size, num_actions]`
- Example: `mean = [[0.2, -0.1, 0.5, ...], ...]` (one mean per action per env)

### Step 3: **Distribution Creation**
**Line 101**: Creates a Normal distribution:
```python
self.distribution = Normal(mean, mean * 0.0 + self.std)
```

- **Mean**: From actor network (varies per observation)
- **Std**: From learnable parameter (same for all observations, but learnable)
- `mean * 0.0 + self.std` ensures std is broadcast correctly
- Result: `Normal(mean=actor_output, std=self.std)`

**Example**:
- Observation 1 → `mean = [0.2, -0.1, 0.5]`, `std = [1.0, 1.0, 1.0]`
- Observation 2 → `mean = [0.3, 0.0, 0.4]`, `std = [1.0, 1.0, 1.0]` (same std!)

### Step 4: **Action Sampling** (During Rollout)
**File**: `scripts/co_rl/core/modules/actor_critic.py`, **Line 103-105**

```python
def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample()  # Sample from Normal(mean, std)
```

**File**: `scripts/co_rl/core/algorithms/ppo.py`, **Line 78**

```python
self.transition.actions = self.actor_critic.act(obs).detach()
```

- Samples actions from `Normal(mean, std)`
- **High std** → More exploration (actions vary more from mean)
- **Low std** → More exploitation (actions closer to mean)
- Shape: `[num_envs, num_actions]`

**Example**:
- `mean = [0.2, -0.1, 0.5]`, `std = [1.0, 1.0, 1.0]`
- Sampled action: `[0.8, -1.2, 1.3]` (could be far from mean due to high std)

### Step 5: **Recording for PPO Update**
**File**: `scripts/co_rl/core/algorithms/ppo.py`, **Lines 80-82**

```python
self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
self.transition.action_mean = self.actor_critic.action_mean.detach()
self.transition.action_sigma = self.actor_critic.action_std.detach()  # This is std!
```

- Records the **std used** when action was sampled (`old_sigma_batch`)
- Needed for importance sampling ratio in PPO

### Step 6: **PPO Update** (Learning)
**File**: `scripts/co_rl/core/algorithms/ppo.py`, **Lines 126-132**

```python
self.actor_critic.act(obs_batch, ...)  # Updates distribution with NEW std
actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
mu_batch = self.actor_critic.action_mean
sigma_batch = self.actor_critic.action_std  # NEW std (may have changed!)
```

**Lines 135-144**: KL Divergence calculation (for adaptive learning rate):
```python
kl = torch.sum(
    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
    / (2.0 * torch.square(sigma_batch))
    - 0.5,
    axis=-1,
)
```

- Compares **old std** (when action was sampled) vs **new std** (current policy)
- If KL too high → policy changed too much → reduce learning rate
- If KL too low → policy not learning → increase learning rate

**Lines 155-161**: Surrogate loss (PPO clip):
```python
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
surrogate = -torch.squeeze(advantages_batch) * ratio
surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
```

- `ratio` depends on log probabilities, which depend on **std**
- If std changes, log_prob changes → ratio changes → affects policy update

**Lines 174-180**: Gradient update:
```python
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
self.optimizer.zero_grad()
loss.backward()  # Gradients flow to std parameter!
nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
self.optimizer.step()  # std gets updated!
```

- **Gradients flow to `self.std` parameter**
- `std` is updated via Adam optimizer
- **Entropy term** (`-entropy_coef * entropy_batch.mean()`) encourages exploration:
  - Higher entropy → more exploration → higher std
  - Lower entropy → less exploration → lower std

### Step 7: **Metric Calculation** (After Update)
**File**: `scripts/co_rl/core/runners/on_policy_runner.py`, **Line 209**

```python
mean_std = self.alg.actor_critic.std.mean()
```

- Takes mean across all action dimensions
- Example: If `std = [0.8, 1.2, 0.9, 1.1, 0.95, 1.05]` → `mean_std = 1.0`

## What the Value Means

### **High std (e.g., 1.0)**
- ✅ **More exploration**: Actions vary significantly from mean
- ✅ **Better for early training**: Discover good strategies
- ❌ **Less stable**: Actions are noisy, harder to learn precise control

### **Low std (e.g., 0.1)**
- ✅ **More exploitation**: Actions close to mean (deterministic-like)
- ✅ **Better for fine-tuning**: Precise control after learning basics
- ❌ **Less exploration**: May get stuck in local optima

### **Typical Evolution**
1. **Start**: `std ≈ 1.0` (high exploration)
2. **Mid-training**: `std ≈ 0.5-0.8` (balanced)
3. **Converged**: `std ≈ 0.1-0.3` (low exploration, high exploitation)

## Why It's Important

1. **Exploration-Exploitation Trade-off**
   - Controls how much the policy explores vs exploits
   - Critical for learning in continuous action spaces

2. **Training Stability**
   - If std changes too fast → policy becomes unstable
   - KL divergence monitoring uses std to detect policy changes

3. **Performance Indicator**
   - Decreasing std → policy becoming more confident
   - Stuck at high std → policy still exploring (may need more training)

4. **Debugging**
   - If std → 0 too fast → policy may be overfitting
   - If std stays high → policy may not be learning effectively

## Example from Your Training

From terminal output:
```
Mean action noise std: 1.00
```

**Interpretation**:
- All 6 actions have `std ≈ 1.0` (initial value)
- Policy is in **high exploration mode**
- Actions will vary significantly from the mean
- This is **normal for iteration 0** - policy hasn't learned yet

**Expected progression**:
- Iteration 0-100: `std ≈ 1.0` (exploring)
- Iteration 100-1000: `std ≈ 0.5-0.8` (learning)
- Iteration 1000+: `std ≈ 0.1-0.3` (exploiting)

## Code Flow Summary

```
1. Initialize: std = [1.0, 1.0, ...] (learnable parameter)
   ↓
2. For each observation:
   - Compute mean = actor(obs)
   - Create distribution = Normal(mean, std)
   - Sample action = distribution.sample()
   ↓
3. Store: old_std, old_mean, action, log_prob
   ↓
4. PPO Update:
   - Compute new mean = actor(obs)
   - Get new std (may have changed!)
   - Compute KL divergence (old_std vs new_std)
   - Compute loss (includes entropy term)
   - Backprop → update std parameter
   ↓
5. Log: mean_std = mean(std across all actions)
```

## Related Metrics

- **Entropy**: Related to std (higher std → higher entropy)
- **KL Divergence**: Uses std in calculation (monitors policy change)
- **Surrogate Loss**: Affected by std through importance sampling ratio

## Configuration

**File**: `lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/agents/co_rl_cfg.py`

```python
policy: CoRlPpoActorCriticCfg = CoRlPpoActorCriticCfg(
    init_noise_std=1.0,  # Initial std value
    ...
)
algorithm: CoRlPpoAlgorithmCfg = CoRlPpoAlgorithmCfg(
    entropy_coef=0.01,  # Controls how much to encourage exploration
    ...
)
```

- `init_noise_std`: Starting value for std (default: 1.0)
- `entropy_coef`: Weight for entropy bonus (higher → more exploration → higher std)

