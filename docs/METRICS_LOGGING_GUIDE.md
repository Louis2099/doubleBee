# Metrics Logging and Visualization Guide

## Overview

The **OnPolicyRunner** automatically logs all training metrics by default. Logging is **enabled** whenever `log_dir` is provided (which is always the case in `train.py`).

## ✅ Is Logging Enabled?

**YES!** Logging is **enabled by default** when you run `train.py`. The runner creates a log directory automatically.

### How to Verify

Check the terminal output when training starts:
```
[INFO] Logging experiment in directory: /path/to/logs/co_rl/doublebee_velocity/ppo
Exact experiment name requested from command line: /path/to/logs/.../2025-11-07_14-51-19_hybrid_stair
```

If you see these messages, logging is **active**.

## 📊 What Metrics Are Logged?

### Core Training Metrics (Always Logged)

These metrics are logged **every iteration**:

1. **Loss Metrics**:
   - `Loss/value_function` - Value function loss (critic loss)
   - `Loss/surrogate` - PPO surrogate loss (actor loss)
   - `Loss/learning_rate` - Current learning rate (may adapt if using adaptive schedule)

2. **Policy Metrics**:
   - `Policy/mean_noise_std` - Mean action noise standard deviation (exploration parameter)

3. **Performance Metrics**:
   - `Perf/total_fps` - Total steps per second (training speed)
   - `Perf/collection time` - Time spent collecting rollouts
   - `Perf/learning_time` - Time spent on policy updates

4. **Training Metrics** (logged when episodes complete):
   - `Train/mean_reward` - Mean episode reward
   - `Train/mean_episode_length` - Mean episode length
   - `Train/mean_reward/time` - Mean reward vs wall-clock time (TensorBoard/Neptune only)
   - `Train/mean_episode_length/time` - Mean episode length vs wall-clock time (TensorBoard/Neptune only)

5. **Environment-Specific Metrics** (from `infos["episode"]` or `infos["log"]`):
   - `Episode_Reward/tracking_lin_vel` - Linear velocity tracking reward
   - `Episode_Reward/tracking_ang_vel` - Angular velocity tracking reward
   - `Episode_Reward/upright` - Upright reward
   - `Episode_Reward/energy` - Energy consumption reward
   - `Episode_Reward/action_rate` - Action rate penalty
   - `Metrics/base_velocity/error_vel_xy` - Velocity tracking error (XY)
   - `Metrics/base_velocity/error_vel_yaw` - Yaw velocity tracking error
   - Any other metrics provided by the environment in `infos["episode"]` or `infos["log"]`

### Code Location

**File**: `scripts/co_rl/core/runners/on_policy_runner.py`

- **Lines 212-226**: Core metrics logging
- **Lines 189-208**: Environment-specific metrics (from `ep_infos`)

## 📁 Where Are Logs Stored?

### Log Directory Structure

```
logs/
└── co_rl/
    └── {experiment_name}/          # e.g., "doublebee_velocity"
        └── {algorithm}/            # e.g., "ppo"
            └── {timestamp}_{run_name}/  # e.g., "2025-11-07_14-51-19_hybrid_stair"
                ├── events.out.tfevents.*  # TensorBoard event files
                ├── model_0.pt             # Saved model checkpoints
                ├── model_500.pt
                ├── params/
                │   ├── agent.yaml         # Agent configuration
                │   ├── agent.pkl          # Agent config (pickle)
                │   ├── env.yaml           # Environment configuration
                │   └── env.pkl            # Environment config (pickle)
                └── git/
                    └── *.diff             # Git diff files
```

### Log Path Components

**File**: `scripts/co_rl/train.py`, **Lines 112-121**

```python
log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
log_dir = os.path.join(log_root_path, log_dir)
```

**Example Path**:
```
/home/yuanliu/Louis_Project/doubleBee/logs/co_rl/doublebee_velocity/ppo/2025-11-07_14-51-19_hybrid_stair
```

## 🎨 How to Visualize Logs

The runner supports **three logging backends**:

### 1. TensorBoard (Default) ✅

**Status**: Enabled by default (no configuration needed)

**How to Use**:

1. **Start TensorBoard**:
   ```bash
   # From project root
   tensorboard --logdir logs/co_rl
   
   # Or for specific experiment
   tensorboard --logdir logs/co_rl/doublebee_velocity/ppo
   ```

2. **Open in Browser**:
   - TensorBoard will start on `http://localhost:6006`
   - Open this URL in your browser

3. **View Metrics**:
   - **SCALARS** tab: All logged metrics
   - **IMAGES** tab: Any logged images (if any)
   - **GRAPHS** tab: Neural network architecture (if logged)

**Features**:
- ✅ Real-time updates (auto-refreshes)
- ✅ Compare multiple runs
- ✅ Smoothing, zooming, filtering
- ✅ Export data as CSV

**Log File Location**:
- TensorBoard event files: `{log_dir}/events.out.tfevents.*`

### 2. Weights & Biases (WandB) 🔬

**Status**: Optional (requires configuration)

**How to Enable**:

1. **Install WandB**:
   ```bash
   pip install wandb
   ```

2. **Login to WandB**:
   ```bash
   wandb login
   ```

3. **Set Environment Variable**:
   ```bash
   export WANDB_USERNAME=your_username
   ```

4. **Run Training with WandB**:
   ```bash
   python scripts/co_rl/train.py \
       --task DoubleBee-Velocity-Flat-StandDrive-v0 \
       --logger wandb \
       --log_project_name doublebee_rl
   ```

**Configuration**:

**File**: `lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/agents/co_rl_cfg.py`

Add to `DoubleBeeCoRlCfg`:
```python
logger: str = "wandb"
wandb_project: str = "doublebee_rl"  # Your WandB project name
```

**Features**:
- ✅ Cloud-based (access from anywhere)
- ✅ Advanced experiment tracking
- ✅ Team collaboration
- ✅ Hyperparameter sweeps
- ✅ Model versioning

**View Metrics**:
- Go to `https://wandb.ai/{username}/{project_name}`

### 3. Neptune 🚀

**Status**: Optional (requires configuration)

**How to Enable**:

1. **Install Neptune**:
   ```bash
   pip install neptune
   ```

2. **Set API Token**:
   ```bash
   export NEPTUNE_API_TOKEN=your_token
   ```

3. **Run Training with Neptune**:
   ```bash
   python scripts/co_rl/train.py \
       --task DoubleBee-Velocity-Flat-StandDrive-v0 \
       --logger neptune \
       --log_project_name your_neptune_project
   ```

**Features**:
- ✅ Enterprise-grade experiment tracking
- ✅ Advanced visualization
- ✅ Model registry
- ✅ Team collaboration

## 🔧 Configuration Options

### Change Logger Type

**Method 1: Command Line** (Recommended)
```bash
python scripts/co_rl/train.py \
    --task DoubleBee-Velocity-Flat-StandDrive-v0 \
    --logger tensorboard  # or "wandb" or "neptune"
```

**Method 2: Config File**

**File**: `lab/doublebee/tasks/manager_based/locomotion/velocity/doublebee_env/agents/co_rl_cfg.py`

```python
@configclass
class DoubleBeeCoRlCfg(CoRlPolicyRunnerCfg):
    logger: str = "tensorboard"  # "tensorboard", "wandb", or "neptune"
    wandb_project: str = "doublebee_rl"  # Only needed for wandb/neptune
    # ... rest of config
```

### Disable Logging (Not Recommended)

To disable logging, you would need to modify `train.py` to pass `log_dir=None` to the runner. **This is not recommended** as you'll lose all training metrics.

## 📈 Viewing Metrics in TensorBoard

### Common TensorBoard Commands

```bash
# View all experiments
tensorboard --logdir logs/co_rl

# View specific experiment
tensorboard --logdir logs/co_rl/doublebee_velocity/ppo/2025-11-07_14-51-19_hybrid_stair

# View on specific port
tensorboard --logdir logs/co_rl --port 6007

# View on remote server (SSH tunnel)
# On local machine:
ssh -L 6006:localhost:6006 user@remote_server
# Then open http://localhost:6006
```

### TensorBoard Interface

1. **SCALARS Tab**:
   - All logged metrics organized by category:
     - `Loss/` - Training losses
     - `Policy/` - Policy metrics
     - `Perf/` - Performance metrics
     - `Train/` - Training statistics
     - `Episode_Reward/` - Reward components
     - `Metrics/` - Environment metrics

2. **Compare Runs**:
   - Select multiple runs in the left sidebar
   - Overlay plots to compare performance

3. **Smoothing**:
   - Adjust smoothing slider (0-1) to reduce noise

4. **Download Data**:
   - Click "Download CSV" to export metric data

## 🔍 Finding Your Logs

### Quick Check

```bash
# List all experiments
ls -la logs/co_rl/doublebee_velocity/ppo/

# Find latest run
ls -t logs/co_rl/doublebee_velocity/ppo/ | head -1

# Check if TensorBoard files exist
ls logs/co_rl/doublebee_velocity/ppo/*/events.out.tfevents.*
```

### Example Output

```
logs/co_rl/doublebee_velocity/ppo/
├── 2025-11-07_14-51-19_hybrid_stair/
│   ├── events.out.tfevents.1762545082.yuanliu-legion.68754.0
│   ├── model_0.pt
│   └── params/
└── 2025-11-07_22-37-56_hybrid_stair/
    ├── events.out.tfevents.1762486844.yuanliu-legion.57107.0
    └── ...
```

## 📝 Summary

| Feature | Status | Location |
|---------|--------|----------|
| **Logging Enabled** | ✅ Yes (by default) | Automatic |
| **Logger Type** | TensorBoard (default) | `train.py` line 80 |
| **Log Directory** | `logs/co_rl/{exp}/{algo}/{timestamp}_{run}` | `train.py` lines 112-121 |
| **TensorBoard Files** | `events.out.tfevents.*` | In log directory |
| **Metrics Logged** | All core + environment metrics | `on_policy_runner.py` lines 212-226 |
| **Visualization** | TensorBoard (default) | `http://localhost:6006` |

## 🚀 Quick Start

1. **Run Training** (logging happens automatically):
   ```bash
   python scripts/co_rl/train.py --task DoubleBee-Velocity-Flat-StandDrive-v0
   ```

2. **Start TensorBoard** (in another terminal):
   ```bash
   tensorboard --logdir logs/co_rl
   ```

3. **View Metrics**:
   - Open `http://localhost:6006` in your browser
   - Navigate to SCALARS tab
   - Select your run from the left sidebar

That's it! All metrics are automatically logged and ready to visualize! 🎉

