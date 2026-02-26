# Isaac Lab for DoubleBee

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/Lab-2.0.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## **✨ Features**
✔️ **DoubleBee Robot**: Two-wheel legged robot with inverted pendulum control  
✔️ **Velocity Tracking**: Advanced velocity control for locomotion tasks  
✔️ **Inverted Pendulum Mode**: Specialized control for self-balancing locomotion  
✔️ **Stack Environment**: Observations can be stacked with arguments  
✔️ **Constraint Manager**: [Constraints as Termination (CaT)](https://arxiv.org/abs/2403.18765) method implementation  
✔️ **CoRL**: Based on [rsl_rl](https://github.com/leggedrobotics/rsl_rl) library, off-policy algorithms implemented via `off_policy_runner`  

## Isaac Lab DoubleBee

DoubleBee is a two-wheel legged robot designed for agile locomotion and inverted pendulum control tasks in Isaac Lab simulation environment. The project focuses on reinforcement learning-based velocity tracking and self-balancing control.

## Available Tasks

### 1. Flat Environment - Stand & Drive
- **Task ID**: `Isaac-Velocity-HybridStair-DoubleBee-v1-ppo`
- **Play Task ID**: `Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo`
- Standard velocity tracking on flat terrain
- Full actuation including propeller control

### 2. Inverted Pendulum Mode
- **Task ID**: `Isaac-Velocity-InvertedPendulum-DoubleBee-v1-ppo`
- **Play Task ID**: `Isaac-Velocity-InvertedPendulum-DoubleBee-Play-v1-ppo`
- Same-level target tracking
- No height scan observations
- No propeller actuation (pure inverted pendulum control)

## Setup
- This repo is tested on Ubuntu 20.04, and we recommend 'local install'

### 1. Install Isaac Sim
```
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html
```

### 2. Install Isaac Lab
```
https://github.com/isaac-sim/IsaacLab
```

### 3. Install DoubleBee package
i. Clone repository
   ```
   git clone https://github.com/yourusername/doubleBee
   cd doubleBee
   ```

ii. Install doubleBee pip package
   - Run it on 'doubleBee' root path
   ```
   conda activate env_isaaclab  # change to your conda env
   pip install -e .
   ```

iii. Unzip assets (USD files)
   - Since git does not correctly upload '.usd' files, you should manually unzip the USD files in assets folder
   ```
   path example: lab/doublebee/assets/data/Robots/DoubleBee/
   ```

## Launch Scripts

### Train DoubleBee
Run it on 'doubleBee' root path:
```bash
python scripts/co_rl/train.py --task {task_name} --algo ppo --num_envs 4096 --headless --num_policy_stacks {stack_number} --num_critic_stacks {stack_number}
```

### Train Examples

#### Standard Flat Environment (Stand & Drive)
```bash
python scripts/co_rl/train.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --algo ppo --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
```

#### Inverted Pendulum Mode
```bash
python scripts/co_rl/train.py --task Isaac-Velocity-InvertedPendulum-DoubleBee-v1-ppo --algo ppo --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
```

### Play DoubleBee
Run it on 'doubleBee' root path:
```bash
python scripts/co_rl/play.py --task {task_name} --algo ppo --num_envs 64 --num_policy_stacks {stack_number} --num_critic_stacks {stack_number} --load_run {folder_name} --checkpoint {checkpoint_file}
```

### Play Examples

#### Standard Play
```bash
python scripts/co_rl/play.py --task Isaac-Velocity-HybridStair-DoubleBee-Play-v1-ppo --algo ppo --num_envs 64 --num_policy_stacks 2 --num_critic_stacks 2 --load_run 2026-02-05_19-31-16_hybrid_stair --checkpoint model_4999.pt
```

#### Inverted Pendulum Play with Video Recording
```bash
python scripts/co_rl/play.py --task Isaac-Velocity-InvertedPendulum-DoubleBee-Play-v1-ppo --video --video_length 1000 --load_run 2026-02-05_19-31-16_hybrid_stair --checkpoint model_4999.pt
```

## Additional Options

### Video Recording
Add `--video` flag and specify length:
```bash
python scripts/co_rl/play.py --task Isaac-Velocity-InvertedPendulum-DoubleBee-Play-v1-ppo --video --video_length 1000 --load_run 2026-02-05_19-31-16_hybrid_stair --checkpoint model_4999.pt
```

### Plotting
Disable plotting with:
```bash
--plot False
```

## Project Structure
```
doubleBee/
├── lab/doublebee/
│   ├── assets/              # Robot USD files and configurations
│   ├── tasks/               # Task definitions and environment configs
│   │   └── manager_based/
│   │       └── locomotion/
│   │           └── velocity/
│   │               ├── doublebee_env/     # Environment configurations
│   │               ├── mdp/               # MDP components (rewards, observations, etc.)
│   │               └── terrain_config/    # Terrain configurations
│   └── isaaclab/            # Custom Isaac Lab extensions
├── scripts/co_rl/           # Training and evaluation scripts
└── logs/                    # Training logs and checkpoints
```

## Acknowledgments

Based on the Isaac Lab framework and inspired by the Flamingo robot project.
