# Test USD Script - Quick Start Guide

## Overview

`test_usd.py` is a smoke test script that loads a USD-based robot into Isaac Sim and performs basic validation. It's designed to quickly verify that:
- The robot USD file loads correctly
- The environment can be created
- Joints and actuators are configured properly
- The robot can be controlled via actions

## Basic Usage

### Quick Test (Default Settings)
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo
```

This will:
- Load the DoubleBee robot with default settings (1 environment)
- Run 10,000 simulation steps
- Test joint movement capabilities
- Display debug information about the robot configuration

### Headless Mode (No GUI)
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --headless
```

### Multiple Environments
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --num_envs 4
```

### Shorter Test Run
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --num_steps 1000
```

## Command-Line Arguments

### Required Arguments
- `--task`: Gym task ID to test (e.g., `Isaac-Velocity-HybridStair-DoubleBee-v1-ppo`)

### Optional Arguments
- `--num_envs`: Number of parallel environments to create (default: `1`)
- `--num_steps`: Number of simulation steps to run (default: `10000`)
- `--disable_fabric`: Disable fabric and use USD I/O operations
- `--hover`: Run a hovering test (wheels stalled, servos at 0, propellers provide thrust)

### Isaac Sim Arguments (via AppLauncher)
- `--headless`: Run without GUI
- `--device`: Device to use (e.g., `cuda:0`, `cpu`)
- `--livestream`: Enable livestream

## Common Use Cases

### 1. Quick Robot Validation
Test if the robot loads correctly:
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --num_steps 100 --headless
```

### 2. Joint Movement Test
Verify that all joints respond to commands:
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --num_steps 200
```

The script will automatically test:
- Wheel joints (velocity control)
- Servo joints (position control)
- Propeller joints (velocity control)

### 3. Hovering Test
Test if propellers can generate enough thrust to hover:
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --hover --num_steps 5000
```

### 4. Multi-Environment Test
Test with multiple parallel environments:
```bash
python scripts/test_usd.py --task Isaac-Velocity-HybridStair-DoubleBee-v1-ppo --num_envs 8 --headless
```

## Output Information

The script provides detailed debug information:

### Robot Information
- Robot type and configuration
- Prim path and USD file path
- Number of instances
- Initial position

### Joint Information
- Available joint names
- Joint types (wheels, servos, propellers)
- Joint indices and mapping
- Actuator configuration

### Action/Control Information
- Action space dimensions
- Action-to-joint mapping verification
- Action manager configuration

### Observation Information
- Observation space structure
- Observation groups and terms
- Observation shapes

### Test Results
At the end, the script provides:
- Joint movement analysis
- Verification that joints respond to commands
- Overall assessment of robot configuration

## Troubleshooting

### Robot Not Visible
If the robot is not visible in the GUI:
1. Check USD file visibility settings
2. Press 'F' key in viewport with robot prim selected
3. Check Stage panel for: `/World/envs/env_0/Doublebee`

### Joints Not Moving
If joints don't respond:
1. Check actuator configuration
2. Verify action-to-joint mapping
3. Check joint limits and effort limits

### Errors During Loading
If you encounter errors:
1. Verify the USD file exists at the configured path
2. Check that all required extensions are loaded
3. Ensure Isaac Sim is properly installed

## Example Output

```
[INFO] Environment 'Isaac-Velocity-HybridStair-DoubleBee-v1-ppo' loaded. Observation shape: {'policy': torch.Size([1, 48])}
[DEBUG] Scene entities: ['robot', 'terrain', 'height_scanner', 'contact_forces', 'light', 'dome_light']
[DEBUG] Robot type: <class 'isaaclab.assets.articulation.Articulation'>
[DEBUG] Robot prim path: /World/envs/env_.*/Doublebee
[DEBUG] Robot num instances: 1
[DEBUG] Robot is spawned successfully!

[JOINT] Available joints: ['leftWheel', 'rightWheel', 'leftPropellerServo', 'rightPropellerServo', 'leftPropeller', 'rightPropeller']
[JOINT] Wheel joints (velocity control): ['leftWheel', 'rightWheel']
[JOINT] Servo joints (position control): ['leftPropellerServo', 'rightPropellerServo']
[JOINT] Propeller joints (velocity control): ['leftPropeller', 'rightPropeller']
```

## Notes

- The script automatically tests joint movement in phases: idle → wheels → servos → propellers
- Each phase runs for 20 steps before switching
- Debug output is printed every 20 steps to reduce verbosity
- The script will run until completion or until you press Ctrl+C

## Related Files

- `scripts/co_rl/train.py`: Training script for RL agents
- `scripts/co_rl/play.py`: Playback script for trained agents
- `lab/doublebee/tasks/`: Task configurations


