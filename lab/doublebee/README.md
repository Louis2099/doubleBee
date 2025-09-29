# DoubleBee Robot Extension for Isaac Lab

This extension provides support for the DoubleBee robot - a two-wheeled robot with propellers for enhanced mobility and terrain traversal.

## Robot Description

The DoubleBee robot features:
- **Two wheels** for ground locomotion
- **Two propellers** with servo-controlled tilt for aerial assistance
- **Hybrid mobility** combining ground and aerial capabilities

## Joint Configuration

The robot has 6 actuated joints:
- `leftWheel` / `rightWheel`: Ground locomotion
- `leftPropellerServo` / `rightPropellerServo`: Propeller tilt control
- `leftPropeller` / `rightPropeller`: Propeller thrust generation

## Usage

```python
from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG

# Use in your environment configuration
env_cfg.robot = DOUBLEBEE_CFG
```

## Installation

1. Install the extension:
```bash
cd lab/doublebee
pip install -e .
```

2. Register with Isaac Lab:
```bash
# Add to your Isaac Lab configuration
export ISAAC_LAB_EXTENSIONS="lab.doublebee"
```

## Configuration

Robot parameters can be adjusted in:
- `assets/doublebee/doublebee_v1.py` - Main robot configuration
- `assets/data/Robots/DoubleBee/config.yaml` - Robot-specific parameters

## License

BSD-3-Clause License
