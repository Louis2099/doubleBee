#!/usr/bin/env python3
"""
State monitor for DoubleBee robot, rocording and visualizing the state of the robot.
Position: x, y, z
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG

class StateMonitor:
    def __init__(self, robot_cfg: ArticulationCfg):
        self.robot_cfg = robot_cfg
        self.robot = sim_utils.create_articulation(self.robot_cfg)
        self.robot.set_name("DoubleBee")
        self.robot.set_position(0.0, 0.0, 0.5)
        self.robot.set_orientation(0.0, 0.0, 0.0)
        self.robot.set_linear_velocity(0.0, 0.0, 0.0)
        self.robot.set_angular_velocity(0.0, 0.0, 0.0)
        pass

    def record_state(self):
        pass

    def visualize_state(self):
        pass

    def close(self):
        pass
