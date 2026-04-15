#!/usr/bin/env python3

# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Test script for DoubleBee robot using Isaac Sim

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

# Now we can import Isaac Sim modules
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading

def main():
    """Main function to test the robot."""
    # 1) Point to your robot root prim
    ROBOT_PATH = "/home/louis/project/doubleBee/lab/doublebee/lab/doublebee/assets/data/Robots/DoubleBee/doubleBee.usd"
    
    # 2) Create simulation context
    simulation_context = SimulationContext()
    
    # 3) Add robot to stage
    add_reference_to_stage(ROBOT_PATH, "/DoubleBee")
    
    # 4) Wait for things to load
    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()
    
    # 5) Initialize physics
    simulation_context.initialize_physics()
    
    # 6) Create articulation and initialize
    robot = Articulation("/DoubleBee")
    robot.initialize()
    
    # 7) Get joint information
    joint_names = robot._dof_names
    print("Joints:", joint_names)
    
    # 8) Start simulation
    simulation_context.play()
    
    # 9) Control the robot
    for i in range(300):
        # Example: set joint positions (you may need to adjust joint names)
        if len(joint_names) >= 2:
            # Set positions for first two joints
            robot.set_joint_positions([[0.5, -0.3]], joint_indices=[0, 1])
        
        # Step simulation
        simulation_context.step(render=True)
    
    # 10) Stop and close
    simulation_context.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()