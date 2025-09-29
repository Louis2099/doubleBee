# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lab.doublebee.assets.doublebee import DOUBLEBEE_ASSETS_DATA_DIR

DOUBLEBEE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Initial height - adjust based on your robot
        joint_pos={
            # Wheel joints - these control ground movement (Z-axis rotation)
            "leftWheel": 0.0,
            "rightWheel": 0.0,
            # Propeller servo joints - these control propeller tilt/angle (Z-axis, ±90°)
            "leftPropellerServo": 0.0,
            "rightPropellerServo": 0.0,
            # Propeller joints - these control propeller rotation (Y-axis rotation)
            "leftPropeller": 0.0,
            "rightPropeller": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    
    soft_joint_pos_limit_factor=0.8,
    
    actuators={
        # Wheel actuators - for ground locomotion
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=["leftWheel", "rightWheel"],
            effort_limit=50.0,  # Adjust based on your motor specs
            velocity_limit=20.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "leftWheel": 0.0,   # Wheels typically have no stiffness
                "rightWheel": 0.0,
            },
            damping={
                "leftWheel": 0.7,   # Some damping for wheel friction
                "rightWheel": 0.7,
            },
            friction={
                "leftWheel": 0.0,
                "rightWheel": 0.0,
            },
            armature={
                "leftWheel": 0.01,
                "rightWheel": 0.01,
            },
        ),
        
        # Propeller servo actuators - for controlling propeller angle/tilt
        "propeller_servos": DelayedPDActuatorCfg(
            joint_names_expr=["leftPropellerServo", "rightPropellerServo"],
            effort_limit=20.0,  # Lower effort for servo motors
            velocity_limit=10.0,
            min_delay=0,
            max_delay=4,
            stiffness={
                "leftPropellerServo": 100.0,  # High stiffness for precise control
                "rightPropellerServo": 100.0,
            },
            damping={
                "leftPropellerServo": 2.0,
                "rightPropellerServo": 2.0,
            },
            friction={
                "leftPropellerServo": 0.0,
                "rightPropellerServo": 0.0,
            },
            armature={
                "leftPropellerServo": 0.01,
                "rightPropellerServo": 0.01,
            },
        ),
        
        # Propeller actuators - for thrust generation
        "propellers": DelayedPDActuatorCfg(
            joint_names_expr=["leftPropeller", "rightPropeller"],
            effort_limit=100.0,  # High effort for propeller thrust
            velocity_limit=50.0,  # High velocity for propeller rotation
            min_delay=0,
            max_delay=4,
            stiffness={
                "leftPropeller": 0.0,  # No stiffness for free rotation
                "rightPropeller": 0.0,
            },
            damping={
                "leftPropeller": 0.1,  # Low damping for free rotation
                "rightPropeller": 0.1,
            },
            friction={
                "leftPropeller": 0.0,
                "rightPropeller": 0.0,
            },
            armature={
                "leftPropeller": 0.01,
                "rightPropeller": 0.01,
            },
        ),
    },
)
