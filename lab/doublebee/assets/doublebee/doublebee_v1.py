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
        # usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee_modified.usd",
        usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee_original.usd",
        activate_contact_sensors=True,
        visible=True,  # Ensure visibility is enabled
        # scale=(1.0, 1.0, 1.0),  # Convert cm to meters (USD was created in cm)
        scale=(0.001, 0.001, 0.001),  # Convert cm to meters (USD was created in cm)
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.7, 0.3),  # Brighter orange/yellow
            metallic=0.0,  # No metallic (metals appear black without proper lighting)
            roughness=0.4,  # Some roughness for better visibility
            emissive_color=(0.1, 0.05, 0.0),  # Slight glow
        ),
        # CRITICAL FIX: Override insane USD masses (USD has BILLIONS of kg!)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=None,  # Will override per-body below
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # Initial height - adjust based on your robot
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
            effort_limit=1.0,  # Adjust based on your motor specs
            velocity_limit=200.0,
            min_delay=0,
            max_delay=0,
            stiffness={
                "leftWheel": 0.0,   # Wheels typically have no stiffness
                "rightWheel": 0.0,
            },
            damping={
                "leftWheel": 10000,   # Some damping for wheel friction
                "rightWheel": 10000,
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
            effort_limit=5.0,  # Lower effort for servo motors
            velocity_limit=2.0,
            min_delay=0,
            max_delay=0,
            stiffness={
                "leftPropellerServo": 50000,  # High stiffness for precise control
                "rightPropellerServo": 50000,
            },
            damping={
                "leftPropellerServo": 1000,
                "rightPropellerServo": 1000,
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
            effort_limit=100.0,  
            velocity_limit=600.0,  # Increased velocity limit
            min_delay=0,
            max_delay=0,
            stiffness={
                "leftPropeller": 0.0,  # No stiffness for velocity control
                "rightPropeller": 0.0,
            },
            damping={
                "leftPropeller": 1000,  # MASSIVELY INCREASED (was 10.0)
                "rightPropeller": 1000,  # τ = damping * (vel_target - vel_current)
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
