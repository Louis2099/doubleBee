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
        usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee_modified.usd",
        # usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee_original.usd",
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
        # NOTE: the original exported USD had nonsensical masses; the
        # doubleBee_modified.usd loaded above is the corrected one.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            # mass=None means DO NOT override -- the USD's own per-body masses are
            # what sim uses. The "will override per-body below" note this replaces
            # was stale: no per-body override exists anywhere in this file, and
            # doubleBee_modified.usd is the version whose masses were already
            # corrected (that is what "modified" refers to).
            #
            # CONFIRMED TOTAL MASS: 4.47 kg -> W = 43.9 N. Every derived number
            # depends on this, so it is recorded here rather than in a comment
            # somewhere downstream:
            #   gravity torque about the wheel axle = 6.10 * sin(theta) N*m
            #   wheels    2 x 0.35 = 0.70 N*m   -> rights  6.6 deg
            #   props     0-375 rad/s @ pi/4    -> 8.07 N*m, rights >90 deg
            #   T/W       0.59 at the 0-375 range, 0.83 at 0-500 (cannot hover)
            # An earlier note put the mass at 2.76 kg, which inflated T/W from
            # 0.38 to 0.62 and made the propeller range look adequate when it was
            # not. See the propeller_vel term in mdp/actions.py.
            mass=None,
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
            effort_limit=0.35,  # Adjust based on your motor specs
            velocity_limit=23.6,
            min_delay=1,
            max_delay=3,
            stiffness={
                "leftWheel": 0.0,   # Wheels typically have no stiffness
                "rightWheel": 0.0,
            },
            damping={
                "leftWheel": 100, # 87.0
                "rightWheel": 100, # 87.0
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
            min_delay=2, # guessed, in sim steps at 0.02s = 40-100ms lag
            max_delay=5, # guessed
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
            effort_limit=5.0,  
            velocity_limit=600.0,  # Increased velocity limit
            # Propellers are the BALANCE actuator and have the longest real lag
            # in the whole chain: MAVLink hop + ESC response + propeller spin-up
            # from idle to hover thrust, which measures ~100-200 ms. These were
            # 0/0, i.e. instantaneous thrust, while the servos already carried
            # 2/5. That mismatch is the direct cause of the divergence seen on
            # hardware 2026-08-21 (transfer3.csv): the policy recovered 69 deg of
            # lean under thrust, overshot straight through vertical to +11 deg,
            # then oscillated apart -- the classic signature of a balance loop
            # tuned with less delay than it actually has.
            # 2-5 steps at the 20 ms control period = 40-100 ms, randomised per
            # reset so the policy cannot latch onto one exact delay.
            min_delay=2,
            max_delay=5,
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
