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
        usd_path=f"{DOUBLEBEE_ASSETS_DATA_DIR}/Robots/DoubleBee/doubleBee_merged.usd",
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
        # NOTE: doubleBee_merged.usd (loaded above) = the model that can rotate,
        # plus weighed mass and measured CoM. doubleBee_modified.usd has an
        # authored diagonalInertia in mm units that PhysX reads as kg*m^2,
        # which makes the body rotationally immovable. Do not go back to it.
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
            # CONFIRMED TOTAL MASS: 3.2182 kg -> W = 31.6 N. Weighed 2026-08-25:
            # frame 4.700 lb + battery 2.395 lb = 7.095 lb. Every derived number
            # depends on it, so it is recorded here rather than downstream:
            #   gravity torque about the wheel axle = 3.21 * sin(theta) N*m
            #   wheels    2 x 0.51 = 1.02 N*m   -> rights 18.5 deg
            #   props     0-375 rad/s @ pi/4    -> 8.07 N*m, rights >90 deg
            #   CoM       101.6 mm above the axle (measured by balancing)
            #   T/W       0.82 at the 0-375 range (cannot hover)
            # This figure has been wrong twice: 2.76 kg (guess), then 4.4665 kg
            # (authored USD masses, 28% high). 3.2182 kg is weighed. Do not
            # revise it from a file again -- put it on a scale.
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
            # DERIVED FROM MEASUREMENT, not a datasheet guess. 0.35 was an
            # estimate ("never validated; estimate 0.2-0.6, start 0.35") and it
            # set the wheels' entire balance authority.
            #
            # db_wheels.py measured 43 rad/s^2 of wheel acceleration under the
            # robot's own weight on 2026-08-20. At the weighed 3.2182 kg:
            #     a   = 43 * 0.0729 m       = 3.13 m/s^2
            #     F   = 3.2182 * 3.13       = 10.1 N
            #     tau = 10.1 * 0.0729       = 0.73 N*m total  -> 0.37 per wheel
            # 0.51 is kept as the LOWER bound from the heavier estimate; the
            # lighter mass makes it, if anything, conservative.
            #
            # Why it matters: gravity torque about the axle is m*g*L*sin(theta)
            # = 3.21*sin(theta) N*m at this mass, so the lean the wheels can
            # still recover is asin(2*effort_limit / 3.21):
            #     0.35 -> 12.6 deg      0.51 -> 18.5 deg
            # With effort_limit 0.35 against the (then believed) 4.4665 kg the
            # recoverable window was only 6.6 deg -- too small for exploration
            # to find, and training stalled: 189 iterations with mean episode
            # length flat at 27-28 steps. Some acceleration trials also had the
            # frame support dragging, which can only have reduced the measured
            # figure, so 0.51 is conservative in both directions.
            effort_limit=0.51,
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
