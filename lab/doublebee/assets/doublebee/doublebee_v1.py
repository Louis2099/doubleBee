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
            # WHEEL COMMAND DELAY. Kept at 1-3 steps (20-60 ms), which is the
            # serial round trip. Do NOT raise this to model the wheels' slow
            # reversal -- that is a SLEW limit, not a dead time, and sim already
            # has it via effort_limit:
            #
            #   bench (db_wheels.py reverse, wheels off the ground):
            #     +10 -> 0 rad/s in 173 ms = 58 rad/s^2 sustained the whole way.
            #     173 ms x 58 = 10.0 rad/s, exactly the speed it had to shed, so
            #     the wheel decelerates from the first instant. Dead time ~= 0.
            #   sim: effort_limit 0.51 / (disc 4.45e-4 + armature 0.01)
            #        = 49 rad/s^2, matching the measured 58 within 20%.
            #
            # I briefly set these to 10-15 on 2026-08-25, reading the 260 ms
            # command-to-response lag from hw_v8.csv as transport delay. It is
            # not: it is the time to slew across a full reversal, and sim
            # reproduces it already. Stacking a transport delay on top would
            # double-count and would very likely have made the task
            # unlearnable. Caught because a wheel-only policy had previously
            # balanced this robot for 10 s -- impossible with 2.5 tau of true
            # dead time, which is what said the model had to be wrong.
            # 2-4 steps = 40-80 ms. Raised from 1-3 on 2026-08-25.
            #
            # This is TRANSPORT delay only -- the round trip from policy output
            # to torque at the wheel: 20 ms control period, the serial write at
            # 38400 baud, the RoboClaw's own control loop, and the ESC/motor
            # electrical response. 1-3 counted little more than the serial hop.
            #
            # Deliberately NOT larger. The wheels' slow reversal (173 ms bench,
            # 260 ms under load) is a SLEW limit, and effort_limit already
            # reproduces it: 0.51/(4.45e-4 + 0.01) = 49 rad/s^2 against the
            # measured 58. Modelling that slew again as dead time would
            # double-count it and very likely make the task unlearnable. A
            # wheel-only policy has balanced this robot for 10 s, which is
            # impossible with 2.5 tau of real dead time -- so the dead time is
            # small and only the slew is large.
            min_delay=2,
            max_delay=4,
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
            # PROPELLER TORQUE. Lowered 5.0 -> 0.2 N*m on 2026-08-25, together with
            # authoring a physical propeller inertia in the USD (1.14e-4 kg*m^2,
            # a 7-inch prop as a rod: m*L^2/12).
            #
            # The props had NO authored inertia, so PhysX computed it from the
            # mm-scale geometry and got 0.238 kg*m^2 -- 2080x too heavy. Against
            # the 5.0 N*m effort_limit that gave ~21 rad/s^2, i.e. 9.5 SECONDS to
            # reach 200 rad/s. Real props get there in ~100 ms. So in sim the
            # propellers were a flywheel the policy could not use for control,
            # and it learned wheels-only locomotion -- measured on hardware as
            # propeller actions averaging -0.4 and thrust below the 7.17 N
            # static-stability threshold.
            #
            # 5.0 N*m was also unphysical in a second way: joint torque REACTS
            # on the airframe, and 5.0 N*m exceeds the entire gravity torque of
            # 3.21 N*m. Counter-rotation cancels it only while both props run
            # together, which is precisely not the case when they are being used
            # to control roll.
            #
            # 0.2 N*m over 1.14e-4 kg*m^2 = 1750 rad/s^2 -> 200 rad/s in ~114 ms,
            # and a reaction torque small against gravity. Both physical.
            #
            # DEPLOYMENT: with props that now reach their commanded speed, sim's
            # achieved omega should approach the full 375 rad/s the action asks
            # for. Re-read `running_max` from the [PROP] print after training and
            # set --prop_rad_s_max to it (was 200 when the props were crippled).
            # RAISED 0.2 -> 0.6 on 2026-08-25, calibrated from measurement.
            #
            # With the armature fixed (0.01 -> 1e-5) the props reached 25.4 rad/s
            # in 0.1 s at 0.2 N*m = 254 rad/s^2, i.e. 787 ms to spin up to 200.
            # That is 13x better than before but still ~6x slower than physical,
            # so the authored 1.14e-4 tensor is evidently not being consumed at
            # face value somewhere in the chain. Rather than chase that, scale
            # the torque by what was actually observed:
            #
            #   0.2 N*m ->  254 rad/s^2 -> 787 ms   reaction  6% of gravity torque
            #   0.6 N*m ->  762 rad/s^2 -> 262 ms   reaction 19%
            #   1.0 N*m -> 1270 rad/s^2 -> 157 ms   reaction 31%
            #
            # 0.6 puts spin-up in the same order as a real ESC and propeller
            # while keeping the airframe reaction torque at 19% of the 3.21 N*m
            # gravity torque -- and counter-rotation cancels most of that except
            # when the props are driven asymmetrically for roll. 1.0 would be
            # more faithful but puts a third of a gravity-torque into the frame
            # exactly when the policy is using the props for control.
            effort_limit=0.6,  
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
            # ARMATURE is ADDED to the link inertia. At 0.01 it is 88x the real
            # propeller (1.14e-4 kg*m^2 authored in the USD) and completely
            # swamps it: measured effective inertia 0.0118, giving 16.9 rad/s^2
            # against the 0.2 N*m effort_limit -- still seconds to spin up.
            #
            # Armature models rotor/gearbox inertia reflected through a
            # reduction. These propellers are DIRECT DRIVE on the motor shaft,
            # so the only reflected inertia is the motor rotor itself, order
            # 1e-5 kg*m^2 for a 2306-class outrunner.
            #
            # With 1e-5 the total is 1.24e-4 and 0.2 N*m gives ~1600 rad/s^2,
            # i.e. 200 rad/s in ~125 ms, which is what a real prop does.
            armature={
                "leftPropeller": 1.0e-5,
                "rightPropeller": 1.0e-5,
            },
        ),
    },
)
