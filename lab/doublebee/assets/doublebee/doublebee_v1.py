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
            effort_limit=0.51,  # RESTORED from model_3500_params/env.yaml. Raised to 0.72
            # on 2026-08-27 to match the measured hardware p90 of 61 rad/s^2, which
            # is a defensible change -- but the run that works used 0.51.
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
            # 2.0 -> 10.0 on 2026-08-27. AT 2.0 THE SERVO CANNOT BALANCE AT ALL.
            #
            # Three independent numbers in this repo already said 10 rad/s and
            # this one line disagreed with all of them:
            #   assets/.../DoubleBee/config.yaml  velocity_limit: 10.0 (this joint)
            #   db_inference.py:1491,1624,2192    "a hobby servo does 86 deg in
            #                                      ~0.15 s" = 10 rad/s, measured
            #   db_inference.py servo_hold_slew_rad_s -- deployment deliberately
            #                                      REFUSES to rate-limit attitude
            #                                      hold to sim's 2.0 because it
            #                                      "loses the race against the fall"
            #
            # Time to traverse the +/-45 deg working range:
            #     2.0 rad/s -> 393 ms = 3.9 tau      10 rad/s -> 79 ms = 0.8 tau
            # against a 102 ms pendulum time constant and a ~600 ms fall to the
            # 70 deg termination. At 2.0 the servo spends the whole fall in
            # transit, so it is not an attitude actuator -- it is a slow trim.
            #
            # CONSEQUENCE, which is what sent us looking: the policy cannot use
            # the servo to catch a fall, so it spends it on the one job that IS
            # reachable at 2 rad/s -- a quasi-static forward tilt for propulsion,
            # paid by reward_progress_to_target (weight 10.0). That produces the
            # asymmetry observed on hardware: the servo appears to help a
            # BACKWARD fall (the propulsion tilt is already restoring, no motion
            # needed) and never a FORWARD one (needs a fast reversal it cannot
            # execute). Forward is the common failure.
            #
            # reward_props_upright (weight 5.0) has been paying for world-vertical
            # thrust this whole time; it was simply not ACHIEVABLE during a fall.
            #
            # REVERTED TO 2.0 on 2026-08-27, SAME DAY. The reasoning above is
            # still believed correct -- 2.0 does not match the real servo -- but
            # raising it MADE THE ROBOT WORSE, and the measurement says why.
            #
            # The policy inherited from model_3500 never learned this channel,
            # because at 2.0 rad/s the actuator filtered whatever it commanded.
            # Measured after the raise (iteration 3622):
            #     [SERVOLOOP] lag1_ac = +0.017     <- servo command is WHITE NOISE
            #                 corr(spos,sact) = +0.014
            #     [SERVOASYM] FWD corr -0.017 | BACK corr +0.080   <- no balance
            #                                                        work either way
            # i.e. the servo head is an untrained output emitting noise. Raising
            # velocity_limit did not give the policy a balance actuator; it gave
            # an untrained noise source 5x the authority. Play confirms it: this
            # checkpoint cannot balance, while model_3500 at 2.0 climbs cleanly.
            #
            # DO NOT re-raise this on a RESUMED checkpoint. It is only safe from
            # a FRESH run, where the servo head is trained against the real speed
            # from the start. Left here as the first thing to try when there is
            # time for a from-scratch run.
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
            # RESTORED to 5.0 N*m. Lowering it to 0.2 then 0.6 on 2026-08-25 was
            # a mistake and cost a 520-iteration run.
            #
            # effort_limit is NOT just spin-up torque -- aerodynamic drag on the
            # propeller grows as omega^2, so the steady-state torque is what sets
            # the TOP SPEED, and therefore the thrust ceiling:
            #
            #   0.2 N*m -> terminal  40 rad/s ->  6.0 N total
            #   0.6 N*m -> terminal  69 rad/s ->  7.2 N total
            #   5.0 N*m -> terminal 200 rad/s -> 13.8 N total
            #
            # The static-stability threshold is 7.17 N (m*g*L/arm at 3.2182 kg,
            # CoM 101.6 mm, prop arm 447.6 mm). At 0.6 the props physically could
            # not reach it, so no reward could ever teach the policy to balance
            # with them. Measured at iteration 520: joint_vel pinned at 44 rad/s
            # against a target of 339, thrust 3.1 N/prop, terrain_levels collapsed
            # to 0, episode length back down to 33.
            #
            # The slow spin-up that motivated lowering this was never the torque
            # -- it was the inertia: no authored tensor (auto-computed 0.238
            # kg*m^2, 2080x too heavy) plus an armature of 0.01 (88x the real
            # propeller). Both are fixed now, so 5.0 N*m gives both a fast
            # spin-up and the full 200 rad/s ceiling.
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
                # 0.0062 gives a response time constant tau = I/D = 20 ms with the
                # corrected inertia (1.14e-4 + 1e-5 armature). Was 1000, which is
                # meaningless for a propeller: tau would be 1.2e-7 s, far below
                # the 5 ms physics step, so the loop overshot every step and the
                # joint velocity had NO relationship to the command -- measured
                # target +26.5 giving actual -167.8, target +288 giving -169.
                #
                # The old 2080x-too-heavy inertia hid this: the same bang-bang
                # moved velocity only 0.1 rad/s per step instead of 202, so it
                # looked smooth. Fixing the inertia exposed that the damping had
                # never been tuned for a real propeller.
                #
                # Damping also sets the TERMINAL speed, because the prop settles
                # where D*(target - w) equals the aerodynamic drag k*w^2
                # (k = 1.25e-4, from 5.0 N*m holding 200 rad/s at the old damping):
                #
                #   D        terminal w   total thrust   tau = I/D   vs 5 ms step
                #   0.0062      114          9.2 N        20 ms         4.0x
                #   0.015       160         11.6 N         8.3 ms       1.7x
                #   0.025       192         13.3 N         5.0 ms       1.0x
                #
                # 0.0062 was tried first and capped thrust at 6.2 N, BELOW the
                # 7.17 N static-stability threshold -- vertical_thrust_support
                # fell from 0.0361 to 0.0115. 0.015 keeps 11.6 N (1.6x threshold)
                # while tau stays comfortably above the physics step. 0.025 gives
                # more thrust but puts tau exactly at the step, where the loop
                # starts oscillating again.
                "leftPropeller": 0.015,
                "rightPropeller": 0.015,
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
