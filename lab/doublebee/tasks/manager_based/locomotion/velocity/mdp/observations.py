# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


##
# Custom observation functions for DoubleBee
##


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame (body frame).
    
    Returns 3D linear velocity [vx, vy, vz] in the robot's body coordinate frame.
    Shape: (num_envs, 3)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_lin_vel_x(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity X-component in the asset's root frame.
    
    Returns forward/backward velocity in robot's body frame.
    Shape: (num_envs, 1)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0].unsqueeze(-1)


def base_lin_vel_y(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity Y-component in the asset's root frame.
    
    Returns lateral (left/right) velocity in robot's body frame.
    Shape: (num_envs, 1)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 1].unsqueeze(-1)


def base_lin_vel_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity Z-component in the asset's root frame.
    
    Returns vertical (up/down) velocity in robot's body frame.
    Shape: (num_envs, 1)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 2].unsqueeze(-1)


def height_scan(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.
    
    Returns the height differences between sensor position and terrain hit points.
    For a 6x6 grid, this returns 36 height values.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for the height scanner sensor
        offset: Offset to subtract from heights (default: 0.5m)
        
    Returns:
        Height scan tensor. Shape: (num_envs, num_rays) - For 6x6 grid: (N, 36)
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # Height = sensor_z_position - hit_point_z - offset
    # sensor.data.pos_w is the sensor origin position
    # sensor.data.ray_hits_w[..., 2] are the Z-coordinates of hit points
    return sensor.data.pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2] - offset


def wheel_velocities(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Wheel joint velocities for DoubleBee.
    
    Returns velocities of left and right wheels.
    Shape: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get wheel joint indices
    wheel_indices = []
    for joint_name in ["leftWheel", "rightWheel"]:
        joint_ids = asset.find_joints(joint_name)[0]
        # Safely convert to list (handle both tensor and list)
        if isinstance(joint_ids, torch.Tensor):
            wheel_indices.extend(joint_ids.cpu().tolist())
        elif isinstance(joint_ids, (list, tuple)):
            wheel_indices.extend(list(joint_ids))
        else:
            # Single value or other type
            wheel_indices.append(int(joint_ids))
    
    if len(wheel_indices) > 0:
        return asset.data.joint_vel[:, wheel_indices]
    else:
        # Fallback: return all joint velocities
        return asset.data.joint_vel


def servo_positions(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Propeller servo joint positions for DoubleBee.
    
    Returns positions of left and right propeller servos (tilt angles).
    Shape: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get servo joint indices
    servo_indices = []
    for joint_name in ["leftPropellerServo", "rightPropellerServo"]:
        joint_ids = asset.find_joints(joint_name)[0]
        # Safely convert to list (handle both tensor and list)
        if isinstance(joint_ids, torch.Tensor):
            servo_indices.extend(joint_ids.cpu().tolist())
        elif isinstance(joint_ids, (list, tuple)):
            servo_indices.extend(list(joint_ids))
        else:
            # Single value or other type
            servo_indices.append(int(joint_ids))
    
    if len(servo_indices) > 0:
        return asset.data.joint_pos[:, servo_indices]
    else:
        # Fallback: return all joint positions
        return asset.data.joint_pos


def propeller_velocities(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Propeller joint velocities for DoubleBee.
    
    Returns velocities of left and right propellers (rotation speeds).
    Shape: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get propeller joint indices
    propeller_indices = []
    for joint_name in ["leftPropeller", "rightPropeller"]:
        joint_ids = asset.find_joints(joint_name)[0]
        # Safely convert to list (handle both tensor and list)
        if isinstance(joint_ids, torch.Tensor):
            propeller_indices.extend(joint_ids.cpu().tolist())
        elif isinstance(joint_ids, (list, tuple)):
            propeller_indices.extend(list(joint_ids))
        else:
            # Single value or other type
            propeller_indices.append(int(joint_ids))
    
    if len(propeller_indices) > 0:
        return asset.data.joint_vel[:, propeller_indices]
    else:
        # Return empty tensor if not found
        return torch.zeros((env.num_envs, 2), device=env.device)


def wheel_contact(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact detection for DoubleBee wheels (per-wheel).
    
    Detects if each wheel is in contact with the ground by checking contact forces.
    Returns separate binary values for left and right wheels.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor with body_names for wheels
        threshold: Force threshold in Newtons to consider as contact (default: 1.0N)
    
    Returns:
        Binary contact indicators. Shape: (num_envs, 2)
        - [left_wheel_contact, right_wheel_contact]
        - 1.0 = wheel in contact with ground
        - 0.0 = wheel not touching ground (airborne)
    
    Example:
        [1.0, 1.0] = both wheels on ground (normal driving)
        [0.0, 0.0] = both wheels airborne (jumping/flying)
        [1.0, 0.0] = only left wheel touching (tipped right)
        [0.0, 1.0] = only right wheel touching (tipped left)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces on all bodies
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # Get the articulation to find wheel body indices
    asset: Articulation = env.scene["robot"]
    
    # Find left and right wheel body indices
    left_wheel_ids = asset.find_bodies("leftWheel")[0]
    right_wheel_ids = asset.find_bodies("rightWheel")[0]
    
    # Extract forces for each wheel
    # Shape: (num_envs, history_length, 3) for each wheel
    if len(left_wheel_ids) > 0 and len(right_wheel_ids) > 0:
        left_wheel_forces = net_contact_forces[:, :, left_wheel_ids[0], :]
        right_wheel_forces = net_contact_forces[:, :, right_wheel_ids[0], :]
        
        # Compute force magnitudes: sqrt(fx^2 + fy^2 + fz^2)
        left_force_mags = torch.norm(left_wheel_forces, dim=-1)   # (num_envs, history_length)
        right_force_mags = torch.norm(right_wheel_forces, dim=-1)  # (num_envs, history_length)
        
        # Get maximum force over history for each wheel
        left_max_force = torch.max(left_force_mags, dim=1)[0]   # (num_envs,)
        right_max_force = torch.max(right_force_mags, dim=1)[0]  # (num_envs,)
        
        # Binary contact for each wheel
        left_contact = (left_max_force > threshold).float()
        right_contact = (right_max_force > threshold).float()
        
        # Stack into shape (num_envs, 2)
        wheel_contacts = torch.stack([left_contact, right_contact], dim=-1)
    else:
        # Fallback: return zeros if wheels not found
        wheel_contacts = torch.zeros((env.num_envs, 2), device=env.device)
    
    return wheel_contacts


##
# Observation Configuration
##



def action_history(env, history_length: int = 5):
    """Last `history_length` actions, NEWEST FIRST, flattened.

    Added 2026-08-26. The policy previously saw one step of action history, and
    that is not enough on this machine.

    MEASURED, hw_v18.csv, across five engaged segments:
        lean -> wheel_action        lag  0 ticks (  0 ms), r = -0.75..-0.80
        wheel_des -> wheel_meas     lag 15 ticks (300 ms), r = +0.74..+0.84

    State estimation is not the problem -- the policy reacts to attitude with no
    measurable lag. The WHEELS are the problem: 300 ms from command to response,
    dominated by the 43 rad/s^2 acceleration limit rather than transport delay.
    Against a 102 ms fall time constant, a policy that cannot see what it has
    already commanded and not yet received is flying blind about half its own
    dynamics. That is why it balances (small corrections) but falls when it
    commits to forward motion (a sustained acceleration whose pitch reaction
    arrives 300 ms later).

    Five steps is 100 ms -- the fast part of the wheel response -- for 4*5 = 20
    dimensions instead of 4. Longer would cover more of the 300 ms but the tail
    is smooth and largely redundant.

    ORDERING IS EXPLICIT AND MUST MATCH DEPLOYMENT:
        [a(t-1) dims..., a(t-2) dims..., ..., a(t-N) dims...]
    i.e. newest first. Written by hand rather than using IsaacLab's
    ObsTerm(history_length=...) precisely so the convention is pinned here and
    db_inference.py can be made to agree by construction. A silent ordering
    mismatch between sim and hardware is invisible in every log we produce.
    """
    act = env.action_manager.action
    n, d = act.shape
    if (not hasattr(env, "_act_hist")) or tuple(env._act_hist.shape) != (n, history_length, d):
        env._act_hist = torch.zeros(n, history_length, d, device=act.device)
        env._act_hist_step = -1

    # This function is called once per OBSERVATION GROUP, and there are two
    # (policy and value). Shifting on every call would advance history twice per
    # environment step. Gate on the step counter so the buffer advances once.
    step = int(getattr(env, "common_step_counter", 0))
    if step != env._act_hist_step:
        env._act_hist_step = step
        env._act_hist = torch.roll(env._act_hist, 1, dims=1)
        env._act_hist[:, 0] = act
        # Freshly reset environments must not inherit the previous episode's
        # commands -- that history describes a robot that no longer exists.
        elb = getattr(env, "episode_length_buf", None)
        if elb is not None:
            fresh = elb == 0
            if bool(fresh.any()):
                env._act_hist[fresh] = 0.0
    return env._act_hist.reshape(n, -1)


@configclass
class ObservationsCfg:
    """Observation specifications for DoubleBee robot.
    
    This configuration defines all observations available to the RL policy.
    DoubleBee has 6 joints total:
    - 2 wheels (leftWheel, rightWheel)
    - 2 servos (leftPropellerServo, rightPropellerServo) 
    - 2 propellers (leftPropeller, rightPropeller)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
        Observation Space Components:
        1. Wheel velocities (2) - Ground locomotion speed
        2. Servo positions (2) - Propeller tilt angles
        3. Propeller velocities (2) - Propeller rotation speeds
        4. Base linear velocity (3) - Robot velocity in body frame [vx, vy, vz]
        5. Base angular velocity (3) - Robot rotation rates [wx, wy, wz]
        6. Base orientation (3) - Projected gravity vector (encodes roll/pitch)
        7. Height scan (36) - 6x6 elevation map around robot
        8. Wheel-ground contact (2) - Binary per wheel: [left, right] (1.0=contact, 0.0=airborne)
        9. Velocity commands (3) - Desired velocities [vx, vy, wz]
        10. Last actions (N) - Previous control actions
        
        Total observations: ~59+ dimensions (exact count depends on action space)
        """

        # ========================================
        # 1. Joint States (DoubleBee-specific)
        # ========================================
        
        # Wheel velocities - Important for ground contact and locomotion
        wheel_vel = ObsTerm(
            func=wheel_velocities,
            scale=0.05,  # Scale down wheel velocities (typ. 0-200 rad/s)
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        
        # RE-ENABLED 2026-08-26. Both were commented out, so the policy commanded
        # servo angle and propeller speed while observing NEITHER.
        #
        # Why that is fatal for this machine specifically: the propellers' whole
        # contribution to balance is T*L_prop*sin(theta - psi), where psi is the
        # thrust axis angle off world-vertical. At psi = theta (props body-fixed)
        # the restoring moment is exactly ZERO -- thrust does nothing at all. So
        # psi is not a detail, it is the difference between the props helping and
        # the props being dead weight, and the policy could not see it.
        #
        # It could not infer it either: this TQC actor is feedforward
        # (is_recurrent = False), so its only memory is `actions` -- ONE step of
        # command history. The servo carries 40-100 ms of delay (min_delay 2,
        # max_delay 5) and a 2.0 rad/s slew limit, so the achieved angle lags the
        # command by 2-5 steps. A memoryless policy cannot integrate that.
        #
        # Cost: obs 36 -> 40. The deployment script MUST be updated to match --
        # see the layout note in the class docstring below.
        servo_pos = ObsTerm(
            func=servo_positions,
            # Already in [-0.785, 0.785] (+/-45 deg, SERVO_POS_LIMIT_RAD), which
            # is the same order as gravity's 1.15 -- no scaling needed.
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # Propeller speed. Terminal is ~158 rad/s at damping 0.015, so scale 0.01
        # puts this channel at 0-1.58, comparable to the other inputs. The 0-600
        # assumption in the original comment predates the actuator fixes.
        propeller_vel = ObsTerm(
            func=propeller_velocities,
            scale=0.01,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # ========================================
        # 2. Base State (Robot body motion)
        # ========================================
        
        # Linear velocity in body frame - Essential for velocity tracking
        base_lin_vel = ObsTerm(
            func=base_lin_vel,
            scale=2.0,  # Emphasize linear velocity for tracking
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        # Angular velocity in body frame -- the RATE term of the balance loop.
        #
        # scale was deliberately set to 0.10 to keep projected_gravity leading on
        # attitude. That intent is RIGHT and is preserved below: gravity still
        # dominates in normal operation. The problem was the noise, not the goal.
        #
        # Measured on hardware (transfer_clamped_props.csv, obs_5:7) against the
        # old scale=0.10 / noise=+/-0.1:
        #
        #     signal p50  0.020        noise std  0.058   ->  SNR 0.35
        #     signal p90  0.093                            ->  SNR 1.6
        #     hard tip, 3.8 rad/s -> 0.38                  ->  SNR 6.6
        #
        # The noise was 3x the typical signal, so for most of every episode this
        # channel sat BELOW its own noise floor. Attenuating a signal and
        # DELETING one are different things, and 0.10 with +/-0.1 noise did the
        # second. An ablation on the 5-action checkpoint confirms it: zeroing
        # this whole block moves the action by 0.0029, against 0.11 for gravity
        # and 0.93 for the command. The policy was not using it at all.
        #
        # For an inverted pendulum that is fatal. Angular velocity is the only
        # observation that reports a fall while the lean is still small enough to
        # recover from. Without it the policy can only respond once the attitude
        # error is already large -- exactly the hardware behaviour: no visible
        # correction, then a tip matching passive pendulum dynamics to the sample
        # interval (13.5 deg -> 70 deg in 0.36 s, predicted 0.36 s).
        #
        # Gravity was never the thing that needed help: at scale 1.15 with
        # +/-0.05 noise its SNR is ~20, so it leads on attitude regardless.
        #
        # scale=0.5 keeps that ordering -- in normal operation ang_vel reads ~0.10
        # against gravity's ~1.15, so gravity is still 10x larger -- while lifting
        # SNR at p50 from 0.35 to ~5 so the channel carries information at all.
        # During a fast tip it reaches ~1.9, comparable to gravity, which is the
        # one moment the rate SHOULD lead. Nothing exceeds ~2, so no channel
        # becomes an outlier. Noise stays nonzero: the real ang_vel comes from
        # quaternion differencing at 100 Hz and is genuinely noisy.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.5,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        
        # Projected gravity - Encodes robot orientation (roll, pitch)
        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=1.15,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # ========================================
        # 3. Terrain Perception (Elevation map)
        # ========================================
        
        # Height scan - 4x4 grid showing terrain elevation around robot
        height_scan = ObsTerm(
            func=height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.0,  # No offset, raw heights
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),  # Clip to reasonable height range
        )
        
        # Wheel contact - Binary indicator if wheels touch ground
        wheel_ground_contact = ObsTerm(
            func=wheel_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "threshold": 1.0,  # 1.0 Newton threshold
            },
        )

        # ========================================
        # 4. Command (Desired behavior)
        # ========================================
        
        # Velocity commands - What the robot should be doing
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # ========================================
        # 5. Action History (For temporal consistency)
        # ========================================
        
        # Action HISTORY, not just the last one. See action_history() above for
        # the measurement that motivates it: 300 ms of wheel actuator lag against
        # a 102 ms fall time constant, with only one step of command history the
        # policy cannot know what it has already asked for.
        # history_length SELECTS THE CHECKPOINT GENERATION:
        #     1 -> obs 40, loads the 2026-08-27_02-28 run (the one that CLIMBS)
        #     5 -> obs 56, loads the 2026-08-27_04-12 run
        # Set to 1 to play or deploy the obs-40 checkpoint; the state_dict is
        # shaped by this and a mismatch is a hard load error, not a silent one.
        actions = ObsTerm(func=action_history, params={"history_length": 1})

        def __post_init__(self):
            """Post-initialization configuration."""
            self.enable_corruption = True  # Add observation noise during training
            self.concatenate_terms = True  # Flatten all observations into single vector

    # ========================================
    # Observation Groups
    # ========================================
    
    # Policy observations - Used by actor network
    policy: PolicyCfg = PolicyCfg()

    # Value observations - Used by critic network (same as policy for now)
    value: PolicyCfg = PolicyCfg()


@configclass
class ObservationsCfgNoHeightScan(ObservationsCfg):
    """Observation config without height scan (e.g. for inverted-pendulum / flat same-level tasks).

    Use when the scene has no height_scanner or when elevation perception is not desired.
    """

    @configclass
    class PolicyCfgNoHeightScan(ObservationsCfg.PolicyCfg):
        """Policy observations without height_scan term."""

        height_scan = None  # Disable elevation map

    policy: PolicyCfgNoHeightScan = PolicyCfgNoHeightScan()
    value: PolicyCfgNoHeightScan = PolicyCfgNoHeightScan()