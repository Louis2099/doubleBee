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
        )
        
        # Servo positions - Critical for propeller orientation control
        servo_pos = ObsTerm(
            func=servo_positions,
            # No scaling needed, positions are already in [-1.57, 1.57] (±90°)
        )
        
        # Propeller velocities - For thrust generation feedback
        propeller_vel = ObsTerm(
            func=propeller_velocities,
            scale=0.01,  # Scale down high propeller speeds (typ. 0-600 rad/s)
        )

        # ========================================
        # 2. Base State (Robot body motion)
        # ========================================
        
        # Linear velocity in body frame - Essential for velocity tracking
        base_lin_vel = ObsTerm(
            func=base_lin_vel,
            scale=2.0,  # Emphasize linear velocity for tracking
        )
        
        # Angular velocity in body frame - For rotation control
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            scale=0.25,  # Reduce magnitude of angular velocity
        )
        
        # Projected gravity - Encodes robot orientation (roll, pitch)
        base_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        # ========================================
        # 3. Terrain Perception (Elevation map)
        # ========================================
        
        # Height scan - 6x6 grid showing terrain elevation around robot
        height_scan = ObsTerm(
            func=height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.0,  # No offset, raw heights
            },
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
        
        # Last actions - Helps policy understand action effects
        actions = ObsTerm(func=mdp.last_action)

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