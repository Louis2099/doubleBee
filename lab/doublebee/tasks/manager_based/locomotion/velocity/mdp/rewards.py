# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


def velocity_direction_alignment(env) -> torch.Tensor:
    """Reward for aligning robot's XY velocity direction with command's XY velocity direction.
    
    Uses cosine similarity (dot product of normalized vectors) to measure alignment,
    scaled by the robot's actual velocity magnitude relative to the command magnitude.
    Returns values in range [-1, 1], where:
    - 1 = perfectly aligned AND moving at or above commanded speed
    - -1 = opposite direction AND moving at or above commanded speed
    - Values scale down toward 0 when velocity magnitude is low
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled alignment reward per environment [num_envs]
    """
    robot = env.scene["robot"]
    cmd_manager = env.command_manager
    
    # Get current robot velocity in body frame (XY components)
    robot_vel_xy = robot.data.root_lin_vel_b[:, :2]  # [num_envs, 2]
    
    # Get command velocity (XY components)
    vel_cmd = cmd_manager.get_command("base_velocity")  # [num_envs, 4]
    cmd_vel_xy = vel_cmd[:, :2]  # [num_envs, 2]
    
    # Compute magnitudes
    robot_vel_mag = torch.norm(robot_vel_xy, dim=1)  # [num_envs]
    cmd_vel_mag = torch.norm(cmd_vel_xy, dim=1)  # [num_envs]
    
    # Normalize vectors (handle zero velocity case)
    robot_vel_norm = robot_vel_xy / (robot_vel_mag.unsqueeze(1) + 1e-6)  # [num_envs, 2]
    cmd_vel_norm = cmd_vel_xy / (cmd_vel_mag.unsqueeze(1) + 1e-6)  # [num_envs, 2]
    
    # Compute cosine similarity (dot product of normalized vectors)
    # This gives alignment in range [-1, 1]
    alignment = torch.sum(robot_vel_norm * cmd_vel_norm, dim=1)  # [num_envs]
    
    # Clamp alignment to [-1, 1] for numerical stability
    alignment = torch.clamp(alignment, min=-0.5, max=1.0) #NOTE: -0.5 to tolerate some misalignment
    
    # Scale by velocity magnitude: use ratio of actual velocity to a reference velocity
    # Since cmd_vel is normalized (magnitude = 1.0), we use a fixed reference velocity (e.g., 2.0 m/s)
    # to tolerate higher velocities and scale the reward appropriately
    reference_velocity = 2.0  # Reference velocity in m/s to tolerate higher speeds
    velocity_scale = robot_vel_mag / reference_velocity  # [num_envs]
    velocity_scale = torch.clamp(velocity_scale, min=0.0, max=1.0)  # Cap at 1.0 to keep reward in [-1, 1]
    
    # Scale alignment by velocity magnitude factor
    # Result: alignment * velocity_scale is in [-1, 1] range
    scaled_alignment = alignment * velocity_scale
    
    return scaled_alignment


def reach_terrain_target(env) -> torch.Tensor:
    """Reward for reaching terrain target positions.
    
    Uses the same target that the command has selected for the current episode.
    Computes distance from robot base to the command's current target.
    Uses exponential reward: exp(-distance² / scale²) to encourage getting closer.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Target reaching reward per environment [num_envs]
    """
    robot = env.scene["robot"]
    
    # Get robot base position in world frame (XY only)
    robot_pos_w = robot.data.root_pos_w[:, :2]  # [num_envs, 2]
    
    # Get the command term to access its selected target
    cmd_manager = env.command_manager
    if "base_velocity" not in cmd_manager._terms:
        # Command not found, return zero reward
        return torch.zeros(robot.num_instances, device=robot.device)
    
    command_term = cmd_manager._terms["base_velocity"]
    
    # Check if this is TerrainTargetDirectionCommand with current_targets_w
    if not hasattr(command_term, "current_targets_w"):
        # Not using terrain target command, fall back to finding nearest target
        terrain = env.scene["terrain"]
        if "target" not in terrain.flat_patches:
            return torch.zeros(robot.num_instances, device=robot.device)
        
        target_patches = terrain.flat_patches["target"]
        terrain_levels = terrain.terrain_levels
        terrain_types = terrain.terrain_types
        env_origins = env.scene.env_origins
        
        level_indices = terrain_levels
        type_indices = terrain_types
        num_patches = target_patches.shape[2]
        targets_relative = target_patches[level_indices, type_indices, :, :]
        targets_world = targets_relative + env_origins.unsqueeze(1)
        targets_xy = targets_world[:, :, :2]
        
        robot_pos_xy_expanded = robot_pos_w.unsqueeze(1)
        distances = torch.norm(targets_xy - robot_pos_xy_expanded, dim=2)
        min_distances = torch.min(distances, dim=1)[0]
    else:
        # Use the command's selected target (aligned with command)
        current_targets_w = command_term.current_targets_w  # [num_envs, 3]
        current_targets_xy = current_targets_w[:, :2]  # [num_envs, 2]
        
        # Compute distance from robot to command's selected target
        distances_xy = robot_pos_w - current_targets_xy  # [num_envs, 2]
        min_distances = torch.norm(distances_xy, dim=1)  # [num_envs]
    
    # Exponential reward: exp(-distance² / scale²)
    # Scale = 2.0 means reward drops to ~0.6 at 2m, ~0.13 at 4m
    scale = 2.0
    rewards = torch.exp(-(min_distances ** 2) / (scale ** 2))
    
    return rewards


def terminal_reward_goal_reached(env) -> torch.Tensor:
    """Terminal reward for successfully reaching the goal.
    
    Returns a positive reward only when the robot reaches the goal (goal_reached constraint is active).
    This is a terminal reward, meaning it's only given when the episode ends due to goal completion.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Terminal reward per environment [num_envs]
        - Positive value (e.g., 10.0) if goal reached
        - 0.0 otherwise
    """
    # Import constraint function to check if goal is reached
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import goal_reached
    
    # Check if goal is reached (constraint is active)
    goal_reached_mask = goal_reached(env, distance_threshold=0.2)  # [num_envs]
    
    # Return positive reward only for environments where goal is reached
    reward_value = 10.0  # Positive terminal reward
    rewards = goal_reached_mask * reward_value
    
    return rewards


def terminal_reward_propeller_collision(env) -> torch.Tensor:
    """Terminal reward (penalty) for propeller collision.
    
    Returns a negative reward only when propellers collide (propeller_collision constraint is active).
    This is a terminal reward, meaning it's only given when the episode ends due to collision.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Terminal penalty per environment [num_envs]
        - Negative value (e.g., -10.0) if propeller collision occurred
        - 0.0 otherwise
    """
    # Import constraint function to check if collision occurred
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import propeller_collision
    from isaaclab.managers import SceneEntityCfg
    
    # Check if propeller collision occurred (constraint is active)
    collision_mask = propeller_collision(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces"),
        threshold=1.0
    )  # [num_envs]
    
    # Return negative reward only for environments where collision occurred
    penalty_value = -10.0  # Negative terminal reward (penalty)
    rewards = collision_mask * penalty_value
    
    return rewards


def terminal_reward_robot_out_of_bounds(env) -> torch.Tensor:
    """Terminal reward (penalty) for robot being thrown out of bounds.
    
    Returns a negative reward only when robot is out of bounds (robot_out_of_bounds constraint is active).
    This is a terminal reward, meaning it's only given when the episode ends due to being out of bounds.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Terminal penalty per environment [num_envs]
        - Negative value (e.g., -10.0) if robot is out of bounds
        - 0.0 otherwise
    """
    # Import constraint function to check if robot is out of bounds
    from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.constraints import robot_out_of_bounds
    
    # Check if robot is out of bounds (constraint is active)
    out_of_bounds_mask = robot_out_of_bounds(
        env,
        max_height=3.0,
        max_xy_distance=6.0
    )  # [num_envs]
    
    # Return negative reward only for environments where robot is out of bounds
    penalty_value = -10.0  # Negative terminal reward (penalty)
    rewards = out_of_bounds_mask * penalty_value
    
    return rewards


def penalize_propeller_efficiency(env) -> torch.Tensor:
    """Penalty for excessive propeller speeds to encourage efficiency.
    
    Computes a penalty based on propeller joint velocities.
    Scales the penalty to [-1, 0] range using e^(-x) - 1 transformation.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for higher propeller speeds
        - Values closer to 0 for low/no propeller speeds
    """
    robot = env.scene["robot"]
    
    # Get propeller joint velocities
    try:
        left_propeller_idx = robot.joint_names.index("leftPropeller")
        right_propeller_idx = robot.joint_names.index("rightPropeller")
        
        propeller_velocities = robot.data.joint_vel[:, [left_propeller_idx, right_propeller_idx]]  # [num_envs, 2]
        
        # Compute raw penalty magnitude (sum of squared velocities)
        raw_penalty_magnitude = torch.sum(torch.square(propeller_velocities), dim=1)  # [num_envs]
        
        # Scale to [-1, 0] using e^(-x) - 1 transformation
        scaled_penalty = torch.exp(-raw_penalty_magnitude) - 1.0  # [num_envs]
        
        return scaled_penalty
    except (ValueError, IndexError):
        # If propeller joints not found, return zero penalty
        return torch.zeros(robot.num_instances, device=robot.device)


def penalize_facing_direction_mismatch(env) -> torch.Tensor:
    """Penalty for mismatch between robot facing direction and target direction.
    
    Reads the angle error directly from vel_command_b[:, 2], which represents the normalized
    angle error between robot's facing direction and the direction to target.
    The angle error is normalized to [-1, 1] range (where 0 = aligned, ±1 = max error).
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for large angle errors (robot not facing target)
        - Values closer to 0 for small/no angle errors (robot facing target)
    """
    cmd_manager = env.command_manager
    
    # Get velocity command which contains angle error in vel_command_b[:, 2]
    vel_cmd = cmd_manager.get_command("base_velocity")  # [num_envs, 3] or [num_envs, 4]
    
    # Extract angle error (normalized to [-1, 1] range)
    # This is the mismatch between robot facing direction and target direction
    if vel_cmd.shape[1] >= 3:
        angle_error_normalized = vel_cmd[:, 2]  # [num_envs] - normalized angle error
    else:
        # Fallback: return zero penalty if command doesn't have ang_vel_z
        robot = env.scene["robot"]
        return torch.zeros(robot.num_instances, device=robot.device)
    
    # Use absolute value to get magnitude of angle error (range: [0, 1])
    angle_error_magnitude = -torch.abs(angle_error_normalized)  # [num_envs]
    
    
    return angle_error_magnitude


def penalize_angular_velocity(env) -> torch.Tensor:
    """Penalty for excessive angular velocity to discourage spinning and unwanted rotations.
    
    Computes a penalty based on weighted sum of roll, pitch, and yaw angular velocities.
    Scales the penalty to [-1, 0] range using e^(-x) - 1 transformation.
    
    Args:
        env: The environment instance
        
    Returns:
        torch.Tensor: Scaled penalty per environment [num_envs] in range [-1, 0]
        - Values closer to -1 for higher angular velocities
        - Values closer to 0 for low/no rotation
    """
    robot = env.scene["robot"]
    
    # Get angular velocity in body frame [num_envs, 3] - (roll, pitch, yaw)
    ang_vel = robot.data.root_ang_vel_b  # [num_envs, 3]
    
    # Extract individual components (using absolute values)
    #NOTE: the forward axis is Y for this robot
    ang_vel_pitch = torch.abs(ang_vel[:, 0])   # [num_envs] - roll (x-axis)
    ang_vel_roll = torch.abs(ang_vel[:, 1]) # [num_envs] - pitch (y-axis)
    ang_vel_yaw = torch.abs(ang_vel[:, 2])    # [num_envs] - yaw (z-axis)
    
    # Weights for each component (can be adjusted based on importance)
    # Yaw is typically most important for spinning, but all rotations should be penalized
    weight_roll = 5.0
    weight_pitch = 3.0
    weight_yaw = 2.0
    
    # Compute weighted sum of squared angular velocities
    # Using squared values to strongly penalize high rotation rates
    weighted_sum = (
        weight_roll * (ang_vel_roll ** 2) +
        weight_pitch * (ang_vel_pitch ** 2) +
        weight_yaw * (ang_vel_yaw ** 2)
    )  # [num_envs]
    
    # Scale to [-1, 0] using e^(-x) - 1 transformation
    scaled_penalty = torch.exp(-weighted_sum) - 1.0  # [num_envs]
    
    return scaled_penalty


@configclass
class RewardsCfg:
    """Reward specifications for DoubleBee velocity tracking task."""

    # ========== Velocity Command Tracking Rewards ==========
    
    # track_lin_vel_xy = RewTerm(
    #     func=lambda env: torch.exp(
    #         -torch.sum(
    #             torch.square(
    #                 env.scene["robot"].data.root_lin_vel_b[:, :2] 
    #                 - env.command_manager.get_command("base_velocity")[:, :2]
    #             ), 
    #             dim=1
    #         )
    #     ),
    #     weight=1.0,
    # )
    # """Horizontal linear velocity tracking (x, y). Exponential reward: exp(-||v_xy - v_cmd_xy||²)"""
    

    # track_lin_vel_z = RewTerm(
    #     func=lambda env: torch.exp(
    #         -torch.square(
    #             env.scene["robot"].data.root_lin_vel_b[:, 2] 
    #             - env.command_manager.get_command("base_velocity")[:, 2]
    #         )
    #     ),
    #     weight=1.0,
    # )
    # """Vertical linear velocity tracking (z). Exponential reward: exp(-||v_z - v_cmd_z||²)"""

    # track_ang_vel_z = RewTerm(
    #     func=lambda env: torch.exp(
    #         -torch.square(
    #             env.scene["robot"].data.root_ang_vel_b[:, 2] 
    #             - env.command_manager.get_command("base_velocity")[:, 3]
    #         )
    #     ),
    #     weight=0.5,
    # )
    # """Yaw angular velocity tracking. Exponential reward: exp(-||ω_z - ω_cmd_z||²)"""

    # ========== Locomotion Direction Rewards ==========
    
    velocity_direction_alignment = RewTerm(
        func=velocity_direction_alignment,
        weight=0.25,
    )
    """Reward for aligning robot's XY velocity direction with command's XY velocity direction.
    Uses cosine similarity: reward = dot(normalize(v_robot_xy), normalize(v_cmd_xy)).
    Range: [-1, 1] where 1 = perfectly aligned, -1 = opposite direction."""
    
    # ========== Target Reaching Rewards ==========
    
    reach_terrain_target = RewTerm(
        func=reach_terrain_target,
        weight=1.0,
    )
    """Reward for reaching terrain target positions.
    Computes distance to nearest target patch from terrain.flat_patches['target'].
    Uses exponential reward: exp(-distance² / scale²) with scale=2.0m."""

    #========== Efficiency Rewards ==========
    
    propeller_efficiency = RewTerm(
        func=penalize_propeller_efficiency,
        weight=0.005,
    )
    """Penalty for excessive propeller speeds to encourage efficiency.
    Computes penalty based on propeller joint velocities, scaled to [-1, 0] using e^(-x) - 1.
    Since thrust ∝ ω², high speeds are inefficient."""

    # ========== Stability Rewards ==========
    
    penalize_facing_mismatch = RewTerm(
        func=penalize_facing_direction_mismatch,
        weight=0.03,
    )
    """Penalty for mismatch between robot facing direction and target direction.
    Reads angle error directly from vel_command_b[:, 2] (normalized angle error in [-1, 1]).
    Penalizes when robot is not facing the target, scaled to [-1, 0] using e^(-x) - 1."""
    
    penalize_rotation = RewTerm(
        func=penalize_angular_velocity,
        weight=0.05,
    )
    """Penalty for excessive angular velocity to discourage spinning.
    Computes penalty based on squared angular velocity magnitude, with emphasis on yaw rotation.
    Strongly penalizes high rotation rates to encourage stable, controlled movement."""

    
    
    # ========== Terminal Rewards ==========
    
    terminal_goal_reached = RewTerm(
        func=terminal_reward_goal_reached,
        weight=1.0,
    )
    """Terminal reward for successfully reaching the goal.
    Returns +10.0 when robot reaches the goal (episode ends due to goal_reached constraint).
    This is a positive terminal reward that encourages task completion."""
    
    terminal_propeller_collision = RewTerm(
        func=terminal_reward_propeller_collision,
        weight=1.0,
    )
    """Terminal reward (penalty) for propeller collision.
    Returns -10.0 when propellers collide (episode ends due to propeller_collision constraint).
    This is a negative terminal reward that penalizes unsafe behavior."""
    
    terminal_robot_out_of_bounds = RewTerm(
        func=terminal_reward_robot_out_of_bounds,
        weight=1.0,
    )
    """Terminal reward (penalty) for robot being thrown out of bounds.
    Returns -10.0 when robot height > 3m or XY distance > 6m from origin (episode ends due to robot_out_of_bounds constraint).
    This is a negative terminal reward that penalizes when the robot is thrown away from the scene."""

    

    # action_smoothness = RewTerm(
    #     func=lambda env: -torch.sum(torch.square(env.action_manager.action), dim=1),
    #     weight=0.001,
    # )
    # """Penalize large action magnitudes to encourage smooth, energy-efficient control."""
