# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aerodynamics functions for DoubleBee propeller thrust and drag.

Adapted from IsaacLab's quadcopter example to work with ManagerBasedRLEnv.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import carb  # For logging


def apply_propeller_aerodynamics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    propeller_joint_names: tuple[str, str],
    propeller_body_names: tuple[str, str],
    thrust_coefficient: float,
    drag_coefficient: float,
    max_thrust_per_propeller: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    visualize: bool = False,
    visualize_scale: float = 0.2,
):
    """
    Apply aerodynamic thrust and drag forces to propellers based on their angular velocity.
    
    This function is inspired by IsaacLab's quadcopter implementation:
    - Thrust is proportional to the square of angular velocity: F = k_t * ω²
    - Drag torque opposes rotation: τ = k_d * ω²
    - Forces are applied in world frame to propeller bodies
    
    The thrust direction is determined by the propeller body orientation in world frame.
    For DoubleBee, propellers rotate around Z-axis, so thrust is along Z-axis in local frame.
    
    Args:
        env: The environment instance (provided automatically by event manager)
        asset_cfg: Configuration for the robot asset
        propeller_joint_names: Names of propeller joints (to get angular velocity)
        propeller_body_names: Names of propeller bodies (to apply forces)
        thrust_coefficient: Thrust coefficient (k_t). Typical range: 0.01-1.0
        drag_coefficient: Drag coefficient (k_d). Typical range: 0.001-0.1
        max_thrust_per_propeller: Maximum thrust force per propeller (N)
    
    Note:
        This function is designed for interval events (mode="interval").
        All parameters except 'env' must be provided in the EventTermCfg params dict.
        The function runs every physics step when interval_range_s=(0.0, 0.0).
    """
    # Get the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get propeller joint indices
    propeller_joint_ids = []
    for joint_name in propeller_joint_names:
        try:
            idx = robot.joint_names.index(joint_name)
            propeller_joint_ids.append(idx)
        except ValueError:
            raise ValueError(
                f"Joint '{joint_name}' not found in robot. Available joints: {robot.joint_names}"
            )
    propeller_joint_ids = torch.tensor(propeller_joint_ids, dtype=torch.long, device=robot.device)
    
    # Get propeller body indices
    propeller_body_ids = []
    for body_name in propeller_body_names:
        try:
            idx = robot.body_names.index(body_name)
            propeller_body_ids.append(idx)
        except ValueError:
            raise ValueError(
                f"Body '{body_name}' not found in robot. Available bodies: {robot.body_names}"
            )
    propeller_body_ids = torch.tensor(propeller_body_ids, dtype=torch.long, device=robot.device)
    
    # Get propeller angular velocities (rad/s) - shape: [num_envs, num_propellers]
    propeller_vel = robot.data.joint_vel[:, propeller_joint_ids]
    
    # Calculate thrust force: F = k_t * ω²
    # Using absolute value to always generate positive thrust
    thrust_magnitude = thrust_coefficient * torch.square(propeller_vel)
    
    # Clamp thrust to maximum value
    thrust_magnitude = torch.clamp(thrust_magnitude, max=max_thrust_per_propeller)
    
    # Get propeller body orientations in world frame
    # Shape: [num_envs, num_propellers, 4] (quaternions)
    propeller_quat_w = robot.data.body_quat_w[:, propeller_body_ids, :]
    
    # Define thrust direction in propeller local frame
    # DoubleBee propellers rotate around Z-axis, so thrust is along Z-axis
    # Shape: [num_envs, num_propellers, 3]
    thrust_local = torch.zeros(env.num_envs, len(propeller_joint_ids), 3, device=robot.device)
    thrust_local[:, :, 2] = thrust_magnitude  # Z-axis thrust
    
    # Rotate thrust vector from local frame to world frame
    # Shape: [num_envs, num_propellers, 3]
    thrust_world = quat_rotate(propeller_quat_w, thrust_local)
    
    # Calculate drag torque: τ = -k_d * sign(ω) * ω²
    # Drag opposes rotation
    drag_torque_magnitude = drag_coefficient * torch.square(propeller_vel)
    drag_torque_magnitude = -torch.sign(propeller_vel) * drag_torque_magnitude
    
    # Drag torque is around the rotation axis (Z-axis for DoubleBee)
    # Shape: [num_envs, num_propellers, 3]
    drag_torque_local = torch.zeros(env.num_envs, len(propeller_joint_ids), 3, device=robot.device)
    drag_torque_local[:, :, 2] = drag_torque_magnitude  # Z-axis torque
    
    # Rotate drag torque to world frame
    drag_torque_world = quat_rotate(propeller_quat_w, drag_torque_local)
    
    # Prepare force and torque tensors for all bodies
    # Shape: [num_envs, num_bodies, 3]
    num_bodies = robot.num_bodies
    external_forces = torch.zeros(env.num_envs, num_bodies, 3, device=robot.device)
    external_torques = torch.zeros(env.num_envs, num_bodies, 3, device=robot.device)
    
    # Assign propeller thrust forces and drag torques
    for i, body_id in enumerate(propeller_body_ids):
        external_forces[:, body_id, :] = thrust_world[:, i, :]
        external_torques[:, body_id, :] = drag_torque_world[:, i, :]
    
    # Apply external forces and torques to the robot
    # Following the quadcopter example's approach
    robot.set_external_force_and_torque(
        external_forces, 
        external_torques,
        body_ids=None  # Apply to all bodies (forces are zero except for propellers)
    )

    # Optional: visualize thrust vectors in the viewport
    if visualize:
        try:
            # Lazy-acquire debug draw interface once
            if not hasattr(apply_propeller_aerodynamics, "_dbg_draw"):
                from omni.isaac.debug_draw import _debug_draw
                apply_propeller_aerodynamics._dbg_draw = _debug_draw.acquire_debug_draw_interface()
            dbg = apply_propeller_aerodynamics._dbg_draw

            # Use first environment for visualization to avoid clutter
            env_id = 0
            # Get body positions (world) for propeller bodies
            prop_pos_w = robot.data.body_pos_w[env_id, propeller_body_ids, :]  # [2, 3]
            prop_force_w = thrust_world[env_id]  # [2, 3]

            # Draw an arrow for each propeller
            for i in range(prop_force_w.shape[0]):
                start = prop_pos_w[i].cpu().numpy()
                end = (prop_pos_w[i] + visualize_scale * prop_force_w[i]).cpu().numpy()
                color = (0.1, 0.9, 0.1, 1.0)  # green
                # Some Isaac versions expose draw_arrows, others draw_lines; try arrows first
                if hasattr(dbg, "draw_arrows"):
                    import numpy as np
                    dbg.draw_arrows(
                        starts=np.asarray([start], dtype=np.float32),
                        ends=np.asarray([end], dtype=np.float32),
                        colors=np.asarray([color], dtype=np.float32),
                        arrow_size=3.0,
                        line_thickness=3.0,
                        duration=0.2,
                    )
                elif hasattr(dbg, "draw_lines"):
                    import numpy as np
                    dbg.draw_lines(
                        points=np.asarray([start, end], dtype=np.float32),
                        colors=np.asarray([color, color], dtype=np.float32),
                        thickness=3.0,
                        duration=0.2,
                    )
        except Exception:
            # Swallow visualization errors silently (e.g., headless mode)
            pass
    # robot.set_external_force_and_torque(
    #     torch.zeros_like(external_forces), 
    #     torch.zeros_like(external_torques),
    #     body_ids=None  # Apply to all bodies (forces are zero except for propellers)
    # )
    
    # Debug output (only for first environment, first few calls)
    if not hasattr(apply_propeller_aerodynamics, '_debug_counter'):
        apply_propeller_aerodynamics._debug_counter = 0

    if apply_propeller_aerodynamics._debug_counter < 5:
        apply_propeller_aerodynamics._debug_counter += 1
        print(f"\n[AERO DEBUG] Step {apply_propeller_aerodynamics._debug_counter}")
        print(f"  Propeller joint IDs: {propeller_joint_ids}")
        print(f"  Propeller body IDs: {propeller_body_ids}")
        
        # Check joint properties in USD/PhysX
        if apply_propeller_aerodynamics._debug_counter == 1:
            print(f"\n  [USD CHECK] Robot prim path: {robot.cfg.prim_path}")
            print(f"  [USD CHECK] Robot joint names: {robot.joint_names}")
            print(f"  [USD CHECK] Robot body names: {robot.body_names}")
            print(f"  [USD CHECK] Joint limits (lower): {robot.data.joint_limits[0, propeller_joint_ids, 0]}")
            print(f"  [USD CHECK] Joint limits (upper): {robot.data.joint_limits[0, propeller_joint_ids, 1]}")
            print(f"  [USD CHECK] Joint velocity limits: {robot.data.soft_joint_vel_limits[0, propeller_joint_ids] if hasattr(robot.data, 'soft_joint_vel_limits') else 'N/A'}")
            print(f"  [USD CHECK] Joint stiffness: {robot.data.joint_stiffness[0, propeller_joint_ids] if hasattr(robot.data, 'joint_stiffness') else 'N/A'}")
            print(f"  [USD CHECK] Joint damping: {robot.data.joint_damping[0, propeller_joint_ids] if hasattr(robot.data, 'joint_damping') else 'N/A'}")
            print(f"  [USD CHECK] Joint friction: {robot.data.joint_friction[0, propeller_joint_ids] if hasattr(robot.data, 'joint_friction') else 'N/A'}")
            print(f"  [USD CHECK] Joint armature: {robot.data.joint_armature[0, propeller_joint_ids] if hasattr(robot.data, 'joint_armature') else 'N/A'}")
            print(f"  [USD CHECK] Body masses: {robot.data.body_masses[0, propeller_body_ids] if hasattr(robot.data, 'body_masses') else 'N/A'}")
            
            # Deep PhysX check
            try:
                masses = robot.root_physx_view.get_masses()[0]
                print(f"\n  [PHYSX CHECK] All body masses: {masses}")
                print(f"  [PHYSX CHECK] Propeller masses specifically: {masses[propeller_body_ids]}")
                print(f"  [PHYSX CHECK] Base body mass: {masses[0]}")
            except Exception as e:
                print(f"\n  [PHYSX CHECK] Could not get masses: {e}")
        
        print(f"\n  Propeller velocities (joint_vel): {propeller_vel[0]}")
        
        # Try reading velocity from PhysX directly
        try:
            all_dof_vels = robot.root_physx_view.get_dof_velocities()
            physx_prop_vels = all_dof_vels[0, propeller_joint_ids]
            print(f"  Propeller velocities (PhysX direct): {physx_prop_vels}")
        except Exception as e:
            print(f"  Propeller velocities (PhysX direct): Failed - {e}")
        
        print(f"  Propeller positions (rad): {robot.data.joint_pos[0, propeller_joint_ids]}")
        print(f"  Propeller target vel (command): {robot.data.joint_vel_target[0, propeller_joint_ids] if hasattr(robot.data, 'joint_vel_target') else 'N/A'}")
        print(f"  Propeller applied torque (IsaacLab): {robot.data.applied_torque[0, propeller_joint_ids]}")
        
        # Check if PhysX is actually receiving forces
        try:
            dof_forces = robot.root_physx_view.get_dof_actuation_forces()
            print(f"  Propeller forces (PhysX actuation): {dof_forces[0, propeller_joint_ids]}")
        except Exception as e:
            print(f"  Propeller forces (PhysX actuation): Failed - {e}")
        print(f"  Thrust magnitude (N): {thrust_magnitude[0]}")
        print(f"  Thrust local (body frame): {thrust_local[0, 0]}")
        print(f"  Thrust world (world frame): {thrust_world[0, 0]}")
        print(f"  Propeller orientations (quat): {propeller_quat_w[0, 0]}")
        print(f"  Robot base position: {robot.data.root_pos_w[0]}")
        print(f"  Robot base HEIGHT (Z): {robot.data.root_pos_w[0, 2]:.4f}m")
        print(f"  Robot base velocity: {robot.data.root_lin_vel_w[0]}")
        print(f"  Robot VERTICAL velocity (Z): {robot.data.root_lin_vel_w[0, 2]:.4f} m/s")

        # Compare thrust Z to weight to explain net vertical accel
        try:
            # Sum world-Z thrust over props
            thrust_z = thrust_world[0, :, 2].sum().item()
            # Approximate weight (N) = mass * g
            masses = robot.root_physx_view.get_masses()[0]
            total_mass = float(masses.sum().item())
            g = 9.81
            weight = total_mass * g
            print(f"  Thrust Z total: {thrust_z:.2f} N | Weight: {weight:.2f} N | Margin: {thrust_z - weight:.2f} N")
        except Exception as e:
            print(f"  Thrust-vs-weight check failed: {e}")
        
        # Check if we're applying ANY forces
        total_thrust = thrust_magnitude[0].sum()
        print(f"  TOTAL THRUST being applied: {total_thrust:.2f} N")
        
        # Are external forces actually set?
        print(f"  External forces applied: {external_forces[0, propeller_body_ids].sum():.2f} N")
        print(f"  External forces Z-axis applied: {external_forces[0, propeller_body_ids, 2].sum():.2f} N")

def apply_simple_propeller_thrust(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    propeller_joint_names: tuple[str, str],
    thrust_coefficient: float,
    max_total_thrust: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Simplified aerodynamics: Apply combined thrust to robot base link.
    
    This is an easier starting point - thrust from both propellers is combined
    and applied to the robot's base link. Good for initial testing.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        propeller_joint_names: Names of propeller joints
        thrust_coefficient: Thrust coefficient (k_t)
        max_total_thrust: Maximum total thrust force (N)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get propeller joint indices
    propeller_joint_ids = torch.tensor(
        [robot.joint_names.index(name) for name in propeller_joint_names],
        dtype=torch.long,
        device=robot.device
    )
    
    # Get propeller angular velocities
    propeller_vel = robot.data.joint_vel[:, propeller_joint_ids]  # [num_envs, 2]
    
    # Calculate total thrust (sum of both propellers)
    # F_total = k_t * (ω_left² + ω_right²)
    total_thrust_magnitude = thrust_coefficient * torch.sum(torch.square(propeller_vel), dim=1)
    total_thrust_magnitude = torch.clamp(total_thrust_magnitude, max=max_total_thrust)
    
    # Get robot base orientation (quaternion)
    base_quat_w = robot.data.root_quat_w  # [num_envs, 4]
    
    # Define thrust direction in robot base frame
    # Assuming robot's up direction is Z-axis in local frame
    thrust_local = torch.zeros(env.num_envs, 3, device=robot.device)
    thrust_local[:, 2] = total_thrust_magnitude  # Z-axis (upward)
    
    # Rotate to world frame
    thrust_world = quat_rotate(base_quat_w, thrust_local)  # [num_envs, 3]
    
    # Apply force to base link (body index 0)
    external_forces = torch.zeros(env.num_envs, robot.num_bodies, 3, device=robot.device)
    external_torques = torch.zeros(env.num_envs, robot.num_bodies, 3, device=robot.device)
    
    external_forces[:, 0, :] = thrust_world  # Apply to base link
    
    # Apply to simulation
    robot.set_external_force_and_torque(external_forces, external_torques, body_ids=None)
    # robot.write_external_force_and_torque_to_sim(torch.zeros_like(external_forces), 
    #                                             torch.zeros_like(external_torques), 
    #                                             body_ids=None)


def apply_thrust_with_tilt_control(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    propeller_joint_names: tuple[str, str],
    propeller_servo_joint_names: tuple[str, str],
    propeller_body_names: tuple[str, str],
    thrust_coefficient: float,
    max_thrust_per_propeller: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Advanced aerodynamics: Apply thrust considering propeller tilt angle.
    
    This version accounts for the propeller servo angles, allowing for vectored thrust.
    The thrust direction is affected by both the propeller servo angle and body orientation.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        propeller_joint_names: Names of propeller rotation joints
        propeller_servo_joint_names: Names of propeller servo (tilt) joints
        propeller_body_names: Names of propeller bodies
        thrust_coefficient: Thrust coefficient
        max_thrust_per_propeller: Maximum thrust per propeller
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get joint and body indices
    propeller_joint_ids = torch.tensor(
        [robot.joint_names.index(name) for name in propeller_joint_names],
        dtype=torch.long,
        device=robot.device
    )
    propeller_body_ids = torch.tensor(
        [robot.body_names.index(name) for name in propeller_body_names],
        dtype=torch.long,
        device=robot.device
    )
    
    # Get propeller angular velocities
    propeller_vel = robot.data.joint_vel[:, propeller_joint_ids]
    
    # Calculate thrust magnitude
    thrust_magnitude = thrust_coefficient * torch.square(propeller_vel)
    thrust_magnitude = torch.clamp(thrust_magnitude, max=max_thrust_per_propeller)
    
    # Get propeller body orientations (already includes servo tilt)
    propeller_quat_w = robot.data.body_quat_w[:, propeller_body_ids, :]
    
    # Thrust direction in propeller local frame (Z-axis)
    thrust_local = torch.zeros(env.num_envs, len(propeller_joint_ids), 3, device=robot.device)
    thrust_local[:, :, 2] = thrust_magnitude
    
    # Rotate to world frame (this automatically accounts for tilt)
    thrust_world = quat_rotate(propeller_quat_w, thrust_local)
    
    # Prepare and apply forces
    external_forces = torch.zeros(env.num_envs, robot.num_bodies, 3, device=robot.device)
    external_torques = torch.zeros(env.num_envs, robot.num_bodies, 3, device=robot.device)
    
    for i, body_id in enumerate(propeller_body_ids):
        external_forces[:, body_id, :] = thrust_world[:, i, :]
    
    robot.set_external_force_and_torque(external_forces, external_torques, body_ids=None)

