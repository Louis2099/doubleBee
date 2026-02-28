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

from .thrust_energy_model import pwm_to_thrust


def apply_propeller_aerodynamics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    propeller_joint_names: tuple[str, str],
    propeller_body_names: tuple[str, str],
    thrust_coefficient: float,
    max_thrust_per_propeller: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    visualize: bool = False,
    visualize_scale: float = 0.2,
):
    """
    Apply aerodynamic thrust forces to propellers based on PWM signal (propeller velocity).
    
    This function uses a regression-based PWM-to-thrust model for realistic thrust calculation.
    - Thrust is computed from PWM signal using pwm_to_thrust() regression model
    - Forces are applied in world frame to propeller bodies
    - No drag torque is applied (removed since using PWM model)
    
    The thrust direction is determined by the propeller body orientation in world frame.
    For DoubleBee, propellers rotate around Z-axis, so thrust is along Z-axis in local frame.
    
    Args:
        env: The environment instance (provided automatically by event manager)
        asset_cfg: Configuration for the robot asset
        propeller_joint_names: Names of propeller joints (to get PWM signal / angular velocity)
        propeller_body_names: Names of propeller bodies (to apply forces)
        thrust_coefficient: Thrust coefficient (unused, kept for compatibility)
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
    
    # Debug: Print propeller velocities for first few steps
    # if not hasattr(apply_propeller_aerodynamics, '_vel_debug_counter'):
    #     apply_propeller_aerodynamics._vel_debug_counter = 0
    
    # if apply_propeller_aerodynamics._vel_debug_counter < 10:
    #     apply_propeller_aerodynamics._vel_debug_counter += 1
    #     print(f"\n[AERO VEL DEBUG] Step {apply_propeller_aerodynamics._vel_debug_counter}")
    #     print(f"  Propeller joint IDs: {propeller_joint_ids}")
    #     print(f"  Propeller velocities (raw joint_vel): {propeller_vel[0]}")
    #     print(f"  Left propeller vel: {propeller_vel[0, 0].item():.2f}, Right propeller vel: {propeller_vel[0, 1].item():.2f}")
    
    """
    NOTE:
    Thrust calculation (from PWM signal):
    propeller_vel is the PWM signal here. pwm_to_thrust returns numpy; convert to tensor.
    """
    pwm_np = propeller_vel.cpu().numpy()
    pwm_np *= 3.25 # scale up to 2000
    
    
    # Debug: Show PWM values before thrust calculation
    # if apply_propeller_aerodynamics._vel_debug_counter <= 10:
    #     print(f"  PWM values: {pwm_np[0]}")
    
    thrust_np = pwm_to_thrust(abs(pwm_np), target="thrust")
    
    # Debug: Show thrust output
    # if apply_propeller_aerodynamics._vel_debug_counter <= 10:
    #     print(f"  Thrust from model: {thrust_np[0]}")
    #     print(f"  Left thrust: {thrust_np[0, 0]:.2f}N, Right thrust: {thrust_np[0, 1]:.2f}N")
    
    thrust_magnitude = torch.as_tensor(
        thrust_np, device=robot.device, dtype=propeller_vel.dtype
    )

    # Calculate thrust force: F = k_t * ω²
    # Using absolute value to always generate positive thrust
    #thrust_magnitude = thrust_coefficient * torch.square(propeller_vel)
    
    # Clamp thrust to maximum value
    # thrust_magnitude = torch.clamp(thrust_magnitude, max=max_thrust_per_propeller)
    
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
    
    # Prepare force tensors for all bodies
    # Shape: [num_envs, num_bodies, 3]
    num_bodies = robot.num_bodies
    external_forces = torch.zeros(env.num_envs, num_bodies, 3, device=robot.device)
    external_torques = torch.zeros(env.num_envs, num_bodies, 3, device=robot.device)  # Zero torques (no drag)
    
    # Assign propeller thrust forces
    for i, body_id in enumerate(propeller_body_ids):
        external_forces[:, body_id, :] = thrust_world[:, i, :]
    
    # Apply external forces to the robot (with zero torques)
    robot.set_external_force_and_torque(
        external_forces, 
        external_torques,  # Zero torques - no drag applied
        body_ids=None  # Apply to all bodies (forces are zero except for propellers)
    )

    # Optional: visualize thrust vectors in the viewport
    if visualize:
        print(f"[AERO VIZ] Attempting to visualize thrust arrows...", flush=True)
        try:
            # Lazy-acquire debug draw interface once
            if not hasattr(apply_propeller_aerodynamics, "_dbg_draw"):
                try:
                    # Try new API first (Isaac Sim 2023+)
                    from omni.isaac.debug_draw import _debug_draw
                    apply_propeller_aerodynamics._dbg_draw = _debug_draw.acquire_debug_draw_interface()
                except (ImportError, ModuleNotFoundError):
                    try:
                        # Try alternative import path
                        import omni.isaac.debug_draw as debug_draw
                        apply_propeller_aerodynamics._dbg_draw = debug_draw._debug_draw.acquire_debug_draw_interface()
                    except (ImportError, ModuleNotFoundError, AttributeError):
                        # Fallback to direct carb debug draw
                        import carb
                        apply_propeller_aerodynamics._dbg_draw = carb.plugins.acquire_interface("omni.isaac.debug_draw.DebugDrawInterface")
                
                if apply_propeller_aerodynamics._dbg_draw is None:
                    print(f"[AERO VIZ] Could not acquire debug draw interface - visualization disabled", flush=True)
                    return
                
                print(f"[AERO VIZ] Debug draw interface acquired successfully", flush=True)
            
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
                print(f"[AERO VIZ] Drew arrow from {start} to {end}, magnitude={torch.norm(prop_force_w[i]).item():.2f}N", flush=True)
        except Exception as e:
            # If debug draw fails, try using Isaac Lab's visualization markers
            print(f"[AERO VIZ] Debug draw visualization failed: {e}", flush=True)
            print(f"[AERO VIZ] Attempting fallback visualization using scene markers...", flush=True)
            try:
                # Fallback: Use Isaac Lab's built-in visualization
                # This prints to console instead of drawing in viewport
                env_id = 0
                prop_pos_w = robot.data.body_pos_w[env_id, propeller_body_ids, :]
                prop_force_w = thrust_world[env_id]
                
                print(f"[AERO VIZ FALLBACK] Env {env_id} thrust visualization:", flush=True)
                for i in range(prop_force_w.shape[0]):
                    pos = prop_pos_w[i].cpu().numpy()
                    force = prop_force_w[i].cpu().numpy()
                    magnitude = torch.norm(prop_force_w[i]).item()
                    direction = force / (magnitude + 1e-6)
                    print(f"[AERO VIZ FALLBACK]   Propeller {i}: pos={pos}, force={force}, mag={magnitude:.2f}N, dir={direction}", flush=True)
                
            except Exception as e2:
                print(f"[AERO VIZ] Fallback visualization also failed: {e2}", flush=True)
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

