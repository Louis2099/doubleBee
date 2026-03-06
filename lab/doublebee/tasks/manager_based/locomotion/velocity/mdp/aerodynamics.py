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
from isaaclab.utils.math import quat_apply 

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import carb  # For logging

from .thrust_energy_model import pwm_to_thrust

# Print action->thrust chain every N steps for env 0 during training
_ACTION_THRUST_PRINT_INTERVAL = 500

# -----------------------------------------------------------------------------
# Joint velocity response (how fast joint_vel tracks vel_target)
# -----------------------------------------------------------------------------
# Propeller joints use a PD actuator with stiffness=0 and damping=1000 (see
# doublebee_v1.py "propellers" actuator). Torque = damping * (vel_target - joint_vel),
# so dynamics are first-order: d(omega)/dt = (damping / I_eff) * (vel_target - omega).
# Time constant: tau = I_eff / damping. With joint armature ~0.01 and link inertia,
# I_eff is small so tau is on the order of milliseconds; response is typically fast.
#
# How to quantify:
#   1. Velocity error: in the [action->thrust] log, vel_error = vel_target - joint_vel.
#      Small steady-state error and fast decay after a step = fast response.
#   2. Step response: set a constant velocity target (e.g. 200 rad/s), log joint_vel
#      every step; time to reach 63% of the step is tau; to 90% is ~2.3*tau.
#   3. From config: tau_approx = (joint_armature + link_inertia_about_axis) / damping.
#      Read robot.data.joint_armature for the propeller joint indices.
# -----------------------------------------------------------------------------


def sample_thrust_scale_dr(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    range_low: float = 0.8,
    range_high: float = 1.2,
    num_propellers: int = 2,
):
    """Domain randomization: sample per-env, per-propeller thrust scale in [range_low, range_high].
    
    Call this from a reset event so each (env, propeller) gets a fixed scale for the episode.
    apply_propeller_aerodynamics will multiply thrust by env.thrust_scale_dr when present.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    device = env.device
    if not hasattr(env, "thrust_scale_dr") or env.thrust_scale_dr is None:
        env.thrust_scale_dr = torch.ones(
            env.num_envs, num_propellers, device=device, dtype=torch.float32
        )
    scale = range_low + (range_high - range_low) * torch.rand(
        len(env_ids), num_propellers, device=device, dtype=env.thrust_scale_dr.dtype
    )
    env.thrust_scale_dr[env_ids] = scale


def _print_action_thrust_chain(env, propeller_vel, pwm, thrust_np):
    """Print policy action -> velocity target -> joint_vel -> PWM -> thrust for env 0 periodically."""
    if not hasattr(_print_action_thrust_chain, "_step_count"):
        _print_action_thrust_chain._step_count = 0
    _print_action_thrust_chain._step_count += 1
    if _print_action_thrust_chain._step_count % _ACTION_THRUST_PRINT_INTERVAL != 0:
        return
    env_id = 0
    # Get propeller action term (raw policy output and processed velocity target)
    raw_action = processed_action = None
    if hasattr(env, "action_manager"):
        try:
            term = env.action_manager.get_term("propeller_vel")
            raw_action = term.raw_actions[env_id].detach().cpu().numpy()
            processed_action = term.processed_actions[env_id].detach().cpu().numpy()
        except Exception:
            pass
    joint_vel = propeller_vel[env_id].detach().cpu().numpy()
    pwm_np = pwm[env_id].detach().cpu().numpy()
    thrust = thrust_np[env_id]
    vel_error = (processed_action - joint_vel) if processed_action is not None else None
    parts = [
        f"step={_print_action_thrust_chain._step_count} env=0",
        f"raw_action(policy)={raw_action}",
        f"vel_target(rad/s)={processed_action}",
        f"joint_vel(rad/s)={joint_vel}",
        f"vel_error(rad/s)={vel_error}" if vel_error is not None else None,
        f"PWM={pwm_np}",
        f"thrust(N)={thrust}",
    ]
    print("[action->thrust] " + " | ".join(p for p in parts if p is not None), flush=True)


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
    Thrust calculation: rotation speed (rad/s) -> PWM [1000, 1600] -> polynomial thrust.
    |rotation speed| maps linearly: 0 rad/s -> 1000, 500 rad/s -> 1600.
    """
    # Map |omega| (rad/s) to PWM: pwm = 1000 + (|omega| / 500) * 600, clamped to [1000, 1600]
    abs_omega = propeller_vel.abs()
    pwm = 1000.0 + (abs_omega / 500.0) * 650.0
    pwm = torch.clamp(pwm, min=1000.0, max=1650.0)
    pwm_np = pwm.cpu().numpy()

    thrust_np = pwm_to_thrust(pwm_np, target="thrust")

    # Training printout: action -> velocity target -> joint_vel -> PWM -> thrust (env 0, every N steps)
    _print_action_thrust_chain(
        env=env,
        propeller_vel=propeller_vel,
        pwm=pwm,
        thrust_np=thrust_np,
    )
    
    thrust_magnitude = torch.as_tensor(
        thrust_np, device=robot.device, dtype=propeller_vel.dtype
    )

    # Domain randomization: ±20% thrust scale per env per propeller (set at reset by sample_thrust_scale_dr)
    if getattr(env, "thrust_scale_dr", None) is not None:
        thrust_magnitude = thrust_magnitude * env.thrust_scale_dr

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
    thrust_world = quat_apply(propeller_quat_w, thrust_local)
    
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

    # Store total thrust magnitude per env (sum over propellers) for logging (e.g. play.py policy I/O)
    env._last_propeller_thrust_total = thrust_magnitude.sum(dim=1)

    # Optional: visualize thrust vectors in the viewport (disabled by default; carb.plugins not in all Isaac builds)
    if visualize:
        # Only attempt once; after failure disable to avoid console spam
        if not getattr(apply_propeller_aerodynamics, "_viz_disabled", False):
            try:
                if not hasattr(apply_propeller_aerodynamics, "_dbg_draw"):
                    try:
                        from omni.isaac.debug_draw import _debug_draw
                        apply_propeller_aerodynamics._dbg_draw = _debug_draw.acquire_debug_draw_interface()
                    except (ImportError, ModuleNotFoundError):
                        try:
                            import omni.isaac.debug_draw as debug_draw
                            apply_propeller_aerodynamics._dbg_draw = debug_draw._debug_draw.acquire_debug_draw_interface()
                        except (ImportError, ModuleNotFoundError, AttributeError):
                            try:
                                import carb
                                if hasattr(carb, "plugins"):
                                    apply_propeller_aerodynamics._dbg_draw = carb.plugins.acquire_interface("omni.isaac.debug_draw.DebugDrawInterface")
                                else:
                                    apply_propeller_aerodynamics._dbg_draw = None
                            except Exception:
                                apply_propeller_aerodynamics._dbg_draw = None

                    if apply_propeller_aerodynamics._dbg_draw is None:
                        apply_propeller_aerodynamics._viz_disabled = True

                if not getattr(apply_propeller_aerodynamics, "_viz_disabled", False):
                    dbg = apply_propeller_aerodynamics._dbg_draw
                    env_id = 0
                    prop_pos_w = robot.data.body_pos_w[env_id, propeller_body_ids, :]
                    prop_force_w = thrust_world[env_id]
                    for i in range(prop_force_w.shape[0]):
                        start = prop_pos_w[i].cpu().numpy()
                        end = (prop_pos_w[i] + visualize_scale * prop_force_w[i]).cpu().numpy()
                        color = (0.1, 0.9, 0.1, 1.0)
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
            except Exception as e:
                apply_propeller_aerodynamics._viz_disabled = True
                print(f"[AERO VIZ] Thrust visualization disabled (one-time): {e}", flush=True)
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

