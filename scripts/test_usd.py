"""
Quick smoke test to load a USD-based robot into Isaac Sim.

This script launches Isaac Sim, creates the requested Isaac Lab task, and
performs a few simulation steps to make sure the asset can be spawned without
errors. It intentionally keeps the interaction minimal so it can be used as a
sanity check while developing new robots.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test."""
    parser = argparse.ArgumentParser(description="Load a USD robot into Isaac Sim and step the simulation.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Velocity-Flat-DoubleBee-v1-ppo",
        help="Gym task ID to instantiate.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments to create.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Number of simulation steps to run after reset.",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--hover",
        action="store_true",
        default=False,
        help="Run a hovering test: wheels stalled, servos at 0, propellers provide thrust to hover.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def create_test_action(env, step, test_phase, wheel_ids, servo_ids, propeller_ids):
    """Create test actions based on the current test phase."""
    import torch
    import numpy as np
    
    # Get action sample and convert to tensor
    action_sample = env.action_space.sample()
    if isinstance(action_sample, np.ndarray):
        action = torch.from_numpy(action_sample).to(env.device)
    else:
        action = action_sample.clone()
    
    # Zero out all actions first
    action.zero_()
    
    # Get action dimension
    action_dim = action.shape[-1]
    
    # Debug: Print test phase and joint IDs
    if step % 20 == 0:
        print(f"[CREATE_ACTION] Step {step}: test_phase='{test_phase}'", flush=True)
        print(f"[CREATE_ACTION] wheel_ids={wheel_ids}, servo_ids={servo_ids}, propeller_ids={propeller_ids}", flush=True)
        print(f"[CREATE_ACTION] action_dim={action_dim}", flush=True)
    
    if test_phase == "idle":
        # No movement - all actions remain zero
        if step % 20 == 0:
            print(f"[CREATE_ACTION] IDLE phase - keeping actions zero", flush=True)
        pass
        
    elif test_phase == "wheels" and wheel_ids is not None:
        # Test wheel velocity control (first 2 actions typically)
        if action_dim >= 2:
            # Apply normalized velocity commands (-1 to 1)
            t = torch.tensor((step + 1) * 0.1, device=action.device)  # Start from 0.1, not 0.0
            wheel_vel = 0.5 * torch.sin(t)  # ±0.5 normalized velocity command
            action[:, :2] = wheel_vel  # First 2 actions for wheels
            if step % 20 == 0:
                print(f"[CREATE_ACTION] WHEELS phase - applied {wheel_vel} to actions[:, :2]", flush=True)
        else:
            if step % 20 == 0:
                print(f"[CREATE_ACTION] WHEELS phase - action_dim {action_dim} < 2, skipping", flush=True)
    elif test_phase == "wheels" and wheel_ids is None:
        if step % 20 == 0:
            print(f"[CREATE_ACTION] WHEELS phase - wheel_ids is None, skipping", flush=True)
            
    elif test_phase == "servos" and servo_ids is not None:
        # Test servo position control (middle 2 actions typically)
        if action_dim >= 4:
            # Apply normalized position commands (-1 to 1, will be scaled by action manager)
            t = torch.tensor((step + 1) * 0.05, device=action.device)  # Start from 0.05, not 0.0
            servo_pos = 0.3 * torch.sin(t)  # ±0.3 normalized position command
            action[:, 2:4] = servo_pos  # Actions 2-3 for servos
        elif action_dim >= 2:
            # Fallback: use last 2 actions
            t = torch.tensor((step + 1) * 0.05, device=action.device)  # Start from 0.05, not 0.0
            servo_pos = 0.3 * torch.sin(t)
            action[:, -2:] = servo_pos
            
    elif test_phase == "propellers" and propeller_ids is not None:
        # Test propeller velocity control (last 2 actions typically)
        if action_dim >= 6:
            # Apply normalized velocity commands (-1 to 1)
            prop_vel = 0.8  # High normalized velocity for testing
            action[:, 4:6] = prop_vel  # Actions 4-5 for propellers
        elif action_dim >= 2:
            # Fallback: use last 2 actions
            prop_vel = 0.8
            action[:, -2:] = prop_vel
    
    return action



def main() -> None:
    args_cli = parse_arguments()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env: Any | None = None
    try:
        import gymnasium as gym
        import numpy as np
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_tasks.utils import parse_env_cfg

        import lab.doublebee.tasks  # noqa: F401  # Register DoubleBee tasks

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        env = gym.make(args_cli.task, cfg=env_cfg)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        obs, info = env.reset()
        print(
            f"[INFO] Environment '{args_cli.task}' loaded. Observation shape: {getattr(obs, 'shape', type(obs))}",
            flush=True,
        )
        
        # Debug: Check if robot exists in scene
        if hasattr(env.unwrapped, 'scene'):
            print(f"[DEBUG] Scene entities: {list(env.unwrapped.scene.keys())}", flush=True)
            try:
                robot = env.unwrapped.scene["robot"]
                print(f"[DEBUG] Robot type: {type(robot)}", flush=True)
                print(f"[DEBUG] Robot prim path: {robot.cfg.prim_path}", flush=True)
                print(f"[DEBUG] Robot num instances: {robot.num_instances}", flush=True)
                if hasattr(robot, 'data') and hasattr(robot.data, 'root_pos_w'):
                    print(f"[DEBUG] Robot root position: {robot.data.root_pos_w[0]}", flush=True)
                print(f"[DEBUG] Robot is spawned successfully!", flush=True)
                print(f"\n{'='*80}", flush=True)
                print(f"VISIBILITY TROUBLESHOOTING:", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"1. If robot is NOT visible in GUI:", flush=True)
                print(f"   - Check USD file visibility settings", flush=True)
                print(f"   - See: scripts/fix_usd_visibility.md for solutions", flush=True)
                print(f"2. Robot USD path: {robot.cfg.spawn.usd_path}", flush=True)
                print(f"3. Try pressing 'F' key in viewport with robot prim selected", flush=True)
                print(f"4. Check Stage panel (left) for: /World/envs/env_0/Robot", flush=True)
                print(f"{'='*80}\n", flush=True)
            except Exception as e:
                print(f"[ERROR] Could not access robot: {e}", flush=True)

        # Test joint movement capabilities
        test_joint_movement = not args_cli.hover
        # Initialize joint IDs as None - will be set if joint movement test succeeds
        wheel_ids = None
        servo_ids = None
        propeller_ids = None
        
        if test_joint_movement:
            print(f"\n{'='*80}", flush=True)
            print(f"JOINT MOVEMENT TEST", flush=True)
            print(f"{'='*80}", flush=True)
            
            try:
                import torch
                robot = env.unwrapped.scene["robot"]
                
                # Get all joint names and indices
                joint_names = robot.joint_names
                print(f"[JOINT] Available joints: {joint_names}", flush=True)
                
                # Check robot dimensions and scaling
                print(f"\n[ROBOT DIMENSIONS] Checking robot size and scaling...", flush=True)
                
                # Get robot base position and dimensions
                base_pos = robot.data.root_pos_w[0]  # [x, y, z]
                print(f"[ROBOT DIMENSIONS] Base position: {base_pos}", flush=True)
                print(f"[ROBOT DIMENSIONS] Base height (Z): {base_pos[2]:.4f} meters", flush=True)
                
                # Check if robot has bounding box information
                if hasattr(robot.data, 'body_pos_w'):
                    body_positions = robot.data.body_pos_w[0]  # [num_bodies, 3]
                    print(f"[ROBOT DIMENSIONS] Body positions shape: {body_positions.shape}", flush=True)
                    
                    # Calculate robot bounding box
                    min_pos = body_positions.min(dim=0)[0]
                    max_pos = body_positions.max(dim=0)[0]
                    robot_size = max_pos - min_pos
                    print(f"[ROBOT DIMENSIONS] Robot bounding box min: {min_pos}", flush=True)
                    print(f"[ROBOT DIMENSIONS] Robot bounding box max: {max_pos}", flush=True)
                    print(f"[ROBOT DIMENSIONS] Robot size (X, Y, Z): {robot_size}", flush=True)
                    print(f"[ROBOT DIMENSIONS] Robot diagonal length: {torch.norm(robot_size):.4f} meters", flush=True)
                
                # Check robot configuration for scaling
                if hasattr(robot, 'cfg'):
                    print(f"[ROBOT DIMENSIONS] Robot config: {robot.cfg}", flush=True)
                    if hasattr(robot.cfg, 'spawn'):
                        spawn_cfg = robot.cfg.spawn
                        print(f"[ROBOT DIMENSIONS] Spawn config: {spawn_cfg}", flush=True)
                        if hasattr(spawn_cfg, 'scale'):
                            print(f"[ROBOT DIMENSIONS] Spawn scale: {spawn_cfg.scale}", flush=True)
                        if hasattr(spawn_cfg, 'position'):
                            print(f"[ROBOT DIMENSIONS] Spawn position: {spawn_cfg.position}", flush=True)
                
                # Check if there's a scale attribute in the robot data
                if hasattr(robot.data, 'scale'):
                    print(f"[ROBOT DIMENSIONS] Robot scale: {robot.data.scale[0]}", flush=True)
                
                # Check PhysX properties for scaling
                try:
                    # Get PhysX view for more detailed info
                    physx_view = robot.root_physx_view
                    if hasattr(physx_view, 'get_global_poses'):
                        poses = physx_view.get_global_poses()
                        print(f"[ROBOT DIMENSIONS] PhysX poses shape: {poses.shape}", flush=True)
                        if poses.shape[1] > 0:  # Check first body
                            first_body_pos = poses[0, 0, :3]  # [x, y, z]
                            print(f"[ROBOT DIMENSIONS] First body PhysX position: {first_body_pos}", flush=True)
                except Exception as e:
                    print(f"[ROBOT DIMENSIONS] Could not get PhysX poses: {e}", flush=True)
                
                # Check if the robot is using the correct USD file
                if hasattr(robot.cfg, 'prim_path'):
                    print(f"[ROBOT DIMENSIONS] Robot prim path: {robot.cfg.prim_path}", flush=True)
                
                # Check scene configuration for robot scaling
                if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'cfg'):
                    scene_cfg = env.unwrapped.scene.cfg
                    if hasattr(scene_cfg, 'robot'):
                        robot_cfg = scene_cfg.robot
                        print(f"[ROBOT DIMENSIONS] Scene robot config: {robot_cfg}", flush=True)
                        if hasattr(robot_cfg, 'spawn'):
                            print(f"[ROBOT DIMENSIONS] Scene robot spawn: {robot_cfg.spawn}", flush=True)
                
                # Identify joint types based on configuration
                wheel_joints = ["leftWheel", "rightWheel"]
                servo_joints = ["leftPropellerServo", "rightPropellerServo"] 
                propeller_joints = ["leftPropeller", "rightPropeller"]
                
                # Find which joints actually exist
                found_wheels = [j for j in wheel_joints if j in joint_names]
                found_servos = [j for j in servo_joints if j in joint_names]
                found_propellers = [j for j in propeller_joints if j in joint_names]
                
                print(f"[JOINT] Wheel joints (velocity control): {found_wheels}", flush=True)
                print(f"[JOINT] Servo joints (position control): {found_servos}", flush=True)
                print(f"[JOINT] Propeller joints (velocity control): {found_propellers}", flush=True)
                
                # Get joint indices
                wheel_ids = torch.tensor([joint_names.index(j) for j in found_wheels], device=robot.device) if found_wheels else None
                servo_ids = torch.tensor([joint_names.index(j) for j in found_servos], device=robot.device) if found_servos else None
                propeller_ids = torch.tensor([joint_names.index(j) for j in found_propellers], device=robot.device) if found_propellers else None
                
                # Safely convert to list (handle both tensor and list)
                if wheel_ids is not None:
                    if isinstance(wheel_ids, torch.Tensor):
                        wheel_ids_list = wheel_ids.cpu().tolist()
                    elif isinstance(wheel_ids, (list, tuple)):
                        wheel_ids_list = list(wheel_ids)
                    else:
                        wheel_ids_list = list(wheel_ids) if hasattr(wheel_ids, '__iter__') else [wheel_ids]
                else:
                    wheel_ids_list = 'None'
                    
                if servo_ids is not None:
                    if isinstance(servo_ids, torch.Tensor):
                        servo_ids_list = servo_ids.cpu().tolist()
                    elif isinstance(servo_ids, (list, tuple)):
                        servo_ids_list = list(servo_ids)
                    else:
                        servo_ids_list = list(servo_ids) if hasattr(servo_ids, '__iter__') else [servo_ids]
                else:
                    servo_ids_list = 'None'
                    
                if propeller_ids is not None:
                    if isinstance(propeller_ids, torch.Tensor):
                        prop_ids_list = propeller_ids.cpu().tolist()
                    elif isinstance(propeller_ids, (list, tuple)):
                        prop_ids_list = list(propeller_ids)
                    else:
                        prop_ids_list = list(propeller_ids) if hasattr(propeller_ids, '__iter__') else [propeller_ids]
                else:
                    prop_ids_list = 'None'
                    
                print(f"[JOINT] Wheel joint indices: {wheel_ids_list}", flush=True)
                print(f"[JOINT] Servo joint indices: {servo_ids_list}", flush=True)
                print(f"[JOINT] Propeller joint indices: {prop_ids_list}", flush=True)
                
                # Check action space - get the actual action space from the unwrapped environment
                try:
                    # Try to get action space from unwrapped environment
                    unwrapped_env = env.unwrapped
                    while hasattr(unwrapped_env, 'env'):
                        unwrapped_env = unwrapped_env.env
                    
                    if hasattr(unwrapped_env, 'action_space'):
                        action_space = unwrapped_env.action_space
                        action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else len(action_space.sample())
                        print(f"[JOINT] Unwrapped action space dimension: {action_dim}", flush=True)
                    else:
                        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else len(env.action_space.sample())
                        print(f"[JOINT] Wrapped action space dimension: {action_dim}", flush=True)
                except Exception as e:
                    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else len(env.action_space.sample())
                    print(f"[JOINT] Action space dimension (fallback): {action_dim}", flush=True)
                    print(f"[JOINT] Action space error: {e}", flush=True)
                
                # Check action manager if available
                if hasattr(env.unwrapped, 'action_manager'):
                    try:
                        # Get action terms list - handle both dict and list formats
                        active_terms = env.unwrapped.action_manager.active_terms
                        if isinstance(active_terms, dict):
                            action_terms = list(active_terms.keys())
                        elif isinstance(active_terms, list):
                            action_terms = [f"term_{i}" for i in range(len(active_terms))]
                        else:
                            action_terms = [str(active_terms)]
                        print(f"[JOINT] Action manager terms: {action_terms}", flush=True)
                        
                        # CRITICAL: Verify joint mapping for each action term
                        print(f"\n[JOINT] ===== ACTION-TO-JOINT MAPPING VERIFICATION =====", flush=True)
                        
                        active_terms = env.unwrapped.action_manager.active_terms
                        if isinstance(active_terms, dict):
                            terms_iter = active_terms.items()
                        elif isinstance(active_terms, list):
                            terms_iter = enumerate(active_terms)
                        else:
                            terms_iter = []
                        
                        for term_identifier, term in terms_iter:
                            if isinstance(term_identifier, int):
                                term_name = f"term_{term_identifier}"
                            else:
                                term_name = term_identifier
                                
                            if hasattr(term, '_joint_names'):
                                expected_joints = term._joint_names
                                print(f"[JOINT] Term '{term_name}':", flush=True)
                                print(f"         Expected joints: {expected_joints}", flush=True)
                                
                                # Get actual joint indices
                                if hasattr(term, '_joint_ids'):
                                    # Safely convert to list (handle both tensor and list)
                                    joint_ids_raw = term._joint_ids[0]  # First env
                                    if isinstance(joint_ids_raw, torch.Tensor):
                                        joint_ids = joint_ids_raw.cpu().tolist()
                                    elif isinstance(joint_ids_raw, (list, tuple)):
                                        joint_ids = list(joint_ids_raw)
                                    elif hasattr(joint_ids_raw, 'cpu'):
                                        joint_ids = joint_ids_raw.cpu().tolist()
                                    elif hasattr(joint_ids_raw, 'tolist'):
                                        joint_ids = joint_ids_raw.tolist()
                                    else:
                                        joint_ids = list(joint_ids_raw) if isinstance(joint_ids_raw, (list, tuple)) else [joint_ids_raw]
                                    actual_joints = [robot.joint_names[i] for i in joint_ids]
                                    print(f"         Actual mapped joints: {actual_joints}", flush=True)
                                    print(f"         Joint indices: {joint_ids}", flush=True)
                                    
                                    # Check for mismatch
                                    if expected_joints != actual_joints:
                                        print(f"         ⚠️  WARNING: Joint order mismatch detected!", flush=True)
                                    else:
                                        print(f"         ✓ Joint mapping is correct", flush=True)
                        print(f"[JOINT] ===============================================\n", flush=True)
                        
                    except AttributeError:
                        # Fallback: try to get terms as a list
                        action_terms = env.unwrapped.action_manager.active_terms
                        print(f"[JOINT] Action manager terms (list): {action_terms}", flush=True)
                
                # Check observation manager structure
                if hasattr(env.unwrapped, 'observation_manager'):
                    print(f"\n[OBS] Observation Manager Structure:", flush=True)
                    obs_mgr = env.unwrapped.observation_manager
                    
                    # Get active observation groups and terms
                    if hasattr(obs_mgr, 'active_terms'):
                        try:
                            # Try dict-style access first (ManagerBasedRLEnv)
                            for group_name, terms in obs_mgr.active_terms.items():
                                print(f"[OBS] Group '{group_name}': {list(terms.keys())}", flush=True)
                                
                                # Get shapes for each term
                                for term_name, term in terms.items():
                                    if hasattr(term, 'shape'):
                                        print(f"[OBS]   - {term_name}: shape {term.shape}", flush=True)
                                    elif hasattr(term, '_shape'):
                                        print(f"[OBS]   - {term_name}: shape {term._shape}", flush=True)
                        except AttributeError:
                            # Fallback for list-style access
                            print(f"[OBS] Active terms (list): {obs_mgr.active_terms}", flush=True)
                
                # Debug: Check actuator configuration
                print(f"[JOINT] Checking actuator configuration...", flush=True)
                try:
                    # Get the robot's actuator configuration
                    if hasattr(robot, 'cfg') and hasattr(robot.cfg, 'actuators'):
                        for actuator_name, actuator_cfg in robot.cfg.actuators.items():
                            print(f"[JOINT] Actuator '{actuator_name}':", flush=True)
                            print(f"         Joints: {actuator_cfg.joint_names_expr}", flush=True)
                            print(f"         Effort limit: {actuator_cfg.effort_limit}", flush=True)
                            print(f"         Velocity limit: {actuator_cfg.velocity_limit}", flush=True)
                            if hasattr(actuator_cfg, 'stiffness'):
                                print(f"         Stiffness: {actuator_cfg.stiffness}", flush=True)
                            if hasattr(actuator_cfg, 'damping'):
                                print(f"         Damping: {actuator_cfg.damping}", flush=True)
                    
                    # Also check the actual joint properties from the robot
                    print(f"[JOINT] Robot joint properties:", flush=True)
                    print(f"         Joint stiffness: {robot.data.joint_stiffness[0]}", flush=True)
                    print(f"         Joint damping: {robot.data.joint_damping[0]}", flush=True)
                    print(f"         Joint effort limits: {robot.data.joint_effort_limit[0]}", flush=True)
                    print(f"         Joint velocity limits: {robot.data.joint_vel_limit[0]}", flush=True)
                    
                    # Check if we have the correct number of joints
                    num_joints = len(robot.joint_names)
                    print(f"[JOINT] Total joints: {num_joints}", flush=True)
                    print(f"[JOINT] Joint names: {robot.joint_names}", flush=True)
                    
                    # Check actuator mapping
                    if hasattr(robot, '_actuators'):
                        print(f"[JOINT] Actuator mapping:", flush=True)
                        for i, actuator in enumerate(robot._actuators):
                            if hasattr(actuator, 'joint_names'):
                                print(f"         Actuator {i}: {actuator.joint_names}", flush=True)
                    
                except Exception as e:
                    print(f"[JOINT] Could not access actuator config: {e}", flush=True)
                
            except Exception as e:
                print(f"[JOINT] Warning: Could not setup joint movement test: {e}", flush=True)
                test_joint_movement = False
                # Reset IDs to None on error
                wheel_ids = None
                servo_ids = None
                propeller_ids = None
        
        # If hover mode requested, run the hover test session and exit
        if args_cli.hover:
            hover_test(env, args_cli.num_steps)
            print("[INFO] Hover session completed. Close the window or press Ctrl+C to exit.", flush=True)
            while simulation_app.is_running():
                simulation_app.update()
            return

        # Run simulation steps with joint movement testing
        joint_positions = []
        joint_velocities = []
        test_phase = "wheels"  # Start with wheels instead of idle
        
        print(f"\n{'='*80}", flush=True)
        print(f"STARTING SIMULATION LOOP", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Test joint movement: {test_joint_movement}", flush=True)
        print(f"Number of steps: {args_cli.num_steps}", flush=True)
        
        for step in range(args_cli.num_steps):
            # Create test actions based on current phase
            if test_joint_movement:
                action = create_test_action(env, step, test_phase, wheel_ids, servo_ids, propeller_ids)
                # Only print action every 20 steps to reduce verbosity
                if step % 20 == 0:
                    print(f"\n[ACTION] Step {step}: Phase={test_phase}", flush=True)
                    print(f"[ACTION] Raw action tensor: {action[0]}", flush=True)
                    print(f"[ACTION] Action shape: {action.shape}", flush=True)
                    print(f"[ACTION] Action device: {action.device}", flush=True)
                    
                    # Debug: Check what action manager receives and outputs
                    if hasattr(env.unwrapped, 'action_manager'):
                        try:
                            print(f"\n[ACTION_MGR] ===== ACTION MANAGER DEBUG =====", flush=True)
                            
                            # Check action manager's raw action buffer
                            if hasattr(env.unwrapped.action_manager, 'action'):
                                print(f"[ACTION_MGR] Manager raw action buffer: {env.unwrapped.action_manager.action[0]}", flush=True)
                            else:
                                print(f"[ACTION_MGR] No raw action buffer found", flush=True)
                            
                            # Handle both dict and list formats for active_terms
                            active_terms = env.unwrapped.action_manager.active_terms
                            
                            if isinstance(active_terms, dict):
                                # Dictionary format: {term_name: term_object}
                                terms_iter = active_terms.items()
                            elif isinstance(active_terms, list):
                                # List format: [term_object1, term_object2, ...]
                                terms_iter = enumerate(active_terms)
                            else:
                                print(f"[ACTION_MGR] Unknown active_terms format: {type(active_terms)}", flush=True)
                                terms_iter = []
                            
                            # Print raw actions per term
                            for term_identifier, term in terms_iter:
                                if isinstance(term_identifier, int):
                                    term_name = f"term_{term_identifier}"
                                else:
                                    term_name = term_identifier
                                
                                print(f"[ACTION_MGR] Term '{term_name}':", flush=True)
                                
                                # Raw actions (what the term receives)
                                if hasattr(term, 'raw_actions'):
                                    raw_action = term.raw_actions[0]
                                    print(f"         Raw action: {raw_action}", flush=True)
                                
                                # Processed actions (after scaling)
                                if hasattr(term, 'action'):
                                    processed_action = term.action[0]
                                    print(f"         Processed action: {processed_action}", flush=True)
                                
                                # Joint commands (what gets sent to robot)
                                if hasattr(term, 'joint_pos_target') and term.joint_pos_target is not None:
                                    joint_cmd = term.joint_pos_target[0]
                                    print(f"         Joint position target: {joint_cmd}", flush=True)
                                
                                if hasattr(term, 'joint_vel_target') and term.joint_vel_target is not None:
                                    joint_cmd = term.joint_vel_target[0]
                                    print(f"         Joint velocity target: {joint_cmd}", flush=True)
                                
                                # Joint effort/torque commands
                                if hasattr(term, 'joint_effort_target') and term.joint_effort_target is not None:
                                    joint_cmd = term.joint_effort_target[0]
                                    print(f"         Joint effort target: {joint_cmd}", flush=True)
                                
                                print(f"", flush=True)  # Empty line for readability
                            
                            print(f"[ACTION_MGR] ======================================\n", flush=True)
                            
                        except Exception as e:
                            print(f"[ACTION_MGR] Debug error: {e}", flush=True)
            else:
                action = env.action_space.sample()
            
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(env.device)
            else:
                action = action.to(env.device)
            # Debug: Check action manager buffer before env.step()
            if test_joint_movement and step % 20 == 0:
                print(f"[PRE_STEP] Action manager buffer before env.step(): {env.unwrapped.action_manager.action[0]}", flush=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Debug: Check if action manager received the raw action after env.step()
            if test_joint_movement and step % 20 == 0:
                print(f"[POST_STEP] Raw action passed to env.step(): {action[0]}", flush=True)
                print(f"[POST_STEP] Action space: {env.action_space}", flush=True)
                print(f"[POST_STEP] Action space shape: {env.action_space.shape if hasattr(env.action_space, 'shape') else 'No shape'}", flush=True)
                
                # Check environment wrappers
                print(f"[POST_STEP] Environment type: {type(env)}", flush=True)
                print(f"[POST_STEP] Unwrapped environment type: {type(env.unwrapped)}", flush=True)
                
                # Check if there are any wrappers
                wrapper_count = 0
                current_env = env
                while hasattr(current_env, 'env'):
                    wrapper_count += 1
                    current_env = current_env.env
                    print(f"[POST_STEP] Wrapper {wrapper_count}: {type(current_env)}", flush=True)
                print(f"[POST_STEP] Total wrappers: {wrapper_count}", flush=True)
                
                if hasattr(env.unwrapped, 'action_manager'):
                    print(f"[POST_STEP] Action manager found", flush=True)
                    print(f"[POST_STEP] Action manager type: {type(env.unwrapped.action_manager)}", flush=True)
                    
                    if hasattr(env.unwrapped.action_manager, 'action'):
                        print(f"[POST_STEP] Action manager buffer after env.step(): {env.unwrapped.action_manager.action[0]}", flush=True)
                        print(f"[POST_STEP] Actions match: {torch.allclose(env.unwrapped.action_manager.action[0], action[0])}", flush=True)
                        
                        # Check if action manager has action space info
                        if hasattr(env.unwrapped.action_manager, 'action_space'):
                            print(f"[POST_STEP] Action manager action space: {env.unwrapped.action_manager.action_space}", flush=True)
                        
                        # Check action manager configuration
                        if hasattr(env.unwrapped.action_manager, 'cfg'):
                            print(f"[POST_STEP] Action manager config: {env.unwrapped.action_manager.cfg}", flush=True)
                        
                        # Check if action manager has action processing method
                        if hasattr(env.unwrapped.action_manager, 'process_actions'):
                            print(f"[POST_STEP] Action manager has process_actions method", flush=True)
                        if hasattr(env.unwrapped.action_manager, 'apply_action'):
                            print(f"[POST_STEP] Action manager has apply_action method", flush=True)
                        
                        # Check if action manager has process_action method (the one actually called)
                        if hasattr(env.unwrapped.action_manager, 'process_action'):
                            print(f"[POST_STEP] Action manager has process_action method", flush=True)
                            
                            # Try to manually call process_action to see what happens
                            try:
                                print(f"[POST_STEP] Manually calling process_action with: {action[0]}", flush=True)
                                env.unwrapped.action_manager.process_action(action)
                                print(f"[POST_STEP] After manual process_action: {env.unwrapped.action_manager.action[0]}", flush=True)
                                
                                # Check if the action gets reset after a short delay
                                import time
                                time.sleep(0.001)  # Small delay
                                print(f"[POST_STEP] After delay: {env.unwrapped.action_manager.action[0]}", flush=True)
                                
                            except Exception as e:
                                print(f"[POST_STEP] Manual process_action failed: {e}", flush=True)
                    else:
                        print(f"[POST_STEP] Action manager has no 'action' attribute", flush=True)
                        print(f"[POST_STEP] Action manager attributes: {[attr for attr in dir(env.unwrapped.action_manager) if not attr.startswith('_')]}", flush=True)
                else:
                    print(f"[POST_STEP] No action manager found", flush=True)
            
            # Debug: Check what was actually applied to the robot after env.step()
            if test_joint_movement and step % 20 == 0:
                try:
                    print(f"\n[POST_STEP] ===== POST-STEP ACTION VERIFICATION =====", flush=True)
                    
                    # Check what the action manager actually sent to the robot
                    if hasattr(env.unwrapped, 'action_manager'):
                        active_terms = env.unwrapped.action_manager.active_terms
                        
                        if isinstance(active_terms, dict):
                            terms_iter = active_terms.items()
                        elif isinstance(active_terms, list):
                            terms_iter = enumerate(active_terms)
                        else:
                            terms_iter = []
                        
                        for term_identifier, term in terms_iter:
                            if isinstance(term_identifier, int):
                                term_name = f"term_{term_identifier}"
                            else:
                                term_name = term_identifier
                                
                            print(f"[POST_STEP] Term '{term_name}' final state:", flush=True)
                            
                            # Check if joint targets were set
                            if hasattr(term, 'joint_pos_target') and term.joint_pos_target is not None:
                                joint_cmd = term.joint_pos_target[0]
                                print(f"         Final joint position target: {joint_cmd}", flush=True)
                            
                            if hasattr(term, 'joint_vel_target') and term.joint_vel_target is not None:
                                joint_cmd = term.joint_vel_target[0]
                                print(f"         Final joint velocity target: {joint_cmd}", flush=True)
                            
                            if hasattr(term, 'joint_effort_target') and term.joint_effort_target is not None:
                                joint_cmd = term.joint_effort_target[0]
                                print(f"         Final joint effort target: {joint_cmd}", flush=True)
                    
                    # Check robot's current joint state
                    robot = env.unwrapped.scene["robot"]
                    try:
                        # Try to read current joint state from PhysX
                        current_pos = robot.root_physx_view.get_dof_positions()[0].cpu().numpy()
                        current_vel = robot.root_physx_view.get_dof_velocities()[0].cpu().numpy()
                        print(f"[POST_STEP] Robot current joint state:", flush=True)
                        print(f"         Position: {current_pos}", flush=True)
                        print(f"         Velocity: {current_vel}", flush=True)
                    except Exception as e:
                        print(f"[POST_STEP] Could not read robot state: {e}", flush=True)
                    
                    print(f"[POST_STEP] ===========================================\n", flush=True)
                    
                except Exception as e:
                    print(f"[POST_STEP] Debug error: {e}", flush=True)
            
            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
            
            # Record joint data for analysis
            if test_joint_movement and step < 200:  # First 200 steps for testing
                try:
                    robot = env.unwrapped.scene["robot"]
                    
                    # CRITICAL FIX: Force articulation data update from PhysX
                    # According to Isaac Lab tutorials, articulation data needs explicit update
                    # This is normally done by scene.update() but we'll double-check
                    if hasattr(robot, 'update'):
                        robot.update(dt=env.unwrapped.physics_dt)
                    
                    # Try reading from PhysX directly as a fallback
                    try:
                        physx_pos = robot.root_physx_view.get_dof_positions()[0].cpu().numpy()
                        physx_vel = robot.root_physx_view.get_dof_velocities()[0].cpu().numpy()
                        
                        # Use PhysX data as ground truth
                        pos = physx_pos
                        vel = physx_vel
                        
                    except Exception as e:
                        # Fallback to robot.data if PhysX read fails
                        pos = robot.data.joint_pos[0].cpu().numpy()
                        vel = robot.data.joint_vel[0].cpu().numpy()
                    
                    joint_positions.append(pos.copy())
                    joint_velocities.append(vel.copy())
                    
                    # Monitor observations to see if joint states are conveyed through obs
                    if step % 20 == 0:
                        print(f"\n[OBS] Step {step}: Observation Analysis", flush=True)
                        
                        # Check observation structure
                        if isinstance(obs, dict):
                            print(f"[OBS] Observation is a dict with keys: {list(obs.keys())}", flush=True)
                            
                            # Check 'policy' group (most common)
                            if 'policy' in obs:
                                policy_obs = obs['policy']
                                if isinstance(policy_obs, torch.Tensor):
                                    policy_obs_np = policy_obs[0].cpu().numpy()
                                    print(f"[OBS] Policy obs shape: {policy_obs.shape}", flush=True)
                                    print(f"[OBS] Policy obs [env 0]: {policy_obs_np}", flush=True)
                                    
                                    # Try to identify joint pos/vel in observations
                                    # According to observation manager, joint_pos is typically first 6 values
                                    # joint_vel is typically next 6 values
                                    if len(policy_obs_np) >= 12:
                                        obs_joint_pos = policy_obs_np[:6]
                                        obs_joint_vel = policy_obs_np[6:12]
                                        print(f"[OBS] Extracted joint_pos from obs: {obs_joint_pos}", flush=True)
                                        print(f"[OBS] Extracted joint_vel from obs: {obs_joint_vel}", flush=True)
                                        print(f"[OBS] Compare with PhysX pos: {pos}", flush=True)
                                        print(f"[OBS] Compare with PhysX vel: {vel}", flush=True)
                        else:
                            # Single tensor observation
                            if isinstance(obs, torch.Tensor):
                                obs_np = obs[0].cpu().numpy() if obs.dim() > 1 else obs.cpu().numpy()
                                print(f"[OBS] Single tensor obs shape: {obs.shape}", flush=True)
                                print(f"[OBS] Obs values: {obs_np}", flush=True)
                    
                    # Print status every 20 steps
                    if step % 20 == 0:
                        print(f"\n[JOINT] Step {step:3d}: Phase={test_phase}", flush=True)
                        if wheel_ids is not None:
                            # Safely convert to numpy array (handle both tensor and list)
                            if hasattr(wheel_ids, 'cpu'):
                                wheel_ids_np = wheel_ids.cpu().numpy()
                            elif hasattr(wheel_ids, 'numpy'):
                                wheel_ids_np = wheel_ids.numpy()
                            else:
                                wheel_ids_np = np.array(wheel_ids)
                            wheel_pos = pos[wheel_ids_np]
                            wheel_vel = vel[wheel_ids_np]
                            print(f"         Wheels - Pos: {wheel_pos}, Vel: {wheel_vel}", flush=True)
                        if servo_ids is not None:
                            # Safely convert to numpy array
                            if hasattr(servo_ids, 'cpu'):
                                servo_ids_np = servo_ids.cpu().numpy()
                            elif hasattr(servo_ids, 'numpy'):
                                servo_ids_np = servo_ids.numpy()
                            else:
                                servo_ids_np = np.array(servo_ids)
                            servo_pos = pos[servo_ids_np]
                            servo_vel = vel[servo_ids_np]
                            print(f"         Servos - Pos: {servo_pos}, Vel: {servo_vel}", flush=True)
                        if propeller_ids is not None:
                            # Safely convert to numpy array
                            if hasattr(propeller_ids, 'cpu'):
                                prop_ids_np = propeller_ids.cpu().numpy()
                            elif hasattr(propeller_ids, 'numpy'):
                                prop_ids_np = propeller_ids.numpy()
                            else:
                                prop_ids_np = np.array(propeller_ids)
                            prop_pos = pos[prop_ids_np]
                            prop_vel = vel[prop_ids_np]
                            print(f"         Props - Pos: {prop_pos}, Vel: {prop_vel}", flush=True)
                        
                        # Also try to read from observation manager directly
                        if hasattr(env.unwrapped, 'observation_manager'):
                            try:
                                obs_mgr = env.unwrapped.observation_manager
                                # Try to compute observations explicitly
                                fresh_obs = obs_mgr.compute()
                                if isinstance(fresh_obs, dict) and 'policy' in fresh_obs:
                                    policy_obs = fresh_obs['policy'][0].cpu().numpy()
                                    print(f"[OBS]    Fresh policy obs (first 12): {policy_obs[:12]}", flush=True)
                            except Exception as e:
                                print(f"[OBS]    Could not get fresh obs: {e}", flush=True)
                
                except Exception as e:
                    if step < 10:  # Only print errors in first few steps
                        print(f"[JOINT] Error recording data: {e}", flush=True)
            
            # Change test phase every 20 steps for faster testing
            if test_joint_movement and step > 0 and step % 20 == 0:
                phases = ["idle", "wheels", "servos", "propellers"]
                current_idx = phases.index(test_phase)
                test_phase = phases[(current_idx + 1) % len(phases)]
                print(f"[JOINT] Switching to test phase: {test_phase}", flush=True)
            term = bool(getattr(terminated, "any", lambda: terminated)())
            trunc = bool(getattr(truncated, "any", lambda: truncated)())
            if term or trunc:
                env.reset()
        
        # Print joint movement test results
        if test_joint_movement and len(joint_positions) > 0:
            print(f"\n{'='*80}", flush=True)
            print(f"JOINT MOVEMENT TEST RESULTS", flush=True)
            print(f"{'='*80}", flush=True)
            
            try:
                import numpy as np
                
                # Convert to numpy arrays for analysis
                pos_array = np.array(joint_positions)
                vel_array = np.array(joint_velocities)
                
                print(f"[JOINT] Recorded {len(joint_positions)} data points", flush=True)
                
                # Analyze each joint type
                if wheel_ids is not None:
                    # Safely convert to numpy array (handle both tensor and list)
                    if hasattr(wheel_ids, 'cpu'):
                        wheel_ids_np = wheel_ids.cpu().numpy()
                    elif hasattr(wheel_ids, 'numpy'):
                        wheel_ids_np = wheel_ids.numpy()
                    else:
                        wheel_ids_np = np.array(wheel_ids)
                    wheel_pos_data = pos_array[:, wheel_ids_np]
                    wheel_vel_data = vel_array[:, wheel_ids_np]
                    
                    wheel_pos_range = np.ptp(wheel_pos_data, axis=0)  # Peak-to-peak range
                    wheel_vel_max = np.max(np.abs(wheel_vel_data), axis=0)
                    
                    print(f"[JOINT] Wheel Analysis:", flush=True)
                    print(f"         Position range: {wheel_pos_range}", flush=True)
                    print(f"         Max velocity: {wheel_vel_max}", flush=True)
                    
                    if np.any(wheel_pos_range > 0.01):
                        print(f"         ✓ Wheels are moving (position control working)", flush=True)
                    else:
                        print(f"         ⚠ Wheels not moving much - check velocity control", flush=True)
                
                if servo_ids is not None:
                    # Safely convert to numpy array
                    if hasattr(servo_ids, 'cpu'):
                        servo_ids_np = servo_ids.cpu().numpy()
                    elif hasattr(servo_ids, 'numpy'):
                        servo_ids_np = servo_ids.numpy()
                    else:
                        servo_ids_np = np.array(servo_ids)
                    servo_pos_data = pos_array[:, servo_ids_np]
                    servo_vel_data = vel_array[:, servo_ids_np]
                    
                    servo_pos_range = np.ptp(servo_pos_data, axis=0)
                    servo_vel_max = np.max(np.abs(servo_vel_data), axis=0)
                    
                    print(f"[JOINT] Servo Analysis:", flush=True)
                    print(f"         Position range: {servo_pos_range}", flush=True)
                    print(f"         Max velocity: {servo_vel_max}", flush=True)
                    
                    if np.any(servo_pos_range > 0.01):
                        print(f"         ✓ Servos are moving (position control working)", flush=True)
                    else:
                        print(f"         ⚠ Servos not moving much - check position control", flush=True)
                
                if propeller_ids is not None:
                    # Safely convert to numpy array
                    if hasattr(propeller_ids, 'cpu'):
                        prop_ids_np = propeller_ids.cpu().numpy()
                    elif hasattr(propeller_ids, 'numpy'):
                        prop_ids_np = propeller_ids.numpy()
                    else:
                        prop_ids_np = np.array(propeller_ids)
                    prop_pos_data = pos_array[:, prop_ids_np]
                    prop_vel_data = vel_array[:, prop_ids_np]
                    
                    prop_pos_range = np.ptp(prop_pos_data, axis=0)
                    prop_vel_max = np.max(np.abs(prop_vel_data), axis=0)
                    
                    print(f"[JOINT] Propeller Analysis:", flush=True)
                    print(f"         Position range: {prop_pos_range}", flush=True)
                    print(f"         Max velocity: {prop_vel_max}", flush=True)
                    
                    if np.any(prop_vel_max > 0.1):
                        print(f"         ✓ Propellers are spinning (velocity control working)", flush=True)
                    else:
                        print(f"         ⚠ Propellers not spinning - check velocity control", flush=True)
                
                # Overall assessment
                print(f"\n[JOINT] Overall Assessment:", flush=True)
                total_movement = np.sum([np.ptp(pos_array[:, i]) for i in range(pos_array.shape[1])])
                if total_movement > 0.1:
                    print(f"         ✓ Joints are responding to commands!", flush=True)
                    print(f"         ✓ Robot configuration appears correct", flush=True)
                else:
                    print(f"         ⚠ Limited joint movement detected", flush=True)
                    print(f"         ⚠ Check actuator configuration and action mapping", flush=True)
                
            except Exception as e:
                print(f"[JOINT] Error analyzing results: {e}", flush=True)
            
            print(f"{'='*80}\n", flush=True)

        print("[INFO] Simulation completed successfully. Close the window or press Ctrl+C to exit.", flush=True)

        while simulation_app.is_running():
            simulation_app.update()
    except Exception as exc:  # pragma: no cover - debugging helper
        print(f"[ERROR] {exc}", flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()

def hover_test(env, num_steps: int) -> None:
    """Run a hovering session: wheels stalled, servos at 0, prop thrust ramps to hover.

    Logic:
    - Compute robot mass and desired hover thrust = g * mass (split across 2 props).
    - Use thrust model F = k_t * omega^2 to compute target prop speeds.
    - Map target speeds to action space using configured scales/order: [wheels(2), servos(2), props(2)].
    - Ramp total thrust from 1.05x to 1.00x over a short warmup, then hold.
    """
    import math
    import torch

    device = env.device if hasattr(env, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover action layout and sizes (fallback to expected [2,2,2])
    wheel_count = 2
    servo_count = 2
    prop_count = 2

    # Scales must match ActionsCfg
    wheel_scale = 20.0
    servo_scale = 2.0
    prop_scale = 600.0

    # Aerodynamics thrust coefficient (must match DoubleBeeEventsCfg)
    thrust_coefficient = 1e-4  # N / (rad/s)^2

    # Compute robot mass
    robot = env.unwrapped.scene["robot"]
    mass = None
    try:
        # Prefer PhysX masses for accuracy
        masses = robot.root_physx_view.get_masses()[0].cpu()
        mass = float(torch.sum(masses).item())
    except Exception:
        try:
            body_masses = robot.data.body_masses[0].cpu()
            mass = float(torch.sum(body_masses).item())
        except Exception:
            exit(1)

    g = 9.81
    hover_total_thrust = g * mass  # Newtons
    print(f"[HOVER] Required Hover total thrust: {hover_total_thrust} N", flush=True)
    # Ramp: first 2 seconds from 1.05x to 1.0x, then hold
    physics_dt = getattr(env.unwrapped, "physics_dt", 0.005)
    ramp_steps = max(1, int(2.0 / physics_dt))

    # Prepare fixed action tensor
    # Try to infer expected shape from action_manager if available
    num_envs = getattr(env, "num_envs", 1)
    action_space = getattr(env, "action_space", None)
    inferred_manager_shape = None
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'action_manager') and hasattr(env.unwrapped.action_manager, 'action'):
        try:
            inferred_manager_shape = tuple(env.unwrapped.action_manager.action.shape)
        except Exception:
            inferred_manager_shape = None
    action_dim = action_space.shape[0] if hasattr(action_space, "shape") else wheel_count + servo_count + prop_count
    if inferred_manager_shape is not None and len(inferred_manager_shape) == 2:
        num_envs = inferred_manager_shape[0]
        action_dim = inferred_manager_shape[1]
    print(f"[HOVER] action_space={action_space}, manager_shape={inferred_manager_shape}, action_dim={action_dim}, num_envs={num_envs}", flush=True)
    action = torch.zeros((num_envs, action_dim), device=device, dtype=torch.float32)

    # Indices assuming order [wheels(2), servos(2), props(2)]
    wheel_slice = slice(0, wheel_count)
    servo_slice = slice(wheel_count, wheel_count + servo_count)
    prop_slice = slice(wheel_count + servo_count, wheel_count + servo_count + prop_count)

    # Sanity: if action_dim doesn't match 6, adapt to last-two-as-props fallback
    if action_dim < (wheel_count + servo_count + prop_count):
        # Use last 2 as props, nothing else
        wheel_slice = slice(0, 0)
        servo_slice = slice(0, 0)
        prop_slice = slice(max(0, action_dim - prop_count), action_dim)
        print(f"[HOVER] Non-standard action_dim={action_dim}, using prop_slice={prop_slice}", flush=True)

    # Fix wheels and servos to zero
    action[:, wheel_slice] = 0.0
    action[:, servo_slice] = 0.0

    print(f"[HOVER] Mass ~ {mass:.3f} kg, total hover thrust target {hover_total_thrust:.2f} N", flush=True)
    print(f"[HOVER] Using k_t={thrust_coefficient}, physics_dt={physics_dt}", flush=True)

    obs, _ = env.reset()

    for step in range(num_steps):
        # Ramp factor from 1.05 -> 1.0 over ramp_steps
        if step < ramp_steps:
            alpha = step / float(ramp_steps)
            thrust_factor = 1.5 - 0.5 * alpha
        else:
            thrust_factor = 1.1

        total_thrust_cmd = thrust_factor * hover_total_thrust
        per_prop_thrust = total_thrust_cmd / float(prop_count)

        # Compute target prop angular velocity from F = k_t * omega^2
        omega = math.sqrt(max(per_prop_thrust, 0.0) / max(thrust_coefficient, 1e-12))

        # Map to action space using scale: target_vel = scale * action
        prop_action_value = omega / prop_scale

        # Clamp to [-1, 1] to respect typical normalized bounds
        prop_action_value = max(-1.0, min(1.0, prop_action_value))

        # Assign opposite signs to the two propellers for gyroscopic balance
        # action[:, prop_slice] selects the propeller columns (shape: [num_envs, 2])
        if (prop_slice.stop - prop_slice.start) >= 2:
            prop_cmd = torch.empty((num_envs, 2), device=action.device, dtype=action.dtype)
            prop_cmd[:, 0] = prop_action_value   # left propeller: +
            prop_cmd[:, 1] = -prop_action_value  # right propeller: -
            action[:, prop_slice] = prop_cmd
        else:
            # Fallback: if only one prop dimension exists, just use positive
            action[:, prop_slice] = prop_action_value

        # Step the environment
        act_tensor = action.to(getattr(env, "device", action.device))
        # Align act_tensor shape with what action_manager expects if present
        try:
            if inferred_manager_shape is not None:
                if tuple(act_tensor.shape) != inferred_manager_shape:
                    if len(inferred_manager_shape) == 2 and inferred_manager_shape[0] == 1 and tuple(act_tensor.shape) == (action_dim,):
                        act_tensor = act_tensor.unsqueeze(0)
                    elif len(inferred_manager_shape) == 1 and tuple(act_tensor.shape) == (1, action_dim):
                        act_tensor = act_tensor.squeeze(0)
        except Exception:
            pass
        if step % 10 == 0:
            print(f"[HOVER] About to step: act_tensor.shape={tuple(act_tensor.shape)} device={act_tensor.device}", flush=True)
            try:
                if hasattr(env.unwrapped, 'action_manager'):
                    print(f"[HOVER] action_manager buffer shape: {tuple(env.unwrapped.action_manager.action.shape)}", flush=True)
            except Exception:
                pass
        obs, reward, terminated, truncated, info = env.step(act_tensor)

        if step % 50 == 0:
            # Basic telemetry
            try:
                root_pos = robot.data.root_pos_w[0]
                root_vel = robot.data.root_lin_vel_w[0]
                print(
                    f"[HOVER] step={step:5d} thrust_factor={thrust_factor:.3f} per_prop_N={per_prop_thrust:.2f} omega={omega:.1f} rad/s | z={root_pos[2]:.3f} vz={root_vel[2]:.3f}",
                    flush=True,
                )
            except Exception:
                pass

        term = bool(getattr(terminated, "any", lambda: terminated)())
        trunc = bool(getattr(truncated, "any", lambda: truncated)())
        if term or trunc:
            obs, _ = env.reset()

        # Keep simulation services updated
        env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

if __name__ == "__main__":
    main()
