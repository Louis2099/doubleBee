# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from lab.doublebee.isaaclab.isaaclab.managers import ConstraintManager

from isaaclab.ui.widgets import ManagerLiveVisualizer

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from .manager_based_constraint_rl_env_cfg import ManagerBasedConstraintRLEnvCfg


class ManagerBasedConstraintRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedConstraintRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedConstraintRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- energy tracking buffers
        self.episode_energy_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.episode_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")
        print(f"[DEBUG Environment Init] max_episode_length = {self.max_episode_length} steps")
        print(f"[DEBUG Environment Init] episode_length_s = {self.cfg.episode_length_s}s")
        print(f"[DEBUG Environment Init] step_dt = {self.step_dt}s")
        print(f"[DEBUG Environment Init] episode_length_buf initialized: {self.episode_length_buf.shape}")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager as ConstraintManager
        self.constraint_manager = ConstraintManager(self.cfg.constraints, self)
        print("[INFO] Constraint Manager: ", self.constraint_manager)

        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "constraint_manager": ManagerLiveVisualizer(manager=self.constraint_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # NOTE: process actions
        # For 6D action space: duplicate left servo/propeller to right with same value (opposite signs via scales)
        # For 4D action space: action manager already handles both joints directly
        # Clone to avoid modifying the original tensor
        action = action.clone()
        
        if action.shape[-1] == 6:
            # 6D action space: apply constraints for backward compatibility
            if action.shape[-1] >= 4:  # Ensure we have at least 4 actions (servos at indices 2 and 3)
                action[..., 3] = action[..., 2]  # Duplicate left servo action to right servo
            if action.shape[-1] >= 6:  # Ensure we have propeller actions (at indices 4 and 5)
                action[..., 5] = action[..., 4]  # Duplicate left propeller action to right propeller
        # else: 4D action space - no duplication needed, action manager handles both joints
        
        # print(f"[INTERNAL CHECK] Action passed to the env.step(): {action}", flush=True)
        self.action_manager.process_action(action.to(self.device))
        # print(f"[INTERNAL CHECK] Action manager processed action: {self.action_manager.action}", flush=True)
        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        if not hasattr(self, '_debug_constraint_counter'):
            self._debug_constraint_counter = 0
        self._debug_constraint_counter += 1
        
        self.reset_buf = self.constraint_manager.compute()
        self.reset_delta = self.constraint_manager.constrained
        self.reset_time_outs = self.constraint_manager.time_outs
        
        # Debug: Log constraint status periodically
        if self._debug_constraint_counter % 500 == 0:
            print(f"[DEBUG Constraints] Step {self._debug_constraint_counter}:")
            print(f"  episode_length_buf[0] = {self.episode_length_buf[0]}")
            print(f"  max_episode_length = {self.max_episode_length}")
            print(f"  reset_buf[0] = {self.reset_buf[0]}")
            print(f"  reset_time_outs[0] = {self.reset_time_outs[0]}")
            print(f"  reset_delta[0] = {self.reset_delta[0]}")
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- energy tracking: calculate power consumption and accumulate energy
        self._update_energy_tracking()

        # -- success tracking: check if robot reached target (uses goal_reached constraint)
        self._update_success_tracking()

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that constrained/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # Debug: Log why episodes are resetting
            if not hasattr(self, '_debug_reset_counter'):
                self._debug_reset_counter = 0
            self._debug_reset_counter += 1
            if self._debug_reset_counter % 10 == 0:  # Log every 10th reset
                print(f"[DEBUG _reset_idx] Resetting {len(reset_env_ids)} envs: {reset_env_ids.tolist()}")
                print(f"  episode_length_buf: {self.episode_length_buf[reset_env_ids].tolist()}")
                print(f"  max_episode_length: {self.max_episode_length}")
                print(f"  time_outs: {self.reset_time_outs[reset_env_ids].tolist()}")
                print(f"  delta (early termination): {self.reset_delta[reset_env_ids].tolist()}")
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_delta, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.constraint_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _update_energy_tracking(self):
        """Calculate instantaneous power from propellers and wheels, then accumulate energy.
        
        Uses the thrust_energy_model API to convert:
        - Propeller PWM -> Power (W)
        - Wheel RPM -> Power (W)
        
        Energy (J) = Power (W) * dt (s)
        """
        from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp.thrust_energy_model import (
            pwm_to_thrust,
            rpm_to_power,
        )
        
        robot = self.scene["robot"]
        
        # Get propeller and wheel joint velocities (radians/sec)
        # Assuming joint ordering: [left_propeller, right_propeller, left_wheel, right_wheel]
        joint_vel = robot.data.joint_vel  # [num_envs, num_joints]
        
        # Convert propeller velocities to PWM (assuming direct mapping or using action)
        # For now, use joint velocity as proxy for PWM (scale appropriately)
        # Note: You may need to adjust this based on your actuator model
        # Assuming propellers are first 2 joints and wheels are last 2 joints
        
        try:
            # Get propeller velocities (first 2 joints) - convert rad/s to PWM range
            # Typical propeller: ~0-3000 rad/s maps to PWM 1000-2000
            prop_left_vel = joint_vel[:, 0]  # rad/s
            prop_right_vel = joint_vel[:, 1]  # rad/s
            
            # Map rad/s to PWM (linear approximation: 0 rad/s -> 1000 PWM, 3000 rad/s -> 2000 PWM)
            pwm_left = 1000.0 + (prop_left_vel.abs() / 500.0).clamp(0, 1) * 650.0
            pwm_right = 1000.0 + (prop_right_vel.abs() / 500.0).clamp(0, 1) * 650.0
            
            # Get wheel velocities (last 2 joints) - convert rad/s to RPM
            wheel_left_vel = joint_vel[:, 2]  # rad/s
            wheel_right_vel = joint_vel[:, 3]  # rad/s
            
            # Convert rad/s to RPM: RPM = (rad/s * 60) / (2 * pi)
            rpm_left = (wheel_left_vel.abs() * 60.0 / (2.0 * 3.14159265359))
            rpm_right = (wheel_right_vel.abs() * 60.0 / (2.0 * 3.14159265359))
            
            # Calculate power for each component (convert to numpy for API, then back to torch)
            power_prop_left = torch.tensor([
                pwm_to_thrust(pwm.item(), target="power") 
                for pwm in pwm_left
            ], device=self.device, dtype=torch.float32)
            
            power_prop_right = torch.tensor([
                pwm_to_thrust(pwm.item(), target="power") 
                for pwm in pwm_right
            ], device=self.device, dtype=torch.float32)
            
            power_wheel_left = torch.tensor([
                rpm_to_power(rpm.item()) 
                for rpm in rpm_left
            ], device=self.device, dtype=torch.float32)
            
            power_wheel_right = torch.tensor([
                rpm_to_power(rpm.item()) 
                for rpm in rpm_right
            ], device=self.device, dtype=torch.float32)
            
            # Total instantaneous power (W) = sum of all components
            total_power = power_prop_left + power_prop_right + power_wheel_left + power_wheel_right
            
            # Energy (J) = Power (W) * time (s)
            energy_step = total_power * self.step_dt
            
            # Accumulate energy for this episode
            self.episode_energy_buf += energy_step
            
        except Exception as e:
            # If energy calculation fails, skip silently to avoid breaking training
            # print(f"[WARNING] Energy tracking failed: {e}")
            pass

    def _update_success_tracking(self):
        """Track if robot has reached the target successfully.
        
        Uses the existing 'goal_reached' constraint from the constraint manager.
        If goal_reached constraint fires (value = 1.0), marks the episode as successful.
        
        Success is tracked cumulatively: once successful, stays successful for the episode.
        """
        # Check if goal_reached constraint exists in constraint manager
        if not hasattr(self.constraint_manager, "_term_values"):
            return
        
        if "goal_reached" not in self.constraint_manager._term_values:
            return
        
        try:
            # Get the goal_reached constraint values [num_envs]
            # 1.0 = goal reached, 0.0 = not reached
            goal_reached_values = self.constraint_manager._term_values["goal_reached"]
            
            # Mark as success if goal_reached fires (once successful, stays successful)
            # Convert to bool: goal_reached_values > 0.5 to handle float comparisons
            is_success = goal_reached_values > 0.5
            
            # Cumulative tracking: once successful, stays successful for the episode
            self.episode_success_buf = torch.logical_or(self.episode_success_buf, is_success)
            
        except Exception as e:
            # If success tracking fails, skip silently
            # print(f"[WARNING] Success tracking failed: {e}")
            pass

    def _log_energy_and_success_metrics(self, env_ids: Sequence[int]):
        """Log energy consumption and success metrics for reset environments.
        
        This is called in _reset_idx() to aggregate metrics over the just-completed episodes.
        
        Args:
            env_ids: List of environment ids being reset.
        """
        if len(env_ids) == 0:
            return
        
        # Convert env_ids to tensor for indexing
        env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        
        # 1. Energy over all completed trajectories in this reset batch
        energy_values = self.episode_energy_buf[env_ids_tensor]
        energy_sum = energy_values.sum().item()
        energy_count = len(env_ids)
        avg_energy = energy_sum / energy_count if energy_count > 0 else 0.0
        self.extras["log"]["Metrics/energy/sum"] = float(energy_sum)
        self.extras["log"]["Metrics/energy/count"] = float(energy_count)
        self.extras["log"]["Metrics/energy/average_consumption"] = avg_energy
        
        # 2. Success rate
        success_count = self.episode_success_buf[env_ids_tensor].sum().item()
        total_count = len(env_ids)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        # Log both numerator and denominator so runners can aggregate exact weighted rate across reset batches.
        self.extras["log"]["Metrics/success/count"] = float(success_count)
        self.extras["log"]["Metrics/success/total"] = float(total_count)
        self.extras["log"]["Metrics/success/rate"] = success_rate
        
        # 3. Energy consumption for successful trajectories only
        successful_mask = self.episode_success_buf[env_ids_tensor]
        if successful_mask.any():
            successful_energy_values = self.episode_energy_buf[env_ids_tensor][successful_mask]
            successful_energy_sum = successful_energy_values.sum().item()
            successful_count = int(successful_mask.sum().item())
            avg_energy_success = successful_energy_sum / successful_count
            self.extras["log"]["Metrics/energy/successful_sum"] = float(successful_energy_sum)
            self.extras["log"]["Metrics/energy/successful_count"] = float(successful_count)
            self.extras["log"]["Metrics/energy/successful_trajectories"] = avg_energy_success
        else:
            # If no successful trajectories, log 0 or NaN
            self.extras["log"]["Metrics/energy/successful_sum"] = 0.0
            self.extras["log"]["Metrics/energy/successful_count"] = 0.0
            self.extras["log"]["Metrics/energy/successful_trajectories"] = 0.0
        
        # Reset the buffers for these environments
        self.episode_energy_buf[env_ids_tensor] = 0.0
        self.episode_success_buf[env_ids_tensor] = False

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_dim)
                })
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        
        # CRITICAL: Apply aligned targets AFTER command manager reset
        # If using aligned initialization (DoubleBeeEventsCfg_PLAY), the event manager
        # stored aligned targets in a buffer during event reset. We must apply them here
        # to override the random targets sampled by command_manager.reset().
        if hasattr(self, "_aligned_targets_buffer"):
            from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import events as mdp_events
            mdp_events.apply_aligned_targets_to_command_manager(self, env_ids)
        
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- constraint manager
        info = self.constraint_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # -- log custom metrics for reset environments
        self._log_energy_and_success_metrics(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
