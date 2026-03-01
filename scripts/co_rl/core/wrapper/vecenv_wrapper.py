# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
import torch
from scripts.co_rl.core.env import VecEnv
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from lab.doublebee.isaaclab.isaaclab.envs.manager_based_constraint_rl_env import ManagerBasedConstraintRLEnv
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg
from scripts.co_rl.core.utils.state_handler import StateHandler

import os
import csv
import numpy as np
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt

class CoRlVecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv, agent_cfg: CoRlPolicyRunnerCfg):
        """
        Args:
            env: The environment to wrap around.
            stack_frames: Number of frames to stack for observations.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        # Get the actual unwrapped environment (handle nested wrappers)
        unwrapped_env = env
        while hasattr(unwrapped_env, 'unwrapped') and unwrapped_env.unwrapped is not unwrapped_env:
            unwrapped_env = unwrapped_env.unwrapped
        
        if not isinstance(unwrapped_env, ManagerBasedRLEnv) and not isinstance(unwrapped_env, DirectRLEnv) and not isinstance(unwrapped_env, ManagerBasedConstraintRLEnv):
            raise ValueError(
                f"The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv, or ManagerBasedConstraintRLEnv. "
                f"Environment type: {type(env)}, Unwrapped type: {type(unwrapped_env)}"
            )

        # initialize the wrapper
        self.env = env
        self.csv_path = os.path.join("logs", f"torque_vel_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.torque_log = []
        self.vel_log = []

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # Determine the number of policy and critic stacks
        self.num_policy_stacks = agent_cfg.num_policy_stacks
        self.num_critic_stacks = agent_cfg.num_critic_stacks

        # Determine if constraint RL is used
        self.use_constraint_rl = agent_cfg.use_constraint_rl
        
        # Check if environment has stacking-specific observation groups
        self.has_stacking_groups = False
        if hasattr(self.unwrapped, "observation_manager"):
            group_obs_dim = self.unwrapped.observation_manager.group_obs_dim
            # Check if this environment has separate stacking groups (like Flamingo)
            if all(key in group_obs_dim for key in ["stack_policy", "none_stack_policy", "stack_critic", "none_stack_critic"]):
                self.has_stacking_groups = True
                print("[INFO] Environment has dedicated stacking observation groups")
            else:
                print(f"[INFO] Environment uses standard observation groups (available: {list(group_obs_dim.keys())})")
                print(f"[INFO] Frame stacking (num_policy_stacks={self.num_policy_stacks}, num_critic_stacks={self.num_critic_stacks}) will be disabled")
    
        # Determine action and observation dimensions
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self.unwrapped.num_actions

        if hasattr(self.unwrapped, "observation_manager"):
            if self.has_stacking_groups:
                # -- Policy observations with stacking (Flamingo-style)
                stack_policy_dim = self.unwrapped.observation_manager.group_obs_dim["stack_policy"][0]
                nonstack_policy_dim = self.unwrapped.observation_manager.group_obs_dim["none_stack_policy"][0]
                self.policy_state_handler = StateHandler(self.num_policy_stacks + 1, stack_policy_dim, nonstack_policy_dim)
                # state handler 내부에서 최종 policy 차원(policy_dim)을 계산합니다.
                self.unwrapped.observation_manager.group_obs_dim["policy"] = (self.policy_state_handler.num_obs,)
                self.num_obs = self.policy_state_handler.num_obs
            else:
                # -- Standard policy observations without stacking (DoubleBee-style)
                self.policy_state_handler = None
                self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.policy_state_handler = None
            self.num_obs = self.unwrapped.num_observations

        # -- Privileged observations (Critic)
        if hasattr(self.unwrapped, "observation_manager"):
            if self.has_stacking_groups:
                # -- Critic observations with stacking (Flamingo-style)
                stack_critic_dim = self.unwrapped.observation_manager.group_obs_dim["stack_critic"][0]
                nonstack_critic_dim = self.unwrapped.observation_manager.group_obs_dim["none_stack_critic"][0]
                self.critic_state_handler = StateHandler(self.num_critic_stacks + 1, stack_critic_dim, nonstack_critic_dim)
                self.unwrapped.observation_manager.group_obs_dim["critic"] = (self.critic_state_handler.num_obs,)
                self.num_privileged_obs = self.critic_state_handler.num_obs
            else:
                # -- Standard critic observations without stacking (DoubleBee-style)
                self.critic_state_handler = None
                if "critic" in self.unwrapped.observation_manager.group_obs_dim:
                    self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
                else:
                    # If no critic group, use policy observations
                    self.num_privileged_obs = self.num_obs
        elif hasattr(self.unwrapped, "num_states"):
            self.critic_state_handler = None
            self.num_privileged_obs = self.unwrapped.num_states
        else:
            self.critic_state_handler = None
            self.num_privileged_obs = 0

        # Observation latency buffer: delays policy obs by N steps.
        # Critic always sees the current obs (asymmetric actor-critic).
        self.obs_latency_steps = agent_cfg.obs_latency_steps
        self._prev_policy_obs: torch.Tensor | None = None
        if self.obs_latency_steps > 0:
            print(f"[INFO] Observation latency enabled: policy sees obs delayed by {self.obs_latency_steps} step(s)")

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        # Policy observations
        if self.policy_state_handler is not None:
            # Use state handler for stacking (Flamingo-style)
            if self.policy_state_handler.stack_buffer is None:
                policy_obs = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
            else:
                policy_obs = self.policy_state_handler.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
            obs_dict["policy"] = policy_obs
        else:
            # Use standard policy observations without stacking (DoubleBee-style)
            policy_obs = obs_dict["policy"]

        # Seed the latency buffer (first call — no delay yet)
        if self.obs_latency_steps > 0 and self._prev_policy_obs is None:
            self._prev_policy_obs = policy_obs.clone()

        # Critic observations
        if self.critic_state_handler is not None:
            # Use state handler for stacking (Flamingo-style)
            if self.critic_state_handler.stack_buffer is None:
                critic_obs = self.critic_state_handler.reset(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            else:
                critic_obs = self.critic_state_handler.update(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            obs_dict["critic"] = critic_obs
        elif "critic" not in obs_dict:
            # If no separate critic observations, use policy observations
            obs_dict["critic"] = policy_obs

        return policy_obs, {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        
        # Policy observations reset
        if self.policy_state_handler is not None:
            # Use state handler for stacking (Flamingo-style)
            policy_obs = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
            obs_dict["policy"] = policy_obs
        else:
            # Use standard policy observations (DoubleBee-style)
            policy_obs = obs_dict["policy"]

        # Seed latency buffer with initial obs (no delay on first step after reset)
        if self.obs_latency_steps > 0:
            self._prev_policy_obs = policy_obs.clone()

        # Critic observations reset
        if self.critic_state_handler is not None:
            # Use state handler for stacking (Flamingo-style)
            critic_obs = self.critic_state_handler.reset(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            obs_dict["critic"] = critic_obs
        elif "critic" not in obs_dict:
            # If no separate critic observations, use policy observations
            obs_dict["critic"] = policy_obs

        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        if not self.use_constraint_rl:
            dones = (terminated | truncated).to(dtype=torch.long)
        else:
            dones = torch.max(terminated, truncated).to(dtype=torch.float32)

        # Update policy observations
        if self.policy_state_handler is not None:
            # Use state handler for stacking (Flamingo-style)
            policy_obs = self.policy_state_handler.update(
                obs_dict["stack_policy"], obs_dict["none_stack_policy"]
            )
            obs_dict["policy"] = policy_obs
        else:
            # Use standard policy observations (DoubleBee-style)
            policy_obs = obs_dict["policy"]

        # Apply observation latency: policy receives obs from the previous step.
        # Freshly-reset envs get the current obs (no stale cross-episode data).
        if self.obs_latency_steps > 0 and self._prev_policy_obs is not None:
            current_policy_obs = policy_obs
            delayed_policy_obs = self._prev_policy_obs.clone()
            # For envs that just reset, use current obs instead of stale previous
            reset_mask = (dones > 0).unsqueeze(-1)  # (num_envs, 1)
            policy_obs = torch.where(reset_mask, current_policy_obs, delayed_policy_obs)
            self._prev_policy_obs = current_policy_obs.clone()
            obs_dict["policy"] = policy_obs

        # Update critic observations (always current, no latency)
        if self.critic_state_handler is not None:
            # Use state handler for stacking (Flamingo-style)
            critic_obs = self.critic_state_handler.update(
                obs_dict["stack_critic"], obs_dict["none_stack_critic"]
            )
            obs_dict["critic"] = critic_obs
        elif "critic" not in obs_dict:
            # If no separate critic observations, use policy observations
            obs_dict["critic"] = policy_obs

        extras["observations"] = obs_dict

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return policy_obs, rew, dones, extras


    def close(self):  # noqa: D102
        return self.env.close()

