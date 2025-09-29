# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SpawnCfg, sim_utils
from isaaclab.utils import configclass

from lab.doublebee.assets.doublebee import DOUBLEBEE_CFG
from lab.doublebee.tasks.manager_based.locomotion.velocity.mdp import (
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class DoubleBeeFlatStandDriveCfg(ManagerBasedRLEnvCfg):
    """Configuration for DoubleBee flat terrain stand and drive environment."""

    # Scene settings
    class Scene(InteractiveSceneCfg):
        """World simulation configuration."""

        # Ground plane
        ground = sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                linear_damping=0.0,
                angular_damping=0.0,
            ),
        )

        # Robot
        robot = DOUBLEBEE_CFG.replace(prim_path="/World/envs/env_.*/Doublebee")

    # MDP settings
    class MDP(ManagerBasedRLEnvCfg.MDP):
        """MDP configuration."""

        # Observations
        observations = ObservationsCfg()

        # Rewards
        rewards = RewardsCfg()

        # Terminations
        terminations = TerminationsCfg()

        # Curriculum
        curriculum = CurriculumCfg()

    # Environment settings
    episode_length_s: float = 20.0
    decimation: int = 4

    # Observation settings
    state_term: ObsTerm = ObsTerm(func=lambda env: env.scene["robot"].data.root_lin_vel_b)
    """Root linear velocity in base frame."""

    # Action settings
    actions: RewTerm = RewTerm(func=lambda env: env.action_manager.get_term("base_actions").raw_actions)
    """Joint actions."""

    # Reward settings
    rewards: RewTerm = RewTerm(func=lambda env: env.reward_manager.compute())
    """Total reward."""

    # Termination settings
    terminations: DoneTerm = DoneTerm(func=lambda env: env.termination_manager.compute())
    """Episode termination."""

    # Curriculum settings
    curriculum: CurriculumTerm = CurriculumTerm(func=lambda env: env.curriculum_manager.compute())
    """Curriculum learning."""
