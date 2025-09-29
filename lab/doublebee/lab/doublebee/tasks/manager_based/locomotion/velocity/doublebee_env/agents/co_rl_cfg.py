# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg


@configclass
class DoubleBeeCoRlCfg(CoRlPolicyRunnerCfg):
    """Configuration for DoubleBee CO-RL agent."""

    # Experiment settings
    experiment_name: str = "doublebee_velocity"
    run_name: str = "stand_drive"
    description: str = "DoubleBee robot velocity control with propellers"

    # Algorithm settings
    algorithm: CoRlPolicyRunnerCfg.AlgorithmCfg = CoRlPolicyRunnerCfg.AlgorithmCfg(
        class_name="PPO",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        clip_range=0.2,
        max_grad_norm=1.0,
    )

    # Policy settings
    policy: CoRlPolicyRunnerCfg.PolicyCfg = CoRlPolicyRunnerCfg.PolicyCfg(
        class_name="ActorCritic",
        hidden_sizes=[512, 256, 128],
        activation="tanh",
        init_noise_std=1.0,
    )

    # Value settings
    value: CoRlPolicyRunnerCfg.ValueCfg = CoRlPolicyRunnerCfg.ValueCfg(
        class_name="Critic",
        hidden_sizes=[512, 256, 128],
        activation="tanh",
    )

    # Training settings
    max_iterations: int = 5000
    save_interval: int = 500
    log_interval: int = 10
    eval_interval: int = 100
    num_eval_episodes: int = 10

    # Environment settings
    num_envs: int = 4096
    episode_length: int = 1000
    seed: int = 42

    # Device settings
    device: str = "cuda:0"
    num_threads: int = 4
