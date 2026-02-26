# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass
from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
)


@configclass
class DoubleBeeCoRlCfg(CoRlPolicyRunnerCfg):
    """Configuration for DoubleBee CO-RL agent."""

    # Experiment settings
    experiment_name: str = "doublebee_velocity"
    run_name: str = "hybrid_stair"
    description: str = "DoubleBee robot velocity control with propellers"

    # Algorithm settings
    empirical_normalization: bool = False
    policy: CoRlPpoActorCriticCfg = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="tanh",
    )
    algorithm: CoRlPpoAlgorithmCfg = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    num_steps_per_env: int = 24

    # Training settings
    max_iterations: int = 5000
    save_interval: int = 500
    log_interval: int = 1000
    eval_interval: int = 100
    num_eval_episodes: int = 10

    # Environment settings
    num_envs: int = 4096
    episode_length: int = 1000
    seed: int = 42

    # Device settings
    device: str = "cuda:0"
    num_threads: int = 4
