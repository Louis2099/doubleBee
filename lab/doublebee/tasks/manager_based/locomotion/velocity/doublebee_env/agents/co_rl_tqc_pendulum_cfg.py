from __future__ import annotations
from isaaclab.utils import configclass
from .co_rl_tqc_cfg import CoRlTqcNetCfg
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg


@configclass
class DoubleBeeInvertedPendulumCoRlTqcCfg(CoRlPolicyRunnerCfg):
    """TQC runner for the wheels-only balance task.

    Added 2026-08-26. Exists only so this can run SIDE BY SIDE with the
    hybrid_stair TQC run -- DoubleBeeCoRlTqcCfg hardcodes
    run_name="hybrid_stair_tqc", so sharing it would make the two runs write to
    the same log directory and overwrite each other's checkpoints.

    Everything else is deliberately identical to DoubleBeeCoRlTqcCfg (same nets,
    same num_steps_per_env, same seed), so the only differences between the two
    runs are the task and the number of envs.
    """

    experiment_name: str = "doublebee_velocity"
    run_name: str = "inverted_pendulum_tqc"
    empirical_normalization: bool = False
    policy: CoRlTqcNetCfg = CoRlTqcNetCfg()
    algorithm: CoRlTqcNetCfg = CoRlTqcNetCfg()
    num_steps_per_env: int = 24
    save_interval: int = 100
    device: str = "cuda:0"
    seed: int = 42
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    logger: str = "tensorboard"
    experiment_description: str = ""
