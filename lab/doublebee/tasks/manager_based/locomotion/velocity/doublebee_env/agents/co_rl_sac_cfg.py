from __future__ import annotations
from isaaclab.utils import configclass
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg, CoRlOffPolicyCfg

@configclass
class CoRlTqcNetCfg(CoRlOffPolicyCfg):
    class_name: str = "SAC"
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]

@configclass
class DoubleBeeCoRlSacCfg(CoRlPolicyRunnerCfg):
    experiment_name: str = "doublebee_velocity"
    run_name: str = "hybrid_stair_sac"
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
