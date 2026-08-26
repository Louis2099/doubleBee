#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from uu import Error

import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from scripts import co_rl
from scripts.co_rl.core.algorithms import SAC
from scripts.co_rl.core.algorithms import TQC
from scripts.co_rl.core.algorithms import TACO
from scripts.co_rl.core.env import VecEnv
from scripts.co_rl.core.modules import ReplayMemory
from scripts.co_rl.core.utils import store_code_state

# class ActionDelayBuffer:
#     """Per-env action delay buffer for sim-to-real DR.
    
#     Each env gets a randomly sampled delay (in steps). push_and_get() stores
#     the latest action and returns the action from delay_steps ago.
#     On reset, the delayed actions for those envs are zeroed so the robot
#     doesn't execute stale pre-reset actions after respawn.
    
#     Args:
#         num_envs: Number of parallel environments.
#         action_dim: Dimension of the action vector.
#         max_delay: Maximum delay in steps (buffer size = max_delay + 1).
#         delay_steps: Per-env delay in steps. Shape [num_envs], dtype long.
#         device: Torch device.
#     """
#     def __init__(self, num_envs: int, action_dim: int, max_delay: int,
#                  delay_steps: torch.Tensor, device: torch.device):
#         self.max_delay = max_delay
#         self.delay_steps = delay_steps.long().to(device)   # [num_envs]
#         # Circular buffer: shape [max_delay+1, num_envs, action_dim]
#         self.buf = torch.zeros(max_delay + 1, num_envs, action_dim, device=device)
#         self.ptr = 0  # points to the slot we write into next

#     def push_and_get(self, action: torch.Tensor) -> torch.Tensor:
#         """Store current action, return per-env delayed action.
        
#         Args:
#             action: Current actions. Shape [num_envs, action_dim].
#         Returns:
#             Delayed actions. Shape [num_envs, action_dim].
#         """
#         # Write current action into buffer at ptr
#         self.buf[self.ptr] = action

#         # For each env, read from (ptr - delay) mod buffer_size
#         # delay_steps[i] steps ago = slot (ptr - delay_steps[i]) % buf_size
#         buf_size = self.max_delay + 1
#         read_ptrs = (self.ptr - self.delay_steps) % buf_size  # [num_envs]

#         # Gather: for each env i, read buf[read_ptrs[i], i, :]
#         # read_ptrs: [num_envs] → expand to [num_envs, 1, action_dim] for gather
#         idx = read_ptrs.view(-1, 1, 1).expand(-1, 1, action.shape[1])  # [num_envs, 1, action_dim]
#         delayed = self.buf.permute(1, 0, 2).gather(1, idx).squeeze(1)  # [num_envs, action_dim]

#         # Advance pointer
#         self.ptr = (self.ptr + 1) % buf_size

#         return delayed

#     def reset(self, env_ids: torch.Tensor):
#         """Zero out buffer for reset envs so stale pre-reset actions aren't executed."""
#         self.buf[:, env_ids, :] = 0.0
#         # Optionally resample per-env delay on reset for DR variety:
#         self.delay_steps[env_ids] = torch.randint(
#             low=1, high=self.max_delay + 1,
#             size=(len(env_ids),),
#             device=self.delay_steps.device
#         )

class OffPolicyRunner:
    """Off-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg

        self.total_steps = 0
        self.device = device
        self.env = env
        self.num_envs = env.num_envs
        obs, extras = self.env.get_observations()  # obs.shape := (num_envs, num_obs)
        obs_dims = obs.shape[1]
        if "critic" in extras["observations"]:
            critic_obs_dims = extras["observations"]["critic"].shape[1]
        else:
            critic_obs_dims = obs_dims

        if self.cfg["algorithm"]["class_name"] == "SAC":
            self.alg: SAC = SAC(
                critic_obs_dims,
                self.env.num_actions,
                actor_hidden_dims=self.cfg["policy"]["actor_hidden_dims"],
                critic_hidden_dims=self.cfg["policy"]["critic_hidden_dims"],
                num_envs=self.num_envs,
                device=self.device,
            )
        elif self.cfg["algorithm"]["class_name"] == "TQC":
            self.alg: TQC = TQC(
                critic_obs_dims,
                self.env.num_actions,
                actor_hidden_dims=self.cfg["policy"]["actor_hidden_dims"],
                critic_hidden_dims=self.cfg["policy"]["critic_hidden_dims"],
                num_envs=self.num_envs,
                device=self.device,
            )
        elif self.cfg["algorithm"]["class_name"] == "TACO":
            self.alg: TACO = TACO(
                critic_obs_dims,
                self.env.num_actions,
                actor_hidden_dims=self.cfg["policy"]["actor_hidden_dims"],
                critic_hidden_dims=self.cfg["policy"]["critic_hidden_dims"],
                num_envs=self.num_envs,
                device=self.device,
            )
        else:
            raise Error("Algorithm not found")

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [co_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()

        # Action delay DR — randomize delay per env between min/max steps
        # Matches sysid: M1=5.6ms, M2=2.6ms at 5ms sim dt → 1-3 steps
        # _delay_steps = torch.randint(
        #     low=1, high=4,  # [1, 3] steps uniformly per env
        #     size=(self.num_envs,),
        #     device=self.device
        # )
        # self._action_delay_buffer = ActionDelayBuffer(
        #     num_envs=self.num_envs,
        #     action_dim=self.env.num_actions,
        #     max_delay=3,
        #     delay_steps=_delay_steps,
        #     device=self.device,
        # )

        # # Pre-fill buffer with initial action so first steps aren't zeros
        # with torch.no_grad():
        #     init_actions = self.alg.act_inference(obs.to(self.device))
        #     for _ in range(3):  # fill all slots
        #         self._action_delay_buffer.push_and_get(init_actions)
                
        obs = obs.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            for i in range(self.num_steps_per_env):
                self.total_steps += self.num_envs
                actions = self.alg.act(obs, self.total_steps)
                # delayed_actions = self._action_delay_buffer.push_and_get(actions)
                # next_obs, rewards, dones, infos = self.env.step(delayed_actions.to(self.env.device))
                next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                # These new variables are in self.env.device (mostly in cuda:0)

                if "time_outs" in infos:
                    timeout_mask = infos["time_outs"].bool()
                    indices_to_update = (dones.bool() & timeout_mask).nonzero(as_tuple=True)[0]
                    dones[indices_to_update] = 0

                # process the step
                self.alg.process_env_step(obs, actions, rewards, next_obs, dones)
                # self.alg.process_env_step(obs, delayed_actions, rewards, next_obs, dones)
                # reset_mask = (dones > 0)
                # if reset_mask.any():
                #     self._action_delay_buffer.reset(reset_mask.nonzero(as_tuple=True)[0])
                obs = next_obs

                if self.log_dir is not None:
                    # Book keeping
                    # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                    # different types of logging data (rewards, curriculum, etc.)
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    reset_mask = (dones > 0) | (timeout_mask if "time_outs" in infos else torch.zeros_like(dones).bool())
                    new_ids = reset_mask.nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

            # if self.total_steps > self.alg.update_after:
                # self.alg.update(update_cnt=self.num_steps_per_env) # * self.env.num_envs
            
            # if len(self.alg.buffer) > self.alg.update_after:   # or self.alg.buffer.size, check attr
            #     self.alg.update(update_cnt=self.num_steps_per_env)

            # print(f"[DEBUG] buffer.size: {self.alg.buffer.size}, update_after: {self.alg.update_after}")

            if self.alg.buffer.size > 50000:   # ~1 iter of fill at 2048 envs before updating
                self.alg.update(update_cnt=self.num_steps_per_env)
                self._grad_steps = getattr(self, "_grad_steps", 0) + self.num_steps_per_env

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            def _to_1d_tensor(value):
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=self.device)
                else:
                    value = value.to(self.device)
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                return value.reshape(-1)

            skip_keys = set()
            def _aggregate_weighted_metric(
                numerator_key: str,
                denominator_key: str,
                output_key: str,
                extra_skip_keys=None,
            ):
                nonlocal ep_string
                numerators = []
                denominators = []
                for ep_info in locs["ep_infos"]:
                    if numerator_key in ep_info and denominator_key in ep_info:
                        numerators.append(_to_1d_tensor(ep_info[numerator_key]))
                        denominators.append(_to_1d_tensor(ep_info[denominator_key]))
                if not denominators:
                    return

                numerator_sum = torch.cat(numerators).sum()
                denominator_sum = torch.cat(denominators).sum()
                if denominator_sum.item() <= 0:
                    return

                weighted_value = numerator_sum / denominator_sum
                self.writer.add_scalar(output_key, weighted_value, locs["it"])
                ep_string += f"""{f'{output_key}:':>{pad}} {weighted_value:.4f}\n"""
                skip_keys.update({output_key, numerator_key, denominator_key})
                if extra_skip_keys is not None:
                    skip_keys.update(extra_skip_keys)
                return numerator_sum, denominator_sum

            success_aggregates = _aggregate_weighted_metric(
                numerator_key="Metrics/success/count",
                denominator_key="Metrics/success/total",
                output_key="Metrics/success/rate",
            )
            if success_aggregates is not None:
                total_success, total_episodes = success_aggregates
                self.writer.add_scalar("Metrics/success/total_successful_trajectories", total_success, locs["it"])
                self.writer.add_scalar("Metrics/success/total_trajectories", total_episodes, locs["it"])
                ep_string += f"""{f'Metrics/success/total_successful_trajectories:':>{pad}} {total_success:.1f}\n"""
                ep_string += f"""{f'Metrics/success/total_trajectories:':>{pad}} {total_episodes:.1f}\n"""

            _aggregate_weighted_metric(
                numerator_key="Metrics/energy/sum",
                denominator_key="Metrics/energy/count",
                output_key="Metrics/energy/average_consumption",
            )
            _aggregate_weighted_metric(
                numerator_key="Metrics/energy/successful_sum",
                denominator_key="Metrics/energy/successful_count",
                output_key="Metrics/energy/successful_trajectories",
            )

            for key in locs["ep_infos"][0]:
                if key in skip_keys:
                    continue
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    infotensor = torch.cat((infotensor, _to_1d_tensor(ep_info[key])))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        # 2026-08-26: surface optimizer state. Without these there is no way to
        # tell a policy that is learning slowly from one that is not updating at
        # all -- which is what hid a replay ratio of 0.125 for 172 iterations.
        if hasattr(self.alg, "last_critic_loss"):
            log_string += (
                f"""{'Critic loss:':>{pad}} {self.alg.last_critic_loss:.4f}\n"""
                f"""{'Actor loss:':>{pad}} {self.alg.last_actor_loss:.4f}\n"""
                f"""{'Alpha (entropy temp):':>{pad}} {self.alg.last_alpha:.4f}\n"""
                f"""{'Buffer size:':>{pad}} {self.alg.buffer.size}\n"""
                f"""{'Grad steps taken:':>{pad}} {getattr(self, '_grad_steps', 0)}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "target_critic_state_dict": self.alg.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "total_steps": self.total_steps,
            "replay_buffer": self.alg.buffer.state_dict(),
            "log_alpha": self.alg.log_alpha.detach(),
            "infos": infos,
        }
        torch.save(saved_dict, path)
        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])           # uncomment
        self.alg.target_critic.load_state_dict(loaded_dict["target_critic_state_dict"])  # uncomment
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])  # uncomment
        self.current_learning_iteration = loaded_dict["iter"]
        self.total_steps = loaded_dict.get("total_steps", 0)

        # if "replay_buffer" in loaded_dict:
        #     self.alg.buffer.load_state_dict(loaded_dict["replay_buffer"])

        # In load(), if checkpoint predates log_alpha saving, set it low manually
        if "log_alpha" in loaded_dict:
            with torch.no_grad():
                self.alg.log_alpha.copy_(loaded_dict["log_alpha"])
        else:
            with torch.no_grad():
                self.alg.log_alpha.fill_(-2.0)  # exp(-2)=0.135, low exploration for fine-tuning       

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor.to(device)
            # self.alg.critic.to(device)
            # self.alg.target_critic.to(device)
        policy = self.alg.act_inference
        return policy

    def train_mode(self):
        self.alg.actor.train()
        self.alg.critic.train()

    def eval_mode(self):
        self.alg.actor.eval()
        self.alg.critic.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
