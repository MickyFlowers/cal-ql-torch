import copy

# nessary imports
import d4rl
import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from cal_ql.cal_ql_sac_trainer import Trainer
from model.model import (FullyConnectedQFunction, SamplerPolicy,
                         TanhGaussianPolicy)
from utils.d4rl_utils import (get_d4rl_dataset_with_mc_calculation,
                              get_hand_dataset_with_mc_calculation)
from utils.logger import WandBLogger
from utils.replay_buffer import ReplayBuffer
from utils.sampler import TrajSampler
from utils.utils import (Timer, batch_to_torch, concatenate_batches,
                         subsample_batch)
from viskit.logging import logger, setup_logger


@hydra.main(config_path="../config", config_name="cal_ql", version_base=None)
def main(cfg: DictConfig):
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )
    if cfg.env in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:

        dataset = get_hand_dataset_with_mc_calculation(
            cfg.env,
            gamma=cfg.cal_ql.discount,
            reward_scale=cfg.reward_scale,
            reward_bias=cfg.reward_bias,
            clip_action=cfg.clip_action,
        )
        use_goal = True
    else:
        dataset = get_d4rl_dataset_with_mc_calculation(
            cfg.env, cfg.reward_scale, cfg.reward_bias, cfg.clip_action, gamma=cfg.cal_ql.discount
        )
        use_goal = False

    assert dataset["next_observations"].shape == dataset["observations"].shape
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    eval_sampler = TrajSampler(gym.make(cfg.env).unwrapped, use_goal, gamma=cfg.cal_ql.discount)
    train_sampler = TrajSampler(
        gym.make(cfg.env).unwrapped,
        use_goal,
        use_mc=True,
        gamma=cfg.cal_ql.discount,
        reward_scale=cfg.reward_scale,
        reward_bias=cfg.reward_bias,
    )
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim,
        action_dim,import copy
import itertools

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from cal_ql.cal_ql_sac_trainer import Trainer
from data.dataset import CalqlDataset
from model.model import ResNetPolicy, ResNetQFunction
from utils.logger import WandBLogger
from utils.replay_buffer import ReplayBuffer
from utils.utils import Timer
from viskit.logging import logger, setup_logger


def dict_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = dict_to_device(v, device)
        else:
            batch[k] = v.to(device=device, non_blocking=True)
    return batch

@hydra.main(config_path="../config", config_name="cal_ql", version_base=None)
def main(cfg: DictConfig):
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )
    dataset = CalqlDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    iter = itertools.cycle(dataloader)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

    policy = ResNetPolicy(
        observation_dim,
        action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        cfg.policy_log_std_multiplier,
        cfg.policy_log_std_offset,
        train_backbone=cfg.train_policy_backbone,
    )
    qf = {}
    qf['qf1'] = ResNetQFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone
    )
    qf['qf2'] = ResNetQFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone
    )
    qf['target_qf1'] = copy.deepcopy(qf['qf1'])
    qf['target_qf2'] = copy.deepcopy(qf['qf2'])
    
    cfg.cal_ql.target_entropy = -np.prod((1, action_dim)).item()

    sac = Trainer(cfg.cal_ql, policy, qf)
    sac.to_device(cfg.device)
    print("compiling sac model...")
    sac.compile(mode=cfg.torch_compile_mode)

    viskit_metrics = {}
    n_train_step_per_epoch = cfg.n_train_step_per_epoch_offline
    cql_min_q_weight = cfg.cql_min_q_weight
    enable_calql = cfg.enable_calql
    use_cql = cfg.use_cql
    total_grad_steps = 0
    train_timer = None
    epoch = 0
    train_metrics = None
    expl_metrics = None
    while True:
        metrics = {"epoch": epoch}
        metrics["grad_steps"] = total_grad_steps
        metrics["epoch"] = epoch
        metrics["train_time"] = 0 if train_timer is None else train_timer()
        if train_metrics is not None:
            metrics.update(train_metrics)
        if expl_metrics is not None:
            metrics.update(expl_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if epoch >= cfg.train_offline_epochs:
            print("Finished Training")
            break

        with Timer() as train_timer:
            for _ in range(n_train_step_per_epoch):
                batch = next(iter)
                batch = dict_to_device(batch, cfg.device)
                train_metrics = sac.train(
                    batch, use_cql=use_cql, cql_min_q_weight=cql_min_q_weight, enable_calql=enable_calql
                )
            total_grad_steps += n_train_step_per_epoch
        epoch += 1


if __name__ == "__main__":
    main()

        cfg.policy_arch,
        cfg.orthogonal_init,
        cfg.policy_log_std_multiplier,
        cfg.policy_log_std_offset,
    )
    qf = {}
    qf['qf1'] = FullyConnectedQFunction(observation_dim, action_dim, cfg.qf_arch, cfg.orthogonal_init)
    qf['qf2'] = FullyConnectedQFunction(observation_dim, action_dim, cfg.qf_arch, cfg.orthogonal_init)
    qf['target_qf1'] = copy.deepcopy(qf['qf1'])
    qf['target_qf2'] = copy.deepcopy(qf['qf2'])
    
    cfg.cal_ql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = Trainer(cfg.cal_ql, policy, qf)
    sac.to_device(cfg.device)
    print("compiling sac model...")
    sac.compile(mode=cfg.torch_compile_mode)
    sampler_policy = SamplerPolicy(sac.policy, cfg.device)

    viskit_metrics = {}
    n_train_step_per_epoch = cfg.n_train_step_per_epoch_offline
    cql_min_q_weight = cfg.cql_min_q_weight
    enable_calql = cfg.enable_calql
    use_cql = cfg.use_cql
    mixing_ratio = cfg.mixing_ratio

    total_grad_steps = 0
    is_online = False
    online_eval_counter = -1
    do_eval = False
    online_rollout_timer = None
    train_timer = None
    epoch = 0
    train_metrics = None
    expl_metrics = None
    while True:
        metrics = {"epoch": epoch}

        if epoch == cfg.n_pretrain_epochs:
            is_online = True
            if cfg.cql_min_q_weight_online >= 0:
                print(f"changing cql alpha from {cql_min_q_weight} to {cfg.cql_min_q_weight_online}")
                cql_min_q_weight = cfg.cql_min_q_weight_online

            if not cfg.online_use_cql and use_cql:
                print("truning off cql during online phase and use sac")
                use_cql = False
                if sac.config.cql_lagrange:
                    model_keys = list(sac.model_keys)
                    model_keys.remove("log_alpha_prime")
                    sac._model_keys = tuple(model_keys)

        """
        Do evaluations when
        1. epoch = 0 to get initial performance
        2. every cfg.offline_eval_every_n_epoch for offline phase
        3. epoch == cfg.n_pretrain_epochs to get offline pre-trained performance
        4. every cfg.online_eval_every_n_env_steps for online phase
        5. when replay_buffer.total_steps >= cfg.max_online_env_steps to get final fine-tuned performance
        """
        do_eval = (
            epoch == 0
            or (not is_online and epoch % cfg.offline_eval_every_n_epoch == 0)
            or (epoch == cfg.n_pretrain_epochs)
            or (is_online and replay_buffer.total_steps // cfg.online_eval_every_n_env_steps > online_eval_counter)
            or (replay_buffer.total_steps >= cfg.max_online_env_steps)
        )

        with Timer() as eval_timer:
            if do_eval:
                print(f"Starting Evaluation for Epoch {epoch}")
                trajs = eval_sampler.sample(
                    sampler_policy, cfg.eval_n_trajs, deterministic=True
                )

                metrics["evaluation/average_return"] = np.mean([np.sum(t["rewards"]) for t in trajs])
                metrics["evaluation/average_traj_length"] = np.mean([len(t["rewards"]) for t in trajs])
                if use_goal:
                    # for adroit envs
                    metrics["evaluation/goal_achieved_rate"] = np.mean([1 in t["goal_achieved"] for t in trajs])
                else:
                    # for d4rl envs
                    metrics["evaluation/average_normalized_return"] = np.mean(
                        [eval_sampler.env.get_normalized_score(np.sum(t["rewards"])) for t in trajs]
                    )
                if is_online:
                    online_eval_counter = replay_buffer.total_steps // cfg.online_eval_every_n_env_steps

                if cfg.save_model:
                    save_data = {"sac": sac, "variant": variant, "epoch": epoch}
                    wandb_logger.save_pickle(save_data, "model.pkl")

        metrics["grad_steps"] = total_grad_steps
        if is_online:
            metrics["env_steps"] = replay_buffer.total_steps
        metrics["epoch"] = epoch
        metrics["online_rollout_time"] = 0 if online_rollout_timer is None else online_rollout_timer()
        metrics["train_time"] = 0 if train_timer is None else train_timer()
        metrics["eval_time"] = eval_timer()
        metrics["epoch_time"] = eval_timer() if train_timer is None else train_timer() + eval_timer()
        if cfg.n_pretrain_epochs >= 0:
            metrics["mixing_ratio"] = mixing_ratio
        if train_metrics is not None:
            metrics.update(train_metrics)
        if expl_metrics is not None:
            metrics.update(expl_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if replay_buffer.total_steps >= cfg.max_online_env_steps:
            print("Finished Training")
            break

        with Timer() as online_rollout_timer:
            if is_online:
                print("collecting online trajs:", cfg.n_online_traj_per_epoch)
                trajs = train_sampler.sample(
                    sampler_policy,
                    n_trajs=cfg.n_online_traj_per_epoch,
                    deterministic=False,
                    replay_buffer=replay_buffer,
                )
                expl_metrics = {}
                expl_metrics["exploration/average_return"] = np.mean([np.sum(t["rewards"]) for t in trajs])
                expl_metrics["exploration/average_traj_length"] = np.mean([len(t["rewards"]) for t in trajs])
                if use_goal:
                    expl_metrics["exploration/goal_achieved_rate"] = np.mean([1 in t["goal_achieved"] for t in trajs])

        with Timer() as train_timer:

            if cfg.n_pretrain_epochs >= 0 and epoch >= cfg.n_pretrain_epochs and cfg.online_utd_ratio > 0:
                n_train_step_per_epoch = np.sum([len(t["rewards"]) for t in trajs]) * cfg.online_utd_ratio

            if cfg.n_pretrain_epochs >= 0:
                if cfg.mixing_ratio >= 0:
                    mixing_ratio = cfg.mixing_ratio
                else:
                    mixing_ratio = dataset["rewards"].shape[0] / (
                        dataset["rewards"].shape[0] + replay_buffer.total_steps
                    )
                batch_size_offline = int(cfg.batch_size * mixing_ratio)
                batch_size_online = cfg.batch_size - batch_size_offline

            for _ in range(n_train_step_per_epoch):
                if is_online:
                    # mix offline and online buffer
                    offline_batch = subsample_batch(dataset, batch_size_offline)
                    online_batch = replay_buffer.sample(batch_size_online)
                    batch = concatenate_batches([offline_batch, online_batch])
                    batch = batch_to_torch(batch, cfg.device)
                else:
                    batch = subsample_batch(dataset, cfg.batch_size)
                    batch = batch_to_torch(batch, cfg.device)
                train_metrics = sac.train(
                    batch, use_cql=use_cql, cql_min_q_weight=cql_min_q_weight, enable_calql=enable_calql
                )
            total_grad_steps += n_train_step_per_epoch
        epoch += 1


if __name__ == "__main__":
    main()
