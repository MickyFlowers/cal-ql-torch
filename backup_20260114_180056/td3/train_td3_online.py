"""TD3 Online Training Script.

This script implements online fine-tuning of a TD3 policy that was
pretrained offline with TD3+BC. It supports:
- Loading pretrained checkpoints from TD3+BC offline training
- Mixed replay buffer (offline + online data)
- Real robot environment interaction
- Periodic evaluation and checkpoint saving
"""

import copy
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.dataset import CalqlDataset
from env.env import UrEnv
from td3.replay_buffer import (
    ImagePreprocessor,
    ImageReplayBuffer,
    MixedReplayBuffer,
    ObservationNormalizer,
)
from td3.td3_model import ResNetDeterministicPolicy, ResNetTD3QFunction
from td3.td3_trainer import TD3Trainer
from utils.logger import WandBLogger
from utils.utils import Timer
from viskit.logging import logger, setup_logger


def dict_to_device(batch, device):
    """Move batch data to device."""
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = dict_to_device(v, device)
        else:
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


def collect_episode(
    env: UrEnv,
    trainer: TD3Trainer,
    replay_buffer: ImageReplayBuffer,
    image_preprocessor: ImagePreprocessor,
    normalizer: ObservationNormalizer,
    max_steps: int = 1000,
    deterministic: bool = False,
) -> dict:
    """Collect a single episode of experience.

    Args:
        env: Robot environment.
        trainer: TD3 trainer for action selection.
        replay_buffer: Buffer to store transitions.
        image_preprocessor: Image preprocessing function.
        normalizer: Observation normalizer.
        max_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic actions.

    Returns:
        Dictionary with episode statistics.
    """
    env.reset()
    env.wait_for_obs()

    episode_reward = 0.0
    episode_steps = 0
    done = False

    obs = env.get_observation()
    proprio = np.concatenate([obs["jnt_obs"], obs["tcp_obs"]], axis=-1)
    proprio_norm = normalizer.normalize_proprio(proprio)
    image = image_preprocessor(obs["img_obs"], training=False)

    while not done and episode_steps < max_steps:
        # Select action
        proprio_tensor = torch.from_numpy(proprio_norm).float()
        image_tensor = torch.from_numpy(image).float()
        action_norm = trainer.select_action(
            proprio_tensor, image_tensor, deterministic=deterministic
        )

        # Denormalize action for environment
        action = normalizer.denormalize_action(action_norm)

        # Step environment
        target_pose = action  # Assuming action is target pose
        env.step(target_pose)

        # Wait for next observation
        import rospy

        rospy.Rate(env.config.ctrl_freq).sleep()

        # Get next observation
        next_obs = env.get_observation()
        next_proprio = np.concatenate([next_obs["jnt_obs"], next_obs["tcp_obs"]], axis=-1)
        next_proprio_norm = normalizer.normalize_proprio(next_proprio)
        next_image = image_preprocessor(next_obs["img_obs"], training=False)

        # Compute reward (task-specific)
        reward = 0.0  # TODO: Implement task-specific reward

        # Check done condition
        done = False  # TODO: Implement done condition

        # Store transition
        replay_buffer.add(
            observation=proprio_norm,
            image=image,
            action=action_norm,
            reward=reward,
            next_observation=next_proprio_norm,
            next_image=next_image,
            done=done,
        )

        # Update for next step
        proprio_norm = next_proprio_norm
        image = next_image
        episode_reward += reward
        episode_steps += 1

    return {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
    }


def evaluate_policy(
    env: UrEnv,
    trainer: TD3Trainer,
    image_preprocessor: ImagePreprocessor,
    normalizer: ObservationNormalizer,
    n_episodes: int = 10,
    max_steps: int = 1000,
) -> dict:
    """Evaluate the current policy.

    Args:
        env: Robot environment.
        trainer: TD3 trainer.
        image_preprocessor: Image preprocessing function.
        normalizer: Observation normalizer.
        n_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.

    Returns:
        Dictionary with evaluation metrics.
    """
    trainer.eval_mode()
    episode_rewards = []
    episode_lengths = []
    successes = []

    for _ in range(n_episodes):
        env.reset()
        env.wait_for_obs()

        episode_reward = 0.0
        episode_steps = 0
        done = False
        success = False

        obs = env.get_observation()
        proprio = np.concatenate([obs["jnt_obs"], obs["tcp_obs"]], axis=-1)
        proprio_norm = normalizer.normalize_proprio(proprio)
        image = image_preprocessor(obs["img_obs"], training=False)

        while not done and episode_steps < max_steps:
            proprio_tensor = torch.from_numpy(proprio_norm).float()
            image_tensor = torch.from_numpy(image).float()
            action_norm = trainer.select_action(proprio_tensor, image_tensor, deterministic=True)
            action = normalizer.denormalize_action(action_norm)

            env.step(action)
            import rospy

            rospy.Rate(env.config.ctrl_freq).sleep()

            next_obs = env.get_observation()
            next_proprio = np.concatenate([next_obs["jnt_obs"], next_obs["tcp_obs"]], axis=-1)
            next_proprio_norm = normalizer.normalize_proprio(next_proprio)
            next_image = image_preprocessor(next_obs["img_obs"], training=False)

            reward = 0.0  # TODO: Implement reward
            done = False  # TODO: Implement done
            success = False  # TODO: Implement success check

            proprio_norm = next_proprio_norm
            image = next_image
            episode_reward += reward
            episode_steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        successes.append(float(success))

    trainer.train_mode()

    return {
        "eval/mean_reward": np.mean(episode_rewards),
        "eval/std_reward": np.std(episode_rewards),
        "eval/mean_length": np.mean(episode_lengths),
        "eval/success_rate": np.mean(successes),
    }


@hydra.main(config_path="../config", config_name="train_td3_online", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg.device)
    variant = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=cfg.seed,
        base_log_dir=cfg.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load offline dataset for mixed replay
    offline_dataset = CalqlDataset(cfg.dataset)
    statistics = offline_dataset.statistics

    # Create preprocessors
    image_preprocessor = ImagePreprocessor(
        image_resize=cfg.dataset.image_resize,
        image_size=cfg.dataset.image_size,
    )
    normalizer = ObservationNormalizer(
        statistics=statistics,
        norm_type=cfg.dataset.norm_type,
    )

    # Create online replay buffer
    online_buffer = ImageReplayBuffer(
        max_size=cfg.replay_buffer_size,
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        image_shape=(3, cfg.dataset.image_size, cfg.dataset.image_size),
        device=str(device),
    )

    # Create mixed replay buffer
    mixed_buffer = MixedReplayBuffer(
        offline_dataset=offline_dataset,
        online_buffer=online_buffer,
        offline_ratio=cfg.offline_ratio,
    )

    # Create models
    observation_dim = cfg.observation_dim
    action_dim = cfg.action_dim

    policy = ResNetDeterministicPolicy(
        observation_dim,
        action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_policy_backbone,
    )

    qf = {}
    qf["qf1"] = ResNetTD3QFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone,
    )
    qf["qf2"] = ResNetTD3QFunction(
        observation_dim,
        action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=cfg.train_q_backbone,
    )
    qf["target_qf1"] = copy.deepcopy(qf["qf1"])
    qf["target_qf2"] = copy.deepcopy(qf["qf2"])

    # Create trainer
    td3 = TD3Trainer(cfg.td3bc, policy, qf)
    td3.to_device(device=device)

    # Load pretrained checkpoint
    if cfg.load_ckpt_path != "":
        print(f"Loading pretrained checkpoint from {cfg.load_ckpt_path}")
        td3.load_checkpoint(cfg.load_ckpt_path, load_optimizer=False)

    # Create environment
    env = UrEnv(cfg.env)

    # Setup checkpoint directory
    if cfg.save_every_n_epoch > 0:
        ckpt_path = os.path.join(
            cfg.ckpt_path, f'{cfg.logging.prefix}_online_{time.strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(ckpt_path, exist_ok=True)

    # Training loop
    viskit_metrics = {}
    total_env_steps = 0
    total_grad_steps = 0
    train_timer = None
    train_metrics = None
    env_metrics = None

    print("Starting online training...")

    for epoch in range(cfg.n_online_epochs):
        metrics = {"epoch": epoch}
        metrics["total_env_steps"] = total_env_steps
        metrics["total_grad_steps"] = total_grad_steps
        metrics["train_time"] = 0 if train_timer is None else train_timer()

        if train_metrics is not None:
            metrics.update(train_metrics)
        if env_metrics is not None:
            metrics.update(env_metrics)

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # Save checkpoint
        if cfg.save_every_n_epoch > 0 and epoch % cfg.save_every_n_epoch == 0 and epoch != 0:
            ckpt_file_path = os.path.join(ckpt_path, f"checkpoint_{epoch:05d}.pt")
            td3.save_checkpoint(ckpt_file_path)

        # Evaluation
        if cfg.eval_every_n_epoch > 0 and epoch % cfg.eval_every_n_epoch == 0 and epoch != 0:
            eval_metrics = evaluate_policy(
                env,
                td3,
                image_preprocessor,
                normalizer,
                n_episodes=cfg.n_eval_episodes,
            )
            metrics.update(eval_metrics)
            print(
                f"Evaluation: reward={eval_metrics['eval/mean_reward']:.2f}, "
                f"success={eval_metrics['eval/success_rate']:.2f}"
            )

        with Timer() as train_timer:
            # Collect environment data
            td3.train_mode()
            env_steps_this_epoch = 0
            episode_rewards = []

            while env_steps_this_epoch < cfg.n_env_steps_per_epoch:
                # Use random actions for warmup
                use_random = total_env_steps < cfg.warmup_steps

                if use_random:
                    # Random exploration
                    env.reset()
                    env.wait_for_obs()
                    for _ in range(min(100, cfg.n_env_steps_per_epoch - env_steps_this_epoch)):
                        obs = env.get_observation()
                        proprio = np.concatenate([obs["jnt_obs"], obs["tcp_obs"]], axis=-1)
                        proprio_norm = normalizer.normalize_proprio(proprio)
                        image = image_preprocessor(obs["img_obs"], training=False)

                        action_norm = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
                        action = normalizer.denormalize_action(action_norm)

                        env.step(action)
                        import rospy

                        rospy.Rate(env.config.ctrl_freq).sleep()

                        next_obs = env.get_observation()
                        next_proprio = np.concatenate(
                            [next_obs["jnt_obs"], next_obs["tcp_obs"]], axis=-1
                        )
                        next_proprio_norm = normalizer.normalize_proprio(next_proprio)
                        next_image = image_preprocessor(next_obs["img_obs"], training=False)

                        reward = 0.0
                        done = False

                        online_buffer.add(
                            proprio_norm,
                            image,
                            action_norm,
                            reward,
                            next_proprio_norm,
                            next_image,
                            done,
                        )

                        env_steps_this_epoch += 1
                        total_env_steps += 1
                else:
                    # Policy-based exploration
                    episode_info = collect_episode(
                        env,
                        td3,
                        online_buffer,
                        image_preprocessor,
                        normalizer,
                        max_steps=min(1000, cfg.n_env_steps_per_epoch - env_steps_this_epoch),
                        deterministic=False,
                    )
                    env_steps_this_epoch += episode_info["episode_steps"]
                    total_env_steps += episode_info["episode_steps"]
                    episode_rewards.append(episode_info["episode_reward"])

            env_metrics = {
                "env/steps_this_epoch": env_steps_this_epoch,
                "env/buffer_size": len(online_buffer),
            }
            if episode_rewards:
                env_metrics["env/mean_episode_reward"] = np.mean(episode_rewards)

            # Training updates
            train_losses = []
            for _ in tqdm(range(cfg.n_train_steps_per_epoch), desc="Training"):
                batch = mixed_buffer.sample(cfg.batch_size)
                batch = dict_to_device(batch, device)
                step_metrics = td3.train(batch)
                train_losses.append(step_metrics.get("critic/qf1_loss", 0))

                def post_process(m):
                    for k, v in m.items():
                        if isinstance(v, torch.Tensor):
                            m[k] = v.detach().item()
                    return m

                train_metrics = post_process(step_metrics)
                total_grad_steps += 1

            train_metrics["train/mean_q_loss"] = np.mean(train_losses)

        epoch += 1

    # Final save
    if cfg.save_every_n_epoch > 0:
        ckpt_file_path = os.path.join(ckpt_path, "checkpoint_final.pt")
        td3.save_checkpoint(ckpt_file_path)

    print("Finished online training!")
    env.close()


if __name__ == "__main__":
    main()
