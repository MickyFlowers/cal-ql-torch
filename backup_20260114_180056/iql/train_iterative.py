"""
Iterative IQL Training with Human-in-the-Loop Intervention.

This script implements iterative offline RL training:
1. Load pretrained checkpoint
2. Rollout with SpaceMouse intervention (human takes over when needed)
3. Collect episodes and label success/failure
4. Train using dual replay buffer strategy (on-policy + off-policy)
5. Repeat

Key features:
- SpaceMouse intervention has priority over policy actions
- Keyboard feedback for success/failure labeling ('s' for success, 'f' for failure)
- Two replay buffers: on-policy (current iteration) and off-policy (historical)
- First iteration trains only on on-policy data
- Subsequent iterations sample uniformly from both buffers
"""

import os
import time
import traceback
from typing import Dict, Optional

import gym
import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from xlib.algo.utils.image_utils import np_buffer_to_pil_image

import env
from iql.iql_trainer import IQLTrainer
from iql.replay_buffer import DualReplayBuffer, EpisodeBuffer
from model.model import ResNetDoubleQFunction, ResNetPolicy, ResNetVFunction
from utils.logger import WandBLogger
from viskit.logging import logger, setup_logger


def normalize(
    data: np.ndarray, statistics: Dict, norm_type: str, epsilon: float = 1e-6
) -> np.ndarray:
    """Normalize data to [-1, 1] range (max_min) or zero mean unit variance (mean_std)."""
    if norm_type == "max_min":
        data_max = np.array(statistics["max"]) + epsilon
        data_min = np.array(statistics["min"]) - epsilon
        data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
    elif norm_type == "mean_std":
        data_mean = np.array(statistics["mean"])
        data_std = np.array(statistics["std"])
        data = (data - data_mean) / data_std
    return data


def denormalize(
    data: np.ndarray, statistics: Dict, norm_type: str, epsilon: float = 1e-6
) -> np.ndarray:
    """Denormalize data from [-1, 1] range (max_min) or standardized (mean_std)."""
    if norm_type == "max_min":
        data_max = np.array(statistics["max"]) + epsilon
        data_min = np.array(statistics["min"]) - epsilon
        data = (data + 1.0) / 2.0 * (data_max - data_min) + data_min
    elif norm_type == "mean_std":
        data_mean = np.array(statistics["mean"])
        data_std = np.array(statistics["std"])
        data = data * data_std + data_mean
    return data


def dict_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move batch data to device."""
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device, non_blocking=True)
    return batch


class IterativeIQLTrainer:
    """
    Iterative IQL Trainer with human-in-the-loop intervention.

    Training workflow:
    1. Collect on_policy_episodes episodes via rollout with human intervention
    2. Train for train_steps_per_iter steps
    3. Move on-policy data to off-policy buffer
    4. Repeat
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load statistics for normalization
        with open(config.statistics_path, "r") as f:
            self.statistics = yaml.safe_load(f)

        # Image transform for policy inference (no augmentation)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.image_resize, config.image_resize)),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize models
        self._init_models()

        # Initialize dual replay buffer
        self.replay_buffer = DualReplayBuffer(
            on_policy_dir=os.path.join(config.buffer_path, "on_policy"),
            off_policy_dir=os.path.join(config.buffer_path, "off_policy"),
            statistics_path=config.statistics_path,
            image_resize=config.image_resize,
            image_size=config.image_size,
            discount=config.discount,
            norm_type=config.proprio_norm_type,
            preload_to_memory=config.get("preload_buffer_to_memory", False),
        )

        # Initialize environment
        self.env = gym.make("ur_env_v0", config=config.env)

        # Counters
        self.total_episodes = 0
        self.total_train_steps = 0
        self.current_iteration = 0
        self._robot_initialized = False

    def _initialize_robot(self):
        """Initialize robot with regrasp sequence (consistent with rollout_bc.py)."""
        if self._robot_initialized:
            return
        print("Initializing robot with regrasp sequence...")
        self.env.reset()
        self.env.regrasp()
        self.env.reset()
        self._robot_initialized = True
        print("Robot initialization complete.")

    def _init_models(self):
        """Initialize policy, Q, and V networks."""
        config = self.config

        # Policy network
        self.policy = ResNetPolicy(
            config.observation_dim,
            config.action_dim,
            config.policy_obs_proj_arch,
            config.policy_out_proj_arch,
            config.hidden_dim,
            config.orthogonal_init,
            config.policy_log_std_multiplier,
            config.policy_log_std_offset,
            train_backbone=config.train_policy_backbone,
        )

        # Double Q-function
        self.qf = ResNetDoubleQFunction(
            config.observation_dim,
            config.action_dim,
            config.q_obs_proj_arch,
            config.q_out_proj_arch,
            config.hidden_dim,
            config.orthogonal_init,
            train_backbone=config.train_q_backbone,
        )

        # V-function
        self.vf = ResNetVFunction(
            config.observation_dim,
            config.v_obs_proj_arch,
            config.v_out_proj_arch,
            config.hidden_dim,
            config.orthogonal_init,
            train_backbone=config.train_v_backbone,
        )

        # Create IQL trainer
        self.iql_trainer = IQLTrainer(config, self.policy, self.qf, self.vf)
        self.iql_trainer.to_device(self.device)

        # Load checkpoint if provided
        # Priority: bc_ckpt_path (BC policy only) > load_ckpt_path (full IQL checkpoint)
        bc_ckpt_path = config.get("bc_ckpt_path", "")
        if bc_ckpt_path and os.path.exists(bc_ckpt_path):
            # Load only BC policy weights, Q and V remain randomly initialized
            print(f"Loading BC policy checkpoint from {bc_ckpt_path}")
            self._load_bc_checkpoint(bc_ckpt_path)
        elif config.load_ckpt_path and os.path.exists(config.load_ckpt_path):
            # Load full IQL checkpoint (policy + Q + V)
            print(f"Loading full IQL checkpoint from {config.load_ckpt_path}")
            self.iql_trainer.load_checkpoint(config.load_ckpt_path)

        # Compile models if enabled
        if config.torch_compile_mode != "disable":
            self.iql_trainer.compile(mode=config.torch_compile_mode)

    def _load_bc_checkpoint(self, filepath: str):
        """
        Load only policy weights from a BC checkpoint.
        Q and V networks remain randomly initialized.

        This is useful for starting iterative training from a BC pretrained policy
        without having pretrained Q and V networks.

        Args:
            filepath: Path to BC checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded BC policy from {filepath}")
        print("Note: Q and V networks are randomly initialized (no pretrained weights)")

    def rollout_episode(self) -> tuple:
        """
        Rollout a single episode with human intervention support.

        Behavior:
        - Starts with policy control
        - Once teleop is enabled, it takes over permanently for this episode
        - When teleop is disabled (enable_teleop: True -> False), episode ends as SUCCESS
        - Human intervention guarantees success
        - Without teleop, use 's'/'f' keyboard to mark success/failure

        Data collection:
        - Only records observation and action during rollout
        - next_observation, reward, done are computed after episode ends via process()

        Returns:
            episode_buffer: Buffer containing episode data.
            success: Whether the episode was successful.
            stats: Dictionary of episode statistics.
        """
        episode_buffer = EpisodeBuffer()
        step_count = 0
        intervention_count = 0
        success = None

        # Track teleop state: once True, stays in teleop mode
        teleop_activated = False
        prev_enable_teleop = False

        # Set policy to eval mode for rollout
        self.policy.eval()

        try:
            # Reset environment (robot should already be initialized)
            self.env.reset()
            self.env.wait_for_obs()

            print("\n--- Episode Start ---")
            print("Controls: SpaceMouse button 0 = enable teleop (takes over permanently)")
            print("          SpaceMouse button 1 = disable teleop (ends episode as SUCCESS)")
            print("Keyboard: 's' = success, 'f' = failure, 'q' = quit without saving")

            while step_count < self.config.max_steps_per_episode:
                start_time = time.time()

                # Get current observation
                observation = self.env.get_observation()
                proprio = observation["ft_obs"]

                # Check SpaceMouse state
                space_mouse_twist, enable_teleop = self.env.get_space_mouse_state()

                # Once teleop is activated, it stays activated for this episode
                if enable_teleop and not teleop_activated:
                    teleop_activated = True
                    print("\n[TELEOP ACTIVATED] Human takes over control")

                # Check if teleop was disabled (True -> False): episode ends as SUCCESS
                if teleop_activated and prev_enable_teleop and not enable_teleop:
                    success = True
                    print("\n[SUCCESS] Teleop disabled - Episode completed successfully")
                    break

                prev_enable_teleop = enable_teleop

                # Determine action source
                if teleop_activated:
                    # Human intervention: use SpaceMouse velocity (even if twist is small)
                    teleop_scale = np.array(self.config.env.teleop_twist_scale, dtype=np.float32)
                    velocity = space_mouse_twist * teleop_scale
                    is_intervention = True
                    intervention_count += 1
                else:
                    # Policy action
                    velocity = self._get_policy_action(observation)
                    is_intervention = False

                # Execute action
                self.env.action(velocity)

                # Wait for next observation
                elapsed = time.time() - start_time
                if elapsed < 1.0 / self.config.freq:
                    time.sleep(1.0 / self.config.freq - elapsed)

                # Store observation and action only
                # next_observation, reward, done will be computed after episode ends
                episode_buffer.add(
                    observation={
                        "ft_obs": proprio.copy(),
                        "img_obs": observation["img_obs"],
                    },
                    action=velocity.copy(),
                    is_intervention=is_intervention,
                )

                step_count += 1

                # Check keyboard for episode control
                key = self.env.get_key()
                if key == "q":
                    print("\n[QUIT] Episode cancelled")
                    episode_buffer.reset()
                    return None, None, None
                elif key == "s" and not teleop_activated:
                    # Manual success mark (only when not in teleop mode)
                    success = True
                    print("\n[SUCCESS] Episode marked as successful")
                    break
                elif key == "f" and not teleop_activated:
                    # Manual failure mark (only when not in teleop mode)
                    success = False
                    print("\n[FAILURE] Episode marked as failed")
                    break

                # Print progress
                if step_count % 50 == 0:
                    interv_pct = 100 * intervention_count / step_count if step_count > 0 else 0
                    mode = "TELEOP" if teleop_activated else "POLICY"
                    print(
                        f"Step {step_count}/{self.config.max_steps_per_episode}, "
                        f"Mode: {mode}, Intervention: {interv_pct:.1f}%"
                    )

        except Exception as e:
            print(f"Error during rollout: {e}")
            traceback.print_exc()
            return None, None, None

        # If episode ended by max steps without explicit termination
        if success is None:
            if teleop_activated:
                # Teleop was active but didn't end normally - still count as success
                # since human was in control
                success = True
                print("\n[SUCCESS] Max steps reached with teleop active")
            else:
                # Policy ran to max steps without teleop - ask user
                print("\nEpisode ended (max steps). Press 's' for success or 'f' for failure:")
                while success is None:
                    key = self.env.get_key()
                    if key == "s":
                        success = True
                        print("[SUCCESS]")
                    elif key == "f":
                        success = False
                        print("[FAILURE]")
                    time.sleep(0.1)

        stats = {
            "steps": step_count,
            "intervention_count": intervention_count,
            "intervention_ratio": intervention_count / max(1, step_count),
            "success": success,
        }

        return episode_buffer, success, stats

    def _get_policy_action(self, observation: Dict) -> np.ndarray:
        """Get action from policy network."""
        # Normalize proprioception
        proprio = observation["ft_obs"]
        proprio_norm = normalize(proprio, self.statistics["proprio"], self.config.proprio_norm_type)
        proprio_tensor = (
            torch.tensor(proprio_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Process image
        image_bytes = observation["img_obs"]
        image = np_buffer_to_pil_image(np.frombuffer(image_bytes, dtype=np.uint8))
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # Get action from policy
        with torch.no_grad():
            action, _ = self.policy(proprio_tensor, image_tensor, deterministic=True)
            action = action.squeeze(0).cpu().numpy()

        # Denormalize action
        action = denormalize(action, self.statistics["action"], self.config.action_norm_type)

        return action

    def collect_episodes(self, num_episodes: int) -> Dict:
        """
        Collect multiple episodes with human intervention.

        Args:
            num_episodes: Number of episodes to collect.

        Returns:
            Dictionary with collection statistics.
        """
        print(f"\n{'='*50}")
        print(f"Collecting {num_episodes} episodes for iteration {self.current_iteration}")
        print(f"{'='*50}")

        collected = 0
        success_count = 0
        total_steps = 0
        total_interventions = 0

        while collected < num_episodes:
            print(f"\n--- Episode {collected + 1}/{num_episodes} ---")

            episode_buffer, success, stats = self.rollout_episode()

            if episode_buffer is None or stats is None:
                print("Episode skipped, retrying...")
                continue

            # Save episode to on-policy buffer
            self.replay_buffer.save_episode(episode_buffer, success)

            # Update counters
            collected += 1
            self.total_episodes += 1
            if success:
                success_count += 1
            total_steps += stats["steps"]
            total_interventions += stats["intervention_count"]

            # Handle episode transition
            if not success:
                # Failed episode: need regrasp
                print("Press 'y' to regrasp and continue...")
                self.env.ur_gripper.move_and_wait_for_pos(0, 255, 100)
                while True:
                    key = self.env.get_key()
                    if key == "y":
                        break
                    time.sleep(0.1)
                self.env.reset()
                self.env.regrasp()
                self.env.reset()
            # Note: For success episodes, reset() will be called at start of next rollout_episode

        return {
            "episodes_collected": collected,
            "success_count": success_count,
            "success_rate": success_count / max(1, collected),
            "total_steps": total_steps,
            "avg_steps_per_episode": total_steps / max(1, collected),
            "intervention_ratio": total_interventions / max(1, total_steps),
        }

    def train_iteration(self, num_steps: int) -> Dict:
        """
        Train for a specified number of gradient steps.

        Args:
            num_steps: Number of training steps.

        Returns:
            Dictionary with training metrics.
        """
        print(f"\n{'='*50}")
        print(f"Training iteration {self.current_iteration} for {num_steps} steps")
        print(f"Buffer stats: {self.replay_buffer.get_stats()}")
        print(f"{'='*50}")

        if self.replay_buffer.on_policy_samples == 0:
            print("Warning: No on-policy samples available for training")
            return {}

        # Set policy to train mode
        self.policy.train()

        # Setup learning rate scheduler if needed
        if self.config.use_lr_scheduler and self.current_iteration == 0:
            total_steps = num_steps * self.config.num_iterations
            self.iql_trainer.setup_lr_scheduler(
                total_steps,
                warmup_ratio=self.config.warmup_ratio,
                min_lr_ratio=self.config.min_lr_ratio,
            )

        # Training loop
        metrics_history = []
        pbar = tqdm(range(num_steps), desc="Training")

        for step in pbar:
            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)
            batch = dict_to_device(batch, self.device)

            # Train step
            metrics = self.iql_trainer.train(batch)
            metrics_history.append(metrics)

            self.total_train_steps += 1

            # Update progress bar
            if step % 10 == 0:
                pbar.set_postfix(
                    {
                        "q_loss": f"{metrics.get('critic/qf_loss', 0):.4f}",
                        "v_loss": f"{metrics.get('critic/v_loss', 0):.4f}",
                        "policy_loss": f"{metrics.get('actor/policy_loss', 0):.4f}",
                    }
                )

        # Aggregate metrics
        avg_metrics = {}
        for key in metrics_history[0].keys():
            values = [m[key] for m in metrics_history if key in m]
            avg_metrics[f"train/{key}"] = np.mean(values)

        return avg_metrics

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.iql_trainer.save_checkpoint(path)
        print(f"Checkpoint saved to {path}")

    def run(self):
        """Run the full iterative training loop."""
        config = self.config

        # Setup logging
        if config.get("use_wandb", False):
            variant = OmegaConf.to_container(config, resolve=True)
            wandb_logger = WandBLogger(config=config.logging, variant=variant)
            setup_logger(
                variant=variant,
                exp_id=wandb_logger.experiment_id,
                seed=config.seed,
                base_log_dir=config.logging.output_dir,
                include_exp_prefix_sub_dir=False,
            )
        else:
            wandb_logger = None

        # Setup checkpoint directory
        ckpt_dir = os.path.join(config.ckpt_path, f'iterative_iql_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_dir, exist_ok=True)

        try:
            # Initialize robot before first iteration
            self._initialize_robot()

            for iteration in range(config.num_iterations):
                self.current_iteration = iteration
                print(f"\n{'#'*60}")
                print(f"# ITERATION {iteration + 1}/{config.num_iterations}")
                print(f"{'#'*60}")

                # 1. Collect episodes
                collect_stats = self.collect_episodes(config.on_policy_episodes)
                print(f"\nCollection stats: {collect_stats}")

                # 2. Train
                train_metrics = self.train_iteration(config.train_steps_per_iter)
                print(f"\nTraining metrics: {train_metrics}")

                # 3. Log metrics
                all_metrics = {
                    "iteration": iteration,
                    "total_episodes": self.total_episodes,
                    "total_train_steps": self.total_train_steps,
                    **{f"collect/{k}": v for k, v in collect_stats.items()},
                    **train_metrics,
                    **{f"buffer/{k}": v for k, v in self.replay_buffer.get_stats().items()},
                }

                if wandb_logger:
                    wandb_logger.log(all_metrics)
                logger.record_dict(all_metrics)
                logger.dump_tabular(with_prefix=False, with_timestamp=False)

                # 4. Save checkpoint
                if (iteration + 1) % config.save_every_n_iter == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_iter_{iteration + 1:04d}.pt")
                    self.save_checkpoint(ckpt_path)

                # 5. Move on-policy data to off-policy buffer
                # Note: We keep ALL data (success + failure) for RL benefit
                self.replay_buffer.end_iteration(keep_only_success=False)

            # Final checkpoint
            final_ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
            self.save_checkpoint(final_ckpt_path)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            interrupt_ckpt_path = os.path.join(ckpt_dir, "checkpoint_interrupted.pt")
            self.save_checkpoint(interrupt_ckpt_path)

        except Exception as e:
            print(f"\nError during training: {e}")
            traceback.print_exc()

        finally:
            self.env.close()
            print("\nTraining completed")


@hydra.main(config_path="../config", config_name="train_iql_iterative", version_base=None)
def main(config: DictConfig):
    """Main entry point."""
    print(OmegaConf.to_yaml(config))

    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Create trainer and run
    trainer = IterativeIQLTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
