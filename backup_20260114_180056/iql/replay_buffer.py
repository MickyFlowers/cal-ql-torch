"""
HDF5-based Replay Buffer for Iterative IQL Training.

Designed for efficient storage and sampling of image-based RL data:
- Episode-based HDF5 storage (compatible with existing dataset format)
- Memory-efficient lazy loading
- Support for on-policy and off-policy buffer separation
"""

import glob
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from xlib.algo.utils.image_utils import np_buffer_to_pil_image


class EpisodeBuffer:
    """
    Temporary buffer for collecting a single episode before saving to disk.

    During rollout, only observations and actions are recorded.
    After episode ends, call process() to generate full transition data
    (next_observations, rewards, dones).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = []  # List of {ft_obs, img_obs}
        self.actions = []
        self.is_intervention = []  # Whether this step was human intervention

        # These are filled by process() after episode ends
        self.next_observations = []
        self.rewards = []
        self.dones = []
        self._processed = False

    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        is_intervention: bool = False,
    ):
        """
        Add a single step to the episode buffer during rollout.
        Only records observation and action - next_obs/reward/done are computed later.
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.is_intervention.append(is_intervention)

    def process(self, success: bool):
        """
        Process the episode after it ends to generate full transition data.

        Args:
            success: Whether the episode was successful (determines final reward).

        This method:
        1. Generates next_observations by shifting observations
        2. Sets sparse reward (1.0 at last step if success, else 0.0)
        3. Sets done flags (1.0 at last step)
        """
        if self._processed:
            return

        episode_length = len(self.observations)
        if episode_length == 0:
            return

        # Generate next_observations: obs[t+1] for each obs[t]
        # For the last step, next_obs = current obs (terminal state)
        self.next_observations = []
        for i in range(episode_length):
            if i < episode_length - 1:
                self.next_observations.append(self.observations[i + 1])
            else:
                # Last step: next_obs is the same as current obs (terminal)
                self.next_observations.append(self.observations[i])

        # Set rewards: sparse reward at the end
        self.rewards = [0.0] * episode_length
        if success:
            self.rewards[-1] = 1.0

        # Set dones: only the last step is done
        self.dones = [0.0] * episode_length
        self.dones[-1] = 1.0

        self._processed = True

    def __len__(self):
        return len(self.observations)

    def is_empty(self):
        return len(self.observations) == 0

    @property
    def is_processed(self):
        return self._processed


class HDF5ReplayBuffer:
    """
    HDF5-based replay buffer for iterative offline RL.

    Features:
    - Stores episodes in HDF5 format compatible with CalqlDataset
    - Lazy loading for memory efficiency
    - Supports separate on-policy and off-policy buffers
    - Automatic statistics computation
    """

    def __init__(
        self,
        save_dir: str,
        statistics_path: str,
        image_resize: int = 256,
        image_size: int = 224,
        discount: float = 0.99,
        norm_type: str = "max_min",
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        preload_to_memory: bool = False,
    ):
        """
        Initialize the HDF5 replay buffer.

        Args:
            save_dir: Directory to save HDF5 episode files.
            statistics_path: Path to statistics.yaml for normalization.
            image_resize: Size to resize images before cropping.
            image_size: Final image size after cropping.
            discount: Discount factor for MC returns.
            norm_type: Normalization type ('max_min' or 'mean_std').
            reward_scale: Scale factor for rewards.
            reward_bias: Bias to add to rewards.
            preload_to_memory: If True, preload all episodes to memory.
        """
        self.save_dir = save_dir
        self.statistics_path = statistics_path
        self.image_resize = image_resize
        self.image_size = image_size
        self.discount = discount
        self.norm_type = norm_type
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.preload_to_memory = preload_to_memory

        os.makedirs(save_dir, exist_ok=True)

        # Load statistics
        if os.path.exists(statistics_path):
            with open(statistics_path, "r") as f:
                self.statistics = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Statistics file not found: {statistics_path}")

        # Image transform for training
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_resize, image_resize)),
                transforms.RandomCrop(image_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Scan existing episodes
        self._refresh_episode_list()

        # Preloaded data cache
        self._preloaded_episodes = None
        if preload_to_memory and len(self.episode_files) > 0:
            self._preload_all_episodes()

    def _refresh_episode_list(self):
        """Refresh the list of episode files."""
        self.episode_files = sorted(glob.glob(os.path.join(self.save_dir, "*.hdf5")))
        self._update_meta()

    def _update_meta(self):
        """Update metadata (episode lengths and cumsum)."""
        self.episode_lengths = []
        for ep_file in self.episode_files:
            with h5py.File(ep_file, "r") as f:
                self.episode_lengths.append(f["actions"].shape[0])
        self._cum_sum = np.cumsum(self.episode_lengths) if self.episode_lengths else np.array([])

    def _preload_all_episodes(self):
        """Preload all episodes into memory."""
        print(f"[HDF5ReplayBuffer] Preloading {len(self.episode_files)} episodes to memory...")
        self._preloaded_episodes = []
        for ep_file in self.episode_files:
            self._preloaded_episodes.append(self._load_episode(ep_file))
        print(f"[HDF5ReplayBuffer] Preloaded {self.total_samples} samples")

    def _load_episode(self, episode_file: str) -> Dict:
        """Load a single episode from HDF5 file."""
        with h5py.File(episode_file, "r") as f:
            ep_data = {
                "ft_obs": f["observations"]["ft_obs"][:],
                "images": f["observations"]["img_obs"][:],
                "actions": f["actions"][:],
                "next_ft_obs": f["next_observations"]["ft_obs"][:],
                "next_images": f["next_observations"]["img_obs"][:],
                "rewards": f["rewards"][:] * self.reward_scale + self.reward_bias,
                "dones": f["dones"][:],
            }

            # Compute MC returns
            episode_length = ep_data["ft_obs"].shape[0]
            mc_returns = np.zeros(episode_length, dtype=np.float32)
            mc_return = 0.0
            for t in reversed(range(episode_length)):
                mc_return = ep_data["rewards"][t] + self.discount * mc_return * (
                    1.0 - ep_data["dones"][t]
                )
                mc_returns[t] = mc_return
            ep_data["mc_returns"] = mc_returns

            # Check success (reward at last step)
            ep_data["success"] = 1.0 if ep_data["rewards"][-1] > 0.5 else 0.0

        return ep_data

    def save_episode(
        self,
        episode_buffer: EpisodeBuffer,
        success: bool,
        episode_idx: Optional[int] = None,
    ) -> str:
        """
        Save an episode to HDF5 file.

        Args:
            episode_buffer: Buffer containing episode data.
            success: Whether the episode was successful.
            episode_idx: Optional episode index. If None, auto-increment.

        Returns:
            Path to the saved HDF5 file.
        """
        if episode_buffer.is_empty():
            return None

        # Process episode data if not already processed
        # This generates next_observations, rewards, dones from raw observations/actions
        if not episode_buffer.is_processed:
            episode_buffer.process(success)

        # Determine episode index
        if episode_idx is None:
            existing_files = glob.glob(os.path.join(self.save_dir, "episode_*.hdf5"))
            if existing_files:
                max_idx = max(
                    [
                        int(os.path.basename(f).replace("episode_", "").replace(".hdf5", ""))
                        for f in existing_files
                    ]
                )
                episode_idx = max_idx + 1
            else:
                episode_idx = 0

        episode_file = os.path.join(self.save_dir, f"episode_{episode_idx:06d}.hdf5")

        # Convert lists to arrays
        ft_obs = np.array([obs["ft_obs"] for obs in episode_buffer.observations], dtype=np.float32)
        actions = np.array(episode_buffer.actions, dtype=np.float32)
        next_ft_obs = np.array(
            [obs["ft_obs"] for obs in episode_buffer.next_observations], dtype=np.float32
        )

        # Handle variable-length image bytes data
        # img_obs is a list of bytes objects (encoded PNG images)
        img_obs_list = [obs["img_obs"] for obs in episode_buffer.observations]
        next_img_obs_list = [obs["img_obs"] for obs in episode_buffer.next_observations]

        # Get rewards and dones from processed episode buffer
        episode_length = len(episode_buffer)
        rewards = np.zeros(episode_length, dtype=np.float32)
        if success:
            rewards[-1] = 1.0  # Sparse reward at the end

        dones = np.zeros(episode_length, dtype=np.float32)
        dones[-1] = 1.0  # Episode ends

        # Save to HDF5
        # Use variable-length dtype for image bytes (compatible with existing dataset format)
        vlen_dtype = h5py.special_dtype(vlen=np.uint8)

        with h5py.File(episode_file, "w") as f:
            obs_grp = f.create_group("observations")
            obs_grp.create_dataset("ft_obs", data=ft_obs, dtype=np.float32)
            # Create variable-length dataset for images
            img_ds = obs_grp.create_dataset("img_obs", shape=(episode_length,), dtype=vlen_dtype)
            for i, img_bytes in enumerate(img_obs_list):
                img_ds[i] = np.frombuffer(img_bytes, dtype=np.uint8)

            next_obs_grp = f.create_group("next_observations")
            next_obs_grp.create_dataset("ft_obs", data=next_ft_obs, dtype=np.float32)
            next_img_ds = next_obs_grp.create_dataset(
                "img_obs", shape=(episode_length,), dtype=vlen_dtype
            )
            for i, img_bytes in enumerate(next_img_obs_list):
                next_img_ds[i] = np.frombuffer(img_bytes, dtype=np.uint8)

            f.create_dataset("actions", data=actions, dtype=np.float32)
            f.create_dataset("rewards", data=rewards, dtype=np.float32)
            f.create_dataset("dones", data=dones, dtype=np.float32)

            # Store intervention flags
            is_intervention = np.array(episode_buffer.is_intervention, dtype=np.float32)
            f.create_dataset("is_intervention", data=is_intervention, dtype=np.float32)

        # Refresh episode list
        self._refresh_episode_list()

        # If preloading, add new episode to cache
        if self.preload_to_memory:
            if self._preloaded_episodes is None:
                self._preloaded_episodes = []
            self._preloaded_episodes.append(self._load_episode(episode_file))

        return episode_file

    def _normalize(self, data: np.ndarray, statistics: Dict, epsilon: float = 1e-6) -> np.ndarray:
        """Normalize data using statistics."""
        if self.norm_type == "max_min":
            data_max = np.array(statistics["max"]) + epsilon
            data_min = np.array(statistics["min"]) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
        elif self.norm_type == "mean_std":
            data_mean = np.array(statistics["mean"])
            data_std = np.array(statistics["std"])
            data = (data - data_mean) / data_std
        return data

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary containing batch data.
        """
        if self.total_samples == 0:
            raise ValueError("Buffer is empty, cannot sample")

        # Random sample indices
        indices = np.random.randint(0, self.total_samples, size=batch_size)
        return self._get_batch(indices)

    def _get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Get a batch of transitions at specified global indices."""
        batch_proprio = []
        batch_image = []
        batch_next_proprio = []
        batch_next_image = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_mc_return = []
        batch_success = []

        for idx in indices:
            # Find episode and sample index
            ep_idx = np.searchsorted(self._cum_sum, idx, side="right")
            sample_idx = idx - (self._cum_sum[ep_idx - 1] if ep_idx > 0 else 0)

            # Load episode data
            if self._preloaded_episodes is not None:
                ep_data = self._preloaded_episodes[ep_idx]
            else:
                ep_data = self._load_episode(self.episode_files[ep_idx])

            # Extract sample
            proprio = ep_data["ft_obs"][sample_idx]
            next_proprio = ep_data["next_ft_obs"][sample_idx]
            image_bytes = ep_data["images"][sample_idx]
            next_image_bytes = ep_data["next_images"][sample_idx]
            action = ep_data["actions"][sample_idx]
            reward = ep_data["rewards"][sample_idx]
            done = ep_data["dones"][sample_idx]
            mc_return = ep_data["mc_returns"][sample_idx]
            success = ep_data["success"]

            # Normalize
            proprio = self._normalize(proprio, self.statistics["proprio"])
            next_proprio = self._normalize(next_proprio, self.statistics["proprio"])
            action = self._normalize(action, self.statistics["action"])

            # Transform images
            image = np_buffer_to_pil_image(image_bytes)
            image = self.image_transform(image)
            next_image = np_buffer_to_pil_image(next_image_bytes)
            next_image = self.image_transform(next_image)

            batch_proprio.append(proprio.astype(np.float32))
            batch_image.append(image)
            batch_next_proprio.append(next_proprio.astype(np.float32))
            batch_next_image.append(next_image)
            batch_action.append(action.astype(np.float32))
            batch_reward.append(reward)
            batch_done.append(done)
            batch_mc_return.append(mc_return)
            batch_success.append(success)

        return {
            "observations": {
                "proprio": torch.from_numpy(np.stack(batch_proprio)),
                "image": torch.stack(batch_image),
            },
            "next_observations": {
                "proprio": torch.from_numpy(np.stack(batch_next_proprio)),
                "image": torch.stack(batch_next_image),
            },
            "action": torch.from_numpy(np.stack(batch_action)),
            "reward": torch.tensor(batch_reward, dtype=torch.float32),
            "done": torch.tensor(batch_done, dtype=torch.float32),
            "mc_return": torch.tensor(batch_mc_return, dtype=torch.float32),
            "success": torch.tensor(batch_success, dtype=torch.float32),
        }

    @property
    def total_samples(self) -> int:
        """Total number of samples in the buffer."""
        return int(np.sum(self.episode_lengths)) if self.episode_lengths else 0

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the buffer."""
        return len(self.episode_files)

    def clear(self):
        """Clear all episodes from the buffer."""
        for ep_file in self.episode_files:
            os.remove(ep_file)
        self._refresh_episode_list()
        self._preloaded_episodes = None

    def move_to(self, target_dir: str):
        """Move all episodes to another directory (e.g., from on-policy to off-policy)."""
        os.makedirs(target_dir, exist_ok=True)

        # Find max index in target directory
        existing_files = glob.glob(os.path.join(target_dir, "episode_*.hdf5"))
        if existing_files:
            max_idx = max(
                [
                    int(os.path.basename(f).replace("episode_", "").replace(".hdf5", ""))
                    for f in existing_files
                ]
            )
            start_idx = max_idx + 1
        else:
            start_idx = 0

        # Move files
        for i, ep_file in enumerate(self.episode_files):
            new_file = os.path.join(target_dir, f"episode_{start_idx + i:06d}.hdf5")
            shutil.move(ep_file, new_file)

        # Clear current buffer state
        self._refresh_episode_list()
        self._preloaded_episodes = None

    def refresh_preloaded_cache(self):
        """Refresh preloaded episodes cache after files have changed."""
        if self.preload_to_memory and len(self.episode_files) > 0:
            self._preload_all_episodes()
        else:
            self._preloaded_episodes = None


class DualReplayBuffer:
    """
    Manages on-policy and off-policy replay buffers for iterative IQL.

    Training strategy:
    - First iteration: Train only on on-policy buffer
    - Subsequent iterations: Sample uniformly from both buffers
    """

    def __init__(
        self,
        on_policy_dir: str,
        off_policy_dir: str,
        statistics_path: str,
        **kwargs,
    ):
        """
        Initialize dual replay buffers.

        Args:
            on_policy_dir: Directory for on-policy (current iteration) data.
            off_policy_dir: Directory for off-policy (historical) data.
            statistics_path: Path to statistics.yaml.
            **kwargs: Additional arguments passed to HDF5ReplayBuffer.
        """
        self.on_policy_buffer = HDF5ReplayBuffer(
            save_dir=on_policy_dir,
            statistics_path=statistics_path,
            **kwargs,
        )
        self.off_policy_buffer = HDF5ReplayBuffer(
            save_dir=off_policy_dir,
            statistics_path=statistics_path,
            **kwargs,
        )
        self.iteration = 0

    def save_episode(
        self,
        episode_buffer: EpisodeBuffer,
        success: bool,
    ) -> str:
        """Save episode to on-policy buffer."""
        return self.on_policy_buffer.save_episode(episode_buffer, success)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.

        First iteration: Sample only from on-policy buffer.
        Subsequent iterations: Sample uniformly from both buffers.
        """
        on_policy_samples = self.on_policy_buffer.total_samples
        off_policy_samples = self.off_policy_buffer.total_samples

        if self.iteration == 0 or off_policy_samples == 0:
            # First iteration or no off-policy data: use only on-policy
            return self.on_policy_buffer.sample(batch_size)

        # Subsequent iterations: sample uniformly (50/50 split)
        on_policy_batch_size = batch_size // 2
        off_policy_batch_size = batch_size - on_policy_batch_size

        # Adjust if one buffer is too small
        if on_policy_samples < on_policy_batch_size:
            on_policy_batch_size = on_policy_samples
            off_policy_batch_size = batch_size - on_policy_batch_size

        batches = []
        if on_policy_batch_size > 0:
            batches.append(self.on_policy_buffer.sample(on_policy_batch_size))
        if off_policy_batch_size > 0:
            batches.append(self.off_policy_buffer.sample(off_policy_batch_size))

        return self._concat_batches(batches)

    def _concat_batches(self, batches: List[Dict]) -> Dict[str, torch.Tensor]:
        """Concatenate multiple batches."""
        if len(batches) == 1:
            return batches[0]

        return {
            "observations": {
                "proprio": torch.cat([b["observations"]["proprio"] for b in batches], dim=0),
                "image": torch.cat([b["observations"]["image"] for b in batches], dim=0),
            },
            "next_observations": {
                "proprio": torch.cat([b["next_observations"]["proprio"] for b in batches], dim=0),
                "image": torch.cat([b["next_observations"]["image"] for b in batches], dim=0),
            },
            "action": torch.cat([b["action"] for b in batches], dim=0),
            "reward": torch.cat([b["reward"] for b in batches], dim=0),
            "done": torch.cat([b["done"] for b in batches], dim=0),
            "mc_return": torch.cat([b["mc_return"] for b in batches], dim=0),
            "success": torch.cat([b["success"] for b in batches], dim=0),
        }

    def end_iteration(self, keep_only_success: bool = False):
        """
        End current iteration: move on-policy data to off-policy buffer.

        Args:
            keep_only_success: If True, only move successful episodes to off-policy.
                             If False (recommended), move all data.
        """
        if keep_only_success:
            # Move only successful episodes
            success_files = []
            for ep_file in self.on_policy_buffer.episode_files:
                with h5py.File(ep_file, "r") as f:
                    rewards = f["rewards"][:]
                    if rewards[-1] > 0.5:  # Success
                        success_files.append(ep_file)

            # Move successful files
            target_dir = self.off_policy_buffer.save_dir
            existing_files = glob.glob(os.path.join(target_dir, "episode_*.hdf5"))
            if existing_files:
                max_idx = max(
                    [
                        int(os.path.basename(f).replace("episode_", "").replace(".hdf5", ""))
                        for f in existing_files
                    ]
                )
                start_idx = max_idx + 1
            else:
                start_idx = 0

            for i, ep_file in enumerate(success_files):
                new_file = os.path.join(target_dir, f"episode_{start_idx + i:06d}.hdf5")
                shutil.move(ep_file, new_file)

            # Remove remaining (failed) episodes from on-policy buffer
            for ep_file in self.on_policy_buffer.episode_files:
                if os.path.exists(ep_file):  # May have been moved already
                    os.remove(ep_file)
        else:
            # Move all data (recommended for RL)
            self.on_policy_buffer.move_to(self.off_policy_buffer.save_dir)

        # Refresh both buffers' file lists and preloaded caches
        self.on_policy_buffer._refresh_episode_list()
        self.on_policy_buffer._preloaded_episodes = None  # Clear on-policy cache (now empty)
        self.off_policy_buffer._refresh_episode_list()
        self.off_policy_buffer.refresh_preloaded_cache()  # Refresh off-policy cache with new files

        # Increment iteration counter
        self.iteration += 1

    @property
    def total_samples(self) -> int:
        """Total samples across both buffers."""
        return self.on_policy_buffer.total_samples + self.off_policy_buffer.total_samples

    @property
    def on_policy_samples(self) -> int:
        return self.on_policy_buffer.total_samples

    @property
    def off_policy_samples(self) -> int:
        return self.off_policy_buffer.total_samples

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "iteration": self.iteration,
            "on_policy_episodes": self.on_policy_buffer.num_episodes,
            "on_policy_samples": self.on_policy_buffer.total_samples,
            "off_policy_episodes": self.off_policy_buffer.num_episodes,
            "off_policy_samples": self.off_policy_buffer.total_samples,
            "total_samples": self.total_samples,
        }
