import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class ImageReplayBuffer(object):
    """Replay buffer for online RL with image observations.

    Stores transitions with both proprioceptive and image observations.
    Supports efficient sampling for TD3/SAC-style algorithms.
    """

    def __init__(
        self,
        max_size: int,
        observation_dim: int,
        action_dim: int,
        image_shape: tuple = (3, 224, 224),
        device: str = "cpu",
    ):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store.
            observation_dim: Dimension of proprioceptive observations.
            action_dim: Dimension of actions.
            image_shape: Shape of image observations (C, H, W).
            device: Device to store tensors on.
        """
        self._max_size = max_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._image_shape = image_shape
        self._device = device

        # Initialize storage
        self._observations = np.zeros((max_size, observation_dim), dtype=np.float32)
        self._images = np.zeros((max_size, *image_shape), dtype=np.float32)
        self._next_observations = np.zeros((max_size, observation_dim), dtype=np.float32)
        self._next_images = np.zeros((max_size, *image_shape), dtype=np.float32)
        self._actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(max_size, dtype=np.float32)
        self._dones = np.zeros(max_size, dtype=np.float32)

        self._next_idx = 0
        self._size = 0
        self._total_steps = 0

    def __len__(self):
        return self._size

    def add(
        self,
        observation: np.ndarray,
        image: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        next_image: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer.

        Args:
            observation: Proprioceptive observation [obs_dim].
            image: Image observation [C, H, W] (already normalized).
            action: Action [action_dim].
            reward: Scalar reward.
            next_observation: Next proprioceptive observation [obs_dim].
            next_image: Next image observation [C, H, W].
            done: Whether the episode ended.
        """
        self._observations[self._next_idx] = observation
        self._images[self._next_idx] = image
        self._actions[self._next_idx] = action
        self._rewards[self._next_idx] = reward
        self._next_observations[self._next_idx] = next_observation
        self._next_images[self._next_idx] = next_image
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_trajectory(
        self,
        observations: np.ndarray,
        images: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        next_images: np.ndarray,
        dones: np.ndarray,
    ):
        """Add a full trajectory to the buffer."""
        for i in range(len(observations)):
            self.add(
                observations[i],
                images[i],
                actions[i],
                rewards[i],
                next_observations[i],
                next_images[i],
                dones[i],
            )

    def sample(self, batch_size: int) -> dict:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary containing batch data with structure matching dataset format.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return self._get_batch(indices)

    def _get_batch(self, indices: np.ndarray) -> dict:
        """Get a batch of transitions at specified indices."""
        observations = {
            'proprio': torch.from_numpy(self._observations[indices]),
            'image': torch.from_numpy(self._images[indices]),
        }
        next_observations = {
            'proprio': torch.from_numpy(self._next_observations[indices]),
            'image': torch.from_numpy(self._next_images[indices]),
        }
        return {
            'observations': observations,
            'next_observations': next_observations,
            'action': torch.from_numpy(self._actions[indices]),
            'reward': torch.from_numpy(self._rewards[indices]),
            'done': torch.from_numpy(self._dones[indices]),
        }

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def size(self):
        return self._size


class MixedReplayBuffer(object):
    """Mixed replay buffer that combines offline dataset with online buffer.

    Useful for offline-to-online RL where we want to mix offline data
    with newly collected online data during training.
    """

    def __init__(
        self,
        offline_dataset,
        online_buffer: ImageReplayBuffer,
        offline_ratio: float = 0.5,
    ):
        """Initialize the mixed replay buffer.

        Args:
            offline_dataset: PyTorch dataset for offline data.
            online_buffer: ImageReplayBuffer for online data.
            offline_ratio: Ratio of offline data in each batch (0-1).
        """
        self.offline_dataset = offline_dataset
        self.online_buffer = online_buffer
        self.offline_ratio = offline_ratio

    def sample(self, batch_size: int) -> dict:
        """Sample a mixed batch from offline and online data.

        Args:
            batch_size: Total batch size.

        Returns:
            Dictionary containing mixed batch data.
        """
        # Calculate split
        if len(self.online_buffer) == 0:
            # No online data yet, use all offline
            offline_batch_size = batch_size
            online_batch_size = 0
        else:
            offline_batch_size = int(batch_size * self.offline_ratio)
            online_batch_size = batch_size - offline_batch_size

        batches = []

        # Sample from offline dataset
        if offline_batch_size > 0:
            offline_indices = np.random.randint(0, len(self.offline_dataset), size=offline_batch_size)
            offline_samples = [self.offline_dataset[i] for i in offline_indices]
            offline_batch = self._collate_samples(offline_samples)
            batches.append(offline_batch)

        # Sample from online buffer
        if online_batch_size > 0:
            online_batch = self.online_buffer.sample(online_batch_size)
            batches.append(online_batch)

        # Concatenate batches
        return self._concat_batches(batches)

    def _collate_samples(self, samples: list) -> dict:
        """Collate a list of samples into a batch dictionary."""
        batch = {
            'observations': {
                'proprio': torch.stack([torch.from_numpy(s['observations']['proprio'])
                                       if isinstance(s['observations']['proprio'], np.ndarray)
                                       else s['observations']['proprio'] for s in samples]),
                'image': torch.stack([s['observations']['image'] for s in samples]),
            },
            'next_observations': {
                'proprio': torch.stack([torch.from_numpy(s['next_observations']['proprio'])
                                       if isinstance(s['next_observations']['proprio'], np.ndarray)
                                       else s['next_observations']['proprio'] for s in samples]),
                'image': torch.stack([s['next_observations']['image'] for s in samples]),
            },
            'action': torch.stack([torch.from_numpy(s['action'])
                                  if isinstance(s['action'], np.ndarray)
                                  else s['action'] for s in samples]),
            'reward': torch.stack([torch.tensor(s['reward']) for s in samples]),
            'done': torch.stack([torch.tensor(s['done']) for s in samples]),
        }
        return batch

    def _concat_batches(self, batches: list) -> dict:
        """Concatenate multiple batch dictionaries."""
        if len(batches) == 1:
            return batches[0]

        result = {
            'observations': {
                'proprio': torch.cat([b['observations']['proprio'] for b in batches], dim=0),
                'image': torch.cat([b['observations']['image'] for b in batches], dim=0),
            },
            'next_observations': {
                'proprio': torch.cat([b['next_observations']['proprio'] for b in batches], dim=0),
                'image': torch.cat([b['next_observations']['image'] for b in batches], dim=0),
            },
            'action': torch.cat([b['action'] for b in batches], dim=0),
            'reward': torch.cat([b['reward'] for b in batches], dim=0),
            'done': torch.cat([b['done'] for b in batches], dim=0),
        }
        return result


class ImagePreprocessor:
    """Preprocessor for converting raw images to tensor format.

    Matches the preprocessing used in the dataset for consistency.
    """

    def __init__(self, image_resize: int = 256, image_size: int = 224):
        """Initialize the preprocessor.

        Args:
            image_resize: Size to resize images to before cropping.
            image_size: Final image size after center cropping.
        """
        # Use center crop for evaluation (no random augmentation)
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_resize, image_resize)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # Use random crop for training (data augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_resize, image_resize)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, image: np.ndarray, training: bool = True) -> np.ndarray:
        """Preprocess an image.

        Args:
            image: Raw image as numpy array (H, W, C) or PIL Image.
            training: If True, use random crop; otherwise center crop.

        Returns:
            Preprocessed image as numpy array (C, H, W).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        transform = self.train_transform if training else self.eval_transform
        processed = transform(image)
        return processed.numpy()


class ObservationNormalizer:
    """Normalizer for proprioceptive observations.

    Uses min-max normalization to scale observations to [-1, 1].
    """

    def __init__(self, statistics: dict, norm_type: str = 'max_min', epsilon: float = 1e-6):
        """Initialize the normalizer.

        Args:
            statistics: Dictionary containing 'proprio' and 'action' statistics
                with 'max' and 'min' keys.
            norm_type: Type of normalization ('max_min' or 'mean_std').
            epsilon: Small constant to avoid division by zero.
        """
        self.statistics = statistics
        self.norm_type = norm_type
        self.epsilon = epsilon

    def normalize_proprio(self, data: np.ndarray) -> np.ndarray:
        """Normalize proprioceptive observation."""
        return self._normalize(data, self.statistics['proprio'])

    def normalize_action(self, data: np.ndarray) -> np.ndarray:
        """Normalize action."""
        return self._normalize(data, self.statistics['action'])

    def denormalize_action(self, data: np.ndarray) -> np.ndarray:
        """Denormalize action back to original scale."""
        return self._denormalize(data, self.statistics['action'])

    def _normalize(self, data: np.ndarray, stats: dict) -> np.ndarray:
        """Apply normalization."""
        if self.norm_type == 'max_min':
            data_max = np.array(stats['max']) + self.epsilon
            data_min = np.array(stats['min']) - self.epsilon
            return ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
        elif self.norm_type == 'mean_std':
            data_mean = np.array(stats['mean'])
            data_std = np.array(stats['std'])
            return (data - data_mean) / data_std
        return data

    def _denormalize(self, data: np.ndarray, stats: dict) -> np.ndarray:
        """Reverse normalization."""
        if self.norm_type == 'max_min':
            data_max = np.array(stats['max']) + self.epsilon
            data_min = np.array(stats['min']) - self.epsilon
            return (data + 1.0) / 2.0 * (data_max - data_min) + data_min
        elif self.norm_type == 'mean_std':
            data_mean = np.array(stats['mean'])
            data_std = np.array(stats['std'])
            return data * data_std + data_mean
        return data
