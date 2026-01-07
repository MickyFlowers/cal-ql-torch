import glob
import io
import os
import pickle
import struct

import h5py
import hydra
import lmdb
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xlib.algo.utils.image_utils import np_buffer_to_pil_image


def calc_meta(data_root, episode_files):
    meta = {"episode_length": []}
    for episode_file in episode_files:
        with h5py.File(episode_file, 'r') as f:
            actions = f['actions'][:]
            episode_length = actions.shape[0]
            meta["episode_length"].append(episode_length)
    meta_file = os.path.join(data_root, "meta.yaml")
    with open(meta_file, 'w') as f:
        yaml.dump(meta, f)
    print(f"Saved meta information to {meta_file}")

def calc_statics(data_root, episode_files):
    proprio_max = None
    proprio_min = None
    action_max = None
    action_min = None
    for file in tqdm(episode_files):
        with h5py.File(file, 'r') as f:
            ft_obs = np.array(f['observations']['ft_obs'][:])
            proprio = ft_obs
            if proprio_max is None:
                proprio_max = np.max(proprio, axis=0)
            else:
                max_candidate = np.max(proprio, axis=0)
                proprio_max = np.maximum(proprio_max, max_candidate)
            if proprio_min is None:
                proprio_min = np.min(proprio, axis=0)
            else:
                min_candidate = np.min(proprio, axis=0)
                proprio_min = np.minimum(proprio_min, min_candidate)
            
            actions = np.array(f['actions'][:])
            if action_max is None:
                action_max = np.max(actions, axis=0)
            else:
                max_candidate = np.max(actions, axis=0)
                action_max = np.maximum(action_max, max_candidate)
            if action_min is None:
                action_min = np.min(actions, axis=0)
            else:
                min_candidate = np.min(actions, axis=0)
                action_min = np.minimum(action_min, min_candidate)
    # save statistics   
    statistics = {
        "proprio": {
            "max": proprio_max.tolist(),
            "min": proprio_min.tolist(),
        },
        "action": {
            "max": action_max.tolist(),
            "min": action_min.tolist(),
        }
    }
    with open(os.path.join(data_root, "statistics.yaml"), 'w') as f:
        yaml.safe_dump(statistics, f, default_flow_style=False, allow_unicode=True)

class CalqlDataset(Dataset):
    def __init__(self, config: DictConfig, preload_to_memory: bool = False):
        """
        CalqlDataset for offline RL training.

        Args:
            config: Dataset configuration
            preload_to_memory: If True, preload all data into memory during initialization.
                              This eliminates I/O latency at epoch boundaries but requires
                              sufficient RAM (dataset size ~33GB for typical datasets).
        """
        super().__init__()
        self.config = config
        self.preload_to_memory = preload_to_memory
        # parser all episode files

        root_path = config.root_path
        self.root_path = root_path
        self.episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))

        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.meta_file = os.path.join(self.root_path, "meta.yaml")
        # Only compute statistics if files don't exist (avoid race condition in multi-GPU)
        if not os.path.exists(self.statics_file):
            calc_statics(self.root_path, self.episode_files)
        if not os.path.exists(self.meta_file):
            calc_meta(self.root_path, self.episode_files)
        self.statistics = self._parser_statstics(self.statics_file)
        self.meta = self._parser_meta(self.meta_file)

        # Pre-compute cumulative sum for faster indexing
        self._cum_sum = np.cumsum(self.meta['episode_length'])

        # File handle cache (will be initialized per worker)
        self._file_handles = {}
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Preload data if requested
        self._preloaded_data = None
        self._preloaded_episodes = None
        if self.preload_to_memory:
            self._preload_all_data()

    def _load_single_episode(self, args):
        """Load a single episode from HDF5 file (for parallel loading)."""
        ep_idx, episode_file, reward_scale, reward_bias, discount = args
        with h5py.File(episode_file, 'r') as f:
            observations = f['observations']
            ft_obs = observations['ft_obs'][:]
            images = observations['img_obs'][:]
            actions = f['actions'][:]
            next_observations = f['next_observations']
            next_ft_obs = next_observations['ft_obs'][:]
            next_images = next_observations['img_obs'][:]
            rewards = f['rewards'][:]
            rewards = rewards * reward_scale + reward_bias
            dones = f['dones'][:]

            # Determine if episode is successful
            success = 1.0 if rewards[-1] == 1.0 else 0.0

            # Precompute MC returns for entire episode
            episode_length = ft_obs.shape[0]
            mc_returns = np.zeros(episode_length, dtype=np.float32)
            mc_return = 0.0
            for t in reversed(range(episode_length)):
                mc_return = rewards[t] + discount * mc_return * (1.0 - dones[t])
                mc_returns[t] = mc_return

            return ep_idx, {
                'ft_obs': ft_obs,
                'images': images,
                'actions': actions,
                'next_ft_obs': next_ft_obs,
                'next_images': next_images,
                'rewards': rewards,
                'dones': dones,
                'mc_returns': mc_returns,
                'success': success,
            }

    def _preload_all_data(self):
        """Preload all episode data into memory using parallel I/O."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing

        print("[CalqlDataset] Preloading all data into memory (parallel)...")
        total_samples = int(np.sum(self.meta['episode_length']))

        # Prepare arguments for parallel loading
        args_list = [
            (ep_idx, episode_file, self.config.reward_scale, self.config.reward_bias, self.config.discount)
            for ep_idx, episode_file in enumerate(self.episode_files)
        ]

        # Use ThreadPoolExecutor for I/O-bound parallel loading
        # Threads work well here because HDF5 releases GIL during I/O
        num_workers = min(32, multiprocessing.cpu_count(), len(self.episode_files))
        self._preloaded_episodes = [None] * len(self.episode_files)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._load_single_episode, args): args[0] for args in args_list}
            with tqdm(total=len(self.episode_files), desc="Preloading episodes") as pbar:
                for future in as_completed(futures):
                    ep_idx, episode_data = future.result()
                    self._preloaded_episodes[ep_idx] = episode_data
                    pbar.update(1)

        print(f"[CalqlDataset] Preloaded {len(self._preloaded_episodes)} episodes, {total_samples} samples")
        self._preloaded_data = True

    def reload(self):
        self.episode_files = glob.glob(os.path.join(self.root_path, "*.hdf5"))
        calc_statics(self.root_path, self.episode_files)
        calc_meta(self.root_path, self.episode_files)
        self.statistics = self._parser_statstics(self.statics_file)
        self.meta = self._parser_meta(self.meta_file)

    def _parser_meta(self, meta_file: str):
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        return meta

    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics

    def __len__(self):
        return np.sum(self.meta['episode_length'])

    
    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data 
    
    def _get_file_handle(self, episode_idx):
        """Get cached file handle or open new one."""
        if episode_idx not in self._file_handles:
            episode_file = self.episode_files[episode_idx]
            self._file_handles[episode_idx] = h5py.File(episode_file, 'r', swmr=True)
        return self._file_handles[episode_idx]

    def __getitem__(self, index):
        episode_idx = np.searchsorted(self._cum_sum, index, side='right')
        sample_idx = index - (self._cum_sum[episode_idx - 1] if episode_idx > 0 else 0)

        # Use preloaded data if available
        if self._preloaded_data and self._preloaded_episodes is not None:
            ep_data = self._preloaded_episodes[episode_idx]
            proprio = ep_data['ft_obs'][sample_idx]
            next_proprio = ep_data['next_ft_obs'][sample_idx]
            image = ep_data['images'][sample_idx]
            next_image = ep_data['next_images'][sample_idx]
            action = ep_data['actions'][sample_idx]
            reward = ep_data['rewards'][sample_idx]
            done = ep_data['dones'][sample_idx]
            mc_return = ep_data['mc_returns'][sample_idx]
            success = ep_data['success']
        else:
            # Fall back to HDF5 file access
            f = self._get_file_handle(episode_idx)

            observations = f['observations']
            ft_obs = observations['ft_obs']
            images = observations['img_obs']
            actions = f['actions']
            next_observations = f['next_observations']
            next_ft_obs = next_observations['ft_obs']
            next_images = next_observations['img_obs']
            rewards = f['rewards'][:]
            rewards = rewards * self.config.reward_scale + self.config.reward_bias
            if rewards[-1] == 1.0:
                success = True
            else:
                success = False

            dones = f['dones']

            episode_length = ft_obs.shape[0]

            proprio = ft_obs[sample_idx]
            next_proprio = next_ft_obs[sample_idx]
            image = images[sample_idx]
            next_image = next_images[sample_idx]
            action = actions[sample_idx]
            reward = rewards[sample_idx]
            done = dones[sample_idx]
            # calc_mc_returns
            mc_return = 0.0
            for t in reversed(range(sample_idx, episode_length)):
                mc_return = rewards[t] + self.config.discount * mc_return * (1.0 - dones[t])

        # Transform images
        image: Image.Image = np_buffer_to_pil_image(image)
        image = self.image_transform(image)
        next_image: Image.Image = np_buffer_to_pil_image(next_image)
        next_image = self.image_transform(next_image)

        # Normalize
        proprio = self._normalize(proprio, self.statistics['proprio'], self.config.norm_type)
        next_proprio = self._normalize(next_proprio, self.statistics['proprio'], self.config.norm_type)
        action = self._normalize(action, self.statistics['action'], self.config.norm_type)

        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }
        next_observations = {
            'proprio': next_proprio.astype(np.float32),
            'image': next_image,
        }
        sample = {
            'observations': observations,
            'next_observations': next_observations,
            'action': action.astype(np.float32),
            'reward': np.array(reward).astype(np.float32),
            'done': np.array(done).astype(np.float32),
            'mc_return': np.array(mc_return).astype(np.float32),
            'success': np.array(success).astype(np.float32),
        }
        return sample

    def __del__(self):
        """Close all cached file handles."""
        for fh in self._file_handles.values():
            try:
                fh.close()
            except:
                pass

class FlowMatchingDataset(Dataset):
    """Dataset for Flow Matching Policy training with action chunks."""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        # parser all episode files

        root_path = config.root_path
        self.horizon = config.horizon
        self.root_path = root_path
        self.episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))

        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.meta_file = os.path.join(self.root_path, "meta.yaml")
        # Only compute statistics if files don't exist (avoid race condition in multi-GPU)
        if not os.path.exists(self.statics_file):
            calc_statics(self.root_path, self.episode_files)
        if not os.path.exists(self.meta_file):
            calc_meta(self.root_path, self.episode_files)
        self.statistics = self._parser_statstics(self.statics_file)
        self.meta = self._parser_meta(self.meta_file)
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(                     
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def reload(self):
        self.episode_files = glob.glob(os.path.join(self.root_path, "*.hdf5"))
        calc_statics(self.root_path, self.episode_files)
        calc_meta(self.root_path, self.episode_files)
        self.statistics = self._parser_statstics(self.statics_file)
        self.meta = self._parser_meta(self.meta_file)

    def _parser_meta(self, meta_file: str):
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        return meta
    
    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics   
    
    def __len__(self):
        return np.sum(self.meta['episode_length'])

    
    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data 
    
    def __getitem__(self, index):
        cum_sum = np.cumsum(self.meta['episode_length'])
        episode_idx = np.searchsorted(cum_sum, index, side='right')
        sample_idx = index - (cum_sum[episode_idx - 1] if episode_idx > 0 else 0)
        episode_file = self.episode_files[episode_idx]
        with h5py.File(episode_file, 'r') as f:
            observations = f['observations']
            ft_obs = observations['ft_obs']
            images = observations['img_obs']
            actions = f['actions']
            proprio = ft_obs[sample_idx]
            image = images[sample_idx]
            image: Image.Image = np_buffer_to_pil_image(image)
            image = self.image_transform(image)
            action = actions[sample_idx: sample_idx + self.horizon]
            if action.shape[0] < self.horizon:
                pad_length = self.horizon - action.shape[0]
                pad_action = np.tile(action[-1:], (pad_length, 1))
                action = np.concatenate([action, pad_action], axis=0)
            proprio = self._normalize(proprio, self.statistics['proprio'], self.config.norm_type)
            action = self._normalize(action, self.statistics['action'], self.config.norm_type)
            
        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }
        sample = {
            'observations': observations,
            'action': action.astype(np.float32),
        }
        return sample

class RoboMimicDataset(Dataset):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.file_path = config.root_path
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(                     
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self._load_data()
        
    
    def _load_data(self):
        with h5py.File(self.file_path, 'r') as f:
            data = f['data']
            all_demo = f['mask']['train'][:] if self.config.split == 'train' else f['mask']['valid'][:]

            self.num_episodes = len(all_demo)
            self.episode_data = []
            self.episode_lengths = []
            print("loading dataset...")
            proprio_max = None
            proprio_min = None
            action_max = None
            action_min = None
            for demo_name in tqdm(all_demo):
                
                demo_name = demo_name.decode('utf-8')
                obs = data[demo_name]['obs']
                next_obs = data[demo_name]['next_obs']
                actions = data[demo_name]['actions'][:]
                dones = data[demo_name]['dones'][:]
                rewards = data[demo_name]['rewards'][:]
                # current obs: eef_pos, eef_quat, jnt_pos
                eef_pos = obs['robot0_eef_pos'][:]
                eef_quat = obs['robot0_eef_quat'][:]
                jnt_pose = obs['robot0_joint_pos'][:]
                proprio = np.concatenate([eef_pos, eef_quat, jnt_pose], axis=-1)
                img_obs = obs['robot0_eye_in_hand_image'][:]

                # next obs: eef_pos, eef_quat, jnt_pos
                next_eef_pos = next_obs['robot0_eef_pos'][:]
                next_eef_quat = next_obs['robot0_eef_quat'][:]
                next_jnt_pose = next_obs['robot0_joint_pos'][:]
                next_proprio = np.concatenate([next_eef_pos, next_eef_quat, next_jnt_pose], axis=-1)
                next_img_obs = next_obs['robot0_eye_in_hand_image'][:]
                # calc statistics
                if proprio_max is None:
                    proprio_max = np.max(proprio, axis=0)
                else:
                    max_candidate = np.max(proprio, axis=0)
                    proprio_max = np.maximum(proprio_max, max_candidate)
                if proprio_min is None:
                    proprio_min = np.min(proprio, axis=0)
                else:
                    min_candidate = np.min(proprio, axis=0)
                    proprio_min = np.minimum(proprio_min, min_candidate)
                if action_max is None:
                    action_max = np.max(actions, axis=0)
                else:
                    max_candidate = np.max(actions, axis=0)
                    action_max = np.maximum(action_max, max_candidate)
                if action_min is None:
                    action_min = np.min(actions, axis=0)
                else:
                    min_candidate = np.min(actions, axis=0)
                    action_min = np.minimum(action_min, min_candidate)

                self.episode_lengths.append(actions.shape[0])
                # store episode data
                observations = {
                    'proprio': proprio.astype(np.float32),
                    'image': img_obs,
                }
                next_observations = {
                    'proprio': next_proprio.astype(np.float32),
                    'image': next_img_obs,
                }
                episode_data = {
                    'observations': observations,
                    'next_observations': next_observations,
                    'action': actions.astype(np.float32),
                    'reward': rewards.astype(np.float32),
                    'done': dones.astype(np.float32),
                }
                self.episode_data.append(episode_data)
                
            self.statistics = {
                "proprio": {
                    "max": proprio_max,
                    "min": proprio_min,
                },
                "action": {
                    "max": action_max,
                    "min": action_min,
                }
            }
            print("finished loading dataset.")

    
    def __len__(self):
        return np.sum(self.episode_lengths)

    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data 
    
    def __getitem__(self, index):
        cum_sum = np.cumsum(self.episode_lengths)
        episode_idx = np.searchsorted(cum_sum, index, side='right')
        sample_idx = index - (cum_sum[episode_idx - 1] if episode_idx > 0 else 0)
        episode_data = self.episode_data[episode_idx]
        proprio = episode_data['observations']['proprio'][sample_idx]
        proprio = self._normalize(proprio, self.statistics['proprio'], self.config.norm_type)
        next_proprio = episode_data['next_observations']['proprio'][sample_idx]
        next_proprio = self._normalize(next_proprio, self.statistics['proprio'], self.config.norm_type)
        img_obs = episode_data['observations']['image'][sample_idx]
        img_obs: Image.Image = Image.fromarray(img_obs)
        img_obs = self.image_transform(img_obs)
        next_img_obs = episode_data['next_observations']['image'][sample_idx]
        next_img_obs: Image.Image = Image.fromarray(next_img_obs)
        next_img_obs = self.image_transform(next_img_obs)
        action = episode_data['action'][sample_idx]
        action = self._normalize(action, self.statistics['action'], self.config.norm_type)
        rewards = episode_data['reward']
        dones = episode_data['done']
        reward = rewards[sample_idx]
        done = dones[sample_idx]
        mc_return = 0.0
        for t in reversed(range(sample_idx, len(rewards))):
            mc_return = rewards[t] + self.config.discount * mc_return * (1.0 - dones[t])
        
        observations = {
            'proprio': proprio.astype(np.float32),
            'image': img_obs,
        }
        next_observations = {
            'proprio': next_proprio.astype(np.float32),
            'image': next_img_obs,
        }
        sample = {
            'observations': observations,
            'next_observations': next_observations,
            'action': action.astype(np.float32),
            'reward': np.array(reward).astype(np.float32),
            'done': np.array(done).astype(np.float32),
            'mc_return': np.array(mc_return).astype(np.float32),
        }
        return sample
        
            
    



def convert_hdf5_to_lmdb(data_root, lmdb_path, episode_files, config):
    """
    Convert HDF5 episode files to LMDB format for faster data loading.

    Args:
        data_root: Root path of the data
        lmdb_path: Path to save the LMDB database
        episode_files: List of HDF5 episode files
        config: Dataset configuration
    """
    # Estimate total size (100GB max by default)
    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(lmdb_path, map_size=map_size)

    total_samples = 0
    episode_lengths = []

    print("Converting HDF5 to LMDB...")

    with env.begin(write=True) as txn:
        for ep_idx, episode_file in enumerate(tqdm(episode_files)):
            with h5py.File(episode_file, 'r') as f:
                observations = f['observations']
                ft_obs = observations['ft_obs'][:]
                images = observations['img_obs'][:]
                actions = f['actions'][:]
                next_observations = f['next_observations']
                next_ft_obs = next_observations['ft_obs'][:]
                next_images = next_observations['img_obs'][:]
                rewards = f['rewards'][:]
                rewards = rewards * config.reward_scale + config.reward_bias
                dones = f['dones'][:]

                episode_length = ft_obs.shape[0]
                episode_lengths.append(episode_length)
                success = 1.0 if rewards[-1] == 1.0 else 0.0

                # Precompute MC returns for the entire episode
                mc_returns = np.zeros(episode_length, dtype=np.float32)
                mc_return = 0.0
                for t in reversed(range(episode_length)):
                    mc_return = rewards[t] + config.discount * mc_return * (1.0 - dones[t])
                    mc_returns[t] = mc_return

                # Store each sample
                for sample_idx in range(episode_length):
                    proprio = ft_obs[sample_idx]
                    next_proprio = next_ft_obs[sample_idx]

                    sample = {
                        'proprio': proprio.astype(np.float32),
                        'image': images[sample_idx],
                        'next_proprio': next_proprio.astype(np.float32),
                        'next_image': next_images[sample_idx],
                        'action': actions[sample_idx].astype(np.float32),
                        'reward': np.array(rewards[sample_idx]).astype(np.float32),
                        'done': np.array(dones[sample_idx]).astype(np.float32),
                        'mc_return': np.array(mc_returns[sample_idx]).astype(np.float32),
                        'success': np.array(success).astype(np.float32),
                    }

                    key = f'{total_samples:09d}'.encode()
                    value = pickle.dumps(sample)
                    txn.put(key, value)
                    total_samples += 1

        # Store metadata
        meta = {
            'total_samples': total_samples,
            'episode_lengths': episode_lengths,
        }
        txn.put(b'__meta__', pickle.dumps(meta))

    env.close()
    print(f"Converted {total_samples} samples to LMDB at {lmdb_path}")
    return total_samples, episode_lengths


class CalqlDatasetLMDB(Dataset):
    """
    LMDB-optimized version of CalqlDataset for faster data loading.

    Key optimizations:
    1. Uses LMDB instead of HDF5 for faster random access
    2. Pre-computes MC returns during conversion
    3. Caches cumsum for episode indexing
    4. Memory-mapped access for efficient I/O
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.lmdb_path = os.path.join(self.root_path, "lmdb_cache")

        # Check if LMDB cache exists
        if not os.path.exists(self.lmdb_path):
            # Convert HDF5 to LMDB
            episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))
            if not episode_files:
                raise ValueError(f"No HDF5 files found in {self.root_path}")

            # Calculate statistics first
            calc_statics(self.root_path, episode_files)
            self.total_samples, self.episode_lengths = convert_hdf5_to_lmdb(
                self.root_path, self.lmdb_path, episode_files, config
            )
        else:
            print(f"Loading from existing LMDB cache: {self.lmdb_path}")

        # Load statistics
        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.statistics = self._parser_statstics(self.statics_file)

        # Lazy initialization for LMDB environment (for multiprocessing compatibility)
        self.env = None

        # Load metadata using temporary env
        tmp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with tmp_env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']
        tmp_env.close()

        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _init_env(self):
        """Lazy initialization of LMDB environment for multiprocessing compatibility."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )

    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics

    def __len__(self):
        return self.total_samples

    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data

    def __getitem__(self, index):
        # Lazy init LMDB env (for multiprocessing DataLoader)
        self._init_env()
        # Read from LMDB
        with self.env.begin() as txn:
            key = f'{index:09d}'.encode()
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {index} not found in LMDB")
            sample = pickle.loads(value)

        # Normalize
        proprio = self._normalize(sample['proprio'], self.statistics['proprio'], self.config.norm_type)
        next_proprio = self._normalize(sample['next_proprio'], self.statistics['proprio'], self.config.norm_type)
        action = self._normalize(sample['action'], self.statistics['action'], self.config.norm_type)

        # Transform images
        image = np_buffer_to_pil_image(sample['image'])
        image = self.image_transform(image)
        next_image = np_buffer_to_pil_image(sample['next_image'])
        next_image = self.image_transform(next_image)

        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }
        next_observations = {
            'proprio': next_proprio.astype(np.float32),
            'image': next_image,
        }

        return {
            'observations': observations,
            'next_observations': next_observations,
            'action': action.astype(np.float32),
            'reward': sample['reward'],
            'done': sample['done'],
            'mc_return': sample['mc_return'],
            'success': sample['success'],
        }

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class BCDatasetLMDB(Dataset):
    """
    Lightweight LMDB dataset specifically optimized for Behavior Cloning.

    Key optimizations compared to CalqlDatasetLMDB:
    1. Only loads single frame (no next_image, next_proprio)
    2. No reward/done/mc_return/success fields
    3. Smaller LMDB records = faster I/O
    4. Single image transform = half the CPU overhead

    Supports random sampling with a fixed ratio for studying data efficiency.
    Uses a fixed seed to ensure consistent sampling across multi-GPU training.

    When sample_ratio < 1.0, the remaining episodes can be used as validation data
    by calling get_validation_dataset() after creating the training dataset.
    """

    def __init__(self, config: DictConfig, sample_ratio: float = 1.0, sample_seed: int = 42,
                 is_validation: bool = False, validation_indices: np.ndarray = None):
        """
        Args:
            config: Dataset configuration
            sample_ratio: Ratio of samples to use (0.0, 1.0]. Default 1.0 uses all data.
            sample_seed: Fixed random seed for sampling to ensure consistency across GPUs.
            is_validation: If True, this is a validation dataset (uses validation_indices).
            validation_indices: Pre-computed indices for validation dataset.
        """
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.sample_ratio = sample_ratio
        self.sample_seed = sample_seed
        self.is_validation = is_validation
        self._validation_indices = validation_indices  # For creating validation dataset
        self.lmdb_path = os.path.join(self.root_path, "lmdb_cache_bc")

        # Use a lock file to prevent race condition in multi-GPU training
        # Only rank 0 should create the LMDB cache, others wait for completion
        lock_file = os.path.join(self.root_path, ".lmdb_cache_bc.lock")
        done_file = os.path.join(self.root_path, ".lmdb_cache_bc.done")

        # Check if we're in distributed mode
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main = local_rank == 0

        # Check if LMDB cache exists and is complete
        if not os.path.exists(done_file):
            if is_main:
                # Main process creates the cache
                if not os.path.exists(self.lmdb_path):
                    episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))
                    if not episode_files:
                        raise ValueError(f"No HDF5 files found in {self.root_path}")

                    calc_statics(self.root_path, episode_files)
                    self.total_samples, self.episode_lengths = self._convert_to_lmdb(episode_files)
                # Create done file to signal completion
                with open(done_file, 'w') as f:
                    f.write('done')
                print(f"[Rank {local_rank}] LMDB cache created: {self.lmdb_path}")
            else:
                # Other processes wait for done file
                print(f"[Rank {local_rank}] Waiting for LMDB cache to be created by rank 0...")
                import time
                while not os.path.exists(done_file):
                    time.sleep(0.5)
                print(f"[Rank {local_rank}] LMDB cache ready, continuing...")
        else:
            print(f"Loading from existing LMDB cache: {self.lmdb_path}")

        # Load statistics
        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.statistics = self._parser_statstics(self.statics_file)

        # Lazy initialization for LMDB environment (for multiprocessing compatibility)
        self.env = None

        # Load metadata using temporary env
        tmp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with tmp_env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']
        tmp_env.close()

        # Random sampling with fixed seed for multi-GPU consistency
        self._setup_sample_indices()

        if self.is_validation:
            self.image_transform = transforms.Compose([
                transforms.Resize((config.image_resize, config.image_resize)),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((config.image_resize, config.image_resize)),
                transforms.RandomCrop(config.image_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _setup_sample_indices(self):
        """Setup random sampling indices at episode level with fixed seed for multi-GPU consistency.

        This samples complete episodes/trajectories rather than individual steps,
        which is more meaningful for behavior cloning from demonstrations.

        When sample_ratio < 1.0, also stores validation indices for remaining episodes.
        """
        # If this is a validation dataset, use the provided validation indices
        if self.is_validation and self._validation_indices is not None:
            self.sample_indices = self._validation_indices
            self.num_samples = len(self.sample_indices)
            self.validation_sample_indices = None  # No further validation split
            return

        if self.sample_ratio >= 1.0:
            # Use all samples
            self.sample_indices = None
            self.num_samples = self.total_samples
            self.validation_sample_indices = None  # No validation data when using all data
        else:
            # Use fixed seed to ensure all GPU processes get the same indices
            rng = np.random.RandomState(self.sample_seed)

            # Sample at episode level, not step level
            num_episodes = len(self.episode_lengths)
            num_episodes_to_sample = max(1, int(num_episodes * self.sample_ratio))

            # Randomly select episodes
            episode_indices = np.arange(num_episodes)
            rng.shuffle(episode_indices)
            selected_episodes = np.sort(episode_indices[:num_episodes_to_sample])
            validation_episodes = np.sort(episode_indices[num_episodes_to_sample:])

            # Compute cumulative lengths to find step ranges for each episode
            cum_lengths = np.cumsum([0] + self.episode_lengths)

            # Collect all step indices from selected episodes (training)
            sample_indices = []
            for ep_idx in selected_episodes:
                start_idx = cum_lengths[ep_idx]
                end_idx = cum_lengths[ep_idx + 1]
                sample_indices.extend(range(start_idx, end_idx))

            self.sample_indices = np.array(sample_indices)
            self.num_samples = len(self.sample_indices)

            # Collect all step indices from remaining episodes (validation)
            validation_indices = []
            for ep_idx in validation_episodes:
                start_idx = cum_lengths[ep_idx]
                end_idx = cum_lengths[ep_idx + 1]
                validation_indices.extend(range(start_idx, end_idx))

            self.validation_sample_indices = np.array(validation_indices) if validation_indices else None

            # Calculate actual steps from selected episodes
            num_val_episodes = len(validation_episodes)
            num_val_steps = len(validation_indices) if validation_indices else 0
            print(f"[BCDatasetLMDB] Train: {num_episodes_to_sample}/{num_episodes} episodes "
                  f"({self.sample_ratio*100:.1f}%) -> {self.num_samples} steps")
            print(f"[BCDatasetLMDB] Val: {num_val_episodes}/{num_episodes} episodes "
                  f"({(1-self.sample_ratio)*100:.1f}%) -> {num_val_steps} steps")

    def _convert_to_lmdb(self, episode_files):
        """Convert HDF5 files to LMDB format (BC-optimized: single frame only)."""
        map_size = 50 * 1024 * 1024 * 1024  # 50GB (smaller than CalqlDatasetLMDB)
        env = lmdb.open(self.lmdb_path, map_size=map_size)

        total_samples = 0
        episode_lengths = []

        print("Converting HDF5 to LMDB for BC...")

        with env.begin(write=True) as txn:
            for ep_idx, episode_file in enumerate(tqdm(episode_files)):
                with h5py.File(episode_file, 'r') as f:
                    observations = f['observations']
                    ft_obs = observations['ft_obs'][:]
                    images = observations['img_obs'][:]
                    actions = f['actions'][:]

                    episode_length = ft_obs.shape[0]
                    episode_lengths.append(episode_length)

                    for i in range(episode_length):
                        # Only store current frame data (no next frame)
                        sample = {
                            'proprio': ft_obs[i].astype(np.float32),
                            'image': images[i],
                            'action': actions[i].astype(np.float32),
                        }

                        key = f'{total_samples:09d}'.encode()
                        value = pickle.dumps(sample)
                        txn.put(key, value)
                        total_samples += 1

            # Store metadata
            meta = {
                'total_samples': total_samples,
                'episode_lengths': episode_lengths,
            }
            txn.put(b'__meta__', pickle.dumps(meta))

        env.close()
        print(f"Converted {total_samples} samples to LMDB at {self.lmdb_path}")
        return total_samples, episode_lengths

    def _init_env(self):
        """Lazy initialization of LMDB environment for multiprocessing compatibility."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )

    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics

    def __len__(self):
        return self.num_samples

    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data

    def __getitem__(self, index):
        # Lazy init LMDB env (for multiprocessing DataLoader)
        self._init_env()

        # Map index to actual LMDB key if sampling is enabled
        if self.sample_indices is not None:
            actual_index = self.sample_indices[index]
        else:
            actual_index = index

        # Read from LMDB
        with self.env.begin() as txn:
            key = f'{actual_index:09d}'.encode()
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {actual_index} not found in LMDB")
            sample = pickle.loads(value)

        # Normalize (only 2 normalizations vs 3 in CalqlDatasetLMDB)
        proprio = self._normalize(sample['proprio'], self.statistics['proprio'], self.config.norm_type)
        action = self._normalize(sample['action'], self.statistics['action'], self.config.norm_type)

        # Transform image (only 1 image vs 2 in CalqlDatasetLMDB)
        image = np_buffer_to_pil_image(sample['image'])
        image = self.image_transform(image)

        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }

        return {
            'observations': observations,
            'action': action.astype(np.float32),
        }

    def get_validation_dataset(self):
        """Create a validation dataset using the remaining episodes not used for training.

        Returns:
            BCDatasetLMDB: Validation dataset, or None if sample_ratio >= 1.0
        """
        if self.validation_sample_indices is None or len(self.validation_sample_indices) == 0:
            return None

        # Create validation dataset with the remaining indices
        val_dataset = BCDatasetLMDB(
            config=self.config,
            sample_ratio=1.0,  # Not used since we provide validation_indices
            sample_seed=self.sample_seed,
            is_validation=True,
            validation_indices=self.validation_sample_indices
        )
        return val_dataset

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class FlowMatchingDatasetLMDB(Dataset):
    """
    LMDB-optimized version of FlowMatchingDataset for faster data loading.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.horizon = config.horizon
        self.lmdb_path = os.path.join(self.root_path, "lmdb_cache_flow_matching")

        # Check if LMDB cache exists
        if not os.path.exists(self.lmdb_path):
            # Convert HDF5 to LMDB
            episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))
            if not episode_files:
                raise ValueError(f"No HDF5 files found in {self.root_path}")

            # Calculate statistics first
            calc_statics(self.root_path, episode_files)
            self.total_samples, self.episode_lengths = self._convert_to_lmdb(episode_files)
        else:
            print(f"Loading from existing LMDB cache: {self.lmdb_path}")

        # Load statistics
        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.statistics = self._parser_statstics(self.statics_file)

        # Lazy initialization for LMDB environment (for multiprocessing compatibility)
        self.env = None

        # Load metadata using temporary env
        tmp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with tmp_env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']
        tmp_env.close()

        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _init_env(self):
        """Lazy initialization of LMDB environment for multiprocessing compatibility."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )

    def _convert_to_lmdb(self, episode_files):
        """Convert HDF5 files to LMDB format."""
        map_size = 100 * 1024 * 1024 * 1024  # 100GB
        env = lmdb.open(self.lmdb_path, map_size=map_size)

        total_samples = 0
        episode_lengths = []

        print("Converting HDF5 to LMDB for FlowMatching...")

        with env.begin(write=True) as txn:
            for ep_idx, episode_file in enumerate(tqdm(episode_files)):
                with h5py.File(episode_file, 'r') as f:
                    observations = f['observations']
                    ft_obs = observations['ft_obs'][:]
                    images = observations['img_obs'][:]
                    actions = f['actions'][:]

                    episode_length = ft_obs.shape[0]
                    episode_lengths.append(episode_length)

                    # Store each sample with action chunks
                    for sample_idx in range(episode_length):
                        proprio = ft_obs[sample_idx]

                        # Get action chunk
                        action_chunk = actions[sample_idx: sample_idx + self.horizon]
                        if action_chunk.shape[0] < self.horizon:
                            pad_length = self.horizon - action_chunk.shape[0]
                            pad_action = np.tile(action_chunk[-1:], (pad_length, 1))
                            action_chunk = np.concatenate([action_chunk, pad_action], axis=0)

                        sample = {
                            'proprio': proprio.astype(np.float32),
                            'image': images[sample_idx],
                            'action': action_chunk.astype(np.float32),
                        }

                        key = f'{total_samples:09d}'.encode()
                        value = pickle.dumps(sample)
                        txn.put(key, value)
                        total_samples += 1

            # Store metadata
            meta = {
                'total_samples': total_samples,
                'episode_lengths': episode_lengths,
            }
            txn.put(b'__meta__', pickle.dumps(meta))

        env.close()
        print(f"Converted {total_samples} samples to LMDB at {self.lmdb_path}")
        return total_samples, episode_lengths

    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics

    def __len__(self):
        return self.total_samples

    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data

    def __getitem__(self, index):
        # Lazy init LMDB env (for multiprocessing DataLoader)
        self._init_env()
        # Read from LMDB
        with self.env.begin() as txn:
            key = f'{index:09d}'.encode()
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {index} not found in LMDB")
            sample = pickle.loads(value)

        # Normalize
        proprio = self._normalize(sample['proprio'], self.statistics['proprio'], self.config.norm_type)
        action = self._normalize(sample['action'], self.statistics['action'], self.config.norm_type)

        # Transform image
        image = np_buffer_to_pil_image(sample['image'])
        image = self.image_transform(image)

        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }

        return {
            'observations': observations,
            'action': action.astype(np.float32),
        }

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class ACTDataset(Dataset):
    """
    Dataset for ACT (Action Chunking with Transformers) training.
    Similar to FlowMatchingDataset but returns action chunks.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.chunk_size = config.chunk_size

        self.episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))

        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.meta_file = os.path.join(self.root_path, "meta.yaml")

        if not os.path.exists(self.statics_file):
            calc_statics(self.root_path, self.episode_files)
        if not os.path.exists(self.meta_file):
            calc_meta(self.root_path, self.episode_files)

        self.statistics = self._parser_statstics(self.statics_file)
        self.meta = self._parser_meta(self.meta_file)

        # Cache cumsum for faster indexing
        self._cumsum = np.cumsum(self.meta['episode_length'])

        # Get dimensions from first sample
        with h5py.File(self.episode_files[0], 'r') as f:
            ft_obs = f['observations']['ft_obs'][0]
            self.proprio_dim = ft_obs.shape[-1]
            self.action_dim = f['actions'][0].shape[-1]

        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _parser_meta(self, meta_file: str):
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        return meta

    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics

    def __len__(self):
        return int(np.sum(self.meta['episode_length']))

    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data

    def __getitem__(self, index):
        # Use cached cumsum
        episode_idx = np.searchsorted(self._cumsum, index, side='right')
        sample_idx = index - (self._cumsum[episode_idx - 1] if episode_idx > 0 else 0)
        episode_file = self.episode_files[episode_idx]

        with h5py.File(episode_file, 'r') as f:
            observations = f['observations']
            ft_obs = observations['ft_obs']
            images = observations['img_obs']
            actions = f['actions']

            proprio = ft_obs[sample_idx]
            image = images[sample_idx]

            # Get action chunk
            action_chunk = actions[sample_idx: sample_idx + self.chunk_size]
            if action_chunk.shape[0] < self.chunk_size:
                pad_length = self.chunk_size - action_chunk.shape[0]
                pad_action = np.tile(action_chunk[-1:], (pad_length, 1))
                action_chunk = np.concatenate([action_chunk, pad_action], axis=0)

        # Transform image
        image = np_buffer_to_pil_image(image)
        image = self.image_transform(image)

        # Normalize
        proprio = self._normalize(proprio, self.statistics['proprio'], self.config.norm_type)
        action_chunk = self._normalize(action_chunk, self.statistics['action'], self.config.norm_type)

        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }

        return {
            'observations': observations,
            'action_chunk': action_chunk.astype(np.float32),
        }


class ACTDatasetLMDB(Dataset):
    """
    LMDB-optimized version of ACTDataset for faster data loading.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.chunk_size = config.chunk_size
        self.lmdb_path = os.path.join(self.root_path, "lmdb_cache_act")

        # Check if LMDB cache exists
        if not os.path.exists(self.lmdb_path):
            episode_files = sorted(glob.glob(os.path.join(self.root_path, "*.hdf5")))
            if not episode_files:
                raise ValueError(f"No HDF5 files found in {self.root_path}")

            calc_statics(self.root_path, episode_files)
            self.total_samples, self.episode_lengths, self.proprio_dim, self.action_dim = self._convert_to_lmdb(episode_files)
        else:
            print(f"Loading from existing LMDB cache: {self.lmdb_path}")

        # Load statistics
        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.statistics = self._parser_statstics(self.statics_file)

        # Lazy initialization for LMDB environment (for multiprocessing compatibility)
        self.env = None

        # Load metadata using temporary env
        tmp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with tmp_env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']
            self.proprio_dim = meta['proprio_dim']
            self.action_dim = meta['action_dim']
        tmp_env.close()

        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.RandomCrop(config.image_size),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + 0.01 * torch.randn_like(x)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _init_env(self):
        """Lazy initialization of LMDB environment for multiprocessing compatibility."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )

    def _convert_to_lmdb(self, episode_files):
        """Convert HDF5 files to LMDB format."""
        map_size = 100 * 1024 * 1024 * 1024
        env = lmdb.open(self.lmdb_path, map_size=map_size)

        total_samples = 0
        episode_lengths = []
        proprio_dim = None
        action_dim = None

        print("Converting HDF5 to LMDB for ACT...")

        with env.begin(write=True) as txn:
            for ep_idx, episode_file in enumerate(tqdm(episode_files)):
                with h5py.File(episode_file, 'r') as f:
                    observations = f['observations']
                    ft_obs = observations['ft_obs'][:]
                    images = observations['img_obs'][:]
                    actions = f['actions'][:]

                    if proprio_dim is None:
                        proprio_dim = ft_obs.shape[-1]
                        action_dim = actions.shape[-1]

                    episode_length = ft_obs.shape[0]
                    episode_lengths.append(episode_length)

                    for sample_idx in range(episode_length):
                        proprio = ft_obs[sample_idx]

                        action_chunk = actions[sample_idx: sample_idx + self.chunk_size]
                        if action_chunk.shape[0] < self.chunk_size:
                            pad_length = self.chunk_size - action_chunk.shape[0]
                            pad_action = np.tile(action_chunk[-1:], (pad_length, 1))
                            action_chunk = np.concatenate([action_chunk, pad_action], axis=0)

                        sample = {
                            'proprio': proprio.astype(np.float32),
                            'image': images[sample_idx],
                            'action_chunk': action_chunk.astype(np.float32),
                        }

                        key = f'{total_samples:09d}'.encode()
                        value = pickle.dumps(sample)
                        txn.put(key, value)
                        total_samples += 1

            meta = {
                'total_samples': total_samples,
                'episode_lengths': episode_lengths,
                'proprio_dim': proprio_dim,
                'action_dim': action_dim,
            }
            txn.put(b'__meta__', pickle.dumps(meta))

        env.close()
        print(f"Converted {total_samples} samples to LMDB at {self.lmdb_path}")
        return total_samples, episode_lengths, proprio_dim, action_dim

    def _parser_statstics(self, statics_file: str):
        with open(statics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        return statistics

    def __len__(self):
        return self.total_samples

    def _normalize(self, data, statistics, norm_type, epsilon=1e-6):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = ((data - data_min) / (data_max - data_min) * 2) - 1.0
        elif norm_type == 'mean_std':
            data_mean = np.array(statistics['mean'])
            data_std = np.array(statistics['std'])
            data = (data - data_mean) / data_std
        return data

    def __getitem__(self, index):
        # Lazy init LMDB env (for multiprocessing DataLoader)
        self._init_env()
        with self.env.begin() as txn:
            key = f'{index:09d}'.encode()
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {index} not found in LMDB")
            sample = pickle.loads(value)

        proprio = self._normalize(sample['proprio'], self.statistics['proprio'], self.config.norm_type)
        action_chunk = self._normalize(sample['action_chunk'], self.statistics['action'], self.config.norm_type)

        image = np_buffer_to_pil_image(sample['image'])
        image = self.image_transform(image)

        observations = {
            'proprio': proprio.astype(np.float32),
            'image': image,
        }

        return {
            'observations': observations,
            'action_chunk': action_chunk.astype(np.float32),
        }

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


@hydra.main(config_path="../config/dataset/", config_name="dataset_defaults", version_base=None)
def test(config):
    import time

    print("Testing original CalqlDataset...")
    dataset = CalqlDataset(config)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    start_time = time.time()

    for _ in tqdm(dataloader):
        pass
    original_time = (time.time() - start_time) / len(dataloader)
    print(f"Original dataset - Average data loading time per batch: {original_time:.4f} seconds")

    print("\nTesting LMDB CalqlDataset...")
    dataset_lmdb = CalqlDatasetLMDB(config)
    dataloader_lmdb = DataLoader(dataset_lmdb, batch_size=64, shuffle=True, num_workers=4)
    start_time = time.time()

    for _ in tqdm(dataloader_lmdb):
        pass
    lmdb_time = (time.time() - start_time) / len(dataloader_lmdb)
    print(f"LMDB dataset - Average data loading time per batch: {lmdb_time:.4f} seconds")
    print(f"Speedup: {original_time / lmdb_time:.2f}x")


if __name__ == "__main__":
    test()
    