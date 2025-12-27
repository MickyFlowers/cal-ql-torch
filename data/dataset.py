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
    for file in episode_files:
        with h5py.File(file, 'r') as f:
            jnt_obs_sum = np.array(f['observations']['jnt_obs'][:])
            tcp_obs_sum = np.array(f['observations']['tcp_obs'][:])
            proprio = np.concatenate([jnt_obs_sum, tcp_obs_sum], axis=-1)
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
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
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
            data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
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
            jnt_obs = observations['jnt_obs']
            tcp_obs = observations['tcp_obs']
            images = observations['img_obs']
            actions = f['actions']
            next_observations = f['next_observations']
            next_jnt_obs = next_observations['jnt_obs']
            next_tcp_obs = next_observations['tcp_obs']
            next_images = next_observations['img_obs']
            rewards = f['rewards'][:]
            rewards = rewards * self.config.reward_scale + self.config.reward_bias
            if rewards[-1] == 1.0:
                success = True
            else:
                success = False

            dones = f['dones']
        
            episode_length = jnt_obs.shape[0]
            
            jnt = jnt_obs[sample_idx]
            tcp = tcp_obs[sample_idx]
            next_jnt = next_jnt_obs[sample_idx]
            next_tcp = next_tcp_obs[sample_idx]
            proprio = np.concatenate([jnt, tcp], axis=-1)
            next_proprio = np.concatenate([next_jnt, next_tcp], axis=-1)
            image = images[sample_idx]
            next_image = next_images[sample_idx]
            image: Image.Image = np_buffer_to_pil_image(image)
            image = self.image_transform(image)
            next_image: Image.Image = np_buffer_to_pil_image(next_image)
            next_image = self.image_transform(next_image)
            action = actions[sample_idx]
            proprio = self._normalize(proprio, self.statistics['proprio'], self.config.norm_type)
            next_proprio = self._normalize(next_proprio, self.statistics['proprio'], self.config.norm_type)
            action = self._normalize(action, self.statistics['action'], self.config.norm_type)
            reward = rewards[sample_idx]
            done = dones[sample_idx]
            # calc_mc_returns
            mc_return = 0.0
            for t in reversed(range(sample_idx, episode_length)):
                mc_return = rewards[t] + self.config.discount * mc_return * (1.0 - dones[t])
            
            
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
    
class DiffusionPolicyDataset(Dataset):
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
            jnt_obs = observations['jnt_obs']
            tcp_obs = observations['tcp_obs']
            images = observations['img_obs']
            actions = f['actions']
            jnt = jnt_obs[sample_idx]
            tcp = tcp_obs[sample_idx]
            proprio = np.concatenate([jnt, tcp], axis=-1)
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
                jnt_obs = observations['jnt_obs'][:]
                tcp_obs = observations['tcp_obs'][:]
                images = observations['img_obs'][:]
                actions = f['actions'][:]
                next_observations = f['next_observations']
                next_jnt_obs = next_observations['jnt_obs'][:]
                next_tcp_obs = next_observations['tcp_obs'][:]
                next_images = next_observations['img_obs'][:]
                rewards = f['rewards'][:]
                rewards = rewards * config.reward_scale + config.reward_bias
                dones = f['dones'][:]

                episode_length = jnt_obs.shape[0]
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
                    proprio = np.concatenate([jnt_obs[sample_idx], tcp_obs[sample_idx]], axis=-1)
                    next_proprio = np.concatenate([next_jnt_obs[sample_idx], next_tcp_obs[sample_idx]], axis=-1)

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

        # Open LMDB environment (read-only)
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )

        # Load metadata
        with self.env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']

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
        if hasattr(self, 'env'):
            self.env.close()


class DiffusionPolicyDatasetLMDB(Dataset):
    """
    LMDB-optimized version of DiffusionPolicyDataset for faster data loading.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.root_path = config.root_path
        self.horizon = config.horizon
        self.lmdb_path = os.path.join(self.root_path, "lmdb_cache_diffusion")

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

        # Open LMDB environment (read-only)
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )

        # Load metadata
        with self.env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']

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

    def _convert_to_lmdb(self, episode_files):
        """Convert HDF5 files to LMDB format."""
        map_size = 100 * 1024 * 1024 * 1024  # 100GB
        env = lmdb.open(self.lmdb_path, map_size=map_size)

        total_samples = 0
        episode_lengths = []

        print("Converting HDF5 to LMDB for DiffusionPolicy...")

        with env.begin(write=True) as txn:
            for ep_idx, episode_file in enumerate(tqdm(episode_files)):
                with h5py.File(episode_file, 'r') as f:
                    observations = f['observations']
                    jnt_obs = observations['jnt_obs'][:]
                    tcp_obs = observations['tcp_obs'][:]
                    images = observations['img_obs'][:]
                    actions = f['actions'][:]

                    episode_length = jnt_obs.shape[0]
                    episode_lengths.append(episode_length)

                    # Store each sample with action chunks
                    for sample_idx in range(episode_length):
                        proprio = np.concatenate([jnt_obs[sample_idx], tcp_obs[sample_idx]], axis=-1)

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
        if hasattr(self, 'env'):
            self.env.close()


class ACTDataset(Dataset):
    """
    Dataset for ACT (Action Chunking with Transformers) training.
    Similar to DiffusionPolicyDataset but returns action chunks.
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
            jnt_obs = f['observations']['jnt_obs'][0]
            tcp_obs = f['observations']['tcp_obs'][0]
            self.proprio_dim = jnt_obs.shape[-1] + tcp_obs.shape[-1]
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
            jnt_obs = observations['jnt_obs']
            tcp_obs = observations['tcp_obs']
            images = observations['img_obs']
            actions = f['actions']

            jnt = jnt_obs[sample_idx]
            tcp = tcp_obs[sample_idx]
            proprio = np.concatenate([jnt, tcp], axis=-1)
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

        # Open LMDB environment
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )

        # Load metadata
        with self.env.begin() as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            self.total_samples = meta['total_samples']
            self.episode_lengths = meta['episode_lengths']
            self.proprio_dim = meta['proprio_dim']
            self.action_dim = meta['action_dim']

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
                    jnt_obs = observations['jnt_obs'][:]
                    tcp_obs = observations['tcp_obs'][:]
                    images = observations['img_obs'][:]
                    actions = f['actions'][:]

                    if proprio_dim is None:
                        proprio_dim = jnt_obs.shape[-1] + tcp_obs.shape[-1]
                        action_dim = actions.shape[-1]

                    episode_length = jnt_obs.shape[0]
                    episode_lengths.append(episode_length)

                    for sample_idx in range(episode_length):
                        proprio = np.concatenate([jnt_obs[sample_idx], tcp_obs[sample_idx]], axis=-1)

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
        if hasattr(self, 'env'):
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
    