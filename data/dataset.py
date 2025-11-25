import glob
import os

import h5py
import hydra
import numpy as np
import torchvision.transforms as transforms
import yaml
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
        self.episode_files = glob.glob(os.path.join(self.root_path, "*.hdf5"))

        self.statics_file = os.path.join(self.root_path, "statistics.yaml")
        self.meta_file = os.path.join(self.root_path, "meta.yaml")
        calc_statics(self.root_path, self.episode_files)
        calc_meta(self.root_path, self.episode_files)
        self.statistics = self._parser_statstics(self.statics_file)
        self.meta = self._parser_meta(self.meta_file)
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
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

    
    def _normalize(self, data, statistics, norm_type, epsilon=1e-8):
        if norm_type == 'max_min':
            data_max = np.array(statistics['max']) + epsilon
            data_min = np.array(statistics['min']) - epsilon
            data = (data - data_min) / (data_max - data_min)
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
        }
        return sample


@hydra.main(config_path="../config/dataset/", config_name="dataset_defaults", version_base=None)
def test(config):
    dataset = CalqlDataset(config)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    import time
    start_time = time.time()
    
    for batch in dataloader:
        print(batch['observations']['proprio'].shape)
        print(batch['observations']['image'].shape)
        print(batch['action'].shape)
        print(batch['action'])
    average_load_time = (time.time() - start_time) / len(dataloader)
    print(f"Average data loading time per batch: {average_load_time:.4f} seconds")
if __name__ == "__main__":
    test()
    