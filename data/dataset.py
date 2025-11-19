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


class CalqlDataset(Dataset):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        # parser all episode files
        root_path = config.root_path
        statics_file = os.path.join(root_path, "statistics.yaml")
        self.episode_files = glob.glob(os.path.join(root_path, "*.hdf5"))
        self.statistics = self._parser_statstics(statics_file)
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
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
        return len(self.episode_files)

    
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
        episode_file = self.episode_files[index]
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
            
            sample_idx = np.random.randint(0, episode_length)
            
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
    