import argparse
import glob
import os

import h5py
import numpy as np
from tqdm import tqdm


def process(args):
    dataset_folder = os.path.normpath(os.path.expanduser(args.data_root))
    base_name = os.path.basename(dataset_folder)
    if base_name == "":
        raise ValueError(f"Invalid data_root: {args.data_root}")
    new_base_name = base_name.replace("raw", "processed", 1)
    new_folder = os.path.join(os.path.dirname(dataset_folder), new_base_name)
    os.makedirs(new_folder, exist_ok=True)
    episode_files = glob.glob(os.path.join(dataset_folder, "*.hdf5"))
    print(f"Loading {len(episode_files)} episodes")
    for episode_file in tqdm(episode_files):
        with h5py.File(episode_file, 'r') as f:
            observations = f['observations']
            ft_obs = observations['ft_obs'][:]
            jnt_obs = observations['jnt_obs'][:]
            tcp_obs = observations['tcp_obs'][:]
            img_obs = observations['img_obs'][:]
            action = f['action'][:]
            
            episode_length = ft_obs.shape[0]
            rewards = np.zeros((episode_length - 1,), dtype=np.float32)
            rewards[-1] = 1.0
            dones = np.zeros((episode_length - 1,), dtype=np.float32)
            dones[-1] = 1.0
        
        episode_file_name = os.path.basename(episode_file)
        new_episode_file = os.path.join(new_folder, episode_file_name)
        with h5py.File(new_episode_file, 'w') as f:
            observations = f.create_group('observations')
            next_observations = f.create_group('next_observations')
            observations.create_dataset('ft_obs', data=ft_obs[:-1])
            observations.create_dataset('tcp_obs', data=tcp_obs[:-1])
            observations.create_dataset('jnt_obs', data=jnt_obs[:-1])
            observations.create_dataset('img_obs', data=img_obs[:-1])
            f.create_dataset('rewards', data=rewards)
            f.create_dataset('dones', data=dones)
            f.create_dataset('actions', data=action[:-1])
            next_observations.create_dataset('ft_obs', data=ft_obs[1:])
            next_observations.create_dataset('tcp_obs', data=tcp_obs[1:])
            next_observations.create_dataset('jnt_obs', data=jnt_obs[1:])
            next_observations.create_dataset('img_obs', data=img_obs[1:])
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to dataset folder")
    args = parser.parse_args()
    process(args=args)
