import multiprocessing as mp
import os
import time
import traceback

import gym
import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from xlib.algo.utils.image_utils import np_buffer_to_pil_image
from xlib.data.hdf5_saver import HDF5BlockSaver
from xlib.data.remote_transfer import RemoteTransfer

import env
from model.model import ResNetPolicy


def normalize(data, statistics, norm_type, epsilon=1e-8):
    if norm_type == 'max_min':
        data_max = np.array(statistics['max']) + epsilon
        data_min = np.array(statistics['min']) - epsilon
        data = (data - data_min) / (data_max - data_min)
    elif norm_type == 'mean_std':
        data_mean = np.array(statistics['mean'])
        data_std = np.array(statistics['std'])
        data = (data - data_mean) / data_std
    return data 


def denormalize(data, statistics, norm_type, epsilon=1e-8):
    if norm_type == 'max_min':
        data_max = np.array(statistics['max']) + epsilon
        data_min = np.array(statistics['min']) - epsilon
        data = data * (data_max - data_min) + data_min
    elif norm_type == 'mean_std':
        data_mean = np.array(statistics['mean'])
        data_std = np.array(statistics['std'])
        data = data * data_std + data_mean
    return data

@hydra.main(config_path="../config", config_name="rollout_online", version_base=None)
def main(config):
    env = gym.make("ur_env_v0", config=config.env)
    try:
        image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(                     
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        first_flag = True
        remote_transfer = RemoteTransfer(
            config.host_name,
            config.port,
            config.user_name,
            key_filepath=config.key_filepath,
        )
        episode_files = remote_transfer.list_remote_dir(config.remote_data_dir)
        episode_files = [f for f in episode_files if f.endswith('.hdf5')]
        num_episodes = len(episode_files)
        # download offline ckpt from remote server
        local_offline_ckpt_file = os.path.join(config.local_ckpt_dir, "offline_ckpt.pth")
        if not os.path.exists(local_offline_ckpt_file):
            remote_transfer.download_file(
                config.remote_offline_ckpt, local_offline_ckpt_file, overwrite=True
            )
        ckpt_state_dict = torch.load(local_offline_ckpt_file, map_location=config.device)
        policy_state_dict = ckpt_state_dict["policy_state_dict"]
        policy = ResNetPolicy(
            config.observation_dim,
            config.action_dim,
            config.policy_obs_proj_arch,
            config.policy_out_proj_arch,
            config.hidden_dim,
            config.orthogonal_init,
            config.policy_log_std_multiplier,
            config.policy_log_std_offset,
            train_backbone=False,
        )
        policy.to(device=config.device)
        policy.load_state_dict(policy_state_dict)
        policy.eval()
        # calculate statistics
        statistics_remote_file = os.path.join(config.remote_data_dir, "statistics.yaml")
        local_statistics_file = os.path.join(config.local_ckpt_dir, "statistics.yaml")
        remote_transfer.download_file(
            statistics_remote_file, local_statistics_file, overwrite=True
        )
        with open(local_statistics_file, 'r') as f:
            statistics = yaml.safe_load(f)
        count = 0
        saver = HDF5BlockSaver(config.save_path, idx=num_episodes)
        online_ckpt_file = None
        while True:
            observation = env.reset()
            if first_flag:
                first_flag = False
            else:
                if count % config.ckpt_download_interval == 0:
                    ckpt_files = remote_transfer.list_remote_dir(config.remote_online_ckpt_dir)
                    if len(ckpt_files) != 0:
                        ckpt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        latest_ckpt_file = ckpt_files[-1]
                        if online_ckpt_file is None or latest_ckpt_file != online_ckpt_file:
                            online_ckpt_file = latest_ckpt_file
                            local_online_ckpt_file = os.path.join(
                                config.local_ckpt_dir, "online_ckpt.pth"
                            )
                            print(f"Downloading online ckpt {online_ckpt_file} ...")
                            remote_transfer.download_file(
                                os.path.join(config.remote_online_ckpt_dir, online_ckpt_file),
                                local_online_ckpt_file,
                                overwrite=True
                            )
                            ckpt_state_dict = torch.load(
                                local_online_ckpt_file, map_location=config.device
                            )
                            policy_state_dict = ckpt_state_dict["policy_state_dict"]
                            policy.load_state_dict(policy_state_dict)
                            policy.to(device=config.device)
                            policy.eval()
                            print(f"Load online ckpt: {online_ckpt_file}")
            while True:
                jnt_obs = observation["jnt_obs"]
                tcp_obs = observation["tcp_obs"]
                proprio = np.concatenate([jnt_obs, tcp_obs], axis=-1)
                proprio = normalize(proprio, statistics['proprio'], config.proprio_norm_type)
                proprio_tensor = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(device=config.device)
                image_bytes = observation["img_obs"]
                image = np_buffer_to_pil_image(np.frombuffer(image_bytes, dtype=np.uint8))
                image = image_transform(image).unsqueeze(0).to(device=config.device)
                with torch.no_grad():
                    action, log_prob = policy(proprio_tensor, image, deterministic=True)
                    action = action.squeeze(0).cpu().numpy()
                action = denormalize(action, statistics['action'], config.action_norm_type)
                # debug
                # step the environment
                next_observations, reward, done, info = env.step(action)
                # record data
                record_data = {
                    "observations": observation,
                    "next_observations": next_observations,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                    "info": info,
                }
                saver.add_frame(record_data)
                observation = next_observations 
                if done:
                    break
            saver.save_episode()
            file_path = os.path.join(
                config.save_path, f"{num_episodes}.hdf5"
            )
            remote_transfer.upload_file(
                file_path,
                os.path.join(config.remote_data_dir, f"{num_episodes}.hdf5"),
                overwrite=False
            )
            num_episodes += 1
            count += 1
        
    except Exception as e:
        traceback.print_exc()
        env.close()
        saver.stop()
        
    env.close()
        
    # print("Initial Observation:", observation)
    

if __name__ == "__main__":
    main()    