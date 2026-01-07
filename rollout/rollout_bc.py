"""
BC (Behavior Cloning) Policy Rollout with Velocity Control

Rollout script for deploying BC policy (ResNetPolicy) in the UR robot environment.
The policy outputs velocity commands directly.
"""

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

import env
from model.model import ResNetPolicy


def normalize(data, statistics, norm_type, epsilon=1e-6):
    """Normalize data to [-1, 1] range (max_min) or zero mean unit variance (mean_std).

    Must match the normalization in data/dataset.py for training consistency.
    """
    if norm_type == 'max_min':
        data_max = np.array(statistics['max']) + epsilon
        data_min = np.array(statistics['min']) - epsilon
        # Normalize to [-1, 1] range - matches training dataset
        data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
    elif norm_type == 'mean_std':
        data_mean = np.array(statistics['mean'])
        data_std = np.array(statistics['std'])
        data = (data - data_mean) / data_std
    return data


def denormalize(data, statistics, norm_type, epsilon=1e-6):
    """Denormalize data from [-1, 1] range (max_min) or standardized (mean_std).

    Inverse of normalize() function.
    """
    if norm_type == 'max_min':
        data_max = np.array(statistics['max']) + epsilon
        data_min = np.array(statistics['min']) - epsilon
        # Denormalize from [-1, 1] range - inverse of training normalization
        data = (data + 1.0) / 2.0 * (data_max - data_min) + data_min
    elif norm_type == 'mean_std':
        data_mean = np.array(statistics['mean'])
        data_std = np.array(statistics['std'])
        data = data * data_std + data_mean
    return data


@hydra.main(config_path="../config", config_name="rollout_bc", version_base=None)
def main(config):
    env = gym.make("ur_env_v0", config=config.env)

    # Success/failure counters
    success_count = 0
    failure_count = 0

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

        # Load checkpoint
        ckpt_path = config.ckpt_path
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt_state_dict = torch.load(ckpt_path, map_location=config.device)
        policy_state_dict = ckpt_state_dict["policy_state_dict"]

        # Create policy model
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

        # Load statistics for normalization
        with open(config.statistics_path, 'r') as f:
            statistics = yaml.safe_load(f)

        # Setup data saver if save_data is enabled
        saver = None
        if config.get('save_data', False):
            os.makedirs(config.save_path, exist_ok=True)
            saver = HDF5BlockSaver(config.save_path, idx=config.get('episode_idx', 0))

        env.reset()
        env.regrasp()
        env.reset()
        step_count = 0
        while True:
            while True:
                start_time = time.time()
                observation = env.get_observation()
                # Extract observations - use ft_obs as proprio (same as training dataset)
                proprio = observation["ft_obs"]

                # Normalize proprioception
                proprio = normalize(proprio, statistics['proprio'], config.proprio_norm_type)
                proprio_tensor = torch.tensor(
                    proprio, dtype=torch.float32
                ).unsqueeze(0).to(device=config.device)

                # Process image
                image_bytes = observation["img_obs"]
                image = np_buffer_to_pil_image(np.frombuffer(image_bytes, dtype=np.uint8))
                image = image_transform(image).unsqueeze(0).to(device=config.device)

                # Get velocity action from policy
                with torch.no_grad():
                    velocity, _ = policy(proprio_tensor, image, deterministic=True)
                    velocity = velocity.squeeze(0).cpu().numpy()

                # Denormalize velocity
                velocity = denormalize(velocity, statistics['action'], config.action_norm_type)

                # Execute velocity command
                env.action(velocity)

                # Save data if enabled
                if saver is not None:
                    record_data = {
                        "observations": observation,
                        "action": velocity,
                    }
                    saver.add_frame(record_data)

                step_count += 1

                # Control execution frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1.0 / config.freq:
                    time.sleep(1.0 / config.freq - elapsed_time)

                # Check keyboard feedback
                key = env.get_key()
                if key == 's':
                    success_count += 1
                    print(f"\n[SUCCESS] Total: success={success_count}, failure={failure_count}")
                    env.regrasp()
                    env.reset()
                    break
                elif key == 'f':
                    failure_count += 1
                    print(f"\n[FAILURE] Total: success={success_count}, failure={failure_count}")
                    print("Press 'y' to regrasp and continue...")
                    env.ur_gripper.move_and_wait_for_pos(0, 255, 100)
                    while True:
                        key = env.keyboard_reader.get_key()
                        if key == 'y':
                            print('y pressed')
                            time.sleep(0.01)
                            break
                    env.reset()
                    env.regrasp()
                    env.reset()
                    step_count = 0
                    print("Environment reset complete. Continuing rollout...")
                    break

                if step_count >= config.max_steps:
                    break

            # Save episode if data saving is enabled
            if saver is not None:
                saver.save_episode()
                print(f"Episode saved to {config.save_path}")
            print(f"\nFinal results: success={success_count}, failure={failure_count}")
    except Exception as e:
        traceback.print_exc()
    finally:
        if saver is not None:
            saver.stop()
        env.close()


if __name__ == "__main__":
    main()
