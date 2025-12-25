"""
Diffusion Policy Rollout

Rollout script for deploying Diffusion Policy in the UR robot environment.
Uses EMA model for inference and supports action horizon prediction.
"""

import os
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
from model.diffusion_policy import DiffusionPolicy
from model.vision_model import VitFeatureExtractor


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


@hydra.main(config_path="../config", config_name="rollout_diffusion", version_base=None)
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

        # Load checkpoint
        ckpt_path = config.ckpt_path
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt_state_dict = torch.load(ckpt_path, map_location=config.device)

        # Determine dtype
        dtype = torch.bfloat16 if config.use_bf16 else torch.float32

        # Create vision encoder using VitFeatureExtractor (same as training)
        vision_encoder = VitFeatureExtractor(
            model_name=config.model_name,
            pretrained=True,
            trainable_layers=None,  # All frozen for inference
            dtype=dtype,
        )
        vision_encoder.to(config.device)

        # Load vision encoder weights (prefer EMA if available)
        if "vision_encoder_ema_state_dict" in ckpt_state_dict:
            vision_encoder.model.load_state_dict(ckpt_state_dict["vision_encoder_ema_state_dict"])
            print("Loaded vision encoder EMA weights")
        elif "vision_encoder_state_dict" in ckpt_state_dict:
            vision_encoder.model.load_state_dict(ckpt_state_dict["vision_encoder_state_dict"])
            print("Loaded vision encoder weights")
        vision_encoder.eval()

        # Create diffusion policy
        policy = DiffusionPolicy(
            action_dim=config.action_dim,
            pred_horizon=config.pred_horizon,
            config=config.dp_config,
            img_token_dim=config.img_token_dim,
            state_token_dim=config.state_token_dim,
            img_cond_len=config.img_cond_len,
            img_pos_embed_config=config.img_pos_embed_config,
            dtype=dtype,
        )
        policy.to(device=config.device)

        # Load policy weights (prefer EMA if available)
        if "policy_ema_state_dict" in ckpt_state_dict:
            policy.load_state_dict(ckpt_state_dict["policy_ema_state_dict"])
            print("Loaded policy EMA weights")
        elif "policy_state_dict" in ckpt_state_dict:
            policy.load_state_dict(ckpt_state_dict["policy_state_dict"])
            print("Loaded policy weights")
        policy.eval()

        # Load statistics for normalization
        with open(config.statistics_path, 'r') as f:
            statistics = yaml.safe_load(f)

        # Setup data saver if needed
        saver = None
        if config.save_data:
            os.makedirs(config.save_path, exist_ok=True)
            saver = HDF5BlockSaver(config.save_path, idx=0)

        episode_count = 0
        while episode_count < config.num_episodes:
            observation = env.reset()
            step_count = 0

            # Action horizon buffer
            action_horizon = None
            horizon_idx = 0

            while True:
                # Extract observations
                jnt_obs = observation["jnt_obs"]
                tcp_obs = observation["tcp_obs"]
                proprio = np.concatenate([jnt_obs, tcp_obs], axis=-1)

                # Normalize proprioception
                proprio = normalize(proprio, statistics['proprio'], config.proprio_norm_type)
                proprio_tensor = torch.tensor(
                    proprio, dtype=torch.float32
                ).unsqueeze(0).to(device=config.device)

                # Process image
                image_bytes = observation["img_obs"]
                image = np_buffer_to_pil_image(np.frombuffer(image_bytes, dtype=np.uint8))
                image = image_transform(image).unsqueeze(0).to(device=config.device)

                # Get action from policy
                with torch.no_grad():
                    # Predict new action horizon when buffer is empty or depleted
                    if action_horizon is None or horizon_idx >= config.action_horizon_steps:
                        # Get image embeddings from vision encoder
                        # VitFeatureExtractor returns (cls_token, patch_tokens)
                        # Training uses patch_tokens (index [1])
                        _, img_tokens = vision_encoder(image)

                        # Predict action sequence
                        action_pred = policy.predict_action(img_tokens, proprio_tensor)
                        action_horizon = action_pred.squeeze(0).float().cpu().numpy()
                        horizon_idx = 0

                    action = action_horizon[horizon_idx]
                    horizon_idx += 1

                # Denormalize action
                action = denormalize(action, statistics['action'], config.action_norm_type)

                # Step environment
                next_observations, reward, done, info = env.step(action)

                # Record data if saving
                if saver is not None:
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
                step_count += 1

                if done or step_count >= config.max_steps:
                    break

            if saver is not None:
                saver.save_episode()

            episode_count += 1
            print(f"Episode {episode_count}/{config.num_episodes} completed with {step_count} steps")

    except Exception as e:
        traceback.print_exc()
    finally:
        env.close()
        if saver is not None:
            saver.stop()


if __name__ == "__main__":
    main()
