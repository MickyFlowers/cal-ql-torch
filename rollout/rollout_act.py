"""
ACT (Action Chunking with Transformers) Policy Rollout

Rollout script for deploying ACT policy in the UR robot environment.
Supports temporal ensemble for smoother action execution.
"""

import time
import traceback

import gym
import hydra
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from xlib.algo.utils.image_utils import np_buffer_to_pil_image

import env
from act.act_model import ACTPolicy, TemporalEnsemble


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


@hydra.main(config_path="../config", config_name="rollout_act", version_base=None)
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
        policy_state_dict = ckpt_state_dict["policy_state_dict"]

        # Create ACT policy model
        policy = ACTPolicy(
            action_dim=config.action_dim,
            proprio_dim=config.observation_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            chunk_size=config.chunk_size,
            dim_feedforward=config.dim_feedforward,
            backbone_name=config.backbone_name,
            pretrained_backbone=False,  # Loading from checkpoint
            train_backbone=False,
        )
        policy.to(device=config.device)
        policy.load_state_dict(policy_state_dict)
        policy.eval()

        # Load statistics for normalization
        with open(config.statistics_path, 'r') as f:
            statistics = yaml.safe_load(f)

        # Setup temporal ensemble if enabled
        temporal_ensemble = None
        if config.temporal_ensemble:
            temporal_ensemble = TemporalEnsemble(
                chunk_size=config.chunk_size,
                action_dim=config.action_dim,
                decay=config.temporal_decay,
            )

        episode_count = 0
        while episode_count < config.num_episodes:
            observation = env.reset()
            step_count = 0

            # Reset temporal ensemble for new episode
            if temporal_ensemble is not None:
                temporal_ensemble.reset()

            # Action chunk buffer for non-ensemble mode
            action_chunk = None
            chunk_idx = 0

            while True:
                start_time = time.time()

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
                    if temporal_ensemble is not None:
                        # Use temporal ensemble for smooth action execution
                        action = policy.get_action(
                            proprio_tensor, image, temporal_ensemble
                        )
                        if action.dim() == 1:
                            action = action.cpu().numpy()
                        else:
                            action = action[0].cpu().numpy()
                    else:
                        # Use action chunking: predict new chunk when needed
                        if action_chunk is None or chunk_idx >= config.chunk_size:
                            pred_actions, _, _ = policy(
                                proprio_tensor, image, deterministic=True
                            )
                            action_chunk = pred_actions.squeeze(0).cpu().numpy()
                            chunk_idx = 0

                        action = action_chunk[chunk_idx]
                        chunk_idx += 1

                # Denormalize action
                action = denormalize(action, statistics['action'], config.action_norm_type)

                # Step environment
                next_observations, reward, done, info = env.step(action)

                observation = next_observations
                step_count += 1

                # Control execution frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1.0 / config.freq:
                    time.sleep(1.0 / config.freq - elapsed_time)

                if done or step_count >= config.max_steps:
                    break

            episode_count += 1
            print(f"Episode {episode_count}/{config.num_episodes} completed with {step_count} steps")

    except Exception as e:
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
