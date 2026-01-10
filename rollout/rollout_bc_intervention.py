"""
BC Policy Rollout with SpaceMouse Intervention for Data Collection

Rollout script for deploying BC policy with human intervention.
When SpaceMouse teleop is enabled, it takes over and records data.
Recording stops when teleop is disabled (enable_teleop: True -> False),
matching teleop.py data collection behavior.
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
    if norm_type == "max_min":
        data_max = np.array(statistics["max"]) + epsilon
        data_min = np.array(statistics["min"]) - epsilon
        data = ((data - data_min) / (data_max - data_min) * 2.0) - 1.0
    elif norm_type == "mean_std":
        data_mean = np.array(statistics["mean"])
        data_std = np.array(statistics["std"])
        data = (data - data_mean) / data_std
    return data


def denormalize(data, statistics, norm_type, epsilon=1e-6):
    """Denormalize data from [-1, 1] range (max_min) or standardized (mean_std)."""
    if norm_type == "max_min":
        data_max = np.array(statistics["max"]) + epsilon
        data_min = np.array(statistics["min"]) - epsilon
        data = (data + 1.0) / 2.0 * (data_max - data_min) + data_min
    elif norm_type == "mean_std":
        data_mean = np.array(statistics["mean"])
        data_std = np.array(statistics["std"])
        data = data * data_std + data_mean
    return data


@hydra.main(config_path="../config", config_name="rollout_bc_intervention", version_base=None)
def main(config):
    env = gym.make("ur_env_v0", config=config.env)

    success_count = 0
    failure_count = 0
    teleop_count = 0

    try:
        image_transform = transforms.Compose([
            transforms.Resize((config.image_resize, config.image_resize)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        ckpt_path = config.ckpt_path
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt_state_dict = torch.load(ckpt_path, map_location=config.device)
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

        with open(config.statistics_path, "r") as f:
            statistics = yaml.safe_load(f)

        # Load F/T bias for first frame correction
        ft_bias = np.array(statistics.get("ft_bias", [0.0] * 6), dtype=np.float32)

        saver = None
        if config.get("save_data", True):
            os.makedirs(config.save_path, exist_ok=True)
            saver = HDF5BlockSaver(config.save_path, idx=config.get("episode_idx", 0))

        teleop_scale = np.array(config.env.teleop_twist_scale, dtype=np.float32)

        env.reset()
        env.regrasp()
        env.reset()

        step_count = 0
        while True:
            start_recording = False
            teleop_done = False
            while True:
                start_time = time.time()
                observation = env.get_observation()
                space_mouse_twist, enable_teleop = env.get_space_mouse_state()

                if enable_teleop:
                    if not start_recording:
                        print("\n[TELEOP] Enabled - start recording intervention.")
                    start_recording = True
                    velocity = space_mouse_twist * teleop_scale

                    if saver is not None:
                        record_data = {
                            "observations": observation,
                            "action": velocity,
                        }
                        saver.add_frame(record_data)
                else:
                    if start_recording:
                        teleop_done = True
                        break

                    proprio = observation["ft_obs"]
                    # Subtract F/T bias (first frame mean from training data)
                    proprio = proprio - ft_bias
                    proprio = normalize(proprio, statistics["proprio"], config.proprio_norm_type)
                    proprio_tensor = torch.tensor(
                        proprio, dtype=torch.float32
                    ).unsqueeze(0).to(device=config.device)

                    image_bytes = observation["img_obs"]
                    image = np_buffer_to_pil_image(np.frombuffer(image_bytes, dtype=np.uint8))
                    image = image_transform(image).unsqueeze(0).to(device=config.device)

                    with torch.no_grad():
                        velocity, _ = policy(proprio_tensor, image, deterministic=True)
                        velocity = velocity.squeeze(0).cpu().numpy()
                    velocity = denormalize(velocity, statistics["action"], config.action_norm_type)

                env.action(velocity)
                step_count += 1

                elapsed_time = time.time() - start_time
                if elapsed_time < 1.0 / config.freq:
                    time.sleep(1.0 / config.freq - elapsed_time)

                if not start_recording:
                    key = env.get_key()
                    if key == "s":
                        success_count += 1
                        print(f"\n[SUCCESS] Total: success={success_count}, failure={failure_count}")
                        env.regrasp()
                        env.reset()
                        break
                    elif key == "f":
                        failure_count += 1
                        print(f"\n[FAILURE] Total: success={success_count}, failure={failure_count}")
                        print("Press 'y' to regrasp and continue...")
                        env.ur_gripper.move_and_wait_for_pos(0, 255, 100)
                        while True:
                            key = env.keyboard_reader.get_key()
                            if key == "y":
                                print("y pressed")
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

            if teleop_done:
                if saver is not None:
                    saver.save_episode()
                    print(f"[TELEOP] Episode saved to {config.save_path}")
                teleop_count += 1
                success_count += 1
                env.regrasp()
                env.reset()
                print(
                    f"[TELEOP] Total: teleop={teleop_count}, success={success_count}, "
                    f"failure={failure_count}"
                )

    except Exception:
        traceback.print_exc()
    finally:
        if saver is not None and hasattr(saver, "stop"):
            saver.stop()
        env.close()


if __name__ == "__main__":
    main()
