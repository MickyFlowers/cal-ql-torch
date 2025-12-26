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
from xlib.algo.utils.transforms import applyDeltaPose6d
from xlib.data.hdf5_saver import HDF5BlockSaver
from xlib.data.remote_transfer import RemoteTransfer

import env
from model.model import ResNetPolicy


@hydra.main(config_path="../config", config_name="teleop", version_base=None)
def main(config):
    env = gym.make("ur_env_v0", config=config.env)
    try:
        count = 0
        saver = HDF5BlockSaver(config.save_path, idx=0)
        while True:
            env.reset()
            start_recording = False
            while True:
                observation = env.get_observation()
                space_mouse_twist, enable_teleop = env.get_space_mouse_state()
                if enable_teleop:
                    start_recording = True
                    target_pose = env.get_target_pose()
                    delta_pose = space_mouse_twist * 0.01 * config.env.teleop_twist_scale  # Scale down the twist for teleoperation
                    next_pose = applyDeltaPose6d(target_pose, delta_pose)
                    env.action(next_pose)
                    record_data = {
                        "observations": observation,
                        "next_observations": next_pose,
                    }
                    saver.add_frame(record_data)
                if start_recording and not enable_teleop:
                    break
            saver.save_episode()
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