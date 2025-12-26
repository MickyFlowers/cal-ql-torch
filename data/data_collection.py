"""
Automatic Data Collection with Velocity Control

Collects data using visual servoing with velocity-based control.
"""

import time
import traceback

import gym
import hydra
import numpy as np
from xlib.algo.vs.vs_controller.pbvs import PBVS
from xlib.data.hdf5_saver import HDF5BlockSaver

import env


@hydra.main(config_path="../config", config_name="data_collection", version_base=None)
def main(config):
    env = gym.make("ur_env_v0", config=config.env)
    try:
        env.wait_for_obs()
        obs = env.get_observation()

        tar_pose = obs["tcp_obs"]
        saver = HDF5BlockSaver(config.save_path)
        pbvs_controller = PBVS(gain=np.array([5.0] * 6))

        env.reset()
        observation = env.get_observation()
        cur_pose = observation["tcp_obs"]

        while np.linalg.norm(cur_pose - tar_pose) > 1e-3:
            start_time = time.time()
            observation = env.get_observation()
            cur_pose = observation["tcp_obs"]

            # Calculate velocity using PBVS
            pbvs_controller.update(cur_pose, tar_pose)
            velocity = pbvs_controller.calc_vel()
            print(f"Velocity: {velocity}")

            # Execute velocity command directly
            env.action(velocity)

            # Record data: observations and velocity action
            record_data = {
                "observations": observation,
                "action": velocity,  # Now action is velocity
            }
            saver.add_frame(record_data)

            elapsed_time = time.time() - start_time
            if elapsed_time < 1.0 / config.freq:
                time.sleep(1.0 / config.freq - elapsed_time)

        saver.save_episode()

    except Exception as e:
        traceback.print_exc()
        env.close()
        saver.stop()

    env.close()


if __name__ == "__main__":
    main()
