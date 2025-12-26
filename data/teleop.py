"""
Teleoperation Data Collection with Velocity Control

Collects demonstration data using SpaceMouse for velocity-based control.
Records observations and velocity actions for training BC/ACT/Diffusion policies.
"""

import time
import traceback

import gym
import hydra
import numpy as np
from xlib.data.hdf5_saver import HDF5BlockSaver

import env


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
                start_time = time.time()
                observation = env.get_observation()
                # Get velocity command from SpaceMouse
                space_mouse_twist, enable_teleop = env.get_space_mouse_state()

                if enable_teleop:
                    start_recording = True
                    # Scale the velocity for teleoperation
                    velocity = space_mouse_twist * config.env.teleop_twist_scale
                    # Execute velocity command
                    env.action(velocity)

                    # Record data: observations and velocity action
                    record_data = {
                        "observations": observation,
                        "action": velocity,  # Now action is velocity
                    }
                    saver.add_frame(record_data)

                if start_recording and not enable_teleop:
                    break

                # Control frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1.0 / config.freq:
                    time.sleep(1.0 / config.freq - elapsed_time)

            saver.save_episode()
            count += 1
            print(f"Episode {count} saved.")

    except Exception as e:
        traceback.print_exc()
        env.close()
        saver.stop()

    env.close()


if __name__ == "__main__":
    main()
