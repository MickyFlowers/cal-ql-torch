import time

import gym
import hydra
import numpy as np
from xlib.algo.utils.transforms import applyDeltaPose6d, velTransform

import env


@hydra.main(config_path="../config/env", config_name="env")
def main(config):
    env = gym.make("ur_env_v0", config=config)
    env.reset()
    # print("Initial Observation:", observation)
    # env.close()
    while True:
        space_mouse_twist, enable_teleop = env.get_space_mouse_state()
        if enable_teleop:
            vel = space_mouse_twist * np.array(
                config.teleop_twist_scale
            )  # Scale down the twist for teleoperation

            env.action(vel)
        # print("Space Mouse Twist:", space_mouse_twist, "Enable Teleop:", enable_teleop)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
