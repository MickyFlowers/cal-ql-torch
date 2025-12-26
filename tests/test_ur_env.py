import time

import gym
import hydra
import numpy as np
from xlib.algo.utils.transforms import applyDeltaPose6d

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
            target_pose = env.get_target_pose()
            delta_pose = space_mouse_twist * 1.0 / 30.0 * np.array(config.teleop_twist_scale)  # Scale down the twist for teleoperation
            next_pose = applyDeltaPose6d(target_pose, delta_pose)
            env.action(next_pose)
        # print("Space Mouse Twist:", space_mouse_twist, "Enable Teleop:", enable_teleop)
        time.sleep(1.0 / 30.0)
    
    

if __name__ == "__main__":
    main()    