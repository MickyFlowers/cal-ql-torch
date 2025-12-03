import gym
import hydra

import env

import time
from xlib.algo.utils.transforms import applyDeltaPose6d

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
            delta_pose = space_mouse_twist * 0.01 * config.teleop_twist_scale  # Scale down the twist for teleoperation
            next_pose = applyDeltaPose6d(target_pose, delta_pose)
            env.step(next_pose)
        # print("Space Mouse Twist:", space_mouse_twist, "Enable Teleop:", enable_teleop)
        time.sleep(0.01)
    
    

if __name__ == "__main__":
    main()    