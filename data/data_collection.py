import multiprocessing as mp
import os
import time
import traceback

import gym
import hydra
import numpy as np
import yaml
from xlib.algo.utils.transforms import applyDeltaPose6d
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
        pbvs_controller = PBVS(gain=np.array([10.0] * 6))
        
        env.reset()
        observation = env.get_observation()
        cur_pose = observation["tcp_obs"]
        while np.linalg.norm(cur_pose - tar_pose) > 1e-3:
            start_time = time.time()
            observation = env.get_observation()
            cur_pose = observation["tcp_obs"]
            pbvs_controller.update(cur_pose, tar_pose)
            vel = pbvs_controller.calc_vel()
            print(vel)
            next_pose = applyDeltaPose6d(cur_pose, vel * config.dt)
            # step the environment
            env.action(next_pose)
            # record data
            record_data = {
                "observations": observation,
                "target_pose": next_pose,
            }
            saver.add_frame(record_data)
            elapsed_time = time.time() - start_time
            if elapsed_time < config.dt:
                time.sleep(config.dt - elapsed_time)
            
        saver.save_episode()
        
    except Exception as e:
        traceback.print_exc()
        env.close()
        saver.stop()
        
    env.close()
        
    # print("Initial Observation:", observation)
    

if __name__ == "__main__":
    main()    