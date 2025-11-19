import multiprocessing as mp
import os
import traceback

import gym
import hydra
import numpy as np
import yaml
from xlib.algo.utils.transforms import applyDeltaPose6d
from xlib.algo.vs.vs_controller.pbvs import PBVS
from xlib.data.hdf5_saver import HDF5Saver

import env


@hydra.main(config_path="../config", config_name="data_collection", version_base=None)
def main(config):
    env = gym.make("ur_env_v0", config=config.env)
    try:
        teach_pose_file = "./config/env/teach_pose.yaml"
        if config.teach_mode:
            env.wait_for_obs()
            obs = env.get_observation()
            tar_pose = {"tar_pose": obs["tcp_obs"].tolist()}
            with open(teach_pose_file, "w") as f:
                yaml.safe_dump(tar_pose, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(teach_pose_file, "r") as f:
                tar_pose = yaml.safe_load(f)["tar_pose"]
            saver = HDF5Saver(config.save_path)
            saver.start()
            pbvs_controller = PBVS(gain=np.array([10.0] * 6))
            for _ in range(config.num_data):
                observation = env.reset()
                cur_pose = observation["tcp_obs"]
                
                while np.linalg.norm(cur_pose - tar_pose) > 1e-3:
                    print(np.linalg.norm(cur_pose - tar_pose))
                    pbvs_controller.update(cur_pose, tar_pose)
                    vel = pbvs_controller.calc_vel()
                    if np.linalg.norm(vel[:3]) > config.max_linear_vel:
                        vel[:3] = vel[:3] / np.linalg.norm(vel[:3]) * config.max_linear_vel
                    if np.linalg.norm(vel[3:]) > config.max_angular_vel:
                        vel[3:] = vel[3:] / np.linalg.norm(vel[3:]) * config.max_angular_vel
                    next_pose = applyDeltaPose6d(cur_pose, vel * config.dt)
                    # step the environment
                    next_observations, reward, done, info = env.step(next_pose)
                    # record data
                    record_data = {
                        "observations": observation,
                        "next_observations": next_observations,
                        "actions": next_pose,
                        "rewards": reward,
                        "dones": done,
                        "info": info,
                    }
                    saver.add_frame(record_data)
                    observation = next_observations 
                    cur_pose = next_observations["tcp_obs"]
                saver.save_episode()
            
            
        
    except Exception as e:
        traceback.print_exc()
        env.close()
        saver.stop()
        
    env.close()
        
    # print("Initial Observation:", observation)
    

if __name__ == "__main__":
    main()    