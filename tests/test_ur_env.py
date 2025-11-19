import gym
import hydra

import env

import time

@hydra.main(config_path="../config/env", config_name="env")
def main(config):
    env = gym.make("ur_env_v0", config=config)
    observation = env.reset()
    print("Initial Observation:", observation)
    # env.close()
    while True:
        time.sleep(1)
    
    

if __name__ == "__main__":
    main()    