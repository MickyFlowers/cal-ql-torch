import argparse
import glob
import os

import h5py
import numpy as np
import yaml


def main(args):
    episode_files = glob.glob(os.path.join(args.data_root, "*.hdf5"))
    proprio_max = None
    proprio_min = None
    action_max = None
    action_min = None   
    for file in episode_files:
        with h5py.File(file, 'r') as f:
            jnt_obs_sum = np.array(f['observations']['jnt_obs'][:])
            tcp_obs_sum = np.array(f['observations']['tcp_obs'][:])
            proprio = np.concatenate([jnt_obs_sum, tcp_obs_sum], axis=-1)
            if proprio_max is None:
                proprio_max = np.max(proprio, axis=0)
            else:
                max_candidate = np.max(proprio, axis=0)
                proprio_max = np.maximum(proprio_max, max_candidate)
            if proprio_min is None:
                proprio_min = np.min(proprio, axis=0)
            else:
                min_candidate = np.min(proprio, axis=0)
                proprio_min = np.minimum(proprio_min, min_candidate)
            
            actions = np.array(f['actions'][:])
            if action_max is None:
                action_max = np.max(actions, axis=0)
            else:
                max_candidate = np.max(actions, axis=0)
                action_max = np.maximum(action_max, max_candidate)
            if action_min is None:
                action_min = np.min(actions, axis=0)
            else:
                min_candidate = np.min(actions, axis=0)
                action_min = np.minimum(action_min, min_candidate)
    # save statistics   
    statistics = {
        "proprio": {
            "max": proprio_max.tolist(),
            "min": proprio_min.tolist(),
        },
        "action": {
            "max": action_max.tolist(),
            "min": action_min.tolist(),
        }
    }
    with open(os.path.join(args.data_root, "statistics.yaml"), 'w') as f:
        yaml.safe_dump(statistics, f, default_flow_style=False, allow_unicode=True)
    
            
            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the image data file.")
    args = parser.parse_args()
    main(args)