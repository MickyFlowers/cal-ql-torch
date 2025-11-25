import argparse
import glob
import os

import h5py
import numpy as np
import yaml
from tqdm import tqdm


def main(args):
    episode_files = glob.glob(os.path.join(args.data_root, "*.hdf5"))
    meta = {"episode_length": []}
    for episode_file in tqdm(episode_files):
        with h5py.File(episode_file, 'r') as f:
            actions = f['actions'][:]
            episode_length = actions.shape[0]
            meta["episode_length"].append(episode_length)
    meta_file = os.path.join(args.data_root, "meta.yaml")
    with open(meta_file, 'w') as f:
        yaml.dump(meta, f)
    print(f"Saved meta information to {meta_file}")

    
            
            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the image data file.")
    args = parser.parse_args()
    main(args)