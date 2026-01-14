"""
Compute the mean of force/torque values at the first frame across all episodes.
"""

import argparse
import glob
import os

import h5py
import numpy as np


def compute_ft_mean(args):
    dataset_folder = os.path.normpath(os.path.expanduser(args.data_root))
    episode_files = glob.glob(os.path.join(dataset_folder, "*.hdf5"))
    episode_files.sort()

    print(f"Found {len(episode_files)} episodes in {dataset_folder}")

    first_frame_ft_list = []
    for episode_file in episode_files:
        with h5py.File(episode_file, "r") as f:
            observations = f["observations"]
            ft_obs = observations["ft_obs"][:]
            first_frame_ft_list.append(ft_obs[0])

    first_frame_ft_array = np.array(first_frame_ft_list)
    mean_ft = np.mean(first_frame_ft_array, axis=0)

    print(f"\nFirst-frame F/T mean across {len(episode_files)} episodes:")
    print(f"  Fx: {mean_ft[0]:.6f}")
    print(f"  Fy: {mean_ft[1]:.6f}")
    print(f"  Fz: {mean_ft[2]:.6f}")
    print(f"  Tx: {mean_ft[3]:.6f}")
    print(f"  Ty: {mean_ft[4]:.6f}")
    print(f"  Tz: {mean_ft[5]:.6f}")
    print(f"\nAs array: {mean_ft.tolist()}")

    return mean_ft


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset folder")
    args = parser.parse_args()
    compute_ft_mean(args)
