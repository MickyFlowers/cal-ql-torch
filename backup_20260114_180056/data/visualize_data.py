import argparse
import glob
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from xlib.algo.utils.image_utils import bytes_to_cv2_image, save_video


def draw_pose(ax, p, axis_len=0.05, lw=2):
    rot_mat = R.from_rotvec(p[3:]).as_matrix()
    t = p[:3]

    axes = np.eye(3)
    colors = ["r", "g", "b"]

    for i in range(3):
        dir_vec = rot_mat[:, i] * axis_len
        ax.plot(
            [t[0], t[0] + dir_vec[0]],
            [t[1], t[1] + dir_vec[1]],
            [t[2], t[2] + dir_vec[2]],
            color=colors[i],
            linewidth=lw,
        )


def make_3d_trajectory_video(poses, future_len=50):
    frames = []
    poses = np.array(poses)
    positions = poses[:, :3]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    min_xyz = positions.min(axis=0)
    max_xyz = positions.max(axis=0)
    pad = 0.05
    ax.set_xlim(min_xyz[0] - pad, max_xyz[0] + pad)
    ax.set_ylim(min_xyz[1] - pad, max_xyz[1] + pad)
    ax.set_zlim(min_xyz[2] - pad, max_xyz[2] + pad)

    for i in range(len(poses)):
        ax.cla()

        # 固定范围避免视频跳动
        ax.set_xlim(min_xyz[0] - pad, max_xyz[0] + pad)
        ax.set_ylim(min_xyz[1] - pad, max_xyz[1] + pad)
        ax.set_zlim(min_xyz[2] - pad, max_xyz[2] + pad)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=45)

        # 绘制过去轨迹
        if i > 1:
            ax.plot(
                positions[:i, 0],
                positions[:i, 1],
                positions[:i, 2],
                color="gray",
                linewidth=2,
                alpha=0.6,
            )

        # 绘制未来轨迹
        j_end = min(i + future_len, len(poses))
        ax.plot(
            positions[i:j_end, 0],
            positions[i:j_end, 1],
            positions[i:j_end, 2],
            color="orange",
            linewidth=2,
            alpha=0.9,
        )

        draw_pose(ax, poses[i], axis_len=0.03)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    return frames


def visualize_data(args):
    episode_files = glob.glob(f"{args.data_root}/*.hdf5")
    os.makedirs(f"{args.data_root}_visualizations", exist_ok=True)
    num_sample = int(args.sample_ratio * len(episode_files))
    sampled_files = random.sample(episode_files, num_sample)

    for file in sampled_files:
        base_name = os.path.basename(file).split(".")[0]
        output_path = os.path.join(f"{args.data_root}_visualizations", f"{base_name}", "video.mp4")
        with h5py.File(file, "r") as f:
            # plot video
            image_bytes = f["observations"]["img_obs"]
            episode_length = image_bytes.shape[0]

            images = []
            for i in range(episode_length):
                cv2_image = bytes_to_cv2_image(image_bytes[i])
                images.append(cv2_image)

            save_video(output_path, images, fps=args.fps)
            # plot pose
            tcp_obs = f["observations"]["tcp_obs"][:]
            frames = make_3d_trajectory_video(tcp_obs, future_len=50)
            pose_video_path = os.path.join(
                f"{args.data_root}_visualizations", f"{base_name}", "traj.mp4"
            )
            save_video(pose_video_path, frames, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to the image data file.")
    parser.add_argument("--sample_ratio", type=float, default=1, help="sample ratio")
    parser.add_argument("--fps", type=int, default=30, help="data fps")
    args = parser.parse_args()
    visualize_data(args)
