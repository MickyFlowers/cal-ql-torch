import numpy as np
from scipy.spatial.transform import Rotation as R
from xlib.algo.utils.transforms import *
from xlib.device.manipulator.ur_robot import UR

# reset right hand pose
# right_base_in_world_mtx = np.load('/home/cyx/project/cal-ql-torch/assets/right_base_to_world.npy')

# right_base_in_world = matrixToPose6d(right_base_in_world_mtx)
# right_ip="172.16.11.68"
# ur_robot = UR(right_ip, right_base_in_world)
# print(ur_robot.tcp_pose)
# euler = np.array([np.pi / 2, -np.pi / 4, -np.pi / 2])
# tcp_pose = np.zeros(6)
# tcp_pose[:3] = np.array([ 0.05,  0.6 , 0.2])
# rot_vec = R.from_euler('xyz', euler).as_rotvec()
# tcp_pose[3:] = rot_vec
# ur_robot.moveToPose(tcp_pose)
# ur_robot.close()

# reset left hand pose
tip_transform = np.zeros(6)
tip_transform[2] = 0.18
left_base_in_world_mtx = np.load('/home/cyx/project/cal-ql-torch/assets/left_base_to_world.npy')
left_base_in_world = matrixToPose6d(left_base_in_world_mtx)
left_ip = "172.16.11.233"
left_ur_robot = UR(left_ip, left_base_in_world)
print(left_ur_robot.tcp_pose)
left_tcp_pose = np.zeros(6)
left_tcp_pose[:3] = np.array([ -0.10,  0.601 , 0.2])

euler = np.array([3.0 / 4.0 * np.pi, 0.0, np.pi / 2])
rot_vec = R.from_euler('xyz', euler).as_rotvec()
left_tcp_pose[3:] = rot_vec
# sample delta_pose
delta_pose = np.zeros(6)
delta_pose[3:] = R.from_euler('xyz', np.array([0.0, 0.0, 0.0])).as_rotvec()
left_tcp_pose = applyDeltaPose6d(left_tcp_pose, delta_pose)
left_tcp_pose = applyDeltaPose6d(left_tcp_pose, invPose6d(tip_transform))



left_ur_robot.moveToPose(left_tcp_pose)

