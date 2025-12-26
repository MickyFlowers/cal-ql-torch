import numpy as np
from xlib.device.manipulator import UR
from xlib.algo.utils.transforms import applyDeltaPose6d, matrixToPose6d
ip="172.16.11.233"
base_to_world = np.load("assets/left_base_to_world.npy")
ur_robot = UR(ip, matrixToPose6d(base_to_world))
print(ur_robot.world_pose)