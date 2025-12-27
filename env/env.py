import threading
import time

import cv2
import gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, WrenchStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
from xlib.algo.controller import VelocityAdmittanceController
from xlib.algo.utils.transforms import *
from xlib.device.keyboard import KeyboardReader
from xlib.device.manipulator import UR
from xlib.device.robotiq.robotiq_gripper import RobotiqGripper


class UrEnv(gym.Env):
    """UR Robot Environment with Velocity Control."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # ur_robot init
        base_to_world_mtx = np.load(config.base_to_world_file)
        self.base_to_world = matrixToPose6d(base_to_world_mtx)
        self.ur_robot = UR(config.ip, self.base_to_world)
        self.admittance_controller = VelocityAdmittanceController(config.M, config.D, config.threshold_high, config.threshold_low)
        # init variables
        self.ft_value = None
        self.img_obs = None
        self.tcp_obs = None
        self.jnt_obs = None
        self.tip_transform = np.zeros(6)
        self.tip_transform[2] = 0.2
        self.cv_bridge = CvBridge()
        self.keyboard_reader = KeyboardReader()
        self.ur_gripper = RobotiqGripper()
        self.ur_gripper.connect(config.ip, 63352)
        self.ur_gripper.activate()
        print("Press y to grasp")
        while True:
            key = self.keyboard_reader.get_key()
            if key == 'y':
                print('y pressed')
                self.ur_gripper.move_and_wait_for_pos(255, 255, 100)
                time.sleep(0.01)
                break
        
        # Action space: 6D velocity (linear + angular)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * 6),
            high=np.array([1.0] * 6),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            "tcp_obs": gym.spaces.Box(shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32),
            "jnt_obs": gym.spaces.Box(shape=(6,), low=-2 * np.pi, high=2 * np.pi, dtype=np.float32),
            "ft_obs": gym.spaces.Box(shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32),
            "img_obs": gym.spaces.Box(shape=(config.image_size, config.image_size, 3), low=0, high=255, dtype=np.uint8),
        })
        self.running = False
        self._env_steps = 0

        # Velocity control
        self.target_velocity = np.zeros(6)

        # ros init
        if not rospy.core.is_initialized():
            rospy.init_node('ur_env_node', anonymous=True)

        rospy.Subscriber(config.ft_sensor_topic, WrenchStamped, self._ft_callback, queue_size=10)
        rospy.Subscriber(config.camera_topic, Image, self._image_callback, queue_size=10)
        rospy.Subscriber(config.spacemouse_twist_topic, Twist, self._spacemouse_callback, queue_size=10)
        rospy.Subscriber(config.spacemouse_buttons_topic, Int8MultiArray, self._spacemouse_buttons_callback, queue_size=10)
        self.space_mouse_twist = np.zeros(6)

        self.enable_teleop = False
        self.timer = rospy.Timer(rospy.Duration(1.0 / config.ctrl_freq), self._control_loop)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spacemouse_callback(self, twist_msg: Twist):
        """Process SpaceMouse twist input as velocity command."""
        self.space_mouse_twist = np.array([
            twist_msg.linear.x,
            twist_msg.linear.y,
            -twist_msg.linear.z,
            twist_msg.angular.x,
            twist_msg.angular.y,
            -twist_msg.angular.z,
        ])
        # Transform to TCP frame
        # self.space_mouse_twist = velTransform(space_mouse_twist, invPose6d(self.ur_robot.tcp_pose)[3:])

    def _spacemouse_buttons_callback(self, buttons_msg: Int8MultiArray):
        buttons = buttons_msg.data
        if self.enable_teleop:
            if buttons[1] == 1:
                self.enable_teleop = False
        else:
            if buttons[0] == 1:
                self.enable_teleop = True

    def get_space_mouse_state(self):
        """Get SpaceMouse twist and enable state."""
        return self.space_mouse_twist.copy(), self.enable_teleop

    def _spin(self):
        rospy.spin()

    def _image_callback(self, img_msg: Image):
        cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_img = cv2.resize(cv_img, (self.config.image_size, self.config.image_size))
        self.img_obs = cv2.imencode(".png", cv_img)[1].tobytes()

    def _ft_callback(self, ft_msg: WrenchStamped):
        ft_value = np.array([
            ft_msg.wrench.force.x,
            ft_msg.wrench.force.y,
            ft_msg.wrench.force.z,
            ft_msg.wrench.torque.x,
            ft_msg.wrench.torque.y,
            ft_msg.wrench.torque.z,
        ])
        self.ft_value = ft_value
        # tcp and jnt observation
        self.tcp_obs = self.ur_robot.tcp_pose
        self.jnt_obs = self.ur_robot.joint_position

    @property
    def env_steps(self):
        return self._env_steps

    def _check_obs(self):
        return self.ft_value is not None and self.img_obs is not None

    def _control_loop(self, event):
        """Velocity control loop."""
        if not (self.running and self._check_obs()):
            return

        # Execute velocity command
        ft_value_in_tcp = self.ft_value.copy()
        ft_value_in_world = velTransform(ft_value_in_tcp, self.ur_robot.tcp_pose[3:])
        target_velocity = self.target_velocity.copy()
        dt = 1.0 / self.config.ctrl_freq
        new_vel = self.admittance_controller.update(dt, target_velocity, self.ur_robot.tcp_velocity, ft_value_in_world)
        # print(new_vel)
        self.ur_robot.applyVel(new_vel, time=dt)
        
        

    def action(self, velocity):
        """Set target velocity (6D: linear + angular).

        Args:
            velocity: 6D array [vx, vy, vz, wx, wy, wz]
        """
        self.target_velocity = velocity.copy()

    def step(self, velocity):
        """Set target velocity (6D: linear + angular).

        Args:
            velocity: 6D array [vx, vy, vz, wx, wy, wz]
        """
        self.target_velocity = velocity.copy()

    def get_observation(self):
        tcp_obs = self.tcp_obs.copy()
        jnt_obs = self.jnt_obs.copy()
        ft_value = self.ft_value.copy()
        img_obs = self.img_obs
        observation = {
            "tcp_obs": tcp_obs,
            "jnt_obs": jnt_obs,
            "ft_obs": ft_value,
            "img_obs": img_obs,
        }
        return observation

    def reset(self):
        # ur stop running
        self.running = False
        self.ur_robot.speedStop()

        # reset observations
        self._env_steps = 0
        self.img_obs = None
        self.ft_value = None
        self.tcp_obs = None
        self.jnt_obs = None
        self.target_velocity = np.zeros(6)

        print("Environment resetting, moving to init pose...")
        reset_pose = np.array(self.config.reset_pose)
        sample = np.random.uniform(self.config.random_lower[3], self.config.random_upper[3])
        delta_pose = np.zeros(6)
        delta_pose[3:] = R.from_euler('xyz', np.array([sample, 0.0, 0.0])).as_rotvec()
        reset_pose = applyDeltaPose6d(reset_pose, self.tip_transform)
        reset_pose = applyDeltaPose6d(reset_pose, delta_pose)
        reset_pose[:3] += np.random.uniform(self.config.random_lower[:3], self.config.random_upper[:3])
        reset_pose = applyDeltaPose6d(reset_pose, invPose6d(self.tip_transform))
        print("Environment resetting, please wait...")
        first_reset_pose = self.ur_robot.tcp_pose.copy()
        first_reset_pose[0] = reset_pose[0]
        self.ur_robot.moveToPose(first_reset_pose, asynchronous=False)
        self.ur_robot.moveToPose(reset_pose, asynchronous=False)
        self.ur_robot.reset_servo_target()
        rospy.Rate(1).sleep()
        self.wait_for_obs()
        print("Environment reset done.")
        self.running = True
        
    def regrasp(self):
        self.running = False
        self.ur_robot.speedStop()
        cur_tcp = self.ur_robot.tcp_pose
        cur_tcp[:3] = np.array([-0.15927285,  0.60000967,  0.33677307])
        euler = np.array([3.0 / 4.0 * np.pi, 0.0, np.pi / 2])
        cur_tcp[3:] = R.from_euler('xyz', euler).as_rotvec()
        cur_tip = applyDeltaPose6d(cur_tcp, self.tip_transform)
        # open gripper
        self.ur_gripper.move_and_wait_for_pos(0, 255, 100)
        sample = np.random.uniform(-0.2, 0.2)
        delta_pose = np.zeros(6)
        delta_pose[3:] = R.from_euler('xyz', np.array([sample, 0.0, 0.0])).as_rotvec()
        rand_tip = applyDeltaPose6d(cur_tip, delta_pose)
        regrasp_pos_rand = np.random.uniform([-0.005, 0.0, -0.005], [0.0, 0.0, 0.005])
        rand_tip[:3] += regrasp_pos_rand
        self.ur_robot.moveToPose(applyDeltaPose6d(rand_tip, invPose6d(self.tip_transform)))
        # close gripper
        self.ur_gripper.move_and_wait_for_pos(255, 255, 100)
        
    def wait_for_obs(self):
        while not self._check_obs():
            rospy.Rate(5).sleep()
        rospy.Rate(1).sleep()

    def close(self):
        self.running = False
        rospy.Rate(1).sleep()
        self.ur_robot.close()
        self.keyboard_reader.close()
