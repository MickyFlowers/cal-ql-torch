import threading

import cv2
import gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, WrenchStamped
from omegaconf import OmegaConf
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
from xlib.algo.controller import AdmittanceController
from xlib.algo.utils.image_utils import compressed_msg_to_bytes
from xlib.algo.utils.random import sample_disturbance
from xlib.algo.utils.transforms import (applyDeltaPose6d, calcPose6dError,
                                        invPose6d, matrixToPose6d,
                                        velTransform)
from xlib.device.keyboard import KeyboardReader
from xlib.device.manipulator import UR


class UrEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ur_robot init
        
        base_to_world_mtx = np.load(config.base_to_world_file)
        self.base_to_world = matrixToPose6d(base_to_world_mtx)
        self.ur_robot = UR(config.ip, self.base_to_world)
        # init variables    
        self.ft_value = None
        self.img_obs = None
        self.tcp_obs = None
        self.jnt_obs = None
        
        self.cv_bridge = CvBridge()
        self.admittance_controller = AdmittanceController(
            config.M, config.D, config.K, config.threshold_high, config.threshold_low
        )
        
        self.keyboard_reader = KeyboardReader()
        self.action_space = gym.spaces.Box(
            low=np.array([-2 * np.pi] * 6),
            high=np.array([2 * np.pi] * 6),
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
        # ros init
        if not rospy.core.is_initialized():
            rospy.init_node('ur_env_node', anonymous=True)
            
        rospy.Subscriber(config.ft_sensor_topic, WrenchStamped, self._ft_callback, queue_size=10)
        rospy.Subscriber(config.camera_topic, Image, self._image_callback, queue_size=10)
        rospy.Subscriber(config.spacemouse_twist_topic, Twist, self._spacemouse_callback, queue_size=10)
        rospy.Subscriber(config.spacemouse_buttons_topic, Int8MultiArray, self._spacemouse_buttons_callback, queue_size=10)
        self.space_mouse_twist = None
        
        self.enable_teleop = False
        self.timer = rospy.Timer(rospy.Duration(1.0 / config.ctrl_freq), self._control_loop)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
    def _spacemouse_callback(self, twist_msg: Twist):
        space_mouse_twist = np.array([
            twist_msg.linear.x,
            twist_msg.linear.y,
            -twist_msg.linear.z,
            twist_msg.angular.x,
            twist_msg.angular.y,
            -twist_msg.angular.z,
        ])
        
        self.space_mouse_twist = velTransform(space_mouse_twist, invPose6d(self.ur_robot.tcp_pose)[3:])
        
    def _spacemouse_buttons_callback(self, buttons_msg: Int8MultiArray):
        buttons = buttons_msg.data
        if self.enable_teleop:
            if buttons[1] == 1:
                self.enable_teleop = False
        else:
            if buttons[0] == 1:
                self.enable_teleop = True
    
    def get_space_mouse_state(self):
        return self.space_mouse_twist, self.enable_teleop
    
    def get_target_pose(self):
        return self.target_pose.copy()
    
    def _spin(self):
        rospy.spin()
    
    def _image_callback(self, img_msg: Image):
        cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        cv_img = cv2.resize(cv_img, (self.config.image_size, self.config.image_size))
        self.img_obs = cv2.imencode(".png", cv_img)[1].tobytes()
        
            
    def _ft_callback(self, ft_msg: WrenchStamped):
        # ft value
        
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
        if (
            self.ft_value is not None
            and self.img_obs is not None
        ):
            return True
        else:
            return False

    def _control_loop(self, event):
        if not (self.running and self._check_obs()):
            return

        ft_value_in_tcp = self.ft_value.copy()
        target_pose = self.target_pose.copy()
        
        tcp_in_target = calcPose6dError(target_pose, self.ur_robot.tcp_pose)
        ft_value_in_target_tcp = velTransform(ft_value_in_tcp, tcp_in_target[3:])
        print(ft_value_in_target_tcp)
        tcp_vel = velTransform(self.ur_robot.tcp_velocity, invPose6d(target_pose)[3:])
        
        new_pose = self.admittance_controller.update(1.0 / self.config.ctrl_freq, tcp_pose=self.ur_robot.tcp_pose, tcp_vel=tcp_vel, target_pose=target_pose ,f_ext=ft_value_in_target_tcp
        )
        self.ur_robot.servoTcp(new_pose, 1.0 / self.config.ctrl_freq)
            

    def action(self, target_pose):
        self.target_pose = target_pose.copy()
        
    def step(self, target_pose):
        self.target_pose = target_pose.copy()

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
        rospy.Rate(1).sleep()
        self.ur_robot.servoStop()
        # self.ur_robot.stop()
        # restet observations
        self._env_steps = 0
        self.img_obs = None
        self.ft_value = None
        self.tcp_obs = None
        self.jnt_obs = None
            
        print("Environment resetting, moving to init pose...")
        reset_pose = np.array(self.config.reset_pose)
        # delta_pose = sample_disturbance(self.config.random_lower, self.config.random_upper)
        # reset_pose = applyDeltaPose6d(reset_pose, delta_pose)
        self.target_pose = reset_pose.copy()
        print("Environment resetting, please wait...")
        first_reset_pose = self.ur_robot.tcp_pose.copy()
        first_reset_pose[0] = reset_pose[0]
        self.ur_robot.moveToPose(first_reset_pose, asynchronous=False)
        self.ur_robot.moveToPose(reset_pose, asynchronous=False)
        self.ur_robot.reset_servo_target()
        rospy.Rate(1).sleep()
        self.wait_for_obs()
        print("waiting for manully start...")
        # print("Press[y] continue")
        # while True:
        #     key = self.keyboard_reader.get_key()
        #     rospy.Rate(1).sleep()
        #     if key == 'y':
        #         break
        print("Environment reset done.")
        self.running = True
    
    def wait_for_obs(self):
        while not self._check_obs():
            rospy.Rate(5).sleep()
        rospy.Rate(1).sleep()

    def close(self):
        self.enable_obs = False
        self.running = False
        rospy.Rate(1).sleep()
        self.ur_robot.close()
        self.keyboard_reader.close()
