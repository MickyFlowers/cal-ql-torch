import threading

import gym
import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped
from omegaconf import OmegaConf
from sensor_msgs.msg import CompressedImage
from xlib.algo.controller import AdmittanceController
from xlib.algo.utils.image_utils import compressed_msg_to_bytes
from xlib.algo.utils.random import sample_disturbance
from xlib.algo.utils.transforms import applyDeltaPose6d
from xlib.device.keyboard import KeyboardReader
from xlib.device.manipulator import UR


class UrEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ur_robot init
        self.ur_robot = UR(config.ip)
        start_pose = self.ur_robot.tcp_pose
        # init variables    
        self.ft_value = None
        self.img_obs = None
        self.tcp_obs = None
        self.jnt_obs = None
        
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
        rospy.Subscriber(config.camera_topic, CompressedImage, self._image_callback, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / config.ctrl_freq), self._control_loop)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        
    def _spin(self):
        rospy.spin()
    
    def _image_callback(self, img_msg: CompressedImage):
        self.img_obs = compressed_msg_to_bytes(img_msg)
            
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
        
        ft_value = self.ft_value.copy()
        target_pose = self.target_pose.copy()

        delta_pose = self.admittance_controller.update(1.0 / self.config.ctrl_freq, x_dot=None, f_ext=ft_value
        )
        new_pose = applyDeltaPose6d(target_pose, delta_pose)
        self.ur_robot.servoTcp(new_pose, 1.0 / self.config.ctrl_freq)
            


    def step(self, target_pose):
        self.target_pose = target_pose.copy()
        rospy.Rate(self.config.action_freq).sleep()
        next_observations = self.get_observation()
        reward = 0.0
        done = False
        info = {"info": "progressing"}
        if self.keyboard_reader.is_pressed('s'):
            reward = 1.0
            done = True
            info["info"] == "success"
        elif self.keyboard_reader.is_pressed('q'):
            done = True
            info["info"] = "quit"
        if self._env_steps >= self.config.max_env_steps - 1:
            done = True
            info["info"] = "max_steps_reached"
        self._env_steps += 1
        return next_observations, reward, done, info

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
        delta_pose = sample_disturbance(self.config.random_lower, self.config.random_upper)
        reset_pose = applyDeltaPose6d(reset_pose, delta_pose)
        self.target_pose = reset_pose.copy()
        print("Environment resetting, please wait...")
        first_reset_pose = reset_pose.copy()
        first_reset_pose[:2] = self.ur_robot.tcp_pose[:2]
        self.ur_robot.moveToPose(first_reset_pose, asynchronous=False)
        self.ur_robot.moveToPose(reset_pose, asynchronous=False)
        rospy.Rate(1).sleep()
        self.wait_for_obs()
        print("Environment reset done.")
        observations = self.get_observation()
        self.running = True
        return observations
    
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
