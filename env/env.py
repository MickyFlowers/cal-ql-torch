import threading
import time

import numpy as np
from xlib.algo.controller import AdmittanceController
from xlib.algo.filter import MovingAverageFilter
from xlib.algo.utils.transforms import velTransform
from xlib.device.keyboard import KeyboardReader
from xlib.device.manipulator import UR
from xlib.device.sensor import Ft300sSensor, RealSenseCamera


class UrEnv(object):
    def __init__(self, config):
        self.config = config
        # ur_robot init
        self.ur_robot = UR(config.ip)
        start_pose = self.ur_robot.tcp_pose

        # observation read thread
        self.ft_sensor = Ft300sSensor(config.port, config.timeout, config.zero_reset)
        self.ft_sensor_filter = MovingAverageFilter(config.window_size)
        self.obs_lock = threading.Lock()
        self.ft_value = None
        self.tcp_obs = None
        self.jnt_obs = None

        # image read thread
        self.camera = RealSenseCamera(
            color_width=config.color_width,
            color_height=config.color_height,
            depth_width=config.depth_width,
            depth_height=config.depth_height,
            frame_rate=config.camera_frame_rate,
            exposure_time=config.camera_exposure_time,
            serial_number=config.camera_serial_number,
        )
        self.img_obs_lock = threading.Lock()
        self.img_obs = None

        # control thread
        self.admittance_controller = AdmittanceController(
            config.M, config.D, config.K, config.threshold_high, config.threshold_los
        )
        self.target_pose = start_pose
        self.target_pose_lock = threading.Lock()

        self.running = False
        self._env_steps = 0
        
        self.keyboard_reader = KeyboardReader()
    
    @property
    def env_steps(self):
        return self._env_steps

    def _observation_read_thread(self):
        while self.running:
            start_time = time.perf_counter()
            ft_value = self.ft_sensor.get_force_torque()
            if ft_value is not None:
                self.ft_sensor_filter.update(ft_value)

            filtered_ft_value = self.ft_sensor_filter.output
            if self.ft_sensor_filter.size == self.config.window_size:
                with self.obs_lock:
                    self.ft_value = filtered_ft_value
                    self.tcp_obs = self.ur_robot.tcp_pose
                    self.jnt_obs = self.ur_robot.joint_position
            # time sleep to avoid busy waiting
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time < 1.0 / self.config.ft_read_freq:
                time.sleep(1.0 / self.config.ft_read_freq - elapsed_time)

    def image_read_thread(self):
        while self.running:
            start_time = time.perf_counter()
            color_image, _ = self.camera.get_frame()
            with self.img_obs_lock:
                self.img_obs = color_image
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time < 1.0 / self.config.camera_frame_rate:
                time.sleep(1.0 / self.config.camera_frame_rate - elapsed_time)

    def _check_obs(self):
        if (
            self.ft_value is not None
            and self.tcp_obs is not None
            and self.jnt_obs is not None
            and self.img_obs is not None
        ):
            return True
        else:
            return False

    def control_thread(self):
        while self.running and self._check_obs():
            start_time = time.perf_counter()
            with self.obs_lock:
                ft_value = self.ft_value.copy()
                tcp_obs = self.tcp_obs.copy()
            with self.target_pose_lock:
                target_pose = self.target_pose.copy()

            ft_value = velTransform(ft_value, tcp_obs[3:])
            new_pose = self.admittance_controller.update(
                tcp_obs, target_pose, 1.0 / self.config.ctrl_freq, x_dot=None, f_ext=ft_value
            )
            self.ur_robot.servoTcp(new_pose, 1.0 / self.config.ctrl_freq)
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time < 1.0 / self.config.ctrl_freq:
                time.sleep(1.0 / self.config.ctrl_freq - elapsed_time)

    def step(self, target_pose):
        
        start_time = time.perf_counter()
        with self.target_pose_lock:
            self.target_pose = target_pose.copy()

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time < 1.0 / self.config.action_freq:
            time.sleep(1.0 / self.config.action_freq - elapsed_time)

        next_observations = self.get_observation()
        
        reward = 0.0
        done = False
        info = {"info": "progressing"}
        if self.keyboard_reader.is_pressed('s'):
            reward = 1.0
            done = True
            info["info"] == "success"
            
        if self._env_steps >= self.config.max_env_steps - 1:
            done = True
            info["info"] = "max_steps_reached"
        self._env_steps += 1
        return next_observations, reward, done, info

    def get_observation(self):
        with self.obs_lock:
            tcp_obs = self.tcp_obs.copy()
            jnt_obs = self.jnt_obs.copy()
            ft_value = self.ft_value.copy()
            img_obs = self.img_obs.copy()
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
        time.sleep(1.0)
        self.ur_robot.stop()
        # restet observations
        self._env_steps = 0
        with self.obs_lock:
            self.img_obs = None
            self.ft_value = None
            self.tcp_obs = None
            self.jnt_obs = None
            
        print("Environment resetting, moving to init pose...")
        # move robot to init pose TODO: sample pose
        pose = None
        with self.target_pose_lock:
            self.target_pose = pose.copy()
        self.ur_robot.moveToPose(pose, asynchronous=False)
        time.sleep(1.0)
        
        # reset start target pose
        
        self.running = True
        
        print("Environment resetting, please wait...")
        while not self._check_obs():
            time.sleep(0.1)
        print("Environment reset done.")
        observations = self.get_observation()
        
        return observations
