# header files needed
from gym import utils
from envs.robot_target_tracking_env_sensors import RobotTargetTrackingEnv
import numpy as np
import torch


class RobotTargetTrackingEnvInterface(RobotTargetTrackingEnv, utils.EzPickle):
    def __init__(self):
        RobotTargetTrackingEnv.__init__(self)
        utils.EzPickle.__init__(self)
