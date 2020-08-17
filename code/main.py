# header files
import torch
import gym
import numpy as np
from gym.envs.registration import register
import logging

logger = logging.getLogger(__name__)
register(
    id='RobotTargetTrackingEnv-v0',
    entry_point='envs:RobotTargetTrackingEnvInterface',
)

env_name = "RobotTargetTrackingEnv-v0"
env = gym.make(env_name)
