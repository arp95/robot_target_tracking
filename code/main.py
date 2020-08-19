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

# initialise environment
env_name = "RobotTargetTrackingEnv-v0"
env = gym.make(env_name)
env.env_parametrization()
env.reset()

# play with environment
for index in range(0, 50):
    env.step([0])
env.render()
print("Done!")
