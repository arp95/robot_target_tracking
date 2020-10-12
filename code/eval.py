# header files
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from gym.envs.registration import register
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from td3 import *
from replay_buffer import *


logger = logging.getLogger(__name__)
register(
    id='RobotTargetTrackingEnv-v0',
    entry_point='envs:RobotTargetTrackingEnvInterface',
)

# initialise environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "RobotTargetTrackingEnv-v0"
env = gym.make(env_name)
env.env_parametrization()
env.reset()

# create TD3 object and load optimal policy
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(0.0005, state_dim, 2, max_action)
policy.load_actor("/home/arpitdec5/Desktop/robot_target_tracking/", "model_sensors_1_targets_2")
step = []
target_1 = []
target_2 = []
target_3 = []
target_4 = []

# eval loop
state = env.reset()
for iter in range(0, 200):
    action, step_size = policy.select_action(state)
    next_state, reward, done, _, var = env.step(action, step_size)
    state = next_state
    env.render()
    env.close()

    step.append(iter)
    for index in range(0, len(var)):
        if(index==0):
            target_1.append(np.linalg.det(var[0]))
        elif(index==1):
            target_2.append(np.linalg.det(var[1]))
        elif(index==2):
            target_3.append(np.linalg.det(var[2]))
        else:
            target_4.append(np.linalg.det(var[3]))
    if(done):
        break

# plot
plt.cla()
plt.title("Plot for scenario: sensors=1 and targets=2")
plt.xlabel("Step")
plt.ylabel("Variance Determinant")
plt.plot(step, target_1, label='target 1')
plt.plot(step, target_2, label='target 2')
#plt.plot(step, target_3, label='target 3')
#plt.plot(step, target_4, label='target 4')
plt.legend()
plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/det_sensors_1_targets_2.png")
