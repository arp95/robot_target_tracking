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

# constants
lr = 0.0001
epochs = 1000
iters = 50

# create TD3 object
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(lr, state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
mean_reward, ep_reward = 0, 0

# training
for epoch in range(0, epochs):
    state = env.reset()
    for iter in range(0, iters):
        action = policy.select_action(state) + torch.normal(0, 0.1, size=env.action_space.shape)
        action = action.clamp(env.action_space.low.item(), env.action_space.high.item())

        next_state, reward, done, reward_info = env.step(action)
        replay_buffer.add((state, action, reward, next_state, np.float(done)))
        state = next_state

        mean_reward += reward
        ep_reward += reward
        if done:
            break

    # update policy
    policy.update(replay_buffer, iter, 100, 0.99, 0.99, 0.2, 2)
    print()
    print("Reward: " + str(ep_reward))
    print()
    ep_reward = 0
