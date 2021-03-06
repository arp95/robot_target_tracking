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
lr = 0.0005
epochs = 1000
iters = 100

# create TD3 object
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(lr, state_dim, 4, max_action)
replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=4)
mean_reward, ep_reward = 0, 0

# training
e = []
r = []
m_e = []
m_r = []
g_e = []
g_r = []
mean_reward = 0
for epoch in range(0, epochs):
    state, _, _, _, _ = env.reset()
    for iter in range(0, iters):
        action_1, step_size_1, action_2, step_size_2 = policy.select_action(state) + torch.normal(0, 0.1, size=env.action_space.shape)
        action_1 = action_1.clamp(env.action_space.low.item(), env.action_space.high.item())
        action_2 = action_2.clamp(env.action_space.low.item(), env.action_space.high.item())

        next_state, reward, done, _, _ = env.step([action_1, action_2], [step_size_1, step_size_2])
        replay_buffer.add((state, torch.tensor([action_1, step_size_1, action_2, step_size_2]), reward, next_state, np.float(done)))
        state = next_state

        mean_reward += reward
        ep_reward += reward
        if done:
            break

    # update policy
    policy.update(replay_buffer, iter, 100, 0.99, 0.99, 0.2, 2)

    # save actor-critic models
    if(epoch>100 and epoch%10==0):
        policy.save("/home/arpitdec5/Desktop/robot_target_tracking/", "sample_1_model_sensors_2_targets_2")

    # print reward
    print()
    print("Epoch: " + str(epoch))
    print("Reward: " + str(ep_reward))
    print()
    e.append(epoch)
    r.append(ep_reward)
    ep_reward = 0

    # mean reward
    if(epoch%50==0 and epoch>0):
        m_e.append(epoch)
        m_r.append(mean_reward/50.0)
        mean_reward = 0


# plot epoch vs reward curve
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.ylim(0, 1000)
plt.plot(e, r, c='blue', label='Cumulative Reward')
plt.plot(m_e, m_r, c='orange', label='Mean Reward')
#plt.plot(g_e, g_r, c='red', label='Greedy Algorithm')
plt.legend()
plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/sample_1_reward_sensors_2_targets_2.png")
