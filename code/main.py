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
LR = 0.1
DISCOUNT = 0.95
epsilon = 0.5
EPOCHS = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPOCHS // 2
epsilon_decay = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# train qlearning
q_table = np.random.uniform(low=-1, high=1, size=(11, 11, 20, 8))
epoch_reward = 0
e = []
r = []

def get_state(state):
    new_state = np.array([0, 0, 0])
    if(state[0]%2 == 0):
        new_state[0] = int(state[0] / 2.0)
    else:
        new_state[0] = int((state[0] - 1) / 2.0) + 1

    if(state[1]%2 == 0):
        new_state[1] = int(state[1] / 2.0)
    else:
        new_state[1] = int((state[1] - 1) / 2.0) + 1

    if(state[2]%3 == 0):
        new_state[2] = int(state[2] / 3.0)
    else:
        new_state[2] = int((state[2] - 1) / 3.0) + 1
    return new_state  


for epoch in range(0, EPOCHS):
    discrete_state = env.reset()
    state = get_state(discrete_state.numpy())
    print(env.action_space.high)
    done = False
    while not done:
        if(np.random.random() > epsilon):
            action = np.argmax(q_table[(state[0], state[1], state[2])])
        else:
            action = np.random.randint(0, 8)
        action_index = action
        if(action == 0):
            action = [0]
        elif(action == 1):
            action = [np.pi / 4]
        elif(action == 2):
            action = [np.pi / 2]
        elif(action == 3):
            action = [3*np.pi / 4]
        elif(action == 4):
            action = [np.pi]
        elif(action == 5):
            action = [225*np.pi / 180]
        elif(action == 6):
            action = [3*np.pi / 2]
        else:
            action = [315*np.pi / 180]

        new_state, reward, done, _ = env.step(action)
        epoch_reward += reward
        new_state = get_state(new_state.numpy())
        
        env.render()

        #if(epoch%100 == 0):
        #    env.render()
        
        if not done:
            max_future_q = np.max(q_table[(new_state[0], new_state[1], new_state[2])])
            current_q = q_table[(state[0], state[1], state[2], action_index)]        
            new_q = ((1 - LR) * current_q) + (LR * (reward + DISCOUNT * max_future_q))
            q_table[(state[0], state[1], state[2], action_index)] = new_q
        state = new_state

    if(END_EPSILON_DECAYING >= epoch >= START_EPSILON_DECAYING):
        epsilon -= epsilon_decay
    if(epoch%1000 == 0):
        e.append(epoch)
        r.append(epoch_reward)
        print("Epoch Reward:")
        print(epoch_reward)
        print()
    epoch_reward = 0
    env.close()

# plot reward curve
plt.xlabel("Episodes")
plt.ylabel("Avg. Reward")
plt.plot(e, r)
plt.savefig("/home/arpitdec5/Desktop/qlearning_reward.png")


# test qlearning
#discrete_state = env.reset()
#state = get_state(discrete_state.numpy())
#done = False
#while not done:
#    action = np.argmax(q_table[(state[0], state[1], state[2])])
#    action_index = action
#    if(action == 0):
#        action = [0]
#    elif(action == 1):
#        action = [np.pi / 4]
#    elif(action == 2):
#        action = [np.pi / 2]
#    elif(action == 3):
#        action = [3*np.pi / 4]
#    elif(action == 4):
#        action = [np.pi]
#    elif(action == 5):
#        action = [225*np.pi / 180]
#    elif(action == 6):
#        action = [3*np.pi / 2]
#    else:
#        action = [315*np.pi / 180]
#
#    new_state, reward, done, _ = env.step(action)
#    new_state = get_state(new_state.numpy())
#    state = new_state    
#
#    env.render()
#    env.close()
