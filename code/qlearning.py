# header files
import gym
import numpy as np
import torch
import random

env = gym.make("MountainCar-v0")
env.reset()

# constants
LR = 0.1
DISCOUNT = 0.95
EPOCHS = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

# run algo
for epoch in range(0, EPOCHS):
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if(epoch%100 == 0):
            env.render()
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]        
            new_q = ((1 - LR) * current_q) + (LR * (reward + DISCOUNT * max_future_q))
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state
    env.close()
