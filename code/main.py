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

# constants
LR = 0.1
DISCOUNT = 0.95
EPOCHS = 2000

# run qlearning algo
q_table = np.random.uniform(low=-1, high=1, size=(21, 21, 21, 8))
print(q_table[(15, 15, 5)])

for epoch in range(0, EPOCHS):
    discrete_state = env.reset()
    state = (max(int(np.round(float(discrete_state[0]))), 19), max(int(np.round(float(discrete_state[1]))), 19), max(int(np.round(float(discrete_state[2]))), 19))
    done = False
    while not done:
        action = np.argmax(q_table[state])
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
        new_state = (max(int(np.round(float(discrete_state[0]))), 19), max(int(np.round(float(discrete_state[1]))), 19), max(int(np.round(float(discrete_state[2]))), 19))
        
        #if(epoch%100 == 0):
        #    env.render()
        
        if not done:
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state + (action_index, )]        
            new_q = ((1 - LR) * current_q) + (LR * (reward + DISCOUNT * max_future_q))
            q_table[state + (action_index, )] = new_q
        state = new_state
    env.close()

print(q_table[(15, 15, 5)])
