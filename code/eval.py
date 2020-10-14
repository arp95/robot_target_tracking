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
from utils import *


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

# eval loop
greedy_cov = []
rl_cov = []
step = []
ratio = []
for index in range(0, 500):
    # init environment
    state, sensors, targets, radii, omegas = env.reset()
    step.append(index+1)

    ############################ Greedy Algo #######################################
    # init ekf variables
    mean_1 = np.asarray([float(targets[0, 0]), float(targets[0, 1])])
    x_true_1, y_true_1 = float(targets[0, 0]), float(targets[0, 1])
    var_1 = [[1, 0], [0, 1]]
    init_pos_1 = [float(targets[0, 0]) - float(radii[0]), float(targets[0, 1])]
    mean_2 = np.asarray([float(targets[1, 0]), float(targets[1, 1])])
    x_true_2, y_true_2 = float(targets[1, 0]), float(targets[1, 1])
    var_2 = [[1, 0], [0, 1]]
    init_pos_2 = [float(targets[1, 0]) - float(radii[1]), float(targets[1, 1])]
    robots_x = []
    robots_y = []
    robots_id = []
    for index in range(0, len(sensors)):
        robots_x.append(float(sensors[index, 0]))
        robots_y.append(float(sensors[index, 1]))
        robots_id.append(index+1)     
    map_height = 20
    map_width = 20
    action_radius = 1.0
    robot_movement_x = [robots_x[0]]
    robot_movement_y = [robots_y[0]]
    true_target_x_1 = [x_true_1]
    true_target_y_1 = [y_true_1]
    prev_target_x_1 = x_true_1
    prev_target_y_1 = y_true_1
    true_target_x_2 = [x_true_2]
    true_target_y_2 = [y_true_2]
    prev_target_x_2 = x_true_2
    prev_target_y_2 = y_true_2
    prev_robot_x = robots_x[0]
    prev_robot_y = robots_y[0]

    # estimate target position after each time step
    avg_val_greedy = 0.0
    i=0
    for t in range(2, 200):
        i += 1

        # update target position
        target_x_mean_1, target_y_mean_1, var_1, x_true_1, y_true_1 = extended_kalman_filter(mean_1[0], mean_1[1], var_1, robots_x, robots_y, robots_id, t, init_pos_1[0], init_pos_1[1], 1, float(omegas[0]), float(radii[0]))
        mean_1 = np.asarray([target_x_mean_1, target_y_mean_1])
        target_x_mean_2, target_y_mean_2, var_2, x_true_2, y_true_2 = extended_kalman_filter(mean_2[0], mean_2[1], var_2, robots_x, robots_y, robots_id, t, init_pos_2[0], init_pos_2[1], 1, float(omegas[1]), float(radii[1]))
        mean_2 = np.asarray([target_x_mean_2, target_y_mean_2])

        # update true target position
        true_target_x_1.append(x_true_1)
        true_target_y_1.append(y_true_1)
        true_target_x_2.append(x_true_2)
        true_target_y_2.append(y_true_2)

        # add robot position for rendering
        robot_movement_x.append(robots_x[0])
        robot_movement_y.append(robots_y[0])
    
        # plot map
        #render_ekf([target_x_mean_1, target_y_mean_1], [target_x_mean_2, target_y_mean_2], var_1, var_2, t, true_target_x_1, true_target_y_1,  true_target_x_2, true_target_y_2, robot_movement_x, robot_movement_y)

        # update robot position
        next_robot_x, next_robot_y, val = update_robot_pos_ekf(robots_x[0], robots_y[0], [target_x_mean_1, target_x_mean_2], [target_y_mean_1, target_y_mean_2], [var_1, var_2], [prev_target_x_1, prev_target_x_2], [prev_target_y_1, prev_target_y_2], action_radius, map_height, map_width, t+1, prev_robot_x, prev_robot_y)
        avg_val_greedy += np.linalg.det(var_1) + np.linalg.det(var_2)
        prev_robot_x = robots_x[0]
        prev_robot_y = robots_y[0]
        robots_x[0] = next_robot_x
        robots_y[0] = next_robot_y
        prev_target_x_1 = target_x_mean_1
        prev_target_y_1 = target_y_mean_1
        prev_target_x_2 = target_x_mean_2
        prev_target_y_2 = target_y_mean_2
    avg_val_greedy = avg_val_greedy/i
    greedy_cov.append(avg_val_greedy)
    #################################################################################


    ############################ RL Algo #######################################
    average_cov_rl = 0.0
    i = 0
    target_1 = []
    target_2 = []
    target_3 = []
    target_4 = []
    for iter in range(0, 200):
        i += 1
        action, step_size = policy.select_action(state)
        next_state, reward, done, _, var = env.step(action, step_size)
        state = next_state
        env.render()
        env.close()

        average_cov_rl += np.linalg.det(var[0]) + np.linalg.det(var[1])
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
    average_cov_rl = average_cov_rl/i
    rl_cov.append(average_cov_rl)
    ratio.append(average_cov_rl/avg_val_greedy)


# plot curve
plt.cla()
plt.title("Plot for scenario: sensors=1 and targets=2")
plt.xlabel("Episodes")
plt.ylabel("Ratio of Avg. Determinant of covariance matrix(RL/greedy)")
plt.plot(step, ratio)
plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/ratio_episode_curve_sensors_1_targets_2.png")
plt.cla()
plt.title("Plot for scenario: sensors=1 and targets=2")
plt.xlabel("Episodes")
plt.ylabel("Avg. Determinant of covariance matrix")
plt.plot(step, greedy_cov, label='Greedy Algo')
plt.plot(step, rl_cov, label='RL Algo')
plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/det_episode_curve_sensors_1_targets_2.png")
