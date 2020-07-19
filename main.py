# header files needed
import numpy as np
#import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from utils import *


# user-defined variables
mean = np.asarray([12, 12])
var = [[1, 0], [0, 1]]
robots_x = [1]
robots_y = [3]
robots_id = [1]
map_height = 20
map_width = 20
stepsize_map = 0.1
sigma_bayesian_hist = 1.0
x_mesh, y_mesh = np.meshgrid(np.arange(0, map_height, stepsize_map), np.arange(0, map_width, stepsize_map))
belief_map = np.ones((int(map_height / stepsize_map), int(map_width / stepsize_map)))
robot_movement_x = []
robot_movement_y = []

# plot the heatmap
def render(t):
    plt.cla()
    plt.contourf(x_mesh, y_mesh, belief_map, cmap=cm.inferno)
    plt.plot(mean[0], mean[1], 'o', c='b')
    plt.plot(robot_movement_x, robot_movement_y, 's', c='r')
    plt.axis('equal')
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s1/" + str(t) + ".png")
    #plt.show()

# compute bayesian histogram for 'm' targets and given robot position
def compute_bayesian_histogram(targets_x_mean, targets_y_mean, robot_x, robot_y, belief_map_height, belief_map_width):
    bayesian_hist = np.zeros((belief_map_height, belief_map_width))
    for index in range(0, len(targets_x_mean)):
        estimated = np.sqrt((targets_x_mean[index] - robot_x)**2 + (targets_y_mean[index] - robot_y)**2)
        for index1 in range(0, belief_map_height):
            for index2 in range(0, belief_map_width):
                true = np.sqrt(((index1*stepsize_map) - robot_x)**2 + ((index2*stepsize_map) - robot_y)**2)
                bayesian_hist[index1, index2] += 1.0 / (np.sqrt(2 * np.pi * sigma_bayesian_hist**2)) * np.exp(-0.5 / sigma_bayesian_hist**2 * (np.abs(true - estimated)**2))
    return bayesian_hist


# estimate target position after each time step
for t in range(2, 20):
    
    # add robot position
    robot_movement_x.append(robots_x[0])
    robot_movement_y.append(robots_y[0])

    # get new target estimate
    target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])

    # compute bayesian histogram and update belief map
    bayesian_hist = compute_bayesian_histogram([target_x_mean], [target_y_mean], robots_x[0], robots_y[0], int(belief_map.shape[0]), int(belief_map.shape[1]))
    belief_map = belief_map * bayesian_hist
    belief_map = belief_map / belief_map.sum()
    
    # plot map
    render(t)

    # update robot position
    robots_x[0] += 1
    robots_y[0] += 1
