# header files needed
import numpy as np
#import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from utils import *


# user-defined variables
mean = np.asarray([12, 12])
x_true = 12
y_true = 12
var = [[1, 0], [0, 1]]
robots_x = [1]
robots_y = [3]
robots_id = [1]
map_height = 20
map_width = 20
stepsize_map = 0.05
sigma_bayesian_hist = 1.0

# variables
x_mesh, y_mesh = np.meshgrid(np.arange(0, map_height, stepsize_map), np.arange(0, map_width, stepsize_map))
belief_map = np.ones((int(map_height / stepsize_map), int(map_width / stepsize_map)))
robot_movement_x = []
robot_movement_y = []

# estimate target position after each time step
for t in range(2, 18):
    
    # add robot position for rendering
    robot_movement_x.append(robots_x[0])
    robot_movement_y.append(robots_y[0])

    # update target position
    #target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    #mean = np.asarray([target_x_mean, target_y_mean])
    #gauss = multivariate_normal(mean, var)
    #prob = gauss.pdf(points)
    x_true, y_true = get_target_position(t, x_true, y_true)

    # compute bayesian histogram and update belief map
    bayesian_hist = compute_bayesian_histogram([x_true], [y_true], robots_x[0], robots_y[0], int(belief_map.shape[0]), int(belief_map.shape[1]), stepsize_map, sigma_bayesian_hist)
    belief_map = belief_map * bayesian_hist
    belief_map = belief_map / belief_map.sum()
    
    # plot map
    render(t, x_mesh, y_mesh, belief_map, x_true, y_true, robot_movement_x, robot_movement_y)

    # update robot position
    robots_x[0] += 1
    robots_y[0] += 1
