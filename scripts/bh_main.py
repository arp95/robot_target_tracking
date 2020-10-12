# header files needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from utils import *


# user-defined variables
x_true = 12
y_true = 12
robots_x = [1]
robots_y = [1]
robots_id = [1]
map_height = 20
map_width = 20
stepsize_map = 0.1
sigma_bayesian_hist = 1.0
action_radius = 2

# variables
x_mesh, y_mesh = np.mgrid[0:map_height:stepsize_map, 0:map_width:stepsize_map]
belief_map = np.ones((int(map_height / stepsize_map), int(map_width / stepsize_map)))
robot_movement_x = []
robot_movement_y = []
true_target_x = []
true_target_y = []
true_target_x.append(x_true)
true_target_y.append(y_true)
robot_movement_x.append(robots_x[0])
robot_movement_y.append(robots_y[0])
prev_target_x = x_true
prev_target_y = y_true
prev_robot_x = robots_x[0]
prev_robot_y = robots_y[0]

# plotting for t=1
bayesian_hist = compute_bayesian_histogram([x_true], [y_true], robots_x[0], robots_y[0], int(belief_map.shape[0]), int(belief_map.shape[1]), stepsize_map, sigma_bayesian_hist)
belief_map = belief_map * bayesian_hist
belief_map = belief_map / belief_map.sum()
render(1, x_mesh, y_mesh, belief_map, true_target_x, true_target_y, robot_movement_x, robot_movement_y)


# estimate target position after each time step
for t in range(2, 200):

    # update target position and add target to list
    x_true, y_true = get_target_position(t, x_true, y_true)
    true_target_x.append(x_true)
    true_target_y.append(y_true)

    # compute bayesian histogram and update belief map
    bayesian_hist = compute_bayesian_histogram([x_true], [y_true], robots_x[0], robots_y[0], int(belief_map.shape[0]), int(belief_map.shape[1]), stepsize_map)
    belief_map = belief_map * bayesian_hist
    belief_map = belief_map / belief_map.sum()
    
    # plot map
    render(t, x_mesh, y_mesh, belief_map, true_target_x, true_target_y, robot_movement_x, robot_movement_y)

    # update robot position
    #next_robot_x, next_robot_y = update_robot_pos(robots_x[0], robots_y[0], x_true, y_true, prev_target_x, prev_target_y, action_radius, map_height, map_width, prev_robot_x, prev_robot_y)

    # add robot position for rendering
    prev_target_x = x_true
    prev_target_y = y_true
    prev_robot_x = robots_x[0]
    prev_robot_y = robots_y[0]
    #robots_x[0] = next_robot_x
    #robots_y[0] = next_robot_y
    robot_movement_x.append(robots_x[0])
    robot_movement_y.append(robots_y[0])
