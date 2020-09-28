# header files needed
import numpy as np
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
robots_x = [18]
robots_y = [2]
robots_id = [1]
map_height = 20
map_width = 20
stepsize_map = 0.1
action_radius = 2

# variables
x_g, y_g = np.mgrid[0:map_height:stepsize_map, 0:map_width:stepsize_map]
points = np.dstack((x_g, y_g))
robot_movement_x = [robots_x[0]]
robot_movement_y = [robots_y[0]]
true_target_x = [x_true]
true_target_y = [y_true]
prev_target_x = x_true
prev_target_y = y_true


# estimate target position after each time step
net_val = 0
for t in range(2, 100):

    # update target position
    target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])

    # update true target position
    true_target_x.append(x_true)
    true_target_y.append(y_true)

    # add robot position for rendering
    robot_movement_x.append(robots_x[0])
    robot_movement_y.append(robots_y[0])

    # update belief map
    #gauss = multivariate_normal(mean, var)
    #prob_ekf = gauss.pdf(points)
    scale = 2, 2
    x, y = get_correlated_dataset(500, var, (mean[0], mean[1]), scale)
    
    # plot map
    plot_ellipse(x, y, mean, true_target_x, true_target_y, target_x_mean, target_y_mean, "/home/arpitdec5/Desktop/robot_target_tracking/s2/" + str(t) + ".png", robots_x[0], robots_y[0], robot_movement_x, robot_movement_y)

    # update robot position
    robots_x[0], robots_y[0], val = update_robot_pos_ekf(robots_x[0], robots_y[0], target_x_mean, target_y_mean, var, prev_target_x, prev_target_y, action_radius, map_height, map_width, t+1)
    net_val += val
    prev_target_x = target_x_mean
    prev_target_y = target_y_mean

print(net_val)
