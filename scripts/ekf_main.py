# header files needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from utils import *


# user-defined variables
mean_1 = np.asarray([10, 10])
x_true_1 = 10
y_true_1 = 10
var_1 = [[1, 0], [0, 1]]
init_pos_1 = [8, 10]
mean_2 = np.asarray([12, 5])
x_true_2 = 12
y_true_2 = 5
var_2 = [[1, 0], [0, 1]]
init_pos_2 = [10, 5]
robots_x = [12.5]
robots_y = [2.5]
robots_id = [1]
map_height = 20
map_width = 20
stepsize_map = 0.1
action_radius = 1

# variables
x_g, y_g = np.mgrid[0:map_height:stepsize_map, 0:map_width:stepsize_map]
points = np.dstack((x_g, y_g))
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
net_val = 0
avg_val = 0
for t in range(2, 200):

    # update target position
    target_x_mean_1, target_y_mean_1, var_1, x_true_1, y_true_1 = extended_kalman_filter(mean_1[0], mean_1[1], var_1, robots_x, robots_y, robots_id, t, init_pos_1[0], init_pos_1[1], 0)
    mean_1 = np.asarray([target_x_mean_1, target_y_mean_1])
    target_x_mean_2, target_y_mean_2, var_2, x_true_2, y_true_2 = extended_kalman_filter(mean_2[0], mean_2[1], var_2, robots_x, robots_y, robots_id, t, init_pos_2[0], init_pos_2[1], 1)
    mean_2 = np.asarray([target_x_mean_2, target_y_mean_2])

    # update true target position
    true_target_x_1.append(x_true_1)
    true_target_y_1.append(y_true_1)
    true_target_x_2.append(x_true_2)
    true_target_y_2.append(y_true_2)

    # add robot position for rendering
    robot_movement_x.append(robots_x[0])
    robot_movement_y.append(robots_y[0])

    # update belief map
    #gauss = multivariate_normal(mean, var)
    #prob_ekf = gauss.pdf(points)
    #scale = 2, 2
    #x, y = get_correlated_dataset(500, var, (mean[0], mean[1]), scale)
    
    # plot map
    render_ekf([target_x_mean_1, target_y_mean_1], [target_x_mean_2, target_y_mean_2], var_1, var_2, t, true_target_x_1, true_target_y_1,  true_target_x_2, true_target_y_2, robot_movement_x, robot_movement_y)
    #plot_ellipse(x, y, mean, true_target_x, true_target_y, target_x_mean, target_y_mean, "/home/arpitdec5/Desktop/robot_target_tracking/s2/" + str(t) + ".png", robots_x[0], robots_y[0], robot_movement_x, robot_movement_y)

    # update robot position
    next_robot_x, next_robot_y, val = update_robot_pos_ekf(robots_x[0], robots_y[0], [target_x_mean_1, target_x_mean_2], [target_y_mean_1, target_y_mean_2], [var_1, var_2], [prev_target_x_1, prev_target_x_2], [prev_target_y_1, prev_target_y_2], action_radius, map_height, map_width, t+1, prev_robot_x, prev_robot_y)
    net_val = np.linalg.det(var_1) + np.linalg.det(var_2)
    avg_val += net_val
    prev_robot_x = robots_x[0]
    prev_robot_y = robots_y[0]
    robots_x[0] = next_robot_x
    robots_y[0] = next_robot_y
    prev_target_x_1 = target_x_mean_1
    prev_target_y_1 = target_y_mean_1
    prev_target_x_2 = target_x_mean_2
    prev_target_y_2 = target_y_mean_2

print(net_val)
print(avg_val/200)
print("Done")
