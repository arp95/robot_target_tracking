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
robots_x = [1]
robots_y = [1]
robots_id = [1]
map_height = 20
map_width = 20
stepsize_map = 0.1
sigma_bayesian_hist = 1.0

# variables
x_mesh, y_mesh = np.mgrid[0:map_height:stepsize_map, 0:map_width:stepsize_map]
belief_map = np.ones((int(map_height / stepsize_map), int(map_width / stepsize_map)))
x_g, y_g = np.mgrid[0:map_height:stepsize_map, 0:map_width:stepsize_map]
points = np.dstack((x_g, y_g))
robot_movement_x = []
robot_movement_y = []
true_target_x = []
true_target_y = []
true_target_x.append(x_true)
true_target_y.append(y_true)
robot_movement_x.append(robots_x[0])
robot_movement_y.append(robots_y[0])

# plotting for t=1
#bayesian_hist = compute_bayesian_histogram([x_true], [y_true], robots_x[0], robots_y[0], int(belief_map.shape[0]), int(belief_map.shape[1]), stepsize_map, sigma_bayesian_hist)
gauss = multivariate_normal(mean, var)
prob_ekf = gauss.pdf(points)
belief_map = prob_ekf
#belief_map = belief_map * bayesian_hist
#belief_map = belief_map / belief_map.sum()
#render(1, x_mesh, y_mesh, belief_map, x_true, y_true, x_true, y_true, robot_movement_x, robot_movement_y)


# estimate target position after each time step
flag1 = True
flag2 = False
flag3 = False
for t in range(2, 160):

    # update target position
    target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    gauss = multivariate_normal(mean, var)
    prob_ekf = gauss.pdf(points)
    scale = 2, 2
    x, y = get_correlated_dataset(500, var, (mean[0], mean[1]), scale)
    #x_true, y_true = get_target_position(t, x_true, y_true)

    # update true target position
    true_target_x.append(x_true)
    true_target_y.append(y_true)

    # add robot position for rendering
    robot_movement_x.append(robots_x[0])
    robot_movement_y.append(robots_y[0])

    # compute bayesian histogram and update belief map
    #bayesian_hist = compute_bayesian_histogram([x_true], [y_true], robots_x[0], robots_y[0], int(belief_map.shape[0]), int(belief_map.shape[1]), stepsize_map, sigma_bayesian_hist)
    belief_map = prob_ekf
    #belief_map = belief_map * bayesian_hist
    #belief_map = belief_map / belief_map.sum()
    
    # plot map
    #render(t, x_mesh, y_mesh, belief_map, true_target_x, true_target_y, target_x_mean, target_y_mean, robot_movement_x, robot_movement_y)
    plot_ellipse(x, y, mean, true_target_x, true_target_y, target_x_mean, target_y_mean, "/home/arpitdec5/Desktop/robot_target_tracking/s2/" + str(t) + ".png", robots_x[0], robots_y[0])

    # update robot position
    if(t < 50):
        if(t % 3 == 0 and robots_y[0] <= 19):
            robots_y[0] += 1
    elif(t>=50 and t < 100):
        if(t % 3 == 0 and robots_y[0] >= 1 and robots_x[0] <= 19):
            robots_y[0] -= 1
            robots_x[0] += 1
    elif(t >= 100):
        if(t % 3 == 0):
            robots_y[0] += 1

    #if(t%5 == 0 and robots_x[0] >= 10 and robots_x[0] > 0):
    #    robots_x[0] -= 1
    #if(t%2 == 0 and robots_y[0] < 10):
    #    robots_y[0] += 1

    #if(t%7 == 0 and robots_y[0] >= 10 and robots_y[0] > 0):
    #    robots_y[0] -= 1

    #print(t)
    #print("Robot x pos: " + str(robots_x[0]) + " y pos: " + str(robots_y[0]))
    #print("Target x pos: " + str(x_true) + " y pos: " + str(y_true))
    #print()
    #print()
