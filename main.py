# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utils import *


# user-defined variables
mean = np.asarray([13, 12])
var = [[1, 0], [0, 1]]
robots_x = [1, 1]
robots_y = [1, 24]
robots_id = [1, 2]

# plot initial target pos
gauss = multivariate_normal(mean, var)
plot_gaussian(gauss)

for t in range(2, 200):
    target_x_mean, target_y_mean, target_sigma, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    var = target_sigma
    gauss = multivariate_normal(mean, target_sigma)
    
    # plot final estimated target pos
    #save_gaussian(gauss, "/home/arpitdec5/Desktop/robot_target_tracking/s/" + str(t) + ".png")
