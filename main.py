# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from utils import *


# user-defined variables
mean = np.asarray([12, 12])
var = [[1, 0], [0, 1]]
robots_x = [1, 1]
robots_y = [3, 8]
robots_id = [1, 2]
x_list = []
y_list = []
map_height = 25
map_width = 25

# estimate target position after each time step
for t in range(2, 80):
    target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    x_list.append(x_true)
    y_list.append(y_true)

    # plot and generate heatmap
    x, y = np.random.multivariate_normal(mean, var, 100000).T
    heatmap, _, _, _ = plt.hist2d(x, y, bins=(map_height, map_width), cmap='Blues', range=[[0, map_height], [0, map_width]], density=True)
    _, _, _, _ = plt.hist2d(x, y, bins=(5*map_height, 5*map_width), cmap='Blues', range=[[0, map_height], [0, map_width]], density=True)
    #plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s1/" + str(t) + ".png")
    #plt.show()
