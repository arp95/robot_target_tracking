# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from utils import *


# user-defined variables
mean = np.asarray([7, 7])
var = [[1, 0], [0, 1]]
robots_x = [1, 1]
robots_y = [3, 8]
robots_id = [1, 2]
x_list = []
y_list = []
map_height = 15
map_width = 15

# estimate target position after each time step
for t in range(2, 60):
    target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    x_list.append(x_true)
    y_list.append(y_true)
    x, y = np.mgrid[0:int(map_height):0.01, 0:int(map_width):.01]
    pos = np.dstack((x, y))
    gauss = multivariate_normal(mean=mean, cov=var)
    plt.contourf(x, y, gauss.pdf(pos), cmap='magma')
    plt.xlim(0, map_height)
    plt.ylim(0, map_width)
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s1/" + str(t) + ".png")
    plt.cla()
    plt.close()
