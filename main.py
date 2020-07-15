# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from utils import *


# user-defined variables
mean = np.asarray([13, 12])
var = [[1, 0], [0, 1]]
robots_x = [1, 1]
robots_y = [1, 24]
robots_id = [1, 2]
x_list = []
y_list = []
map_height = 25
map_width = 25
scale = map_height, map_width

# estimate target position after each time step
for t in range(2, 100):
    target_x_mean, target_y_mean, var, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    x_list.append(x_true)
    y_list.append(y_true)
    x, y = get_correlated_dataset(500000, var, (mean[0], mean[1]), scale)
    
    # plot heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(map_height, map_width), density=True)
    print(heatmap)
    plt.xlim(0, map_height)
    plt.ylim(0, map_width)
    extent = [0, map_height, 0, map_width]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/1.png")
    plt.show()
    break
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s1/" + str(t) + ".png")
    plt.cla()
    plt.close()
