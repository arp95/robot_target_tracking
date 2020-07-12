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

for t in range(2, 100):
    target_x_mean, target_y_mean, target_sigma, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    var = target_sigma
    x_list.append(x_true)
    y_list.append(y_true)
    
    # plot
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
    scale = 25, 25
    x, y = get_correlated_dataset(500, var, (mean[0], mean[1]), scale)
    confidence_ellipse(x, y, ax_nstd, n_std=3, edgecolor='firebrick')
    ax_nstd.scatter(mean[0], mean[1], c='b', s=3)
    ax_nstd.legend()
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.scatter(x_list, y_list, color='r')
    plt.scatter([1, 1], [1, 24], color='b')
    plt.scatter([target_x_mean], [target_y_mean], color='b')
    plt.plot([1, target_x_mean], [1, target_y_mean], color='b')
    plt.plot([1, target_x_mean], [24, target_y_mean], color='b')
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s/" + str(t) + ".png")
    plt.cla()
    plt.close()
