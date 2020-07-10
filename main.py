# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utils import *


# user-defined variables
mean = np.asarray([4, 20])
var = [[1, 0], [0, 1]]
robots_x = [1, 5]
robots_y = [1, 5]
robots_id = [1, 2]

# plot initial target pos
gauss = multivariate_normal(mean, var)
x, y = np.mgrid[0:25:100j, 0:25:100j]
z = np.dstack((x, y))
plt.contourf(x, y, gauss.pdf(z))
plt.show()

for t in range(2, 500):
    target_x_mean, target_y_mean, target_sigma, x_true, y_true = extended_kalman_filter(mean[0], mean[1], var, robots_x, robots_y, robots_id, t)
    mean = np.asarray([target_x_mean, target_y_mean])
    var = target_sigma
    gauss = multivariate_normal(mean, target_sigma)
    

# plot final estimated target pos
x, y = np.mgrid[0:25:100j, 0:25:100j]
z = np.dstack((x, y))
plt.contourf(x, y, gauss.pdf(z))
plt.show()
