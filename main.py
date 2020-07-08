# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utils import *


# get the new target position
target_x_mean, target_y_mean, target_sigma, _, _ = extended_kalman_filter(4, 20, 0.5, [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], 2)
mean = np.asarray([target_x_mean, target_y_mean])
gauss = multivariate_normal(mean, target_sigma)

# plot gaussian
x, y = np.mgrid[0:25:100j, 0:25:100j]
z = np.dstack((x, y))
plt.contourf(x, y, gauss.pdf(z))
plt.show()



