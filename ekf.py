# header files
import numpy as np
import math


# predict the target location
def predict_target(x_hat_t, y_hat_t, sigma_hat_t, robots_x, robots_y, robots_id, t):
    
    # for simulation purposes, find true target 
    sigma_z = 0.2
    omega = 100 
    x_true = cos((t-1) / omega) + 3
    y_true = sin((t-1) / omega) + 20
    noise = sigma_z * np.random.randn(1000, 1)

    z_true = np.zeros(len(robots_x))
    for i in range(0, len(robots_x)):
        z_true[i] = np.linalg.norm([[robots_x[i] - x_true], [robots_y[i] - y_true]]) + noise[robots_id[i]]
    

    # filter part
