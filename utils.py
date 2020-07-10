# header files needed
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# code for the filter
def extended_kalman_filter(target_xhat_t, target_yhat_t, target_sigma_t, robots_x, robots_y, robots_id, t):
    
    # get z_true using true target motion
    omega = 10
    sigma_z = 0.2
    x_true = np.cos((t-1) / omega) + 12
    y_true = np.sin((t-1) / omega) + 12
    noise = sigma_z * np.random.randn(1000, 1)
        
    z_true = np.zeros((len(robots_x), 1))
    for index in range(0, len(robots_x)):
        z_true[index][0] = np.linalg.norm([[robots_x[index] - x_true], [robots_y[index] - y_true]], 2) + noise[robots_id[index]]

    # filter code
    q_matrix = 0.2 * np.eye(2)
    x_matrix = np.array([[target_xhat_t], [target_yhat_t]])
    sigma_matrix = target_sigma_t + q_matrix
    
    z_pred = np.zeros((len(robots_x), 1))
    h_matrix = np.zeros((len(robots_x), 2))
    for index in range(0, len(robots_x)):
        z_pred[index][0] = np.linalg.norm([[x_matrix[0][0] - robots_x[index]], [x_matrix[1][0] - robots_y[index]]], 2)
        h_matrix[index][0] = (-1.0 / z_pred[index]) * (robots_x[index] - x_matrix[0][0])
        h_matrix[index][1] = (-1.0 / z_pred[index]) * (robots_y[index] - x_matrix[1][0])
        
    res = (z_true - z_pred)
    r_matrix = sigma_z * sigma_z * np.eye(len(robots_x))
    s_matrix = np.matmul(np.matmul(h_matrix, sigma_matrix), h_matrix.T) + r_matrix
    k_matrix = np.matmul(np.matmul(sigma_matrix, h_matrix.T), np.linalg.inv(s_matrix))    

    x_matrix_tplus1 = x_matrix + (np.matmul(k_matrix, res))
    sigma_matrix_tplus1 = np.matmul(np.matmul((np.eye(2) - np.matmul(k_matrix, h_matrix)), sigma_matrix), (np.eye(2) - np.matmul(k_matrix, h_matrix)).T) + np.matmul(np.matmul(k_matrix, r_matrix), k_matrix.T)
    target_xhat_tplus1 = x_matrix_tplus1[0][0]
    target_yhat_tplus1 = x_matrix_tplus1[1][0]
    return (target_xhat_tplus1, target_yhat_tplus1, sigma_matrix_tplus1, x_true, y_true)

# plot gaussian
def plot_gaussian(gauss):
    x, y = np.mgrid[0:25:100j, 0:25:100j]
    z = np.dstack((x, y))
    plt.contourf(x, y, gauss.pdf(z))
    plt.show()

# save gaussian
def save_gaussian(gauss, path):
    x, y = np.mgrid[0:25:100j, 0:25:100j]
    z = np.dstack((x, y))
    plt.contourf(x, y, gauss.pdf(z))
    plt.savefig(path)
