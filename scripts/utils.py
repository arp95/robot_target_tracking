# header files needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi, cos, sin
from random import random


# plot the heatmap
def render(t, x_mesh, y_mesh, belief_map, x_target, y_target, robot_movement_x, robot_movement_y):
    """
        Inputs:
        t: time step
        x_mesh: the x-coordinates
        y_mesh: the y-coordinates
        belief_map: the map containing the probabilities
        x_target: the target x-coordinate
        y_target: the target y-coordinate
        robot_movement_x: the list of robot paths x-coordinates
        robot_movement_y: the list of robot paths y-coordinates
    """
    plt.cla()
    plt.title("Greedy algorithm using BH(Target moving fast)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(x_mesh, y_mesh, belief_map, cmap=cm.inferno)
    plt.plot(x_target, y_target, 'b--')
    plt.plot(x_target[len(x_target) - 1], y_target[len(y_target) - 1], 'o', c='b', marker='*')
    if(len(robot_movement_x) < 8):
        plt.plot(robot_movement_x, robot_movement_y, 'r--')
    else:
        plt.plot(robot_movement_x[-8:], robot_movement_y[-8:], 'r--')
    plt.scatter(robot_movement_x[len(robot_movement_x) - 1], robot_movement_y[len(robot_movement_y) - 1], color='r', marker='D')
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s1/" + str(t) + ".png")
    #plt.show()


# compute bayesian histogram for 'm' targets and given robot position
def compute_bayesian_histogram(targets_x_true, targets_y_true, robot_x, robot_y, belief_map_height, belief_map_width, stepsize_map, sigma_bayesian_hist=1.0):
    """
        Inputs:
        targets_x_true: the true position of target x-coordinate
        targets_y_true: the true position of target y-coordinate
        robot_x: the position of robot x-coordinate
        robot_y: the position of robot y-coordinate 
        belief_map_height: the environment dimensions, height
        belief_map_width: the environment dimensions, width
        stepsize_map: equal to 0.1
        sigma_bayesian_hist: equal to 1

        Outputs:
        bayesian_hist: the belief map of dimensions (belief_map_height, belief_map_width) containing probabilities      
    """
    noise = sigma_bayesian_hist * np.random.randn(1, 1)
    bayesian_hist = np.zeros((belief_map_height, belief_map_width))
    for index in range(0, len(targets_x_true)):
        estimated = np.sqrt((targets_x_true[index] - robot_x)**2 + (targets_y_true[index] - robot_y)**2) + noise[0][0]
        for index1 in range(0, belief_map_height):
            for index2 in range(0, belief_map_width):
                true = np.sqrt(((index1*stepsize_map) - robot_x)**2 + ((index2*stepsize_map) - robot_y)**2)
                bayesian_hist[index1, index2] += 1.0 / (np.sqrt(2 * np.pi * sigma_bayesian_hist**2)) * np.exp(-0.5 / sigma_bayesian_hist**2 * (np.abs(true - estimated)**2))
    return bayesian_hist


# get target estimate
def get_target_position(t, x_true, y_true):
    """
        Inputs:
        t: time step
        x_true: the true position of target x-coordinate
        y_true: the true position of target y-coordinate

        Outputs:
        (x_true, y_true): the target position at next time step      
    """
    omega = 33
    x_true = 3*np.cos((t-1) / omega) + 9
    y_true = 3*np.sin((t-1) / omega) + 12
    return (x_true, y_true)


# reference: https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
        Inputs:
        x: the x-coordinate datapoints
        y: the y-coordinate datapoints
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# reference: https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


# code for the filter
def extended_kalman_filter(target_xhat_t, target_yhat_t, target_sigma_t, robots_x, robots_y, robots_id, t):
    """
        Inputs:
        target_xhat_t: the estimated target position x-coordinate
        target_yhat_t: the estimated target position y-coordinate
        robots_x: the position of robots x-coordinate
        robots_y: the position of robots y-coordinate
        robots_id: the ids of robots
        t: the time step

        Outputs:
        (target_xhat_tplus1, target_yhat_tplus1, sigma_matrix_tplus1, x_true, y_true): the predicted target position      
    """
    # get z_true using true target motion
    omega = 33
    sigma_z = 1.0
    x_true = 3*np.cos((t-1) / omega) + 9
    y_true = 3*np.sin((t-1) / omega) + 12
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
    #plt.show()


# save gaussian
def save_gaussian(gauss, path):
    x, y = np.mgrid[0:25:100j, 0:25:100j]
    z = np.dstack((x, y))
    plt.contourf(x, y, gauss.pdf(z))
    plt.savefig(path)


# plot confidence ellipse
def plot_ellipse(x, y, mean, x_list, y_list, target_x_mean, target_y_mean, path, robot_x, robot_y, robot_movement_x, robot_movement_y):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
    confidence_ellipse(x, y, ax_nstd, n_std=1, edgecolor='firebrick')
    ax_nstd.scatter(mean[0], mean[1], c='b', s=1)
    ax_nstd.legend()
    plt.title("Greedy algorithm using EKF(Target moving slow)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot(x_list, y_list, 'b--')
    plt.scatter(x_list[len(x_list)-1], y_list[len(y_list)-1], color='b', marker='*')
    plt.scatter([target_x_mean], [target_y_mean], color='b', marker="s")
    if(len(robot_movement_x) < 8):
        plt.plot(robot_movement_x, robot_movement_y, 'r--')
    else:
        plt.plot(robot_movement_x[-8:], robot_movement_y[-8:], 'r--')
    plt.scatter(robot_x, robot_y, color='r', marker='D')    
    plt.plot([robot_x, target_x_mean], [robot_y, target_y_mean], color='r')
    plt.savefig(path)
    plt.cla()
    plt.close()


def points_in_circle_np(radius, x0=0, y0=0):
    """
        Inputs:
        radius: the radius of the circular region around the current robot position
        x0: the x-coordinate of the robot position
        y0: the y-coordinate of the robot position

        Outputs: 
        points: the action set to be used for deciding the robot trajectory      
    """
    thetas = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]
    points = []
    for theta in thetas:
        theta = (theta * pi) / 180.0
        points.append((float(x0 + cos(theta) * radius), float(y0 + sin(theta) * radius)))
    return points


# choose optimal action
def update_robot_pos_ekf(robot_x, robot_y, target_x, target_y, var, prev_target_x, prev_target_y, radius, map_height, map_width, t):
    """
        Inputs:
        robot_x: the position of robot x-coordinate
        robot_y: the position of robot y-coordinate 
        target_x: the position of target x-coordinate
        target_y: the position of target y-coordinate
        var: the uncertainty in target position
        prev_target_x: the previous position of target x-coordinate
        prev_target_y: the previous position of target y-coordinate
        radius: the radius of the circular region around the current robot position
        map_height: the environment dimensions, height
        map_width: the environment dimensions, width
        t: the time step

        Outputs:
        (best_action[0], best_action[1]): updated robot position     
    """
    action_set = points_in_circle_np(radius, robot_x, robot_y)
    best_val = np.log(np.linalg.det(var))
    best_action = (robot_x, robot_y)
    for action in action_set:
        curr_robot_x = action[0]
        curr_robot_y = action[1]
        if(curr_robot_x >= 1 and curr_robot_x < (map_height-1) and curr_robot_y >= 1 and curr_robot_y < (map_width-1)):
            _, _, var, _, _ = extended_kalman_filter(target_x, target_y, var, [curr_robot_x], [curr_robot_y], [1], t)
            val = np.log(np.linalg.det(var))
            if(val < best_val):
                best_val = val
                best_action = (curr_robot_x, curr_robot_y)
    return (best_action[0], best_action[1], -best_val)


# choose optimal action
def update_robot_pos(robot_x, robot_y, target_x, target_y, prev_target_x, prev_target_y, radius, map_height, map_width):
    """
        Inputs:
        robot_x: the position of robot x-coordinate
        robot_y: the position of robot y-coordinate 
        target_x: the position of target x-coordinate
        target_y: the position of target y-coordinate
        prev_target_x: the previous position of target x-coordinate
        prev_target_y: the previous position of target y-coordinate
        radius: the radius of the circular region around the current robot position
        map_height: the environment dimensions, height
        map_width: the environment dimensions, width

        Outputs:
        (best_action[0], best_action[1]): updated robot position
    """
    action_set = points_in_circle_np(radius, robot_x, robot_y)
    alpha_opt = 10000000
    dist_opt = 10000000
    best_action = (0, 0)
    for action in action_set:
        curr_robot_x = action[0]
        curr_robot_y = action[1]
        if(curr_robot_x >= 1 and curr_robot_x < (map_height-1) and curr_robot_y >= 1 and curr_robot_y < (map_width-1)):
            m1 = (prev_target_y - robot_y) / (prev_target_x - robot_x + 1e-8)
            m2 = (target_y - curr_robot_y) / (target_x - curr_robot_x + 1e-8)
            d1 = np.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            d2 = np.sqrt((target_x - curr_robot_x)**2 + (target_y - curr_robot_y)**2)
            if(np.abs((m1 * m2) + 1) < alpha_opt):
                alpha_opt = np.abs((m1 * m2) + 1)
                dist_opt = np.abs((d1 / d2) - 1)
                best_action = action
    return (best_action[0], best_action[1])
