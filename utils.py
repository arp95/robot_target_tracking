# header files needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# plot the heatmap
def render(t, x_mesh, y_mesh, belief_map, x_target, y_target, target_x_mean, target_y_mean, robot_movement_x, robot_movement_y):
    plt.cla()
    plt.contourf(x_mesh, y_mesh, belief_map, cmap=cm.inferno)
    plt.plot(x_target, y_target, 'o', c='b')
    plt.plot(target_x_mean, target_y_mean, 'o', c='r')
    plt.plot(robot_movement_x, robot_movement_y, 's', c='r')
    plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s1/" + str(t) + ".png")
    #plt.show()

# compute bayesian histogram for 'm' targets and given robot position
def compute_bayesian_histogram(targets_x_mean, targets_y_mean, robot_x, robot_y, belief_map_height, belief_map_width, stepsize_map, sigma_bayesian_hist):
    bayesian_hist = np.zeros((belief_map_height, belief_map_width))
    for index in range(0, len(targets_x_mean)):
        estimated = np.sqrt((targets_x_mean[index] - robot_x)**2 + (targets_y_mean[index] - robot_y)**2)
        for index1 in range(0, belief_map_height):
            for index2 in range(0, belief_map_width):
                true = np.sqrt(((index1*stepsize_map) - robot_x)**2 + ((index2*stepsize_map) - robot_y)**2)
                bayesian_hist[index1, index2] += 1.0 / (np.sqrt(2 * np.pi * sigma_bayesian_hist**2)) * np.exp(-0.5 / sigma_bayesian_hist**2 * (np.abs(true - estimated)**2))
    return bayesian_hist

# get target estimate
def get_target_position(t, x_true, y_true):
    omega = 100
    x_true = 3*np.cos((t-1) / omega) + 11
    y_true = 3*np.sin((t-1) / omega) + 12
    #x_true = x_true + 0.05
    #y_true = y_true + 0.05
    return (x_true, y_true)

# reference: https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
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
    
    # get z_true using true target motion
    omega = 50
    sigma_z = 0.2
    x_true = 2*np.cos((t-1) / omega) + 10
    y_true = 2*np.sin((t-1) / omega) + 12
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
def plot_ellipse(x, y, mean, x_list, y_list, target_x_mean, target_y_mean, path, robot_x, robot_y):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
    confidence_ellipse(x, y, ax_nstd, n_std=1, edgecolor='firebrick')
    ax_nstd.scatter(mean[0], mean[1], c='b', s=1)
    ax_nstd.legend()
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.scatter(x_list, y_list, color='r')
    plt.scatter([target_x_mean], [target_y_mean], color='b')
    plt.plot([robot_x, target_x_mean], [robot_y, target_y_mean], color='b')
    #plt.plot([1, target_x_mean], [14, target_y_mean], color='b')
    #plt.plot(robot_x, robot_y, color='b')
    #plt.show()
    plt.savefig(path)
    plt.cla()
    plt.close()
