# header files needed
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


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
    omega = 10
    sigma_z = 0.2
    x_true = 5*np.cos((t-1) / omega) + 11
    y_true = 5*np.sin((t-1) / omega) + 12
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
def plot_ellipse(x, y, mean, x_list, y_list, target_x_mean, target_y_mean, path):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
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
    plt.savefig(path)
    plt.cla()
    plt.close()
