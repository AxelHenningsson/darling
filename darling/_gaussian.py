"""Compatibility entry points for the original fitting-branch Gaussian ufuncs."""

import numba
import numpy as np

from .properties._gaussian import (
    adam_update,
    clip_value,
    fit_gaussian_2d,
    gaussian_2d,
    gaussian_pred_map,
    nrmse,
    params_from_moments,
    residual_map,
)


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32,
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(m,n),(m,n),(m,n),(p),(p),(p,p),()->(p),()",
    nopython=True,
    target="parallel",
)
def moments_2d_gaussian(data, x, y, dum, stat_mean, stat_cov, weight_power, res, rmse):
    """Compute first moments by fitting a 2D Gaussian using statistical initialization.

    This function uses the Gaussian fitting procedure to refine the statistical
    moments estimates for better precision, especially in noisy data.

    Args:
        data (numpy.ndarray): Intensity data, shape (m, n)
        x (numpy.ndarray): X-coordinate grid, shape (m, n)
        y (numpy.ndarray): Y-coordinate grid, shape (m, n)
        dum (numpy.ndarray): Dummy array for numba shapes
        stat_mean (numpy.ndarray): Statistical mean vector [x0, y0]
        stat_cov (numpy.ndarray): Statistical covariance matrix
        weight_power (float): Weight power for the weight function
        res (numpy.ndarray): Output array for results of the mean
        rmse (numpy.ndarray): Output array for results of the RMSE

    Returns:
        numpy.ndarray: Refined mean coordinates [x0, y0]
    """
    if np.sum(data) < 1e-10:
        res[0] = 0.0
        res[1] = 0.0
        rmse[0] = 0.0
        return

    params = fit_gaussian_2d(
        data.flatten(), x.flatten(), y.flatten(), stat_mean, stat_cov, weight_power
    )
    res[0] = params[1]  # x0
    res[1] = params[2]  # y0

    rmap = residual_map(data, x, y, params)
    rmse[0] = nrmse(data, rmap)


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32,
            numba.float32[:],
            numba.float32[:],
            numba.float32[:, :],
        )
    ],
    "(m,n),(m,n),(m,n),(p),(p),(p,p),()->(p),(),(p,p)",
    nopython=True,
    target="parallel",
)
def fit_2d_gaussian(
    data, x, y, dum, stat_mean, stat_cov, weight_power, res_mean, rmse, res_cov
):
    """Fit a 2D Gaussian once and return both mean and covariance.

    This function fits a 2D Gaussian to the data using statistical initialization,
    then extracts both the mean (x0, y0) and covariance matrix from the fitted
    parameters in a single pass.

    Args:
        data (numpy.ndarray): Intensity data, shape (m, n)
        x (numpy.ndarray): X-coordinate grid, shape (m, n)
        y (numpy.ndarray): Y-coordinate grid, shape (m, n)
        dum (numpy.ndarray): Dummy array for numba shapes
        stat_mean (numpy.ndarray): Statistical mean vector [x0, y0]
        stat_cov (numpy.ndarray): Statistical covariance matrix
        weight_power (float): Weight power for the weight function
        res_mean (numpy.ndarray): Output array for mean [x0, y0]
        rmse (numpy.ndarray): Output array for RMSE
        res_cov (numpy.ndarray): Output array for covariance matrix

    Returns:
        None: Results are written to res_mean, rmse, and res_cov
    """
    if np.sum(data) < 1e-10:
        res_mean[0] = 0.0
        res_mean[1] = 0.0
        rmse[0] = 0.0
        res_cov[0, 0] = 0.0
        res_cov[1, 1] = 0.0
        res_cov[0, 1] = res_cov[1, 0] = 0.0
        return

    params = fit_gaussian_2d(
        data.flatten(), x.flatten(), y.flatten(), stat_mean, stat_cov, weight_power
    )

    # Extract mean
    res_mean[0] = params[1]  # x0
    res_mean[1] = params[2]  # y0

    # Extract covariance from sigma_x, sigma_y, and rho
    sigma_x = max(params[3], 1e-10)
    sigma_y = max(params[4], 1e-10)
    rho = params[5]

    res_cov[0, 0] = sigma_x**2
    res_cov[1, 1] = sigma_y**2
    res_cov[0, 1] = res_cov[1, 0] = rho * sigma_x * sigma_y

    # Compute RMSE
    rmap = residual_map(data, x, y, params)
    rmse[0] = nrmse(data, rmap)
