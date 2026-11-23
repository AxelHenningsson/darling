"""Generalized ufuncs for fitting a Gaussian with precomputed intensity moments."""

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

    Args:
        data (:obj:`numpy.ndarray`): Intensity data, shape (m, n)
        x (:obj:`numpy.ndarray`): X-coordinate grid, shape (m, n)
        y (:obj:`numpy.ndarray`): Y-coordinate grid, shape (m, n)
        dum (:obj:`numpy.ndarray`): Dummy array for numba shapes
        stat_mean (:obj:`numpy.ndarray`): Statistical mean vector [x0, y0]
        stat_cov (:obj:`numpy.ndarray`): Statistical covariance matrix
        weight_power (:obj:`float`): Weight power for the weight function
        res (:obj:`numpy.ndarray`): Output array for results of the mean
        rmse (:obj:`numpy.ndarray`): Output array for results of the RMSE

    Returns:
        tuple: Fitted center [x0, y0] and peak-normalized RMS residual.
    """
    if np.sum(data) < 1e-10:
        res[0] = 0.0
        res[1] = 0.0
        rmse[0] = 0.0
        return

    params = fit_gaussian_2d(
        data.flatten(), x.flatten(), y.flatten(), stat_mean, stat_cov, weight_power
    )
    res[0] = params[1]
    res[1] = params[2]

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

    Args:
        data (:obj:`numpy.ndarray`): Intensity data, shape (m, n)
        x (:obj:`numpy.ndarray`): X-coordinate grid, shape (m, n)
        y (:obj:`numpy.ndarray`): Y-coordinate grid, shape (m, n)
        dum (:obj:`numpy.ndarray`): Dummy array for numba shapes
        stat_mean (:obj:`numpy.ndarray`): Statistical mean vector [x0, y0]
        stat_cov (:obj:`numpy.ndarray`): Statistical covariance matrix
        weight_power (:obj:`float`): Weight power for the weight function
        res_mean (:obj:`numpy.ndarray`): Output array for mean [x0, y0]
        rmse (:obj:`numpy.ndarray`): Output array for RMSE
        res_cov (:obj:`numpy.ndarray`): Output array for covariance matrix

    Returns:
        tuple: Fitted center, peak-normalized RMS residual and covariance.
    """
    if np.sum(data) < 1e-10:
        res_mean[0] = 0.0
        res_mean[1] = 0.0
        rmse[0] = 0.0
        res_cov[:2, :2] = 0.0
        return

    params = fit_gaussian_2d(
        data.flatten(), x.flatten(), y.flatten(), stat_mean, stat_cov, weight_power
    )

    res_mean[0] = params[1]
    res_mean[1] = params[2]

    sigma_x = max(params[3], 1e-10)
    sigma_y = max(params[4], 1e-10)
    rho = params[5]

    res_cov[0, 0] = sigma_x**2
    res_cov[1, 1] = sigma_y**2
    res_cov[0, 1] = res_cov[1, 0] = rho * sigma_x * sigma_y

    rmap = residual_map(data, x, y, params)
    rmse[0] = nrmse(data, rmap)
