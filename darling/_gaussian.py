"""Fast Gaussian fitting module using pre-computed statistical moments and Adam optimization.

This module provides functions for fitting 2D Gaussian distributions to data using
moment-based initialization and Adam optimization.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def clip_value(x, min_val, max_val):
    """Clip a scalar value between minimum and maximum values.

    Args:
        x (float): Value to be clipped
        min_val (float): Minimum allowed value
        max_val (float): Maximum allowed value

    Returns:
        float: Clipped value between min_val and max_val
    """
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x


@numba.jit(nopython=True)
def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, rho):
    """Calculate the value of a 2D Gaussian function with correlation at point (x,y).

    Args:
        x (float): X-coordinate
        y (float): Y-coordinate
        amplitude (float): Peak height of the Gaussian
        x0 (float): X-coordinate of the center
        y0 (float): Y-coordinate of the center
        sigma_x (float): Standard deviation in x-direction
        sigma_y (float): Standard deviation in y-direction
        rho (float): Correlation coefficient between x and y (-1 to 1)

    Returns:
        float: Value of the 2D Gaussian at point (x,y)
    """
    dx = x - x0
    dy = y - y0

    sigma_x = max(sigma_x, 1e-10)
    sigma_y = max(sigma_y, 1e-10)

    z = (
        (dx / sigma_x) ** 2
        + (dy / sigma_y) ** 2
        - 2 * rho * dx * dy / (sigma_x * sigma_y)
    )
    denom = 2 * (1 - rho**2)

    return amplitude * np.exp(-z / denom)


@numba.jit(nopython=True)
def params_from_moments(data_flat, stat_mean, stat_cov):
    """Initialize Gaussian parameters directly from statistical moments.

    Args:
        data_flat (numpy.ndarray): Flattened intensity data
        stat_mean (numpy.ndarray): Statistical mean vector [x0, y0]
        stat_cov (numpy.ndarray): Statistical covariance matrix [[Ïƒ_xx, Ïƒ_xy], [Ïƒ_xy, Ïƒ_yy]]

    Returns:
        numpy.ndarray: Initial parameter estimates [amplitude, x0, y0, sigma_x, sigma_y, rho]
    """
    total_intensity = np.sum(data_flat)
    if total_intensity < 1e-10:
        return np.zeros(6)

    amplitude = np.max(data_flat)
    if amplitude < 1e-10:
        return np.zeros(6)

    x0, y0 = stat_mean[0], stat_mean[1]

    sigma_x = np.sqrt(max(stat_cov[0, 0], 1e-10))
    sigma_y = np.sqrt(max(stat_cov[1, 1], 1e-10))

    denom = sigma_x * sigma_y
    if denom < 1e-10:
        rho = 0.0
    else:
        rho = clip_value(stat_cov[0, 1] / denom, -0.99, 0.99)

    return np.array([amplitude, x0, y0, sigma_x, sigma_y, rho])


@numba.jit(nopython=True)
def adam_update(
    param, m, v, grad, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """Perform one Adam optimization step.

    Args:
        param (float): Current parameter value
        m (float): First moment estimate
        v (float): Second moment estimate
        grad (float): Gradient of the loss with respect to the parameter
        t (int): Iteration number
        alpha (float): Learning rate
        beta1 (float): Exponential decay rate for first moment
        beta2 (float): Exponential decay rate for second moment
        epsilon (float): Small constant to prevent division by zero

    Returns:
        tuple: Updated parameter value, updated first moment, updated second moment
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad

    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # Update
    param -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    return param, m, v

@numba.jit(nopython=True)
def gaussian_pred_map(x, y, p):
    out = np.empty_like(x, dtype=np.float32)
    ax, x0, y0, sx, sy, r = p[0], p[1], p[2], max(p[3],1e-10), max(p[4],1e-10), p[5]
    inv = 1.0 / (2.0 * (1.0 - r*r))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dx = x[i,j]-x0; dy = y[i,j]-y0
            z = (dx/sx)*(dx/sx) + (dy/sy)*(dy/sy) - 2.0*r*dx*dy/(sx*sy)
            out[i,j] = ax * np.exp(-z * inv)
    return out

@numba.jit(nopython=True)
def residual_map(data, x, y, params):
    pred = gaussian_pred_map(x, y, params)
    return pred - data.astype(np.float32)  

@numba.jit(nopython=True)
def nrmse(data, res):
    denom = max(np.max(data.astype(np.float32)), 1e-12)
    return np.sqrt(np.mean(res*res)) / denom

@numba.jit(nopython=True)
def fit_gaussian_2d(
    data_flat, x_flat, y_flat, stat_mean, stat_cov, weight_power, max_iter=5000, tolerance=1e-3
):
    """Fit a 2D Gaussian using Adam optimization with adaptive learning rates.

    This function fits a 2D Gaussian to the provided data points using Adam optimization.
    It uses statistical moments for initialization and implements an adaptive learning
    rate strategy based on the optimization progress.

    Args:
        data_flat (numpy.ndarray): Flattened intensity data
        x_flat (numpy.ndarray): Flattened x-coordinates
        y_flat (numpy.ndarray): Flattened y-coordinates
        stat_mean (numpy.ndarray): Statistical mean vector [x0, y0]
        stat_cov (numpy.ndarray): Statistical covariance matrix
        weight_power (float): Weight power for the weight function
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance for error reduction

    Returns:
        numpy.ndarray: Fitted parameters [amplitude, x0, y0, sigma_x, sigma_y, rho]
    """
    params = params_from_moments(data_flat, stat_mean, stat_cov)

    if np.sum(data_flat) < 1e-10:
        return params

    scale = np.max(data_flat)
    if scale < 1e-10:
        return params

    data_norm = data_flat / scale
    params[0] /= scale

    base_rates = np.array([0.01, 0.006, 0.006, 0.03, 0.03, 0.005])

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 1

    min_error = np.inf
    best_params = params.copy()
    patience = 50
    patience_counter = 0

    error_history = np.zeros(3)
    error_idx = 0

    threshold = np.max(data_norm) * 0.2

    for _ in range(max_iter):
        grad = np.zeros_like(params)
        total_error = 0
        active_points = 0

        for i in range(len(data_flat)):
            if data_norm[i] < threshold:
                continue

            active_points += 1
            pred = gaussian_2d(
                x_flat[i],
                y_flat[i],
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
            )

            weight = data_norm[i] ** weight_power

            residual = (pred - data_norm[i]) * weight
            total_error += residual**2

            if abs(residual) < 1e-10:
                continue

            dx = x_flat[i] - params[1]
            dy = y_flat[i] - params[2]
            sigma_x = max(params[3], 1e-10)
            sigma_y = max(params[4], 1e-10)
            rho = params[5]

            z = (
                (dx / sigma_x) ** 2
                + (dy / sigma_y) ** 2
                - 2 * rho * dx * dy / (sigma_x * sigma_y)
            )
            exp_term = np.exp(-z / (2 * (1 - rho**2)))

            grad[0] += residual * exp_term
            grad[1] += (
                residual
                * params[0]
                * exp_term
                * (dx / (sigma_x**2) - rho * dy / (sigma_x * sigma_y))
                / (1 - rho**2)
            )
            grad[2] += (
                residual
                * params[0]
                * exp_term
                * (dy / (sigma_y**2) - rho * dx / (sigma_x * sigma_y))
                / (1 - rho**2)
            )
            grad[3] += (
                residual * params[0] * exp_term * dx**2 / (sigma_x**3) / (1 - rho**2)
            )
            grad[4] += (
                residual * params[0] * exp_term * dy**2 / (sigma_y**3) / (1 - rho**2)
            )
            grad[5] += (
                residual
                * params[0]
                * exp_term
                * (rho / (1 - rho**2) - dx * dy / (sigma_x * sigma_y))
            )

        if active_points > 0:
            total_error /= active_points

            error_history[error_idx] = total_error
            error_idx = (error_idx + 1) % 3

            if t > 3:
                error_trend = (
                    error_history[(error_idx - 1) % 3] - error_history[error_idx % 3]
                ) / error_history[error_idx % 3]
            else:
                error_trend = -1.0

            alphas = base_rates.copy()

            if t < 50:
                if error_trend > 0:
                    alphas *= 1.2
                elif error_trend > -0.01:
                    alphas *= 0.8
            else:
                if error_trend > 0:
                    alphas *= 1.05
                elif error_trend > -0.01:
                    alphas *= 0.9

            shape_error = (
                abs(params[3] - best_params[3]) / best_params[3]
                + abs(params[4] - best_params[4]) / best_params[4]
            )
            if shape_error > 0.1:
                alphas[3:5] *= 1.2

            for i in range(len(params)):
                params[i], m[i], v[i] = adam_update(
                    params[i], m[i], v[i], grad[i], t, alpha=alphas[i]
                )

            params[0] = max(params[0], 0)
            params[3] = max(params[3], 0.5 * np.sqrt(stat_cov[0, 0]))
            params[4] = max(params[4], 0.5 * np.sqrt(stat_cov[1, 1]))
            params[5] = clip_value(params[5], -0.99, 0.99)

            if total_error < min_error:
                min_error = total_error
                best_params = params.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience or total_error < tolerance:
                params = best_params
                break

            t += 1

    params[0] *= scale
    return params

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
            numba.float32[:],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32,
            numba.float32[:, :],
        )
    ],
    "(n,m),(p),(k,q),(p),(p,p),()->(p,p)",
    nopython=True,
    target="parallel",
)
def covariance_2d_gaussian(data, first_moments, points, dum, stat_cov, weight_power, res):
    """Compute covariance by fitting a 2D Gaussian using statistical initialization.

    This function uses the Gaussian fitting procedure to refine the statistical
    covariance estimates for better precision, especially in noisy data.

    Args:
        data (numpy.ndarray): Intensity data, shape (n, m)
        first_moments (numpy.ndarray): Mean vector [x0, y0]
        points (numpy.ndarray): Coordinate points, shape (k, q)
        dum (numpy.ndarray): Dummy array for numba shapes
        stat_cov (numpy.ndarray): Statistical covariance matrix
        weight_power (float): Weight power for the weight function
        res (numpy.ndarray): Output array for results

    Returns:
        numpy.ndarray: Refined covariance matrix [[Ïƒ_xx, Ïƒ_xy], [Ïƒ_xy, Ïƒ_yy]]
    """
    if np.sum(data) < 1e-10:
        res[0, 0] = 0.0
        res[1, 1] = 0.0
        res[0, 1] = res[1, 0] = 0.0
        return

    params = fit_gaussian_2d(
        data.flatten(), points[0], points[1], first_moments, stat_cov, weight_power
    )

    sigma_x = max(params[3], 1e-10)
    sigma_y = max(params[4], 1e-10)
    rho = params[5]

    res[0, 0] = sigma_x**2
    res[1, 1] = sigma_y**2
    res[0, 1] = res[1, 0] = rho * sigma_x * sigma_y




