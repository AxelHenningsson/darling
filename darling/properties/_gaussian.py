"""Fit a correlated 2D Gaussian to each detector pixel in a mosaicity scan."""

import numba
import numpy as np


def fit_2D_gaussian(
    data, coordinates, weight_power=1.0, mask=None, *, stat_mean=None, stat_cov=None,
    tolerance=1e-3,
):
    """Fit one correlated Gaussian to both motor axes of each detector pixel.

    The model has a single peak and zero background. The fit uses motor
    samples at or above 20% of the maximum intensity for each pixel.
    For a noisy motor scan, try smoothing each detector pixel's motor data
    before fitting. The domains example uses
    ``darling.filters.gaussian_filter(data.astype(np.float64), sigma=1.5, axis=(0, 1))``,
    where sigma is measured in motor steps along each axis.

    Args:
        data (:obj:`numpy.ndarray`): Nonnegative intensities of shape (ny, nx, m, n).
        coordinates (:obj:`numpy.ndarray`): Two motor grids of shape (m, n).
        weight_power (:obj:`float`): Exponent in the residual weight
            ``(I / Imax)**weight_power``. Defaults to 1.
        tolerance (float): Absolute target for the mean squared weighted residual
            over included samples, with intensities normalized by the pixel peak.
            Defaults to 1e-3. The fit can also stop after 50 evaluations without
            improvement or 5000 Adam steps, returning the lowest-loss iterate.
        mask (:obj:`numpy.ndarray`, optional): Boolean array of shape (ny, nx).
            True selects a pixel for fitting. Defaults to all pixels.
        stat_mean (:obj:`numpy.ndarray`, optional): Initial centers of shape
            (ny, nx, 2). Defaults to the intensity-weighted mean.
        stat_cov (:obj:`numpy.ndarray`, optional): Initial covariances of shape
            (ny, nx, 2, 2). Defaults to the intensity-weighted covariance.

    Returns:
        :obj:`numpy.ndarray`: Float64 array of shape (ny, nx, 7) containing
            [amplitude, mu0, mu1, sigma0, sigma1, rho, valid]. The sigmas are
            marginal widths in motor units and rho is the correlation
            between the two motor coordinates. Valid fits have a final value
            of one. Masked or unusable pixels return seven zeros.
    """
    data = np.asarray(data)
    grid = np.ascontiguousarray(coordinates, dtype=np.float32)
    if data.ndim != 4 or grid.shape != (2, *data.shape[2:]) or not grid.size:
        raise ValueError("expected data (ny, nx, m, n) and two motor grids (2, m, n)")
    if not np.isfinite(grid).all() or not np.isfinite(weight_power) or weight_power < 0:
        raise ValueError(
            "coordinates and weight_power must be finite, with weight_power >= 0"
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if data.dtype not in (np.uint16, np.float32, np.float64):
        if data.dtype.kind not in "iuf":
            raise ValueError("expected real intensity data")
        data = data.astype(np.float64)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != data.shape[:2] or mask.dtype != bool:
            raise ValueError("mask must be boolean with the detector shape")
    return _fit_map(
        data,
        grid,
        np.float32(weight_power),
        float(tolerance),
        mask,
        _seed(stat_mean, (*data.shape[:2], 2)),
        _seed(stat_cov, (*data.shape[:2], 2, 2)),
    )


@numba.njit(cache=True)
def clip_value(x, min_val, max_val):
    """Clip a scalar value between minimum and maximum values.

    Args:
        x (:obj:`float`): Value to be clipped
        min_val (:obj:`float`): Minimum allowed value
        max_val (:obj:`float`): Maximum allowed value

    Returns:
        :obj:`float`: Clipped value between min_val and max_val
    """
    if x < min_val:
        return min_val
    if x > max_val:
        return max_val
    return x


@numba.njit(cache=True)
def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, rho):
    """Calculate the value of a 2D Gaussian function with correlation at point (x,y).

    Args:
        x (:obj:`float`): X-coordinate
        y (:obj:`float`): Y-coordinate
        amplitude (:obj:`float`): Peak height of the Gaussian
        x0 (:obj:`float`): X-coordinate of the center
        y0 (:obj:`float`): Y-coordinate of the center
        sigma_x (:obj:`float`): Standard deviation in x-direction
        sigma_y (:obj:`float`): Standard deviation in y-direction
        rho (:obj:`float`): Correlation coefficient between x and y (-1 to 1)

    Returns:
        :obj:`float`: Value of the 2D Gaussian at point (x,y)
    """
    dx = x - x0
    dy = y - y0

    z = (
        (dx / sigma_x) ** 2
        + (dy / sigma_y) ** 2
        - 2 * rho * dx * dy / (sigma_x * sigma_y)
    )
    denom = 2 * (1 - rho**2)

    return amplitude * np.exp(-z / denom)


@numba.njit(cache=True)
def params_from_moments(data_flat, stat_mean, stat_cov):
    """Initialize Gaussian parameters directly from statistical moments.

    Args:
        data_flat (:obj:`numpy.ndarray`): Flattened intensity data
        stat_mean (:obj:`numpy.ndarray`): Statistical mean vector [x0, y0]
        stat_cov (:obj:`numpy.ndarray`): 2 x 2 covariance matrix of the motor coordinates.

    Returns:
        :obj:`numpy.ndarray`: Initial parameter estimates [amplitude, x0, y0, sigma_x, sigma_y, rho]
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


@numba.njit(cache=True)
def adam_update(
    param, m, v, grad, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """Perform one Adam optimization step.

    Args:
        param (:obj:`float`): Current parameter value
        m (:obj:`float`): First moment estimate
        v (:obj:`float`): Second moment estimate
        grad (:obj:`float`): Gradient of the loss with respect to the parameter
        t (:obj:`int`): Iteration number
        alpha (:obj:`float`): Learning rate
        beta1 (:obj:`float`): Exponential decay rate for first moment
        beta2 (:obj:`float`): Exponential decay rate for second moment
        epsilon (:obj:`float`): Small constant to prevent division by zero

    Returns:
        tuple: Updated parameter value, updated first moment, updated second moment
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad

    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    param -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    return param, m, v


@numba.njit(cache=True)
def gaussian_pred_map(x, y, p):
    out = np.empty_like(x, dtype=np.float32)
    amplitude, x0, y0 = p[0], p[1], p[2]
    sx, sy = max(p[3], 1e-10), max(p[4], 1e-10)
    r = p[5]
    inv = 1.0 / (2.0 * (1.0 - r * r))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dx = x[i, j] - x0
            dy = y[i, j] - y0
            z = (
                (dx / sx) * (dx / sx)
                + (dy / sy) * (dy / sy)
                - 2.0 * r * dx * dy / (sx * sy)
            )
            out[i, j] = amplitude * np.exp(-z * inv)
    return out


@numba.njit(cache=True)
def residual_map(data, x, y, params):
    pred = gaussian_pred_map(x, y, params)
    return pred - data.astype(np.float32)


@numba.njit(cache=True)
def nrmse(data, res):
    denom = max(np.max(data.astype(np.float32)), 1e-12)
    return np.sqrt(np.mean(res * res)) / denom


@numba.njit(cache=True)
def _weighted_mse_and_grad(params, x, y, observed, weights):
    """Mean squared weighted residual and its six analytic derivatives."""
    amplitude, mx, my, sx, sy, rho = params
    denom = 1.0 - rho**2
    grad = np.zeros(6)
    loss = 0.0
    for i in range(observed.size):
        u, v = (x[i] - mx) / sx, (y[i] - my) / sy
        distance = u**2 + (v - rho * u)**2 / denom
        basis = np.exp(-0.5 * distance)
        prediction = amplitude * basis
        residual = weights[i] * (prediction - observed[i])
        loss += residual**2

        # One weight is in the residual, the other in its derivative.
        weighted = residual * weights[i]
        factor = weighted * prediction / denom
        center_x = factor * (u - rho * v) / sx
        center_y = factor * (v - rho * u) / sy
        grad[0] += weighted * basis
        grad[1] += center_x
        grad[2] += center_y
        grad[3] += center_x * u
        grad[4] += center_y * v
        grad[5] += factor * (u * v - rho * distance)
    return loss / observed.size, grad * (2.0 / observed.size)


@numba.njit(cache=True)
def fit_gaussian_2d(
    data_flat,
    x_flat,
    y_flat,
    stat_mean,
    stat_cov,
    weight_power,
    max_iter=5000,
    tolerance=1e-3,
):
    """Fit a 2D Gaussian using Adam optimization with adaptive learning rates.

    Args:
        data_flat (:obj:`numpy.ndarray`): Flattened intensity data
        x_flat (:obj:`numpy.ndarray`): Flattened x-coordinates
        y_flat (:obj:`numpy.ndarray`): Flattened y-coordinates
        stat_mean (:obj:`numpy.ndarray`): Statistical mean vector [x0, y0]
        stat_cov (:obj:`numpy.ndarray`): Statistical covariance matrix
        weight_power (:obj:`float`): Weight power for the weight function
        max_iter (:obj:`int`): Maximum number of iterations
        tolerance (:obj:`float`): Absolute target for the mean squared weighted residual

    Returns:
        :obj:`numpy.ndarray`: Fitted parameters [amplitude, x0, y0, sigma_x, sigma_y, rho]
    """
    params = params_from_moments(data_flat, stat_mean, stat_cov)

    if np.sum(data_flat) < 1e-10:
        return params

    scale = np.max(data_flat)
    if scale < 1e-10:
        return params

    data_norm = data_flat / scale
    params[0] /= scale

    # Initial step sizes for amplitude, x0, y0, sigma_x, sigma_y and rho.
    base_rates = np.array([0.01, 0.006, 0.006, 0.03, 0.03, 0.01])

    m = np.zeros_like(params)
    v = np.zeros_like(params)

    min_error = np.inf
    best_params = params.copy()
    patience = 50
    patience_counter = 0

    error_history = np.zeros(3)
    error_idx = 0

    active = data_norm >= np.max(data_norm) * 0.2
    x, y = x_flat[active], y_flat[active]
    observed = data_norm[active]
    weights = observed**weight_power

    # Score the initial guess and each updated iterate, including the final one.
    for step in range(max_iter + 1):
        total_error, grad = _weighted_mse_and_grad(params, x, y, observed, weights)
        if not np.isfinite(total_error):
            break
        shape_error = (
            abs(params[3] - best_params[3]) / best_params[3]
            + abs(params[4] - best_params[4]) / best_params[4]
        )
        if total_error < min_error:
            min_error = total_error
            best_params = params.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        if total_error <= tolerance or patience_counter >= patience or step == max_iter:
            break

        t = step + 1
        error_history[error_idx] = total_error
        error_idx = (error_idx + 1) % 3

        if t > 3:
            error_trend = (
                error_history[(error_idx - 1) % 3] - error_history[error_idx % 3]
            ) / error_history[error_idx % 3]
        else:
            error_trend = -1.0

        alphas = base_rates.copy()

        if error_trend > 0:
            alphas *= 1.2 if t < 50 else 1.05
        elif error_trend > -0.01:
            alphas *= 0.8 if t < 50 else 0.9

        if shape_error > 0.1:
            # Speed up updates to the shape parameters when their drift is large
            alphas[3:5] *= 1.2

        for i in range(len(params)):
            params[i], m[i], v[i] = adam_update(
                params[i], m[i], v[i], grad[i], t, alpha=alphas[i]
            )

        params[0] = max(params[0], 0)
        params[3] = max(params[3], 0.5 * np.sqrt(stat_cov[0, 0]))
        params[4] = max(params[4], 0.5 * np.sqrt(stat_cov[1, 1]))
        params[5] = clip_value(params[5], -0.99, 0.99)

    best_params[0] *= scale
    return best_params


@numba.njit(cache=True)
def _fit_pixel(data, grid, points, weight_power, tolerance, i, j, mean_map, cov_map):
    x, y = grid[0], grid[1]
    params = np.zeros(7, dtype=np.float64)
    if not np.isfinite(data).all() or np.min(data) < 0:
        return params
    total = np.sum(data)
    if np.max(data) == np.min(data):
        return params

    flat = data.flatten()
    # Keep the float32 moment reductions in this order. Rounding changes
    # the initial parameters and can change when the optimizer stops.
    if mean_map is None:
        first = np.array(
            [np.sum(data * x) / total, np.sum(data * y) / total], dtype=np.float32
        )
    else:
        first = mean_map[i, j]
    if cov_map is None:
        offsets = points.copy()
        offsets[0] -= first[0]
        offsets[1] -= first[1]
        weighted = (offsets * flat).astype(np.float32)
        second = (np.dot(weighted, offsets.T) / total).astype(np.float32)
    else:
        second = cov_map[i, j]
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or second[0, 0] <= 0
        or second[1, 1] <= 0
    ):
        return params
    p = fit_gaussian_2d(
        flat, points[0], points[1], first, second, weight_power, tolerance=tolerance
    )
    if not np.isfinite(p).all() or p[3] <= 0 or p[4] <= 0:
        return params
    error = nrmse(data, residual_map(data, x, y, p))
    if np.isfinite(error):
        params[:6] = p
        params[6] = 1
    return params


@numba.njit(cache=True, parallel=True)
def _fit_map(data, grid, weight_power, tolerance, mask, first, second):
    rows, cols = data.shape[:2]
    params = np.zeros((rows, cols, 7), dtype=np.float64)
    points = grid.reshape(2, -1)
    for index in numba.prange(rows * cols):
        i, j = index // cols, index % cols
        if mask is None or mask[i, j]:
            params[i, j] = _fit_pixel(
                data[i, j], grid, points, weight_power, tolerance, i, j, first, second
            )
    return params


def _seed(value, shape):
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"expected moment map with shape {shape}, got {value.shape}")
    return value
