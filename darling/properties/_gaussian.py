"""Single correlated 2D Gaussian used for Darling mosaicity scans.

The residual weighting, Adam updates, width bounds, and stopping criterion
follow Adam Cretton's fitting branch (b7ba50a).
"""

import numba
import numpy as np


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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
        data_flat (numpy.ndarray): Flattened intensity data
        stat_mean (numpy.ndarray): Statistical mean vector [x0, y0]
        stat_cov (numpy.ndarray): Statistical covariance matrix [[σ_xx, σ_xy], [σ_xy, σ_yy]

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


@numba.njit(cache=True)
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


@numba.njit(cache=True)
def gaussian_pred_map(x, y, p):
    out = np.empty_like(x, dtype=np.float32)
    ax, x0, y0, sx, sy, r = p[0], p[1], p[2], max(p[3], 1e-10), max(p[4], 1e-10), p[5]
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
            out[i, j] = ax * np.exp(-z * inv)
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

    # Per-parameter base learning rates for Adam.
    # Order corresponds to params = [A, x0, y0, sigma_x, sigma_y, rho].
    # These are starting step sizes; we adapt them each iteration based on
    # error trend and simple heuristics to stabilize/focus the optimization.
    base_rates = np.array([0.01, 0.006, 0.006, 0.03, 0.03, 0.01])

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

            # Begin with the base rates and adapt per-iteration
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


@numba.njit(cache=True)
def _fit_pixel(data, grid, points, weight_power, i, j, mean_map, cov_map):
    x, y = grid[0], grid[1]
    params = np.zeros(7, dtype=np.float64)
    if not np.isfinite(data).all() or np.min(data) < 0:
        return params, np.nan
    total = np.sum(data)
    if total == 0:
        return params, 0.0
    if np.max(data) == np.min(data):
        return params, np.nan

    flat = data.flatten()
    # The reference implementation rounds both moment maps to float32 before
    # Adam sees them. Preserve its reduction order as the stopping rule is
    # sensitive to small variations in the starting point.
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
        return params, np.nan
    p = fit_gaussian_2d(flat, points[0], points[1], first, second, weight_power)
    if not np.isfinite(p).all() or p[3] <= 0 or p[4] <= 0:
        return params, np.nan
    error = nrmse(data, residual_map(data, x, y, p))
    if np.isfinite(error):
        params[:6] = p
        params[6] = 1
    return params, error


@numba.njit(cache=True, parallel=True)
def _fit_map(data, grid, weight_power, mask, first, second):
    rows, cols = data.shape[:2]
    params = np.zeros((rows, cols, 7), dtype=np.float64)
    errors = np.full((rows, cols), np.nan, dtype=np.float32)
    points = grid.reshape(2, -1)
    for index in numba.prange(rows * cols):
        i, j = index // cols, index % cols
        if mask is None or mask[i, j]:
            params[i, j], errors[i, j] = _fit_pixel(
                data[i, j], grid, points, weight_power, i, j, first, second
            )
    return params, errors


def _seed(value, shape):
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"expected moment map with shape {shape}, got {value.shape}")
    return value


def _fit(data, coordinates, weight_power, stat_mean, stat_cov, mask):
    data = np.asarray(data)
    grid = np.ascontiguousarray(coordinates, dtype=np.float32)
    if data.ndim != 4 or grid.shape != (2, *data.shape[2:]) or not grid.size:
        raise ValueError("expected data (ny, nx, m, n) and two motor grids (2, m, n)")
    if not np.isfinite(grid).all() or not np.isfinite(weight_power) or weight_power < 0:
        raise ValueError(
            "coordinates and weight_power must be finite, with weight_power >= 0"
        )
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
        mask,
        _seed(stat_mean, (*data.shape[:2], 2)),
        _seed(stat_cov, (*data.shape[:2], 2, 2)),
    )


def fit_2D_gaussian(
    data, coordinates, weight_power=6.0, mask=None, *, stat_mean=None, stat_cov=None
):
    """Fit one correlated Gaussian to both motor axes of each detector pixel.

    data has shape (ny, nx, m, n), and coordinates contains the two (m, n)
    motor grids. mask selects detector pixels. Optional stat_mean and stat_cov
    supply starting values of shape (ny, nx, 2) and (ny, nx, 2, 2).

    Return (ny, nx, 7) with [amplitude, mu0, mu1, sigma0, sigma1, rho, valid].
    The last field is 1 for a finite fit and 0 for a masked or unusable scan.
    Intensities must be nonnegative. Supply background-corrected data if needed.
    Starting at intensity moments, fit samples at or above 20% of the local
    maximum, multiplying residuals by (I / Imax)**weight_power. There is no
    fitted background. Both sigmas are marginal widths in motor units.
    """
    return _fit(data, coordinates, weight_power, stat_mean, stat_cov, mask)[0]


def fit_gaussian(
    data, coordinates, weight_power=1.0, stat_mean=None, stat_cov=None, *, mask=None
):
    """Original mosaicity API: return float32 center, covariance and peak-normalized RMSE.

    For data of shape (ny, nx, m, n) and two (m, n) motor grids, the outputs
    have shapes (ny, nx, 2), (ny, nx, 2, 2), and (ny, nx). The error uses
    all motor samples and is divided by the maximum observed intensity.
    Pass weight_power=6 for the bright-signal mosaicity maps. Empty pixels return
    zeros. For masked or unusable nonempty scans the error is NaN. The parameter
    map interface in curvefit.fit_2D_gaussian also exposes amplitude and validity.
    """
    params, errors = _fit(data, coordinates, weight_power, stat_mean, stat_cov, mask)
    sx, sy, rho = params[..., 3], params[..., 4], params[..., 5]
    cov = np.zeros((*params.shape[:2], 2, 2), dtype=np.float32)
    cov[..., 0, 0] = sx**2
    cov[..., 1, 1] = sy**2
    cov[..., 0, 1] = cov[..., 1, 0] = rho * sx * sy
    return params[..., 1:3].astype(np.float32), cov, errors
