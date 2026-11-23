"""Fit a correlated 2D Gaussian to each detector pixel in a mosaicity scan."""

import numba
import numpy as np

from ..filters._filters import _gaussian_kernel_1d


def fit_2D_gaussian(
    data, coordinates, weight_power=1.0, mask=None, *, stat_mean=None, stat_cov=None,
    tolerance=1e-3, return_background=False,
):
    """Fit one correlated Gaussian to both motor axes of each detector pixel.

    The model is a correlated Gaussian plus one constant offset per detector pixel.
    A preliminary plane supplies the detrended signal for peak initialization
    and residual weights. Adam fits the peak and offset to the measured intensities.
    Optional motor-axis smoothing is applied before fitting. For the domains
    scan, use a Gaussian sigma of 0.5 motor steps along each axis.

    Args:
        data (:obj:`numpy.ndarray`): Nonnegative intensities of shape (ny, nx, m, n).
        coordinates (:obj:`numpy.ndarray`): Two motor grids of shape (m, n).
        weight_power (:obj:`float`): Exponent in the residual weight
            ``max(signal / peak, 0.2)**weight_power``, using intensities above
            the initial plane. The floor gives background samples weight.
            Defaults to 1. Zero gives uniform weights.
        tolerance (float): Absolute target for the mean squared weighted residual
            over the scan, with intensities normalized by the initial peak height
            above background and squared weights normalized to sum to one.
            Defaults to 1e-3. The fit can also stop after 50 evaluations without
            improvement or 5000 Adam steps, returning the lowest-loss iterate.
        mask (:obj:`numpy.ndarray`, optional): Boolean array of shape (ny, nx).
            True selects a pixel for fitting. Defaults to all pixels.
        stat_mean (:obj:`numpy.ndarray`, optional): Initial centers of shape
            (ny, nx, 2). Defaults to the background-subtracted intensity mean.
        stat_cov (:obj:`numpy.ndarray`, optional): Initial covariances of shape
            (ny, nx, 2, 2). Defaults to the background-subtracted intensity covariance.

        return_background (bool): Also return the fitted background coefficients.

    Returns:
        :obj:`numpy.ndarray`: Float64 array of shape (ny, nx, 7) containing
            [amplitude, mu0, mu1, sigma0, sigma1, rho, valid]. The sigmas are
            marginal widths in motor units and rho is the correlation
            between the two motor coordinates. Valid fits have a final value
            of one. Masked or unusable pixels return seven zeros. With
            ``return_background=True``, returns ``(fit, background)``, where
            ``background[..., :] = [0, 0, offset]``.
    """
    data = np.asarray(data)
    grid = np.ascontiguousarray(coordinates, dtype=np.float64)
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
    points = grid.reshape(2, -1)
    origin = points.mean(axis=1)
    span = np.ptp(points, axis=1)
    if np.any(span == 0):
        raise ValueError("motor coordinates must span a two-dimensional scan")
    points = (points - origin[:, None]) / span[:, None]
    design = np.column_stack((points.T, np.ones(points.shape[1])))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("motor coordinates must span a two-dimensional scan")
    mean = _seed(stat_mean, (*data.shape[:2], 2))
    cov = _seed(stat_cov, (*data.shape[:2], 2, 2))
    if mean is not None:
        mean = (mean - origin) / span
    if cov is not None:
        cov = cov / np.outer(span, span)
    fit, background = _fit_map(
        data, points, np.ascontiguousarray(np.linalg.pinv(design)), origin, span,
        float(weight_power), float(tolerance), mask, mean, cov, return_background,
    )
    return (fit, background) if return_background else fit



def combined_fwhm(fit, coordinates=None, *, smoothing_sigma=0.0, truncate=4.0, radius=None):
    """Combine the two marginal Gaussian FWHMs in motor units.

    By default, return the observed combined width, including any broadening
    from smoothing. To correct for a kernel applied to both motor axes, pass
    its scalar sigma (motor steps), truncate and optional radius as used
    in ``filters.gaussian_filter``. The discrete kernel covariance is mapped
    through the two motor step vectors and subtracted from the fitted covariance.
    This bulk broadening correction assumes a Gaussian peak away from scan edges
    on an approximately affine grid, with step variations below 0.02 scan steps.
    It does not correct boundary truncation or bias from fitting a non-Gaussian peak.

    Returns a detector map with NaN for invalid fits and for intrinsic
    covariances that are unresolved in either principal direction.
    """
    fit = np.asarray(fit)
    if fit.ndim < 1 or fit.shape[-1] != 7:
        raise ValueError("expected the seven parameters from fit_2D_gaussian")
    if not np.isscalar(smoothing_sigma) or not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be a finite nonnegative scalar")
    kernel_cov = np.zeros((2, 2))
    if smoothing_sigma > 0:
        grid = np.asarray(coordinates, dtype=np.float64)
        if grid.ndim != 3 or grid.shape[0] != 2 or min(grid.shape[1:]) < 2 or not np.isfinite(grid).all():
            raise ValueError("smoothing correction requires two finite 2D motor grids")
        if not np.isfinite(truncate) or truncate <= 0:
            raise ValueError("truncate must be finite and positive")
        if radius is None:
            radius = int(round(truncate * smoothing_sigma))
        if not isinstance(radius, (int, np.integer)) or radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        increments = [np.diff(grid, axis=axis).reshape(2, -1) for axis in (1, 2)]
        steps = np.column_stack([d.mean(axis=1) for d in increments])
        if np.linalg.matrix_rank(steps) < 2:
            raise ValueError("smoothing correction requires independent motor axes")
        inverse = np.linalg.inv(steps)
        for axis, delta in enumerate(increments):
            if not np.allclose(inverse @ delta, np.eye(2)[:, axis, None], atol=0.02, rtol=0):
                raise ValueError("smoothing correction requires an approximately affine motor grid")
        kernel = _gaussian_kernel_1d(smoothing_sigma, radius)
        variance = np.dot(kernel, np.arange(-radius, radius + 1, dtype=float)**2)
        kernel_cov = variance * (steps @ steps.T)

    sx, sy, rho = fit[..., 3], fit[..., 4], fit[..., 5]
    xx = sx**2 - kernel_cov[0, 0]
    yy = sy**2 - kernel_cov[1, 1]
    xy = rho * sx * sy - kernel_cov[0, 1]
    resolved = (fit[..., 6] == 1) & (xx > 0) & (yy > 0) & (xx*yy > xy**2)
    return np.sqrt(8 * np.log(2) * np.where(resolved, xx + yy, np.nan))


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
    if total_intensity <= 0:
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
def gaussian_pred_map(x, y, p, background=0.0):
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
            out[i, j] = amplitude * np.exp(-z * inv) + background
    return out


@numba.njit(cache=True)
def residual_map(data, x, y, params, background=0.0):
    pred = gaussian_pred_map(x, y, params, background)
    return pred - data.astype(np.float32)


@numba.njit(cache=True)
def nrmse(data, res):
    denom = max(np.max(data.astype(np.float32)), 1e-12)
    return np.sqrt(np.mean(res * res)) / denom


@numba.njit(cache=True)
def _weighted_mse_and_grad(params, x, y, observed, weights):
    """Loss and gradient for a Gaussian plus a constant with unit-norm residual weights."""
    amplitude, mx, my, sx, sy, rho, offset = params
    denom = 1.0 - rho**2
    grad = np.zeros(7)
    loss = 0.0
    for i in range(observed.size):
        u, v = (x[i] - mx) / sx, (y[i] - my) / sy
        distance = u**2 + (v - rho * u)**2 / denom
        basis = np.exp(-0.5 * distance)
        peak = amplitude * basis
        prediction = peak + offset
        residual = weights[i] * (prediction - observed[i])
        loss += residual**2

        # One weight is in the residual, the other in its derivative.
        weighted = residual * weights[i]
        factor = weighted * peak / denom
        center_x = factor * (u - rho * v) / sx
        center_y = factor * (v - rho * u) / sy
        grad[0] += weighted * basis
        grad[1] += center_x
        grad[2] += center_y
        grad[3] += center_x * u
        grad[4] += center_y * v
        grad[5] += factor * (u * v - rho * distance)
        grad[6] += weighted
    return loss, grad * 2.0


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
    weight_signal=None,
):
    """Fit a Gaussian plus a constant to background-centered intensities."""
    if weight_signal is None:
        weight_signal = data_flat
    scale = np.max(weight_signal)
    if scale <= 0:
        return np.zeros(7)
    observed = data_flat / scale
    signal = weight_signal / scale
    params = np.zeros(7)
    params[:6] = params_from_moments(np.maximum(signal, 0.0), stat_mean, stat_cov)
    # At fixed center and shape, height and offset are linear coefficients.
    basis = gaussian_2d(
        x_flat, y_flat, 1.0, params[1], params[2], params[3], params[4], params[5]
    )
    design = np.column_stack((basis, np.ones(observed.size)))
    weights = np.maximum(signal, 0.2) ** weight_power
    weights /= np.linalg.norm(weights)
    linear, _, rank, _ = np.linalg.lstsq(
        design * weights.reshape(-1, 1), observed * weights, rcond=-1.0
    )
    if rank == 2 and linear[0] > 0:
        params[0] = linear[0]
        params[6] = linear[1]
    base_rates = np.array([0.01, 0.006, 0.006, 0.03, 0.03, 0.01, 0.01])

    m = np.zeros_like(params)
    v = np.zeros_like(params)

    min_error = np.inf
    best_params = params.copy()
    patience = 50
    patience_counter = 0

    error_history = np.zeros(3)
    error_idx = 0

    x, y = x_flat, y_flat

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

        # Update log-widths so each step preserves positivity, even for
        # narrow peaks and inflated initial covariance estimates.
        grad[3:5] *= params[3:5]
        params[3:5] = np.log(params[3:5])
        for i in range(len(params)):
            params[i], m[i], v[i] = adam_update(
                params[i], m[i], v[i], grad[i], t, alpha=alphas[i]
            )

        params[0] = max(params[0], 0)
        params[3:5] = np.exp(params[3:5])
        params[5] = clip_value(params[5], -0.99, 0.99)

    best_params[0] *= scale
    best_params[6] *= scale
    return best_params


@numba.njit(cache=True)
def _fit_pixel(data, points, projection, origin, span, power, tolerance, mean_map, cov_map, i, j):
    out = np.zeros(8)
    if not np.isfinite(data).all() or np.min(data) < 0 or np.min(data) == np.max(data):
        return out
    flat = data.flatten().astype(np.float64)
    x, y = points
    plane = projection @ flat
    signal = flat - (plane[0]*x + plane[1]*y + plane[2])
    height = np.max(signal)
    if height <= 1e-7 * np.max(flat):
        return out
    # Exclude low-amplitude fluctuations from the moment seed only.
    seed = np.where(signal >= 0.05 * height, signal, 0.0)
    # A few sparse scans have all above-threshold samples on one motor
    # line. Use the remaining positive signal to initialize those pixels.
    if cov_map is None:
        selected = seed > 0
        if np.ptp(x[selected]) == 0 or np.ptp(y[selected]) == 0:
            seed = np.maximum(signal, 0.0)
    total = np.sum(seed)
    if mean_map is None:
        mean = np.array([np.sum(seed*x), np.sum(seed*y)]) / total
    else:
        mean = mean_map[i, j]
    if cov_map is None:
        offsets = points - mean.reshape(2, 1)
        cov = (offsets * seed) @ offsets.T / total
    else:
        cov = cov_map[i, j]
    if not np.isfinite(mean).all() or not np.isfinite(cov).all() or min(cov[0,0], cov[1,1]) <= 0:
        return out

    p = fit_gaussian_2d(
        flat - plane[2], x, y, mean, cov, power,
        tolerance=tolerance, weight_signal=signal,
    )
    if not np.isfinite(p).all() or p[0] <= 0 or min(p[3], p[4]) <= 0:
        return out
    out[:6] = p[:6]
    out[1:3] = origin + p[1:3] * span
    out[3:5] = p[3:5] * span
    out[6] = 1
    out[7] = plane[2] + p[6]
    return out


@numba.njit(cache=True, parallel=True)
def _fit_map(data, points, projection, origin, span, power, tolerance, mask, mean, cov, return_background):
    rows, cols = data.shape[:2]
    fit = np.zeros((rows, cols, 7))
    background = np.zeros((rows, cols, 3)) if return_background else np.empty((0, 0, 3))
    for index in numba.prange(rows * cols):
        i, j = index // cols, index % cols
        if mask is None or mask[i, j]:
            p = _fit_pixel(data[i, j], points, projection, origin, span, power, tolerance, mean, cov, i, j)
            fit[i, j] = p[:7]
            if return_background:
                background[i, j, 2] = p[7]
    return fit, background


def _seed(value, shape):
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"expected moment map with shape {shape}, got {value.shape}")
    return value
