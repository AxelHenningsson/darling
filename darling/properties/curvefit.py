import numba
import numpy as np

from .models import func_and_grad_gaussian_with_linear_background


def fit_1D_gaussian(
    data,
    coordinates,
    n_iter_gauss_newton=7,
    mask=None,
):
    """Fit analytical gaussian + linear background for each pixel in a 2D image.

    The input volume is assumed to have shape (ny, nx, m), where the
    last axis corresponds to the 1D curves to be fitted with
    a Gaussian + linear background. For each data[i, j, :] the function fits

    f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    and stores the five fitted parameters: [A, mu, sigma, k, m, success].

    Args:
        data (:obj:`numpy.ndarray`): 3D array of shape (ny, nx, m). Arbitrary dtypes are supported.
        coordinates (:obj:`numpy array`): numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid. Shape=(ndim, m, n, ...). where ndim=1 for a rocking scan,
            ndim=2 for a mosaicity scan, etc.
        n_iter_gauss_newton (:obj:`int`): Number of Gauss-Newton iterations to use for the fit. Defaults to 7.
        mask (:obj:`numpy.ndarray`): 2D array of shape (ny, nx) with dtype bool. Defaults to None. If provided, only the pixels where mask is True will be fitted.
    Returns:
        :obj:`numpy.ndarray`: Output array of shape (ny, nx, 5) of dtype float64 with
            parameters [A, mu, sigma, k, m, success] for each (i, j).
            Here A is the amplitude, mu is the mean, sigma is the standard deviation
            of the Gaussian, k is the background slope, m is the background intercept,
            and success is 0 if the fit failed, 1 if it succeeded.
    """
    if len(coordinates) != 1:
        raise ValueError(
            f"coordinates must be a 1d tuple but got {len(coordinates)} dimensions"
        )
    if data.ndim != 3:
        raise ValueError(
            f"data must be a 3d numpy array but got {data.ndim} dimensions"
        )

    return fit_gaussian_with_linear_background_1D(
        data,
        coordinates[0],
        n_iter_gauss_newton,
        mask,
    )


@numba.njit(cache=True)
def estimate_initial_linear_trend(data, x):
    """Estimate an initial linear background k_bg * x + m_bg.

    This function computes the least-squares fit of a straight line
        y = k_bg * x + m_bg
    to the input data. It uses the standard analytic formulas for
    simple linear regression.

    Args:
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.

    Returns:
        tuple:
            k_bg (:obj:`float`): Initial estimate of the background slope.
            m_bg (:obj:`float`): Initial estimate of the background intercept.
    """
    n = x.shape[0]

    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Sxy = 0.0
    for k in range(n):
        t = x[k]
        yk = data[k]
        Sx += t
        Sy += yk
        Sxx += t * t
        Sxy += t * yk

    den = n * Sxx - Sx * Sx
    if den != 0.0:
        k_bg = (n * Sxy - Sx * Sy) / den
        m_bg = (Sy - k_bg * Sx) / n
    else:
        k_bg = 0.0
        m_bg = Sy / n

    return k_bg, m_bg


@numba.njit(cache=True)
def _estimate_initial_gaussian_params(data, x, k_bg, m_bg):
    """Estimate initial Gaussian parameters on top of a linear background.

    The function first subtracts the linear background k_bg * x + m_bg
    from the data and then computes simple intensity-weighted moments
    using only the positive background-subtracted values.

    From these moments it estimates:
        mu    from the first moment,
        sigma from the second central moment,
        A     from the maximum background-subtracted value.

    Args:
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        k_bg (:obj:`float`): Background slope estimate.
        m_bg (:obj:`float`): Background intercept estimate.

    Returns:
        tuple:
            A (:obj:`float`): Initial amplitude estimate.
            sigma (:obj:`float`): Initial standard deviation estimate.
            mu (:obj:`float`): Initial mean estimate.
    """
    n = x.shape[0]

    sumw = 0.0
    sumwx = 0.0
    sumwxx = 0.0
    max_w = 0.0

    for i in range(n):
        t = x[i]
        yk = data[i] - (k_bg * t + m_bg)
        if yk > 0.0:
            sumw += yk
            sumwx += yk * t
            sumwxx += yk * t * t
            if yk > max_w:
                max_w = yk

    if sumw <= 0.0:
        return 0.0, 0.0, 0.0

    mu = sumwx / sumw
    var = sumwxx / sumw - mu * mu
    if var <= 0.0:
        return 0.0, 0.0, 0.0

    sigma = np.sqrt(var)
    if sigma <= 1e-8:
        return 0.0, 0.0, 0.0

    if max_w <= 0.0:
        A = np.max(data)
    else:
        A = max_w

    return A, sigma, mu


@numba.njit(cache=True)
def _assemble_normal_equations(params, x, data, func_and_grad):
    """Assemble the normal equations for Gauss-Newton method.

    The normal equations are given by
        H = sum( J^T J ) (approx. Hessian matrix)
        g = -sum( J^T r ) (approx. gradient vector)

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (n,).
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        func_and_grad (callable): Function that maps (params, x_k) -> (f_k, grad_k).

    Returns:
        tuple:
            H (:obj:`numpy.ndarray`): Approx. Hessian matrix of shape (n, n).
            g (:obj:`numpy.ndarray`): Exact gradient vector of shape (n,).
    """
    n = len(params)
    H = np.zeros((n, n))
    g = np.zeros(n)
    for i in range(x.shape[0]):
        t = x[i]
        yk = data[i]
        f, grad = func_and_grad(params, t)
        r = yk - f
        for a in range(n):
            ga = grad[a]
            g[a] += -ga * r
            for b in range(n):
                H[a, b] += ga * grad[b]
    return H, g


@numba.njit(cache=True)
def _gauss_newton_iteration(params, x, data, func_and_grad):
    """Perform a single Gauss-Newton iteration for a generic model.

    Args:
        params (:obj:`numpy.ndarray`): Current parameter vector of shape (n,).
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        func_and_grad (callable): Function that maps (params, x_k) -> (f_k, grad_k).

    Returns:
        tuple:
            params_new (:obj:`numpy.ndarray`): Updated parameter vector.
            success (:obj:`bool`): False if the step failed (e.g. singular or ill-conditioned matrix).
    """
    H, g = _assemble_normal_equations(params, x, data, func_and_grad)
    try:
        params += np.linalg.solve(H, -g)
    except Exception:
        return params, False
    return params, True


@numba.njit(cache=True)
def gauss_newton_fit_1D(data, x, initial_params, n_iter, func_and_grad):
    """Fit a generic 1D model using Gauss-Newton iterations.

    Args:
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        initial_params (sequence): Initial parameter values (tuple, list, or array).
        n_iter (:obj:`int`): Number of Gauss-Newton iterations.
        func_and_grad (callable): Function that maps (params, x_k) -> (f_k, grad_k).

    Returns:
        tuple:
            params (:obj:`numpy.ndarray`): Fitted parameters.
            success (:obj:`int`): 0 if the fit failed, 1 if it succeeded.
    """
    params = np.asarray(initial_params, dtype=np.float64).copy()
    x = np.asarray(x, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    for _ in range(n_iter):
        params, success = _gauss_newton_iteration(params, x, data, func_and_grad)
        if not success:
            return params, 0.0

    return params, 1.0


@numba.njit(parallel=True)
def fit_callable_model_1D(
    data,
    x,
    initial_guess,
    func_and_grad,
    n_iter,
    mask,
):
    """Fit a callable model to 1D data.

    Args:
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        initial_guess (:obj:`numpy.ndarray`): Initial parameter values of shape (data.shape[0], data.shape[1], n).
        func_and_grad (:obj:`callable`): Function that maps (params, x_k) -> (f_k, grad_k). must be decorated with numba.njit
        n_iter (:obj:`int`): Number of Gauss-Newton iterations.
        mask (:obj:`numpy.ndarray`): 2D array of shape (ny, nx) with dtype bool.
            Only the pixels where mask is True will be fitted.

    Returns:
        :obj:`numpy.ndarray`: Output array of shape (ny, nx, initial_guess.shape[-1] + 1) with
            the fitted parameters and success flag for each pixel.
    """
    ny, nx, m = data.shape
    out = np.zeros((ny, nx, initial_guess.shape[-1] + 1), dtype=np.float64)
    x64 = x.astype(np.float64)

    for i in range(ny):
        for j in numba.prange(nx):
            if mask is not None and not mask[i, j]:
                continue
            else:
                y64 = data[i, j].astype(np.float64)
                fitted_params, success = gauss_newton_fit_1D(
                    y64,
                    x64,
                    initial_params=initial_guess[i, j, :],
                    n_iter=n_iter,
                    func_and_grad=func_and_grad,
                )
                out[i, j, 0 : len(fitted_params)] = fitted_params
                out[i, j, len(fitted_params)] = success

    return out


@numba.njit(parallel=True)
def fit_gaussian_with_linear_background_1D(
    data,
    x,
    n_iter,
    mask,
):
    """Fit a 1D Gaussian + linear background for each pixel in a 2D image.

    The input volume is assumed to have shape (ny, nx, m), where the
    last axis corresponds to the 1D traces along x. For each (i, j)
    the function fits

        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    and stores the five fitted parameters.

    Args:
        data (:obj:`numpy.ndarray`): 3D array of shape (ny, nx, m)
            containing the data traces.
        x (:obj:`numpy.ndarray`): 1D array of positions of length m.
        n_iter (:obj:`int`): Number of Gauss-Newton iterations per trace.
        mask (:obj:`numpy.ndarray`): 2D array of shape (ny, nx) with dtype bool.
            Only the pixels where mask is True will be fitted.
        func_and_grad (callable): Function that maps (params, x_k) -> (f_k, grad_k).

    Returns:
        :obj:`numpy.ndarray`: Output array of shape (ny, nx, 6) with
            parameters [A, sigma, mu, k_bg, m_bg, success] for each (i, j).
            Here A is the amplitude, sigma is the standard deviation,
            mu is the mean of the Gaussian, k_bg is the background slope,
            m_bg is the background intercept, and success is 0 if the fit failed,
            1 if it succeeded.
    """
    ny, nx, m = data.shape
    out = np.zeros((ny, nx, 6), dtype=np.float64)
    x64 = x.astype(np.float64)

    for i in range(ny):
        for j in numba.prange(nx):
            if mask is not None and not mask[i, j]:
                continue

            y64 = data[i, j].astype(np.float64)
            k_bg, m_bg = estimate_initial_linear_trend(y64, x64)
            A, sigma, mu = _estimate_initial_gaussian_params(y64, x64, k_bg, m_bg)

            initial_params = np.array([A, sigma, mu, k_bg, m_bg], dtype=np.float64)

            if sigma <= 1e-8 or A == 0.0:
                out[i, j, 0:5] = initial_params
                continue
            else:
                fitted_params, success = gauss_newton_fit_1D(
                    y64,
                    x64,
                    initial_params,
                    n_iter,
                    func_and_grad=func_and_grad_gaussian_with_linear_background,
                )
                out[i, j, 0 : len(fitted_params)] = fitted_params
                out[i, j, len(fitted_params)] = success

    return out


if __name__ == "__main__":
    pass
