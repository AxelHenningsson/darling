import numba
import numpy as np
from numba import prange


@numba.njit(cache=True)
def func_grad(A, sigma, mu, k, m, x):
    """Evaluate Gaussian with linear background and its gradient.

    The function is defined as
        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    The gradient is taken with respect to the parameters
        [A, sigma, mu, k, m]
    in that order.

    Args:
        A (:obj:`float`): Amplitude of the Gaussian peak.
        sigma (:obj:`float`): Standard deviation of the Gaussian peak.
        mu (:obj:`float`): Mean position of the Gaussian peak.
        k (:obj:`float`): Slope of the linear background.
        m (:obj:`float`): Intercept of the linear background.
        x (:obj:`float`): Position at which to evaluate the function.

    Returns:
        tuple:
            func (:obj:`float`): Function value f(x).
            grad (:obj:`numpy.ndarray`): Gradient vector of shape (5,)
                ordered as [df/dA, df/dsigma, df/dmu, df/dk, df/dm].
    """
    res = mu - x
    res2 = res * res
    s2 = sigma * sigma
    s3 = s2 * sigma
    l1 = np.exp(-0.5 * res2 / s2)
    func = A * l1 + k * x + m
    grad = np.array(
        [
            l1,
            A * res2 * l1 / s3,
            -A * res * l1 / s2,
            x,
            1.0,
        ]
    )
    return func, grad


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
def estimate_initial_gaussian_params(data, x, k_bg, m_bg):
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
            mu (:obj:`float`): Initial mean estimate.
            sigma (:obj:`float`): Initial standard deviation estimate.
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

    return A, mu, sigma


@numba.njit(cache=True)
def assemble_normal_equations(A, sigma, mu, k_bg, m_bg, x, data):
    """Assemble the normal equations for Gauss-Newton method for the Gaussian + linear model.

    The normal equations are given by
        H = sum( J^T J ) (approx. Hessian matrix)
        g = -sum( J^T r ) (approx. gradient vector)

    Args:
        A (:obj:`float`): Current amplitude.
        sigma (:obj:`float`): Current standard deviation.
        mu (:obj:`float`): Current mean.
        k_bg (:obj:`float`): Current background slope.
        m_bg (:obj:`float`): Current background intercept.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.

    Returns:
        tuple:
            H (:obj:`numpy.ndarray`): Approx. Hessian matrix of shape (5, 5).
            g (:obj:`numpy.ndarray`): Exact gradient vector of shape (5,).
    """
    H = np.zeros((5, 5))
    g = np.zeros(5)
    for i in range(x.shape[0]):
        t = x[i]
        yk = data[i]
        f, grad = func_grad(A, sigma, mu, k_bg, m_bg, t)
        r = yk - f
        for a in range(5):
            ga = grad[a]
            g[a] += -ga * r
            for b in range(5):
                H[a, b] += ga * grad[b]
    return H, g


@numba.njit(cache=True)
def gauss_newton_iteration(A, sigma, mu, k_bg, m_bg, x, data):
    """Perform a single Gauss-Newton iteration for a Gaussian + linear model.

    The model is
        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k_bg * x + m_bg

    Using the current parameter values, this function computes the
    Gauss-Newton step by building the normal equations

        H = sum( J^T J )
        g = -sum( J^T r )

    where r = y - f and J is the gradient of f with respect to
    [A, sigma, mu, k_bg, m_bg].

    Args:
        A (:obj:`float`): Current amplitude.
        sigma (:obj:`float`): Current standard deviation.
        mu (:obj:`float`): Current mean.
        k_bg (:obj:`float`): Current background slope.
        m_bg (:obj:`float`): Current background intercept.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.

    Returns:
        tuple:
            A_new (:obj:`float`): Updated amplitude.
            sigma_new (:obj:`float`): Updated standard deviation.
            mu_new (:obj:`float`): Updated mean.
            k_new (:obj:`float`): Updated background slope.
            m_new (:obj:`float`): Updated background intercept.
            success (:obj:`bool`): False if the step failed (e.g. singular or ill-conditioned matrix).
    """

    # get Hessian matrix and gradient vector
    H, g = assemble_normal_equations(A, sigma, mu, k_bg, m_bg, x, data)

    # solve the normal equations
    try:
        delta = np.linalg.solve(H, -g)
    except:
        # Singular or ill-conditioned matrix
        return A, sigma, mu, k_bg, m_bg, False

    # update the parameters
    A_new = A + delta[0]
    sigma_new = sigma + delta[1]
    mu_new = mu + delta[2]
    k_new = k_bg + delta[3]
    m_new = m_bg + delta[4]

    # check if the new standard deviation is zero or negative
    if sigma_new <= 1e-8:
        return A, sigma, mu, k_bg, m_bg, False

    return A_new, sigma_new, mu_new, k_new, m_new, True


@numba.njit(cache=True)
def gauss_newton_fit_1D(data, x, n_iter):
    """Fit a Gaussian peak on top of a linear background to a 1D trace.

    The model is
        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    This function:
        1. Estimates an initial linear trend (k, m).
        2. Estimates initial Gaussian parameters (A, mu, sigma)
           on the background-subtracted data.
        3. Runs a fixed number of Gauss-Newton iterations to refine
           all five parameters jointly.

    Args:
        data (:obj:`numpy.ndarray`): 1D array of data values y_k.
        x (:obj:`numpy.ndarray`): 1D array of positions x_k.
        n_iter (:obj:`int`): Number of Gauss-Newton iterations.

    Returns:
        tuple:
            A (:obj:`float`): Fitted Gaussian amplitude.
            mu (:obj:`float`): Fitted Gaussian mean.
            sigma (:obj:`float`): Fitted Gaussian standard deviation.
            k_bg (:obj:`float`): Fitted background slope.
            m_bg (:obj:`float`): Fitted background intercept.
            success (:obj:`int`): 0 if the fit failed, 1 if it succeeded.
    """
    k_bg, m_bg = estimate_initial_linear_trend(data, x)
    A, mu, sigma = estimate_initial_gaussian_params(data, x, k_bg, m_bg)

    if sigma <= 1e-8 or A == 0.0:
        return 0.0, 0.0, 0.0, k_bg, m_bg, 0

    # Perform the Gauss-Newton iterations to refine the parameters
    # see also: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    for it in range(n_iter):
        A, sigma, mu, k_bg, m_bg, success = gauss_newton_iteration(
            A, sigma, mu, k_bg, m_bg, x, data
        )
        # if the fit failed, return the initial parameters and 0 for success
        if not success:
            return 0.0, 0.0, 0.0, k_bg, m_bg, 0

    return A, mu, sigma, k_bg, m_bg, 1


@numba.njit(parallel=True, cache=True)
def fit_gaussian_with_linear_background_1D(data, x, n_iter, mask):
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
        mask (:obj:`numpy.ndarray`): 2D array of shape (ny, nx) with dtype bool. Only the pixels where mask is True will be fitted.

    Returns:
        :obj:`numpy.ndarray`: Output array of shape (ny, nx, 6) with
            parameters [A, mu, sigma, k, m, success] for each (i, j).
            Here A is the amplitude, mu is the mean, sigma is the standard deviation
            of the Gaussian, k is the background slope, m is the background intercept,
            and success is 0 if the fit failed, 1 if it succeeded.
    """
    ny, nx, m = data.shape
    out = np.zeros((ny, nx, 6), dtype=np.float64)
    x64 = x.astype(np.float64)
    for i in prange(ny):
        for j in range(nx):
            if mask is not None and not mask[i, j]:
                pass
            else:
                A, mu, sigma, k_bg, m_bg, success = gauss_newton_fit_1D(
                    data[i, j].astype(np.float64),
                    x64,
                    n_iter,
                )
                out[i, j, 0] = A
                out[i, j, 1] = mu
                out[i, j, 2] = sigma
                out[i, j, 3] = k_bg
                out[i, j, 4] = m_bg
                out[i, j, 5] = success
    return out


if __name__ == "__main__":
    pass
