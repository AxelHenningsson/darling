import numba
import numpy as np


@numba.njit(cache=True)
def linear_background(params, x):
    """Evaluate a linear background.

    The linear background is defined as

        f(x) = k * x + m

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (2,), where
            params[0] is the background slope (k_bg),
            params[1] is the background intercept (m_bg).
        x (:obj:`float` or :obj:`numpy.ndarray`): Position(s) at which to
            evaluate the function.

    Returns:
        func (:obj:`float` or :obj:`numpy.ndarray`): Function value f(x).

    """
    k_bg, m_bg = params
    return k_bg * x + m_bg


@numba.njit(cache=True)
def gaussian(params, x):
    """Evaluate a Gaussian peak.

    The Gaussian peak is defined as

        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2))

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (3,), where
            params[0] is the amplitude of the Gaussian peak (A),
            params[1] is the standard deviation of the Gaussian peak (sigma),
            params[2] is the mean position of the Gaussian peak (mu).
        x (:obj:`float` or :obj:`numpy.ndarray`): Position(s) at which to
            evaluate the function.

    Returns:
        func (:obj:`float` or :obj:`numpy.ndarray`): Function value f(x).
    """
    A, sigma, mu = params
    s2 = sigma * sigma
    dx = x - mu
    dx2 = dx * dx
    return A * np.exp(-0.5 * dx2 / s2)


@numba.njit(cache=True)
def lorentzian(params, x):
    """Evaluate a Lorentzian peak.

    The Lorentzian peak is defined as

        f(x) = A / (1 + ((x - mu) / gamma)**2)

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (3,), where
            params[0] is the amplitude of the Lorentzian peak (A),
            params[1] is the mean position of the Lorentzian peak (mu),
            params[2] is the half-width-at-half-maximum of the Lorentzian peak (gamma),
        x (:obj:`float` or :obj:`numpy.ndarray`): Position(s) at which to
            evaluate the function.
    """
    A, mu, gamma = params
    dx = x - mu
    dx2 = dx * dx
    g2 = gamma * gamma
    return A / (1 + dx2 / g2)


@numba.njit(cache=True)
def pseudo_voigt(params, x):
    """Evaluate a pseudo-Voigt peak.

    The pseudo-Voigt line shape is defined as a linear combination of
    a Gaussian and a Lorentzian profile with common center mu,
    """
    A, sigma, mu, gamma, eta = params
    s2 = sigma * sigma
    dx = x - mu
    dx2 = dx * dx
    g2 = gamma * gamma
    G = np.exp(-0.5 * dx2 / s2)
    L = 1 / (1 + dx2 / g2)
    return A * (eta * L + (1.0 - eta) * G)


@numba.njit(cache=True)
def pseudo_voigt_with_linear_background(params, x):
    """Evaluate a pseudo-Voigt peak with linear background.

    The pseudo-Voigt line shape is defined as a linear combination of
    a Gaussian and a Lorentzian profile with common center mu,

        G(x) = exp(-(x - mu)**2 / (2 * sigma**2))
        L(x) = 1 / (1 + ((x - mu) / gamma)**2)

        S(x) = eta * L(x) + (1 - eta) * G(x)

    The full model is

        f(x) = A * S(x) + k * x + m

    where A is the peak amplitude, sigma is the Gaussian width,
    gamma is the Lorentzian half-width-at-half-maximum,
    eta is the mixing parameter (0 <= eta <= 1), and k, m define a
    linear background.

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (7,), where
            params[0] is the peak amplitude (A),
            params[1] is the Gaussian width parameter (sigma),
            params[2] is the peak center (mu),
            params[3] is the Lorentzian width parameter (gamma),
            params[4] is the mixing parameter (eta),
            params[5] is the background slope (k),
            params[6] is the background intercept (m).
        x (:obj:`float` or :obj:`numpy.ndarray`): Position(s) at which to
            evaluate the function.

    Returns:
        func (:obj:`float` or :obj:`numpy.ndarray`):
            Function value f(x).
    """
    S = pseudo_voigt(params[:5], x)
    B = linear_background(params[5:], x)
    return S + B


@numba.njit(cache=True)
def gaussian_with_linear_background(params, x):
    """Evaluate a Gaussian peak with linear background.

    The Gaussian peak is defined as

        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (5,), where
            params[0] is the amplitude of the Gaussian peak (A),
            params[1] is the standard deviation of the Gaussian peak (sigma),
            params[2] is the mean position of the Gaussian peak (mu),
            params[3] is the background slope (k),
            params[4] is the background intercept (m).
        x (:obj:`float` or :obj:`numpy.ndarray`): Position(s) at which to
            evaluate the function.

    Returns:
        func (:obj:`float` or :obj:`numpy.ndarray`): Function value f(x).
    """
    G = gaussian(params[:3], x)
    B = linear_background(params[3:], x)
    return G + B


@numba.njit(cache=True)
def func_and_grad_gaussian_with_linear_background(params, x):
    """Evaluate Gaussian with linear background and its gradient.

    The function is defined as
        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    The gradient is taken with respect to the parameters
        A, sigma, mu, k, m = *params
    in that order.

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (5,), where
        params[0] is the amplitude of the Gaussian peak (A),
        params[1] is the standard deviation of the Gaussian peak (sigma),
        params[2] is the mean position of the Gaussian peak (mu),
        params[3] is the slope of the linear background (k),
        params[4] is the intercept of the linear background (m).
        x (:obj:`float`): Position at which to evaluate the function.

    Returns:
        tuple:
            func (:obj:`float`): Function value f(x).
            grad (:obj:`numpy.ndarray`): Gradient vector of shape (5,)
                ordered as [df/dA, df/dsigma, df/dmu, df/dk, df/dm].
    """
    A, sigma, mu, k_bg, m_bg = params
    res = mu - x
    res2 = res * res
    s2 = sigma * sigma
    s3 = s2 * sigma
    l1 = np.exp(-0.5 * res2 / s2)
    func = A * l1 + k_bg * x + m_bg
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
def func_and_grad_gaussian(params, x):
    """Evaluate Gaussian and its gradient.

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (3,), where
        params[0] is the amplitude of the Gaussian peak (A),
        params[1] is the standard deviation of the Gaussian peak (sigma),
        params[2] is the mean position of the Gaussian peak (mu).
        x (:obj:`float`): Position at which to evaluate the function.

    Returns:
        tuple:
            func (:obj:`float`): Function value f(x).
            grad (:obj:`numpy.ndarray`): Gradient vector of shape (3,)
                ordered as [df/dA, df/dsigma, df/dmu].
    """
    A, sigma, mu = params
    res = mu - x
    res2 = res * res
    s2 = sigma * sigma
    s3 = s2 * sigma
    l1 = np.exp(-0.5 * res2 / s2)
    func = A * l1
    grad = np.array(
        [
            l1,
            A * res2 * l1 / s3,
            -A * res * l1 / s2,
        ]
    )
    return func, grad


@numba.njit(cache=True)
def func_and_grad_pseudo_voigt_with_linear_background(params, x):
    """Evaluate pseudo-Voigt with linear background and its gradient.

    The pseudo-Voigt line shape is defined as a linear combination of
    a Gaussian and a Lorentzian profile with common center mu,

        G(x) = exp(-(x - mu)**2 / (2 * sigma**2))
        L(x) = 1 / (1 + ((x - mu) / gamma)**2)

        S(x) = eta * L(x) + (1 - eta) * G(x)

    The full model is

        f(x) = A * S(x) + k_bg * x + m_bg

    where A is the peak amplitude, sigma is the Gaussian width,
    gamma is the Lorentzian half-width-at-half-maximum,
    eta is the mixing parameter (0 <= eta <= 1), and k_bg, m_bg
    define a linear background.

    The gradient is taken with respect to the parameters
        A, sigma, mu, gamma, eta, k_bg, m_bg = *params
    in that order.

    Args:
        params (:obj:`numpy.ndarray`): Array of parameters of shape (7,), where
            params[0] is the peak amplitude (A),
            params[1] is the Gaussian width parameter (sigma),
            params[2] is the peak center (mu),
            params[3] is the Lorentzian width parameter (gamma),
            params[4] is the mixing parameter (eta),
            params[5] is the background slope (k_bg),
            params[6] is the background intercept (m_bg).
        x (:obj:`float`): Position at which to evaluate the function.

    Returns:
        tuple:
            func (:obj:`float`): Function value f(x).
            grad (:obj:`numpy.ndarray`): Gradient vector of shape (7,)
                ordered as [df/dA, df/dsigma, df/dmu,
                           df/dgamma, df/deta, df/dk_bg, df/dm_bg].
    """
    A, sigma, mu, gamma, eta, k_bg, m_bg = params

    dx = x - mu

    u = dx / sigma
    u2 = u * u
    v = dx / gamma
    v2 = v * v

    G = np.exp(-0.5 * u2)
    denom = 1.0 + v2
    L = 1.0 / denom

    S = eta * L + (1.0 - eta) * G

    func = A * S + k_bg * x + m_bg

    dG_dsigma = u2 * G / sigma
    dG_dmu = u * G / sigma

    denom2 = denom * denom
    dL_dmu = 2.0 * v / (gamma * denom2)
    dL_dgamma = 2.0 * v2 / (gamma * denom2)

    dS_dsigma = (1.0 - eta) * dG_dsigma
    dS_dmu = eta * dL_dmu + (1.0 - eta) * dG_dmu
    dS_dgamma = eta * dL_dgamma
    dS_deta = L - G

    dfdA = S
    dfdsigma = A * dS_dsigma
    dfdmu = A * dS_dmu
    dfdgamma = A * dS_dgamma
    dfdeta = A * dS_deta
    dfdk = x
    dfdm = 1.0

    grad = np.array(
        [
            dfdA,
            dfdsigma,
            dfdmu,
            dfdgamma,
            dfdeta,
            dfdk,
            dfdm,
        ]
    )

    return func, grad


@numba.njit(cache=True)
def func_and_grad_gaussian_mixture(params, x, dim):
    """Evaluate an anisotropic Gaussian mixture (Cholesky parameterisation) and its gradient.

    Args:
        params (:obj:`numpy.ndarray`): Flat parameter vector of shape
            ``(number_of_gaussians * parameters_per_gaussian,)`` where
            ``parameters_per_gaussian = 1 + dim + dim * (dim + 1) // 2``.
            For each Gaussian component the parameters are ordered as::

                [amplitude,
                 mu_0, ..., mu_{dim-1},
                 alpha_0, ..., alpha_{dim-1},
                 L_off_0, ..., L_off_{dim*(dim-1)//2 - 1}]

            Here :math:`L` is the lower-triangular Cholesky factor of the precision matrix,
            with ``L[i, i] = alpha_i ** 2`` and the strictly lower-triangular entries
            ``L[i, j]`` (for ``i > j``) taken from ``L_off_*`` in row-major order over
            the lower triangle.
        x (:obj:`numpy.ndarray`): Evaluation point of shape ``(dim,)``.
        dim (:obj:`int`): Spatial dimension of the Gaussian mixture (1, 2, or 3).

    Returns:
        tuple:
            function_value (:obj:`float`): Value of the Gaussian mixture
                :math:`f(x) = \\sum_k A_k \\exp\\bigl(-\\tfrac{1}{2} \\lVert L_k^T (x - \\mu_k) \\rVert^2\\bigr)`.
            gradient (:obj:`numpy.ndarray`): Gradient vector of shape ``(params.size,)``,
                containing :math:`\\partial f(x)/\\partial \\text{params}` in the same
                ordering as ``params``.
    """
    n_params = params.size
    params_per_gaussian = 1 + dim + dim * (dim + 1) // 2
    n_gaussians = n_params // params_per_gaussian

    grad = np.zeros(n_params, dtype=params.dtype)
    f_val = 0.0

    for k in range(n_gaussians):
        offset = k * params_per_gaussian

        A = params[offset]

        mu_start = offset + 1
        mu_end = mu_start + dim

        r = np.empty(dim, dtype=params.dtype)
        for i in range(dim):
            r[i] = x[i] - params[mu_start + i]

        L_param_start = mu_end

        alpha = np.empty(dim, dtype=params.dtype)
        for i in range(dim):
            alpha[i] = params[L_param_start + i]

        L_off_start = L_param_start + dim

        L = np.zeros((dim, dim), dtype=params.dtype)
        for i in range(dim):
            L[i, i] = alpha[i] * alpha[i]

        idx_off = 0
        for i in range(1, dim):
            for j in range(i):
                L[i, j] = params[L_off_start + idx_off]
                idx_off += 1

        u = np.zeros(dim, dtype=params.dtype)
        for j in range(dim):
            s = 0.0
            for i in range(dim):
                s += L[i, j] * r[i]
            u[j] = s

        t = 0.0
        for j in range(dim):
            t += u[j] * u[j]

        ell = np.exp(-0.5 * t)
        f_k = A * ell
        f_val += f_k

        grad[offset] += ell

        Lu = np.zeros(dim, dtype=params.dtype)
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += L[i, j] * u[j]
            Lu[i] = s

        for i in range(dim):
            grad[mu_start + i] += A * ell * Lu[i]

        for i in range(dim):
            grad[L_param_start + i] += -2.0 * A * ell * alpha[i] * u[i] * r[i]

        idx_off = 0
        for i in range(1, dim):
            for j in range(i):
                param_index = L_off_start + idx_off
                grad[param_index] += -A * ell * u[j] * r[i]
                idx_off += 1

    return f_val, grad


if __name__ == "__main__":
    pass
