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


if __name__ == "__main__":
    pass
