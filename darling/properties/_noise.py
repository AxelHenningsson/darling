import numba
import numpy as np

from .curvefit import gauss_newton_fit_1D
from .models import func_and_grad_gaussian


def _check_estimate_white_noise_inputs(
    data,
    inital_guess,
    truncate,
    max_iterations,
    convergence_tol,
    loop_outer_dims,
    gauss_newton_refine,
    n_iter_gauss_newton,
):
    if not isinstance(loop_outer_dims, bool):
        raise TypeError(
            f"Loop outer dimensions must be a boolean but got {type(loop_outer_dims)}"
        )
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "data must be a numpy.ndarray, got type {}".format(type(data).__name__)
        )

    if loop_outer_dims and data.ndim not in (3, 4, 5):
        raise ValueError(
            "data must have 3, 4, or 5 dimensions when loop_outer_dims is True, got ndim={}".format(
                data.ndim
            )
        )
    elif not loop_outer_dims and data.ndim not in (1, 2, 3):
        raise ValueError(
            "data must have 1, 2, or 3 dimensions when loop_outer_dims is False, got ndim={}".format(
                data.ndim
            )
        )

    if inital_guess is not None and not isinstance(inital_guess, tuple):
        raise TypeError(
            f"inital_guess must be a tuple of two floats but got {type(inital_guess)}"
        )

    if inital_guess is not None and len(inital_guess) != 2:
        raise ValueError(
            f"inital_guess must be a tuple of two floats but got {len(inital_guess)}"
        )

    if inital_guess is not None and not isinstance(inital_guess[0], (int, float)):
        raise TypeError(f"Mean must be a number but got {type(inital_guess[0])}")

    if inital_guess is not None and not isinstance(inital_guess[1], (int, float)):
        raise TypeError(
            f"Standard deviation must be a number but got {type(inital_guess[1])}"
        )

    if not isinstance(truncate, (int, float)):
        raise TypeError(f"Truncate must be a number but got {type(truncate)}")

    if truncate <= 0.0:
        raise ValueError(f"Truncate must be strictly positive but got {truncate:.2f}")

    if not isinstance(max_iterations, int):
        raise TypeError(
            f"Max iterations must be an integer but got {type(max_iterations)}"
        )

    if max_iterations <= 0:
        raise ValueError(
            f"Max iterations must be a positive integer but got {max_iterations}"
        )

    if not np.isscalar(convergence_tol):
        raise TypeError(
            f"Convergence tolerance must be a number but got {type(convergence_tol)}"
        )

    if convergence_tol <= 0.0:
        raise ValueError(
            f"Convergence tolerance must be strictly positive but got {convergence_tol:.2f}"
        )

    if not isinstance(gauss_newton_refine, bool):
        raise TypeError(
            f"Gauss-Newton refine must be a boolean but got {type(gauss_newton_refine)}"
        )

    if not isinstance(n_iter_gauss_newton, int):
        raise TypeError(
            f"Number of Gauss-Newton iterations must be an integer but got {type(n_iter_gauss_newton)}"
        )

    if n_iter_gauss_newton <= 0:
        raise ValueError(
            f"Number of Gauss-Newton iterations must be a positive integer but got {n_iter_gauss_newton}"
        )


def _estimate_initial_guess(data, inital_guess, loop_outer_dims, n_estimates=5):
    # Robust (but rough) estimate of the noise floor by sampling random pixels.
    # The assumption is that the majority of the data is noise (at least 25%)
    if inital_guess is None and loop_outer_dims:
        inital_guess_estimates = np.zeros((n_estimates, 2))
        for k in range(inital_guess_estimates.shape[0]):
            i = np.random.randint(0, data.shape[0])
            j = np.random.randint(0, data.shape[1])
            random_pixel_data = np.sort(data[i, j].flatten())
            N = np.maximum(1, len(random_pixel_data) // 4)
            inital_guess_estimates[k, 0] = random_pixel_data[0:N].mean()
            inital_guess_estimates[k, 1] = random_pixel_data[0:N].std()
        return np.median(inital_guess_estimates, axis=0)
    elif inital_guess is None and not loop_outer_dims:
        random_pixel_data = np.sort(data.flatten())  # we have only one pixel..
        N = np.maximum(1, len(random_pixel_data) // 4)
        return random_pixel_data[0:N].mean(), random_pixel_data[0:N].std()
    else:
        # use the provided initial guess if user wants to override the default.
        return inital_guess[0], inital_guess[1]


def _estimate_white_noise(
    data,
    inital_guess,
    truncate,
    max_iterations,
    convergence_tol,
    loop_outer_dims,
    gauss_newton_refine,
    n_iter_gauss_newton,
):
    _check_estimate_white_noise_inputs(
        data,
        inital_guess,
        truncate,
        max_iterations,
        convergence_tol,
        loop_outer_dims,
        gauss_newton_refine,
        n_iter_gauss_newton,
    )

    mean_guess, std_guess = _estimate_initial_guess(data, inital_guess, loop_outer_dims)
    if loop_outer_dims:  # the user sent us a pixel array, we need to estimate the mean and standard deviation for each pixel independently.
        mean, std = _estimate_white_noise_parallel(
            data,
            truncate,
            mean_guess,
            std_guess,
            max_iterations,
            convergence_tol,
        )

        if gauss_newton_refine:
            mean, std = _curve_fit_white_noise_parallel(
                data,
                truncate,
                mean,
                std,
                n_iter_gauss_newton,
            )
    else:  # the user sent us data from a single pixel, no need to go parallel.
        data_flat = data.copy().reshape(1, 1, -1)
        mean, std = _estimate_white_noise_parallel(
            data_flat,
            truncate,
            mean_guess,
            std_guess,
            max_iterations,
            convergence_tol,
        )

        if gauss_newton_refine:
            mean, std = _curve_fit_white_noise_1D(
                data_flat,
                truncate,
                mean,
                std,
                n_iter_gauss_newton,
            )

    return mean, std


@numba.njit(parallel=True, cache=True)
def _estimate_white_noise_parallel(
    data,
    truncate,
    mean_guess,
    std_guess,
    max_iterations,
    convergence_tol,
):
    mean = np.full(data.shape[:2], mean_guess, dtype=np.float64)
    std = np.full(data.shape[:2], std_guess, dtype=np.float64)
    has_converged = np.full(data.shape[:2], False, dtype=bool)

    for _ in range(max_iterations):
        for i in numba.prange(data.shape[0]):
            for j in range(data.shape[1]):
                # Skipping pixels that have already converged.
                if has_converged[i, j]:
                    continue

                old_mean = float(mean[i, j])
                old_std = float(std[i, j])

                # Updates the estimate of the mean and standard deviation using arithmetic tail statistics.
                new_mean, new_std = _iterate_white_noise_tail_statistics(
                    data[i, j].ravel().astype(np.float64),
                    truncate,
                    old_mean,
                    old_std,
                )

                res_mean = np.abs(new_mean - old_mean)
                res_std = np.abs(new_std - old_std)

                mean[i, j] = new_mean
                std[i, j] = new_std

                # Mark as converged of the statistics are not moving anymore.
                if res_mean < convergence_tol and res_std < convergence_tol:
                    has_converged[i, j] = True

    return mean, std


@numba.njit(parallel=True)
def _curve_fit_white_noise_parallel(
    data,
    truncate,
    mean,
    std,
    n_iter_gauss_newton,
):
    # Refines an inital map of mean and standard deviation using a Gauss-Newton fit
    # of a Gaussian to a histogram constructed from the truncated samples.
    new_mean = np.empty(data.shape[:2], dtype=np.float64)
    new_std = np.empty(data.shape[:2], dtype=np.float64)
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            new_mean[i, j], new_std[i, j], success = _curve_fit_white_noise_1D(
                data[i, j].ravel(),
                truncate,
                mean[i, j],
                std[i, j],
                n_iter_gauss_newton,
            )
            if not success:
                new_mean[i, j] = mean[i, j]
                new_std[i, j] = std[i, j]
    return new_mean, new_std


@numba.njit
def _curve_fit_white_noise_1D(
    data,
    truncate,
    mean,
    std,
    n_iter_gauss_newton,
):
    # Fits a Gaussian to a histogram constructed from the truncated samples.
    if std <= 0.0:
        return mean, std, 0

    # build the histogram from the tail samples.
    hist, bin_edges_start, bin_size = _histogram_white_noise_1d(
        data,
        truncate,
        mean,
        std,
    )

    A = hist.max()

    bin_edges = np.linspace(
        bin_edges_start,
        bin_edges_start + bin_size * (len(hist) - 1),
        len(hist),
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    initial_params = np.array([A, std, mean])

    # Fit a parametric Gaussian curve to the histogram.
    fitted_params, success = gauss_newton_fit_1D(
        hist,
        bin_centers,
        initial_params,
        n_iter=n_iter_gauss_newton,
        func_and_grad=func_and_grad_gaussian,
    )

    return fitted_params[2], fitted_params[1], success


@numba.njit(cache=True)
def _histogram_white_noise_1d(data_flat, truncate, mean, std):
    # Construct a 1D histogram from tail samples that lie in
    # the inlier range (mean - truncate * std, mean + truncate * std).
    width = truncate * std
    cut = mean + width

    bin_size = std / 2.0  # TODO: hardcoded (but robust) bin size, could be improved..
    nbins = int((2.0 * width) / bin_size) + 1
    bin_edges_start = mean - width

    hist = np.zeros(nbins, dtype=np.float64)
    inv = 1.0 / bin_size
    n = len(data_flat)

    for i in range(n):
        x = data_flat[i]
        if x < cut and x > bin_edges_start:
            idx = int((x - bin_edges_start) * inv)
            if idx < 0:
                continue
            if idx >= nbins:
                idx = nbins - 1
            hist[idx] += 1

    return hist, bin_edges_start, bin_size


@numba.njit(cache=True)
def _iterate_white_noise_tail_statistics(
    data_flat,
    truncate,
    mean,
    std,
):
    # Computes the arithmetic mean and standard deviation of the tail samples that lie in
    # the inlier range (mean - truncate * std, mean + truncate * std).
    new_mean = 0.0
    new_std = 0.0
    count = 0

    cut = mean + truncate * std
    low = mean - truncate * std
    n = len(data_flat)

    for i in range(n):
        x = data_flat[i]
        if x < cut and x > low:
            count += 1
            new_mean += x

    if count == 0:
        return mean, std

    new_mean /= count

    for i in range(n):
        x = data_flat[i]
        if x < cut and x > low:
            dx = x - new_mean
            new_std += dx * dx

    if count > 1:
        new_std /= count - 1
    else:
        return new_mean, 0.0

    if new_std <= 0.0:
        return new_mean, 0.0
    else:
        new_std = np.sqrt(new_std)
        return new_mean, new_std


def estimate_white_noise(
    data,
    inital_guess=None,
    truncate=3.5,
    max_iterations=5,
    convergence_tol=1e-3,
    loop_outer_dims=True,
    gauss_newton_refine=True,
    n_iter_gauss_newton=3,
):
    """Build a per-pixel Gaussian white-noise model from a multidimensional data array.

    The method separates (approximately) additive white noise from signal by repeatedly
    estimating a Gaussian from the low end portion of the distribution (defined by a
    symmetric truncation around a current mean value). Optionally, it refines the estimate
    using a Gauss-Newton curve fit to a histogram.

    This function is usefull for predicting accurate darks with associated uncertainties from
    collected diffraction data. This method of predicting darks has 3 major benefits:

    - No additional dark frames need to be collected.
    - The metrics are often statistically much more robust than independently collected dark frames due to the
      possibly large number of samples (a 20 x 30 mosa scan gives 600 samples per pixel which often can correspond
      to 400-500 dark samples per pixel). This is especially apparent for the estimation of the standard deviation
      of the dark field which is sensetive to the sample size.
    - The conditions of the dark estimate is guaranteed to be exactly the same as the conditions of the data.

    NOTE: This method is only valid as long as the data signal is somewhat sparse and clearly
    separated from the noise floor. I.e for each detector pixel, the majority of readouts are
    dark-redouts, without any diffraction signal.

    Example usecase:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import darling

        _, data, motors = darling.io.assets.mosaicity_scan()
        mean, std = darling.properties.estimate_white_noise(data)

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        im = ax[0].imshow(mean, cmap="plasma")
        ax[0].text(1, 4, "Estimate of Noise Mean", fontweight="bold", color="k")
        fig.colorbar(im, ax=ax[0], fraction=0.0335, pad=0.02)
        im = ax[1].imshow(std, cmap="spring")
        ax[1].text(
            1, 4, "Estimate of Noise Standard-Deviation", fontweight="bold", color="k"
        )
        fig.colorbar(im, ax=ax[1], fraction=0.0335, pad=0.02)
        for a in ax.flatten():
            for spine in a.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/white_noise.png

    Args:
        data (:obj:`numpy.ndarray`): If loop_outer_dims is ``True``, array of shape
            ``(a, b, m)``, ``(a, b, m, n)``, or ``(a, b, m, n, o)``. For each detector pixel
            ``(i, j)``, the values ``data[i, j, ...]`` form the sample distribution across
            scan coordinates. If loop_outer_dims is ``False``, array of shape ``(m,)``, ``(m, n)``,
            or ``(m, n, o)``, containing the sample distribution across scan coordinates.
        inital_guess (:obj:`tuple` of :obj:`float`): Initial guess for the noise mean
            and standard deviation. This is your expected mean noise level and standard
            deviation of the noise. If not provided, a robust (but rough) estimate of
            the noise floor is computed by sampling random pixels. The assumption is that
            the majority of the data is noise (at least 25%).
        truncate (:obj:`float`): Truncation factor that defines the inlier range
            ``(mean - truncate * std, mean + truncate * std)`` used when estimating
            mean and standard deviation. Defaults to ``3.5``. For each iteration, only the
            data points within this range are used to update the estimate of the mean and
            standard deviation.
        max_iterations (:obj:`int`): Maximum number of tail-statistics iterations.
            Defaults to ``5``. I.e the algorithm will iterate at most 5 times to update
            the estimate of the mean and standard deviation.
        convergence_tol (:obj:`float`): Absolute convergence tolerance for both mean
            and standard deviation updates. Defaults to ``1e-3``. I.e the algorithm will
            stop iterating if the absolute difference between the new and old estimate of
            the mean and standard deviation is less than this value.
        loop_outer_dims (:obj:`bool`): If ``True``, estimate mean and standard deviation
            independently for each pixel, returning arrays of shape ``(a, b)``. I.e the
            algorithm will estimate the mean and standard deviation for each pixel independently.
            If ``False``, estimate a single (global) mean and standard deviation by
            flattening the entire array. Defaults to ``True``.
        gauss_newton_refine (:obj:`bool`): If ``True``, refine the estimates using a
            Gauss-Newton fit of a Gaussian to a histogram constructed from the truncated
            samples. Defaults to ``True``. I.e the algorithm will refine the estimate of
            the mean and standard deviation using a Gauss-Newton fit of a Gaussian to a
            histogram constructed from the truncated samples.
        n_iter_gauss_newton (:obj:`int`): Number of Gauss-Newton iterations used during
            refinement. Only used if ``gauss_newton_refine`` is ``True``. Defaults to ``3``.
            I.e the algorithm will perform 3 Gauss-Newton iterations to refine the estimate
            of the mean and standard deviation.

    Returns:
        mean, std (:obj:`tuple`): Estimated noise mean and standard deviation.

        - If ``loop_outer_dims`` is ``True``, both are ``numpy.ndarray`` of shape ``(a, b)``.
        - If ``loop_outer_dims`` is ``False``, both are scalar floats.
    """
    return _estimate_white_noise(
        data,
        inital_guess,
        truncate,
        max_iterations,
        convergence_tol,
        loop_outer_dims,
        gauss_newton_refine,
        n_iter_gauss_newton,
    )
