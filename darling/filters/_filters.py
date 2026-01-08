import warnings

import numba
import numpy as np


def snr_threshold(
    data,
    mean,
    sigma,
    primary_threshold,
    secondary_threshold=None,
    copy=False,
    loop_outer_dims=True,
):
    """Threshold data based on a local signal-to-noise ratio (SNR) criterion.

    Each data bin is classified using a Gaussian noise model defined by the
    input mean and sigma. The signal-to-noise ratio (SNR) is defined as::

        SNR = (local_value - mean) / sigma

    Thresholding rules:

        SNR > primary_threshold
            Always kept.

        SNR <= secondary_threshold
            Always set to zero.

        secondary_threshold < SNR <= primary_threshold
            Kept only if at least one immediate neighbour has
            SNR > primary_threshold.
            Neighbourhood sizes are 3, 9 and 27 in 1D, 2D and 3D respectively.

    Dimensionality handling:

        When ``loop_outer_dims=True`` (default), the first two dimensions of
        the data array are treated as outer dimensions (detector rows and
        columns). Thresholding is applied independently and in parallel to
        each trailing block.

            Supported shapes:
                shape=(a, b, m)
                shape=(a, b, m, n)
                shape=(a, b, m, n, o)

            In this case, mean and sigma must be arrays of shape=(a, b).

        When ``loop_outer_dims=False``, thresholding is applied directly to
        the data array:

            Supported shapes:
                shape=(m,)
                shape=(m, n)
                shape=(m, n, o)

            In this case, mean and sigma must be scalars.

    Example:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import darling

        _, data, motors = darling.io.assets.mosaicity_scan()
        a, b, m, n = data.shape
        mean = 100 * np.ones((a, b))
        sigma = 10 * np.ones((a, b))
        primary_threshold = 3
        secondary_threshold = 1

        filtered_data = darling.filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        im = ax[0].imshow(data[20, 10, 0:31, 10:36], norm='log', cmap='jet')
        ax[0].text(1, 4, "Raw Data at pixel (20, 10)", fontsize=22, fontweight="bold")
        fig.colorbar(im, ax=ax[0], fraction=0.054, pad=0.04)
        im = ax[1].imshow(filtered_data[20, 10, 0:31, 10:36], norm='log', cmap='jet')
        ax[1].text(1, 4, "SNR Filtered Data at pixel (20, 10) ", fontsize=22, fontweight="bold")
        # annotate the  primary threshold
        ax[1].text(1, 6, f"Primary Threshold: SNR >= {primary_threshold}", fontsize=20)
        # annotate the secondary threshold
        ax[1].text(1, 8, f"Secondary Threshold: SNR >= {secondary_threshold} ", fontsize=20)
        fig.colorbar(im, ax=ax[1], fraction=0.054, pad=0.04)
        for a in ax.flatten():
            for spine in a.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/snr.png

    Args:
        data (:obj:`numpy.ndarray`):
            Input data array to threshold.
        mean (:obj:`int`, :obj:`float` or :obj:`numpy.ndarray`):
            Mean of the local Gaussian noise model.
        sigma (:obj:`int`, :obj:`float` or :obj:`numpy.ndarray`):
            Standard deviation of the local Gaussian noise model.
        primary_threshold (:obj:`float`):
            Primary SNR threshold. Must be greater than zero.
        secondary_threshold (:obj:`float`, optional):
            Secondary SNR threshold. Defaults to None, in which case only
            the primary threshold is applied and no neighbour check is
            performed. Must be less than the primary threshold.
        copy (:obj:`bool`, optional):
            Whether to return a thresholded copy of the data. Defaults to False.
        loop_outer_dims (:obj:`bool`, optional):
            Whether to loop over the outer dimensions (a, b). Defaults to True.

    Returns:
        :obj:`numpy.ndarray`:
            The thresholded data (may be a copy or a view depending on ``copy``).
    """

    _check_input_snr_threshold(
        data, mean, sigma, primary_threshold, secondary_threshold, copy, loop_outer_dims
    )

    if secondary_threshold is None:
        secondary_threshold = np.inf

    if copy:
        out = data.copy()
    else:
        out = data

    if loop_outer_dims is False:
        trailing_dims = data.ndim
        dummy = (1,) * trailing_dims
        return _threshold_snr_nd(
            out,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            dummy,
        )
    else:
        trailing_dims = data.ndim - 2
        dummy = (1,) * trailing_dims
        return _threshold_parallel_snr(
            out,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            dummy,
        )


@numba.njit(inline="always", cache=True)
def _snr(x, mean, sigma):
    """Signal-to-noise ratio.

    The signal-to-noise ratio is defined in the classical engineering sense as the ratio
    of the signal to the noise assuming a Gaussian noise model. Crudely simplified:

        returns --> infinite - x is purely signal
        returns --> 0.0 - x is purely noise

    """
    return (x - mean) / sigma


@numba.njit(cache=True)
def _threshold_snr_1d(data, mean, sigma, primary_threshold, secondary_threshold):
    m = data.shape[0]
    tmp = np.empty(data.shape, np.uint8)
    for i in range(m):
        pval = _snr(data[i], mean, sigma)
        if pval > primary_threshold:
            tmp[i] = 1
        elif pval > secondary_threshold:
            tmp[i] = 2
        else:
            data[i] = 0.0
            tmp[i] = 0

    if secondary_threshold != np.inf:
        for i in range(m):
            if tmp[i] != 2:
                continue

            i0 = 0 if i == 0 else -1
            i1 = 1 if i == m - 1 else 2
            has_neighbour_1 = False
            for di in range(i0, i1):
                if tmp[i + di] == 1:
                    has_neighbour_1 = True
                    break
            if not has_neighbour_1:
                data[i] = 0.0
    return data


@numba.njit(cache=True)
def _threshold_snr_2d(data, mean, sigma, primary_threshold, secondary_threshold):
    m, n = data.shape
    tmp = np.empty(data.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            pval = _snr(data[i, j], mean, sigma)
            if pval > primary_threshold:
                tmp[i, j] = 1
            elif pval > secondary_threshold:
                tmp[i, j] = 2
            else:
                data[i, j] = 0.0
                tmp[i, j] = 0

    if secondary_threshold != np.inf:
        for i in range(m):
            i0 = 0 if i == 0 else -1
            i1 = 1 if i == m - 1 else 2
            for j in range(n):
                if tmp[i, j] != 2:
                    continue

                j0 = 0 if j == 0 else -1
                j1 = 1 if j == n - 1 else 2
                has_neighbour_1 = False
                for di in range(i0, i1):
                    for dj in range(j0, j1):
                        if tmp[i + di, j + dj] == 1:
                            has_neighbour_1 = True
                            break
                    if has_neighbour_1:
                        break
                if not has_neighbour_1:
                    data[i, j] = 0.0
    return data


@numba.njit(cache=True)
def _threshold_snr_3d(
    data,
    mean,
    sigma,
    primary_threshold,
    secondary_threshold,
):
    m, n, o = data.shape
    tmp = np.empty(data.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            for k in range(o):
                pval = _snr(data[i, j, k], mean, sigma)
                if pval > primary_threshold:
                    tmp[i, j, k] = 1
                elif pval > secondary_threshold:
                    tmp[i, j, k] = 2
                else:
                    data[i, j, k] = 0.0
                    tmp[i, j, k] = 0

    if secondary_threshold != np.inf:
        for i in range(m):
            i0 = 0 if i == 0 else -1
            i1 = 1 if i == m - 1 else 2
            for j in range(n):
                j0 = 0 if j == 0 else -1
                j1 = 1 if j == n - 1 else 2
                for k in range(o):
                    if tmp[i, j, k] != 2:
                        continue

                    k0 = 0 if k == 0 else -1
                    k1 = 1 if k == o - 1 else 2
                    has_neighbour_1 = False
                    for di in range(i0, i1):
                        for dj in range(j0, j1):
                            for dk in range(k0, k1):
                                if tmp[i + di, j + dj, k + dk] == 1:
                                    has_neighbour_1 = True
                                    break
                            if has_neighbour_1:
                                break
                        if has_neighbour_1:
                            break
                    if not has_neighbour_1:
                        data[i, j, k] = 0.0
    return data


@numba.njit(cache=True)
def _threshold_snr_nd(data, mean, sigma, primary_threshold, secondary_threshold, dummy):
    dim = len(dummy)
    if dim == 1:
        return _threshold_snr_1d(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
        )
    elif dim == 2:
        return _threshold_snr_2d(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
        )
    elif dim == 3:
        return _threshold_snr_3d(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
        )


@numba.njit(parallel=True, cache=True)
def _threshold_parallel_snr(
    data, mean, sigma, primary_threshold, secondary_threshold, trailing_dims
):
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = _threshold_snr_nd(
                data[i, j],
                mean[i, j],
                sigma[i, j],
                primary_threshold,
                secondary_threshold,
                trailing_dims,
            )
    return data


def gaussian_filter(
    data,
    sigma,
    truncate=4.0,
    radius=None,
    axis=None,
    copy=False,
    loop_outer_dims=True,
):
    """Apply a separable, intensity-preserving Gaussian filter to a data array.

    The filter is implemented as a separable convolution with 1D Gaussian kernels
    along the requested trailing axes. The 1D Gaussian kernel is defined as::

        g(x) = exp(-x**2 / (2 * sigma**2))

    A discrete kernel of width ``2 * radius + 1`` is constructed for each filtered
    axis and normalised to sum to 1 (intensity preserving). Convolution is performed
    with zero padding at the boundaries (samples outside the array are treated as 0).

    NOTE: For integer data, the filtering will not be intensity preserving due to casting.
    Such data will raise a UserWarning.

    Dimensionality handling:

        When ``loop_outer_dims=True`` (default), the first two dimensions of the data
        array are treated as outer dimensions (detector rows and columns). The filter
        is applied independently and in parallel to each trailing block.

            Supported shapes:
                shape=(a, b, m)
                shape=(a, b, m, n)
                shape=(a, b, m, n, o)

        When ``loop_outer_dims=False``, the filter is applied directly to the data array:

            Supported shapes:
                shape=(m,)
                shape=(m, n)
                shape=(m, n, o)

    The ``axis`` argument always refers to the trailing dimensions only (never the
    outer (a, b) dimensions). For example, for ``shape=(a, b, m, n, o)``, the valid
    axis values refer to (m, n, o), i.e. ``axis=(0, 1, 2)`` or equivalently
    ``axis=(-3, -2, -1)``.

    Kernel size:

        If ``radius`` is None, the radius is inferred per filtered axis as::

            radius = round(truncate * sigma)

        The resulting kernel width is ``2 * radius + 1``. Larger sigma or radius
        produces stronger smoothing.

    Example:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import darling

        _, data, motors = darling.io.assets.mosaicity_scan()

        filtered_data = darling.filters.gaussian_filter(
            data.astype(np.float64),
            sigma = (1.0, 1.0),
            axis = (0, 1),
            radius = (3, 5),
            copy=True,
            loop_outer_dims=True,
        )

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        im = ax[0].imshow(data[20, 10, 3:27, 10:36], vmin=95, vmax=140, cmap="viridis")
        ax[0].text(1, 4, "Raw Data at pixel (20, 10)", fontsize=22, fontweight="bold")
        fig.colorbar(im, ax=ax[0], fraction=0.0425, pad=0.02)
        im = ax[1].imshow(filtered_data[20, 10, 3:27, 10:36], vmin=95, vmax=140, cmap="viridis")
        ax[1].text(1, 4, "Filtered Data at pixel (20, 10) ", fontsize=22, fontweight="bold")
        fig.colorbar(im, ax=ax[1], fraction=0.0425, pad=0.02)
        for a in ax.flatten():
            for spine in a.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/gaussian_filter.png

    Args:
        data (:obj:`numpy.ndarray`):
            The data to filter. Must be one of shape=(a, b, m, n, o),
            shape=(a, b, m, n) or shape=(a, b, m) when ``loop_outer_dims`` is True.
            When ``loop_outer_dims`` is False, the data array must be one of
            shape=(m, n, o), shape=(m, n) or shape=(m,).
        sigma (:obj:`int`, :obj:`float` or :obj:`tuple`):
            Standard deviation(s) of the Gaussian kernel. If a scalar, the same
            sigma is used for all trailing dimensions. If a tuple, it must have
            the same length as ``axis`` (when axis is provided), and each filtered
            axis uses its corresponding sigma value.
        truncate (:obj:`float`, optional):
            Truncation in units of sigma used to infer the radius when ``radius`` is
            not provided. Defaults to 4.0.
        radius (:obj:`int` or :obj:`tuple`, optional):
            Kernel radius/radii. If provided, the kernel width is ``2 * radius + 1``.
            If a tuple, it must have the same length as ``axis`` (when axis is
            provided). Defaults to None (infer from ``truncate`` and ``sigma``).
        axis (:obj:`int` or :obj:`tuple`, optional):
            Trailing axis/axes to filter. Defaults to None, in which case all trailing
            dimensions are filtered. Axis values may be negative and are interpreted
            relative to the trailing block (not the full array when ``loop_outer_dims``
            is True).
        copy (:obj:`bool`, optional):
            Whether to return a filtered copy of the data. Defaults to False, in which
            case the input array is modified in place.
        loop_outer_dims (:obj:`bool`, optional):
            Whether to loop over the outer dimensions (a, b) and filter each trailing
            block independently. Defaults to True.

    Returns:
        :obj:`numpy.ndarray`:
            The filtered data (may be a copy or a view depending on ``copy``).

    Raises:
        UserWarning: If the input data is of integer type. For integer data, the filtering will not be intensity
        preserving due to float -> integer casting.
    """
    if loop_outer_dims is False:
        trailing_dims = data.ndim
        _check_inputs_gaussian_filter(
            data, sigma, truncate, radius, axis, loop_outer_dims
        )
        trailing_dims = data.ndim
        kernels, axis = _get_kernels_and_axis(
            data, sigma, truncate, radius, axis, trailing_dims
        )
        return _gaussian_filter_single_pixel(data, kernels, axis, copy)
    else:
        _check_inputs_gaussian_filter(
            data, sigma, truncate, radius, axis, loop_outer_dims
        )
        trailing_dims = data.ndim - 2
        kernels, axis = _get_kernels_and_axis(
            data, sigma, truncate, radius, axis, trailing_dims
        )
        return _gaussian_filter_parallel(data, kernels, axis, copy)


def _get_kernels_and_axis(data, sigma, truncate, radius, axis, trailing_dims=None):
    if trailing_dims is None:
        trailing_dims = data.ndim - 2

    if axis is None:
        axis = tuple(range(trailing_dims))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    if isinstance(sigma, (int, float)):
        sigma = [float(sigma)] * trailing_dims

    if radius is None:
        radius = np.array(
            [
                int(np.round(float(truncate) * float(sigma[i])))
                for i in range(len(sigma))
            ]
        )
    elif isinstance(radius, (int, float)):
        radius = np.array([radius] * trailing_dims)
    elif isinstance(radius, (list, tuple, np.ndarray)):
        radius = np.array([int(r) for r in radius])

    for i in range(len(sigma)):
        if sigma[i] <= 0:
            raise ValueError(f"Positive sigma parameter is required but got {sigma}")

    kernels = [_gaussian_kernel_1d(sigma[i], radius[i]) for i in range(len(sigma))]

    # kernels is always of len equal to the number of trailing dimensions
    # the len of the kernel tuple can then be used to infer the data dim
    # at compile-time for numba, which otherwise struggles with dynamic
    # shapes.
    if len(kernels) != trailing_dims:
        # this has to be dummy numpy array for branch prediction to not fail..
        _kernels = [np.array([0.0], dtype=np.float64)] * trailing_dims
        for ax, k in zip(axis, kernels):
            _kernels[ax] = k
        kernels = tuple(_kernels)
    else:
        kernels = tuple(kernels)

    return kernels, axis


@numba.njit(cache=True)
def _gaussian_filter_single_pixel(
    data,
    kernels,
    axis,
    copy,
):
    if copy:
        out = data.copy()
    else:
        out = data
    return _convolve_nd(out, kernels, axis)


@numba.njit(parallel=True, cache=True)
def _gaussian_filter_parallel(data, kernels, axis, copy):
    if copy:
        out = data.copy()
    else:
        out = data
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = _convolve_nd(out[i, j], kernels, axis)
    return out


def _gaussian_kernel_1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


@numba.njit(cache=True)
def _convolve_1d(data, kernel):
    n = data.size
    m = kernel.size
    out = np.zeros((n,), dtype=data.dtype)
    half = m // 2
    for i in range(n):
        s = 0.0
        for j in range(m):
            k = i + j - half
            if 0 <= k < n:
                s += data[k] * kernel[j]
        out[i] = s
    return out


@numba.njit(cache=True)
def _convolve_nd(data, kernels, axis):
    if len(kernels) == 1:
        data = _convolve_1d(data, kernels[axis[0]])
    elif len(kernels) == 2:
        data = _convolve_2d(data, kernels, axis)
    elif len(kernels) == 3:
        data = _convolve_3d(data, kernels, axis)
    else:
        raise ValueError("Data must be 1D, 2D or 3D")
    return data


@numba.njit(cache=True)
def _convolve_2d(data, kernels, axis):
    m, n = data.shape
    for a in axis:
        if a == 0 or a == -2:
            for i in range(m):
                data[i, :] = _convolve_1d(data[i, :], kernels[0])
        if a == 1 or a == -1:
            for j in range(n):
                data[:, j] = _convolve_1d(data[:, j], kernels[1])
    return data


@numba.njit(cache=True)
def _convolve_3d(data, kernels, axis):
    m, n, o = data.shape

    for a in axis:
        if a == 0 or a == -3:
            for j in range(n):
                for k in range(o):
                    data[:, j, k] = _convolve_1d(data[:, j, k], kernels[0])
        if a == 1 or a == -2:
            for i in range(m):
                for k in range(o):
                    data[i, :, k] = _convolve_1d(data[i, :, k], kernels[1])
        if a == 2 or a == -1:
            for i in range(m):
                for j in range(n):
                    data[i, j, :] = _convolve_1d(data[i, j, :], kernels[2])
    return data


def _check_input_snr_threshold(
    data,
    mean,
    sigma,
    primary_threshold,
    secondary_threshold,
    copy,
    loop_outer_dims,
):
    if not isinstance(copy, bool):
        raise ValueError(f"Copy must be a boolean but got {type(copy)}")

    if not isinstance(loop_outer_dims, bool):
        raise ValueError(
            f"Loop outer dimensions must be a boolean but got {type(loop_outer_dims)}"
        )

    if not isinstance(data, np.ndarray):
        raise ValueError(f"Data must be a numpy array but got {type(data)}")

    if not isinstance(primary_threshold, (float, int)):
        raise ValueError(
            f"Primary threshold must be a float or int when loop_outer_dims is False but got {type(primary_threshold)}"
        )

    if primary_threshold < 0.0:
        raise ValueError("Primary threshold must be greater than or equal to 0.0")

    if secondary_threshold is not None and not isinstance(
        secondary_threshold, (float, int)
    ):
        raise ValueError(
            f"Secondary threshold must be a float or int or None but got {type(secondary_threshold)}"
        )

    if secondary_threshold is not None and secondary_threshold < 0.0:
        raise ValueError("Secondary threshold must be greater than or equal to 0.0")

    if secondary_threshold is not None and secondary_threshold > primary_threshold:
        raise ValueError(
            "Secondary threshold must be less than or equal to the primary threshold"
        )

    if loop_outer_dims is False:
        if data.ndim not in [1, 2, 3]:
            raise ValueError(
                f"Data must be 1D, 2D or 3D array when loop_outer_dims is False but got {data.ndim}D array"
            )

        if not isinstance(mean, (int, float)):
            raise ValueError(
                f"Mean must be an int or float when loop_outer_dims is False but got {type(mean)}"
            )
        if not isinstance(sigma, (int, float)):
            raise ValueError(
                f"Sigma must be an int or float when loop_outer_dims is False but got {type(sigma)}"
            )
    else:
        if data.ndim not in [3, 4, 5]:
            raise ValueError(
                f"Data must be 3D, 4D or 5D array when loop_outer_dims is True but got {data.ndim}D array"
            )
        if not isinstance(mean, np.ndarray):
            raise ValueError(
                f"Mean must be a numpy array when loop_outer_dims is True but got {type(mean)}"
            )
        if not isinstance(sigma, np.ndarray):
            raise ValueError(
                f"Sigma must be a numpy array when loop_outer_dims is True but got {type(sigma)}"
            )
        if mean.shape != sigma.shape:
            raise ValueError(
                f"Mean and sigma must have the same shape when loop_outer_dims is True but got {mean.shape} != {sigma.shape}"
            )
        if mean.shape[0] != data.shape[0] or mean.shape[1] != data.shape[1]:
            raise ValueError(
                f"Mean and and data must have the same shape for the first two dimensions when loop_outer_dims is True but got {mean.shape[0]} != {data.shape[0]} or {mean.shape[1]} != {data.shape[1]}"
            )
        if sigma.shape[0] != data.shape[0] or sigma.shape[1] != data.shape[1]:
            raise ValueError(
                f"Sigma and data must have the same shape for the first two dimensions when loop_outer_dims is True but got {sigma.shape[0]} != {data.shape[0]} or {sigma.shape[1]} != {data.shape[1]}"
            )


def _check_inputs_gaussian_filter(
    data, sigma, truncate, radius, axis, loop_outer_dims, supress_warnings=False
):
    if axis is not None and isinstance(sigma, tuple) and not isinstance(axis, tuple):
        raise ValueError("You passed a tuple of sigma but axis is not a tuple")

    if (
        axis is not None
        and isinstance(sigma, tuple)
        and isinstance(axis, tuple)
        and len(sigma) != len(axis)
    ):
        raise ValueError(
            f"The lengths of the sigma and axis tuples do not match {len(sigma)} != {len(axis)}"
        )

    if loop_outer_dims is False and data.ndim not in [1, 2, 3]:
        raise ValueError(
            f"loop_outer_dims was set to False but the provided data is ndim = {data.ndim}, expected 1D 2D or 3D array"
        )

    if loop_outer_dims is True and data.ndim not in [3, 4, 5]:
        raise ValueError(
            f"loop_outer_dims was set to True but the provided data is ndim = {data.ndim}, expected 3D, 4Dor 5D array"
        )

    if not isinstance(data, np.ndarray):
        raise ValueError(f"Data must be a numpy array but got {type(data)}")

    if not isinstance(sigma, (int, float, list, tuple, np.ndarray)):
        raise ValueError(
            f"Sigma must be a number, list, tuple or numpy array but got {type(sigma)}"
        )

    if not isinstance(truncate, (int, float)):
        raise ValueError(f"Truncate must be a number but got {type(truncate)}")

    if truncate <= 0:
        raise ValueError(f"Positive truncate parameter is required but got {truncate}")

    if isinstance(radius, tuple) and not isinstance(sigma, tuple):
        raise ValueError("You passed a tuple of radius but sigma is not a tuple")

    if radius is not None and isinstance(radius, int):
        radius = (radius,)
    elif radius is not None and isinstance(radius, tuple):
        if axis is not None and len(radius) != len(axis):
            raise ValueError(
                f"Radius tuple must have length equal to the number of axes but got {len(radius)} != {len(axis)}"
            )
        for r in radius:
            if r <= 0:
                raise ValueError(f"Positive radius parameter is required but got {r}")
    elif radius is not None:
        raise ValueError(f"Radius must be an int or tuple but got {type(radius)}")

    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"Positive sigma parameter is required but got {sigma}")
    elif isinstance(sigma, tuple):
        if axis is not None and len(sigma) != len(axis):
            raise ValueError(
                f"Sigma tuple must have length equal to the number of axes but got {len(sigma)} != {len(axis)}"
            )

        for s in sigma:
            if s <= 0:
                raise ValueError(
                    f"Positive sigma parameter is required but got {sigma}"
                )
    else:
        raise ValueError(f"Sigma must be an int, float or tuple but got {type(sigma)}")

    trailing_dims = data.ndim - 2 if loop_outer_dims is True else data.ndim

    feasible_axis_values = [i for i in range(trailing_dims)] + [
        -i for i in range(1, trailing_dims + 1)
    ]

    if axis is not None and isinstance(axis, int):
        if axis not in feasible_axis_values:
            raise ValueError(
                f"Axis must be one of {feasible_axis_values} but got {axis}"
            )
    elif axis is not None and isinstance(axis, tuple):
        for a in axis:
            if not isinstance(a, int):
                raise ValueError(f"Axis must be an integer but got {type(a)}")
            if a not in feasible_axis_values:
                raise ValueError(
                    f"Axis must be one of {feasible_axis_values} but got {a}"
                )
    elif axis is not None:
        raise ValueError(f"Axis must be an integer or tuple but got {type(axis)}")
    else:
        pass

    if not supress_warnings:
        if np.issubdtype(data.dtype, np.integer):
            warnings.warn(
                f"data is of integer type {data.dtype}, hence Gaussian filtering will not be intensity preserving due to float -> integer casting.",
                UserWarning,
                stacklevel=2,
            )


def _get_filter_parameters_from_dict(data, loop_outer_dims, filter):
    """Get kernels and axis from a dictionary of filter parameters.

    This is a helper function for darling.properties.peaks() to get the
    kernels and axis from a dictionary of filter parameters with the
    end goal of applying smoothing on the fly, before peak detection.

    Args:
        data (:obj:`numpy.ndarray`): Input data array.
        loop_outer_dims (:obj:`bool`): Whether to loop over the outer dimensions.
        filter (:obj:`dict`): Dictionary of filter parameters.
            Must contain the keys "sigma", "truncate", "radius", "axis".
    Returns:
        :obj:`tuple`: Tuple of kernels and axis and threshold.
    """
    filter_dict = {
        "sigma": None,
        "truncate": 4.0,
        "radius": None,
        "axis": None,
        "threshold": None,
    }
    for key in filter_dict:
        if filter is not None and key in filter:
            filter_dict[key] = filter[key]

    threshold = (
        np.min(data) if filter_dict["threshold"] is None else filter_dict["threshold"]
    )

    if filter_dict["sigma"] is None:
        return None, None, threshold, filter_dict

    sigma = filter_dict["sigma"]
    truncate = filter_dict["truncate"]
    radius = filter_dict["radius"]
    axis = filter_dict["axis"]

    trailing_dims = data.ndim - 2 if loop_outer_dims is True else data.ndim

    _check_inputs_gaussian_filter(
        data, sigma, truncate, radius, axis, loop_outer_dims, supress_warnings=True
    )

    kernels, axis = _get_kernels_and_axis(
        data, sigma, truncate, radius, axis, trailing_dims
    )

    return kernels, axis, threshold, filter_dict


if __name__ == "__main__":
    pass
