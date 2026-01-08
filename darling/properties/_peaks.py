"""This module defines a discrete peak searcher algorithm for 1D, 2D and 3D data.

This module enables segmentation of data into domains of interest based on a local maxima climber algorithm.
Features from these domains are extracted using arithmetic operations. The algorithm is implemented in
numba and can be run in parallel across 2D grids of data-blocks, where each data block (i,j) is a 1D, 2D or 3D array.

The features extracted from the segmented domains are stored in a feature table which is a dictionary
with keys corresponding to the features extracted from the segmented domains.

Specifically, for each located peak, the following features are currently supported:

    - **sum_intensity**: Sum of the intensity values in the segmented domain.
    - **max_intensity**: Integrated intensity in the segmented domain.
    - **number_of_pixels**: Number of pixels in the segmented domain.
    - **maxima_axis_0**: Coordinate of the pixel with the highest intensity along axis=0 in the the segmented domain.
    - **maxima_axis_1**: Coordinate of the pixel with the highest intensity along axis=1 in the the segmented domain.
    - **maxima_axis_2**: Coordinate of the pixel with the highest intensity along axis=2 in the the segmented domain.
    - **mean_axis_0**: Arithmetric Mean coordinate along axis=0 in the the segmented domain.
    - **mean_axis_1**: Arithmetric Mean coordinate along axis=1 in the the segmented domain.
    - **mean_axis_2**: Arithmetric Mean coordinate along axis=2 in the the segmented domain.
    - **variance_axis_0**: Variance of the coordinates along axis=0 in the the segmented domain.
    - **variance_axis_1**: Variance of the coordinates along axis=1 in the the segmented domain.
    - **variance_axis_2**: Variance of the coordinates along axis=2 in the the segmented domain.
    - **variance_axis_0_axis_1**: Covariance of the coordinates along axis=0 and axis=1 in the the segmented domain.
    - **variance_axis_0_axis_2**: Covariance of the coordinates along axis=0 and axis=2 in the the segmented domain.
    - **variance_axis_1_axis_2**: Covariance of the coordinates along axis=1 and axis=2 in the the segmented domain.

When motor coordinate arrays are provided, units are inferred from the coordinate arrays. This supports
non-equidistant grids. When no motor coordinate arrays are provided, the features are extracted from the
data array itself using integer (np.indices) coordinates to proxy the motor coordinates.
"""

import warnings

import numba
import numpy as np

from ..filters._filters import (
    _convolve_1d,
    _convolve_2d,
    _convolve_3d,
    _get_filter_parameters_from_dict,
)
from ._constants import _FEATURE_MAPPING
from ._peakmap import PeakMap


def gaussian_mixture(data, k=3, coordinates=None):
    # deprecated early January 2025, OK to remove early February 2025
    # gaussian_mixture was a very bad name for this function, it should have been called peaks() instead
    # it does not do a gaussian mixture model, it does a peak search and feature extraction.
    # an actual implementation of a generic gaussian mixture model is on my wishlist for darling...
    # Axel Henningsson 2025-12-29
    warnings.warn(
        "darling.properties.gaussian_mixture() is deprecated, use darling.properties.peaks() instead (which now also supports higher and lower dimensional data arrays)",
        DeprecationWarning,
        stacklevel=2,
    )
    return peaks(data, k, coordinates)


def peaks(
    data,
    k=3,
    coordinates=None,
    loop_outer_dims=True,
    filter=None,
):
    """Find peaks/domains on a 2D grid of 1D, 2D or 3D data-blocks and extract features from them.

    A peak/domain is defined as a connected domain of pixels surrounding a local maximum.

    NOTE: Zero valued pixels are treated as void. When thresholding is used,
        data values less than the threshold are set to zero and are thus treated as void.

    For a data array of shape (a, b, m, n, o), each primary pixel (i, j) contains a
    (m, n, o) sub-array that is analyzed as a 3D image. Local maxima are identified
    within this sub-array, and segmentation is performed to assign a label to
    each secondary pixel using a local maxima climber algorithm. For algorithm details, see
    darling.properties.local_max_label().

    Each segmented region is treated as a Gaussian, with mean and covariance
    extracted for each label. Additionally, a set of features is computed
    for each label. for algorithm details, see darling.properties.extract_features().

    This function implements support for data arrays of the following shapes:
    - (a, b, m, n, o) for 3D data (when loop_outer_dims is True)
    - (a, b, m, n) for 2D data (when loop_outer_dims is True)
    - (a, b, m) for 1D data (when loop_outer_dims is True)
    - (m ,n, o) for single pixel data (when loop_outer_dims is False)
    - (m, n) for single pixel data (when loop_outer_dims is False)
    - (m,) for single pixel data (when loop_outer_dims is False)


    Specifically, for each located peak, the following features are extracted:

    - **sum_intensity**: Sum of the intensity values in the segmented domain.
    - **max_intensity**: Integrated intensity in the segmented domain.
    - **number_of_pixels**: Number of pixels in the segmented domain.
    - **maxima_axis_0**: Coordinate of the pixel with the highest intensity along axis=0 in the the segmented domain.
    - **maxima_axis_1**: Coordinate of the pixel with the highest intensity along axis=1 in the the segmented domain.
    - **maxima_axis_2**: Coordinate of the pixel with the highest intensity along axis=2 in the the segmented domain.
    - **mean_axis_0**: Arithmetric Mean coordinate along axis=0 in the the segmented domain.
    - **mean_axis_1**: Arithmetric Mean coordinate along axis=1 in the the segmented domain.
    - **mean_axis_2**: Arithmetric Mean coordinate along axis=2 in the the segmented domain.
    - **variance_axis_0**: Variance of the coordinates along axis=0 in the the segmented domain.
    - **variance_axis_1**: Variance of the coordinates along axis=1 in the the segmented domain.
    - **variance_axis_2**: Variance of the coordinates along axis=2 in the the segmented domain.
    - **variance_axis_0_axis_1**: Covariance of the coordinates along axis=0 and axis=1 in the the segmented domain.
    - **variance_axis_0_axis_2**: Covariance of the coordinates along axis=0 and axis=2 in the the segmented domain.
    - **variance_axis_1_axis_2**: Covariance of the coordinates along axis=1 and axis=2 in the the segmented domain.

    Example:

    .. code-block:: python

        import darling
        import matplotlib.pyplot as plt

        # import a small data set from assets known to
        # comprise crystalline domains
        _, data, coordinates = darling.io.assets.domains()

        # Segment domains / peaks and extract features from them
        peakmap = darling.properties.peaks(data, k=3, coordinates=coordinates)

        # this is a dict like structure that can be accessed like this:
        sum_intensity_second_strongest_peak = peakmap["sum_intensity"][..., 1]

        # plot the mean in the first motor direction for the strongest peak
        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(peakmap["mean_axis_0"][..., 0], cmap="plasma")
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/domains.png

    Args:
        data (:obj:`numpy array`): Array of shape (a, b, m, n, o) for 3D data, (a, b, m, n) for 2D data,
            (a, b, m) for 1D data. Alternatively, when loop_outer_dims is False, the data array can be of
            shape (m, n, o) for 3D data, (m, n) for 2D data, (m,) for 1D data.
        k (:obj:`int`): The number of peaks to keep. Defaults to 3.
            this means that the k strongest peaks, with respect to integrated intensity, will be kept.
            The remaining peaks will be ignored.
        coordinates (:obj:`numpy array`): array specifying the coordinates of
            shape=(3, m, n, ...) for 3D data, shape=(2, m, n) for 2D data, shape=(1, m) for 1D data.
        loop_outer_dims (:obj:`bool`): whether to loop over the outer dimensions of the data array.
            Defaults to True in which case the data array must be at least 3D and at most 5D.
            If False, the data array must be 1D, 2D or 3D (i.e data for a single pixel)
        filter (:obj:`dict`): A dictionary of filter parameters defining a Gaussian filter and thresholding.
            Defaults to None, in which case no filter is applied. The filter dictionary must contain the keys:
            - **sigma**: Standard deviation of the Gaussian filter. Defaults to None.
            - **truncate**: Truncation of the Gaussian filter. Defaults to 4.0.
            - **radius**: Radius of the Gaussian filter. Defaults to None.
            - **axis**: Axis of the Gaussian filter. Defaults to None.
            - **threshold**: Threshold for peak detection. Defaults to np.min(data).
            When both sigma and threshold are set, thresholding is applied before the Gaussian filtering.
            Values less than or equal the threshold are not considered during peak / domain segmentation.

    Returns:
        :obj:`darling.properties.PeakMap` : a wrapper for the per peak features as a dictionary with keys corresponding to:
            - **sum_intensity**: Sum of the intensity values in the segmented domain.
            - **max_intensity**: Integrated intensity in the segmented domain.
            - **number_of_pixels**: Number of pixels in the segmented domain.
            - **maxima_axis_0**: Coordinate of the pixel with the highest intensity along axis=0 in the the segmented domain.
            - **maxima_axis_1**: Coordinate of the pixel with the highest intensity along axis=1 in the the segmented domain.
            - **maxima_axis_2**: Coordinate of the pixel with the highest intensity along axis=2 in the the segmented domain.
            - **mean_axis_0**: Arithmetric Mean coordinate along axis=0 in the the segmented domain.
            - **mean_axis_1**: Arithmetric Mean coordinate along axis=1 in the the segmented domain.
            - **mean_axis_2**: Arithmetric Mean coordinate along axis=2 in the the segmented domain.
            - **variance_axis_0**: Variance of the coordinates along axis=0 in the the segmented domain.
            - **variance_axis_1**: Variance of the coordinates along axis=1 in the the segmented domain.
            - **variance_axis_2**: Variance of the coordinates along axis=2 in the the segmented domain.
            - **variance_axis_0_axis_1**: Covariance of the coordinates along axis=0 and axis=1 in the the segmented domain.
            - **variance_axis_0_axis_2**: Covariance of the coordinates along axis=0 and axis=2 in the the segmented domain.
            - **variance_axis_1_axis_2**: Covariance of the coordinates along axis=1 and axis=2 in the the segmented domain.

            For loop_outer_dims is True, these are fields across the 2D outer dimensions of the data array such that
            features["max_intensity"][i, j, k] is the integrated intensity in the segmented domain for pixel (i, j) and peak number k.
            features["number_of_pixels"][i, j, k] is the number of pixels in the segmented domain for pixel (i, j) and peak number k.
            ...etc....

            The PeakMap object provides additional convenience methods for manipulating the features table, such as sorting.
            See the PeakMap class for more details.

    """
    kernels, axis, threshold, filter_to_apply = _get_filter_parameters_from_dict(
        data, loop_outer_dims, filter
    )
    _check_inputs_peaks(data, k, coordinates, loop_outer_dims, filter_to_apply)
    if loop_outer_dims:
        trailing_dims = data.ndim - 2
        features_array = _peaksearch_parallel(
            data,
            k,
            coordinates,
            kernels,
            axis,
            threshold,
        )
    else:
        trailing_dims = data.ndim
        labeled_array, nlabels = local_max_label(data, loop_outer_dims, filter_to_apply)
        features_array = _extract_features(labeled_array, data, coordinates, nlabels, k)
    feature_table = _build_feature_table(features_array, trailing_dims)
    return PeakMap(feature_table)


def _check_inputs_peaks(data, k, coordinates, loop_outer_dims, filter):
    """Basic sanity checks on the inputs to the peaks function."""
    trailing_dims = data.ndim - 2 if loop_outer_dims is True else data.ndim

    if filter is not None and not isinstance(filter, dict):
        raise ValueError(f"filter must be a dictionary but got {type(filter)}")
    if filter is not None and "sigma" not in filter:
        raise ValueError(f"filter must contain a 'sigma' key but got {filter.keys()}")
    if filter is not None and "truncate" not in filter:
        raise ValueError(
            f"filter must contain a 'truncate' key but got {filter.keys()}"
        )
    if filter is not None and "radius" not in filter:
        raise ValueError(f"filter must contain a 'radius' key but got {filter.keys()}")
    if filter is not None and "axis" not in filter:
        raise ValueError(f"filter must contain a 'axis' key but got {filter.keys()}")

    if k < 0:
        raise ValueError(f"k must be larger than 0 but got {k}")
    if loop_outer_dims is True and data.ndim not in [3, 4, 5]:
        raise ValueError(
            f"data must be 3D, 4D or 5D when loop_outer_dims is True but got {data.ndim}D array"
        )
    if loop_outer_dims is False and data.ndim not in [1, 2, 3]:
        raise ValueError(
            f"data must be 1D, 2D or 3D when loop_outer_dims is False but got {data.ndim}D array"
        )
    if coordinates is not None and not isinstance(coordinates, np.ndarray):
        raise ValueError(
            f"coordinates must be a numpy array but got {type(coordinates)}"
        )
    if coordinates is not None and coordinates.shape[0] != trailing_dims:
        raise ValueError(
            f"coordinates.shape[0] must be {trailing_dims} for such data but got {coordinates.ndim}"
        )
    if coordinates is not None and coordinates.ndim != trailing_dims + 1:
        raise ValueError(
            f"coordinates must have {trailing_dims + 1} dimensions for such data but got {coordinates.ndim} dimensions"
        )
    if loop_outer_dims is True and data.shape[2:] != coordinates.shape[1:]:
        raise ValueError(
            f"missmatch between data blocks and coordinates: data.shape[2:]={data.shape[2:]} != coordinates.shape[1:]={coordinates.shape[1:]}"
        )
    if loop_outer_dims is False and data.shape != coordinates.shape[1:]:
        raise ValueError(
            f"missmatch between data blocks and coordinates: data.shape={data.shape} != coordinates.shape[1:]={coordinates.shape[1:]}"
        )


def label_sparse(data):
    # deprecated early January 2025, OK to remove early February 2025
    warnings.warn(
        "label_sparse is deprecated, use local_max_label instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return local_max_label(data)


def local_max_label(data, loop_outer_dims=True, filter=None):
    """Assigns pixels in a 1D, 2D or 3D image to the closest local maxima.

    The algorithm proceeds as follows:

    1. For a given pixel, find the highest-valued neighbor.
    2. Move the pixel to this neighbor:

        a. If the neighbor is already labeled, propagate the label back to the pixel.
        b. If the pixel is a local maximum, assign it a new label.
        c. Otherwise, repeat step 1 until a label is assigned.

    This process ensures that each pixel is assigned to the nearest local maximum
    through a gradient ascent type climb.

    To illustrate how the local maxclimber algorithm can separate overlapping gaussians
    we can consider the following example:

    NOTE: Zero valued pixels are treated as void. When thresholding is used,
        data values less than the threshold are set to zero and are thus treated as void.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import darling

        # Create a synthetic image with 9 gaussians with a lot of overlap
        rng = np.random.default_rng(42)
        x, y = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
        img = np.zeros((512, 512))
        for i in range(127, 385, 127):
            for j in range(127, 385, 127):
                img += np.exp(-((x - i) ** 2 + (y - j) ** 2) / (2 * rng.uniform(31, 61) ** 2))

        # Label the image following the local max climber algorithm, since we are only labeling
        # a single pixel, we set loop_outer_dims to False. (To label a grid of pixels, set loop_outer_dims to True.)
        labeled_array, nfeatures = darling.properties.local_max_label(img, loop_outer_dims=False)

        # The segmented image shows how the local maxclimber algorithm has segmented the image
        # into 9 regions splitting the overlapping gaussians.
        fig, ax = plt.subplots(1, 2, figsize=(14,7))
        im = ax[0].imshow(img)
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        im = ax[1].imshow(labeled_array, cmap='tab20')
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[0].set_title("Original image")
        ax[1].set_title("Labeled image")
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/labeloverlap.png

    Args:
        data (:obj:`numpy.ndarray`): a 1D, 2D or 3D data map to process.
        loop_outer_dims (:obj:`bool`): whether to loop over the outer dimensions of the data array.
            Defaults to True, in which case the data array must be 1D, 2D or 3D.
            If True, the data array must be at least 3D and at most 5D.
        filter (:obj:`dict`): A dictionary of filter parameters defining a Gaussian filter and thresholding.
            Defaults to None, in which case no filter is applied. The filter dictionary must contain the keys:
            - **sigma**: Standard deviation of the Gaussian filter. Defaults to None.
            - **truncate**: Truncation of the Gaussian filter. Defaults to 4.0.
            - **radius**: Radius of the Gaussian filter. Defaults to None.
            - **axis**: Axis of the Gaussian filter. Defaults to None.
            - **threshold**: Threshold for peak detection. Defaults to np.min(data).
            When both sigma and threshold are set, thresholding is applied before the Gaussian filtering.
            Values less than or equal the threshold are not considered during peak / domain segmentation.

    Returns:
        labeled_array (:obj:`numpy.ndarray`): same shape as the input ``data`` with integer labels for each pixel.
        number_of_labels (:obj:`int`): the number of labels assigned to the data map
    """

    kernels, axis, threshold, _ = _get_filter_parameters_from_dict(
        data, loop_outer_dims, filter
    )

    if loop_outer_dims:
        trailing_dims = data.ndim - 2
        if trailing_dims == 1:
            return _local_max_label_1D_parallel(data, kernels, axis, threshold)
        elif trailing_dims == 2:
            return _local_max_label_2D_parallel(data, kernels, axis, threshold)
        elif trailing_dims == 3:
            return _local_max_label_3D_parallel(data, kernels, axis, threshold)
        else:
            raise ValueError(
                f"When loop_outer_dims is True, trailing dimensions of data must be 1D, 2D or 3D but found {trailing_dims} trailing dimensions"
            )
    else:
        trailing_dims = data.ndim
        labeled_array, tmpi, tmpj, tmpk = _allocate_climber_arrays(
            data.shape, np.prod(data.shape)
        )
        if trailing_dims == 1:
            return _local_max_label_1D(
                data, labeled_array, tmpi, kernels, axis, threshold
            )
        elif trailing_dims == 2:
            return _local_max_label_2D(
                data, labeled_array, tmpi, tmpj, kernels, axis, threshold
            )
        elif trailing_dims == 3:
            return _local_max_label_3D(
                data, labeled_array, tmpi, tmpj, tmpk, kernels, axis, threshold
            )
        else:
            raise ValueError(
                f"When loop_outer_dims is False, total dimensions of data must be 1D, 2D or 3D but found {trailing_dims} dimensions"
            )


def extract_features(labeled_array, data, k, coordinates=None):
    """Extract features from a 1D, 2D or 3D labeled array.

    Args:
        labeled_array (:obj:`numpy.ndarray`): Label array with shape (m,n,o) for 3D data, (m,n) for
            2D data, (m,) for 1D data
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (m,n,o) for 3D data,
            (m,n) for 2D data, (m,) for 1D data
        k (:obj:`int`): number of segmented domains to keep features for, the first k peaks with highest
            integrated intensity will be kept, remaining peaks are ignored. Defaults to 3.
        coordinates (:obj:`tuple`): A tuple with the coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0, 1, 2 of the data array.
            and to have shapes (1, m, n), (2, m, n) or (3, m, n, o) for 1D, 2D or 3D data respectively.
            Defaults to None in which case integer indices are assumed to prevail.
            (i.e motors of the axis are np.arange(m), np.arange(n), np.arange(o))

    Returns:
        :obj:`darling.properties.PeakMap` : a wrapper for the per peak features as a dictionary with keys corresponding to:
            - **sum_intensity**: Sum of the intensity values in the segmented domain.
            - **max_intensity**: Integrated intensity in the segmented domain.
            - **number_of_pixels**: Number of pixels in the segmented domain.
            - **maxima_axis_0**: Coordinate of the pixel with the highest intensity along axis=0 in the the segmented domain.
            - **maxima_axis_1**: Coordinate of the pixel with the highest intensity along axis=1 in the the segmented domain.
            - **maxima_axis_2**: Coordinate of the pixel with the highest intensity along axis=2 in the the segmented domain.
            - **mean_axis_0**: Arithmetric Mean coordinate along axis=0 in the the segmented domain.
            - **mean_axis_1**: Arithmetric Mean coordinate along axis=1 in the the segmented domain.
            - **mean_axis_2**: Arithmetric Mean coordinate along axis=2 in the the segmented domain.
            - **variance_axis_0**: Variance of the coordinates along axis=0 in the the segmented domain.
            - **variance_axis_1**: Variance of the coordinates along axis=1 in the the segmented domain.
            - **variance_axis_2**: Variance of the coordinates along axis=2 in the the segmented domain.
            - **variance_axis_0_axis_1**: Covariance of the coordinates along axis=0 and axis=1 in the the segmented domain.
            - **variance_axis_0_axis_2**: Covariance of the coordinates along axis=0 and axis=2 in the the segmented domain.
            - **variance_axis_1_axis_2**: Covariance of the coordinates along axis=1 and axis=2 in the the segmented domain.

            For loop_outer_dims is True, these are fields across the 2D outer dimensions of the data array such that
            features["max_intensity"][i, j, k] is the integrated intensity in the segmented domain for pixel (i, j) and peak number k.
            features["number_of_pixels"][i, j, k] is the number of pixels in the segmented domain for pixel (i, j) and peak number k.
            ...etc....

            The PeakMap object provides additional convenience methods for manipulating the features table, such as sorting.
            See the PeakMap class for more details.
    """
    coordinates = np.indices(data.shape) if coordinates is None else coordinates
    nlabels = np.max(labeled_array)
    features_array = _extract_features(labeled_array, data, coordinates, nlabels, k)
    feature_table = _build_feature_table(
        features_array, trailing_dims=labeled_array.ndim
    )
    return PeakMap(feature_table)


def _peaksearch_parallel(data, k, coordinates, kernels, axis, threshold):
    """Parallel wrapper for peaksearch for 1D, 2D or 3D data. See these functions for docs and algorithm details."""
    trailing_dims = data.ndim - 2
    if trailing_dims == 1:
        return _peaksearch_parallel_1D(data, coordinates, k, kernels, axis, threshold)
    elif trailing_dims == 2:
        return _peaksearch_parallel_2D(data, coordinates, k, kernels, axis, threshold)
    elif trailing_dims == 3:
        return _peaksearch_parallel_3D(data, coordinates, k, kernels, axis, threshold)
    else:
        raise ValueError(
            f"Trailing dimensions of data array are expected to be 1D, 2D or 3D but found {trailing_dims} trailing dimensions with total shape of data.shape={data.shape}"
        )


def _build_feature_table(features_array, trailing_dims):
    """Build a feature table (dictionary of numpy arrays) from a features array (raw numpy array).

    Args:
        features_array (:obj:`numpy.ndarray`): The nd array of features of shape (a, b, 15, k) where
        a is detector rows, b is detector columns, 15 is the number of features and k is the number
        of local-max labeled peak features.
        trailing_dims (:obj:`int`): the number of trailing dimensions of the data array. (1, 2 or 3)

    Returns:
        feature_table (:obj:`dict` of `numpy.ndarray`): A dictionary with keys corresponding to
            the _FEATURE_MAPPING keys. These are local-max labeled peak features.
    """
    feature_table = {}
    feature_mapping = _FEATURE_MAPPING[trailing_dims]
    for key in feature_mapping:
        feature_table[key] = features_array[..., feature_mapping[key], :]
    return feature_table


@numba.njit(cache=True)
def _allocate_climber_arrays_per_thread(data_shape, max_iterations, nthreads):
    labeled_array = np.zeros((nthreads, *data_shape), dtype=np.uint16)
    tmpi = np.zeros((nthreads, max_iterations), dtype=np.uint16)
    tmpj = np.zeros((nthreads, max_iterations), dtype=np.uint16)
    tmpk = np.zeros((nthreads, max_iterations), dtype=np.uint16)
    return labeled_array, tmpi, tmpj, tmpk


@numba.njit(cache=True)
def _allocate_climber_arrays(data_shape, max_iterations):
    labeled_array = np.zeros(data_shape, dtype=np.uint16)
    tmpi = np.zeros((max_iterations,), dtype=np.uint16)
    tmpj = np.zeros((max_iterations,), dtype=np.uint16)
    tmpk = np.zeros((max_iterations,), dtype=np.uint16)
    return labeled_array, tmpi, tmpj, tmpk


@numba.njit(parallel=True)
def _local_max_label_1D_parallel(data, kernels, axis, threshold):
    """Parallel wrapper for local_max_label_1D for 1D data. See this function for docs and algorithm details."""
    a, b, m = data.shape
    labels = np.empty((a, b, m), dtype=np.uint16)
    nlabels = np.empty((a, b), dtype=np.int32)

    nthreads = numba.get_num_threads()
    labeled_array, tmpi, _, _ = _allocate_climber_arrays_per_thread((m,), m, nthreads)

    # TODO: apply the smoothing + thresholding on the fly

    for p in numba.prange(a * b):
        t = numba.get_thread_id()
        i = p // b
        j = p % b

        labeled_array[t].fill(0)

        labels[i, j], nlabels[i, j] = _local_max_label_1D(
            data[i, j],
            labeled_array[t],
            tmpi[t],
            kernels,
            axis,
            threshold,
        )

    return labels, nlabels


@numba.njit(parallel=True)
def _local_max_label_2D_parallel(data, kernels, axis, threshold):
    """Parallel wrapper for local_max_label_2D for 2D data. See this function for docs and algorithm details."""
    a, b, m, n = data.shape
    labels = np.empty((a, b, m, n), dtype=np.uint16)
    nlabels = np.empty((a, b), dtype=np.int32)

    nthreads = numba.get_num_threads()
    labeled_array, tmpi, tmpj, _ = _allocate_climber_arrays_per_thread(
        (m, n), m * n, nthreads
    )

    for p in numba.prange(a * b):
        t = numba.get_thread_id()
        i = p // b
        j = p % b

        labeled_array[t].fill(0)

        labels[i, j], nlabels[i, j] = _local_max_label_2D(
            data[i, j],
            labeled_array[t],
            tmpi[t],
            tmpj[t],
            kernels,
            axis,
            threshold,
        )

    return labels, nlabels


@numba.njit(parallel=True)
def _local_max_label_3D_parallel(data, kernels, axis, threshold):
    """Parallel wrapper for local_max_label_3D for 3D data. See this function for docs and algorithm details."""
    a, b, m, n, o = data.shape
    labels = np.empty((a, b, m, n, o), dtype=np.uint16)
    nlabels = np.empty((a, b), dtype=np.int32)

    nthreads = numba.get_num_threads()
    labeled_array, tmpi, tmpj, tmpk = _allocate_climber_arrays_per_thread(
        (m, n, o), m * n * o, nthreads
    )

    # TODO: apply the smoothing + thresholding on the fly

    for p in numba.prange(a * b):
        t = numba.get_thread_id()
        i = p // b
        j = p % b

        labeled_array[t].fill(0)

        labels[i, j], nlabels[i, j] = _local_max_label_3D(
            data[i, j],
            labeled_array[t],
            tmpi[t],
            tmpj[t],
            tmpk[t],
            kernels,
            axis,
            threshold,
        )

    return labels, nlabels


@numba.njit(cache=True)
def _local_max_label_1D(data_block, labeled_array, tmpi, kernels, axis, threshold):
    """Assigns pixels in a 1D image to the closest local maxima.

    The algorithm proceeds as follows:

    1. For a given pixel, find the highest-valued neighbor.
    2. Move the pixel to this neighbor:

        a. If the neighbor is already labeled, propagate the label back to the pixel.
        b. If the pixel is a local maximum, assign it a new label.
        c. Otherwise, repeat step 1 until a label is assigned.

    This process ensures that each pixel is assigned to the nearest local maximum
    through a gradient ascent type climb.

    NOTE: data values of exactly zero are treated as void. When thresholding is used,
        data values less than the threshold are set to zero and are thus treated as void.

    Args:
        data (:obj:`numpy.ndarray`): a 3D data map to process. shape=(m,)
        labeled_array (:obj:`numpy.ndarray`): shape=(m,n) with integer labels for each pixel.
        tmpi (:obj:`numpy.ndarray`): shape=(m,) with integer indices of the pixels in the data map.
        tmpj (:obj:`numpy.ndarray`): shape=(m,) with integer indices of the pixels in the data map.
        tmpk (:obj:`numpy.ndarray`): shape=(m,) with integer indices of the pixels in the data map.
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied.
        threshold (:obj:`float`): Data values less than the threshold are set to zero before peak detection.

    Returns:
        labeled_array (:obj:`numpy.ndarray`): shape=(m,) with integer labels for each pixel.
        number_of_labels (:obj:`int`): the number of labels assigned to the data map. This is the number peaks found.
    """

    data = _apply_filters_1D(data_block, kernels, axis, threshold)
    m = data.shape[0]
    max_iterations = m
    label = 1

    # The current label to propagate is stored in label_to_propagate.
    # when a pixel reaches a labeled pixel or a local maxima, the label is propagated
    # back to the climbing pixel through the path stored in tmpi.
    # thus, for each climb many pixels will be labeled.
    label_to_propagate = -1

    for ii in range(0, m):
        # if the pixel is already labeled or has zero intensity, skip it
        if labeled_array[ii] > 0 or data[ii] == 0:
            continue

        # while climbing is True, the current pixel is climbing to a local maxima
        # it will stop once it reaches a labeled pixel or a local maxima.
        climbing = True
        iterations = 0
        i = ii  # current location of the climbing pixel is stored in i

        # we limit the number of climb moves of the pixel to m
        # i.e. the pixel can move to any pixel in the image.
        while climbing and iterations < max_iterations:
            # max_val is the intensity of the most intense neighbor of the pixel
            # or the intensity of the pixel itself if it is a local maxima.
            max_val = data[i]
            tmpi[iterations] = i

            # max_i is the location of the most intense neighbor of the pixel
            max_i = i

            # Now we are ready to move the pixel to the most intense neighbor
            # of the pixel. We search through all neighbors of the pixel and
            # find the most intense one.
            for di in [-1, 0, 1]:
                # skip the current pixel itself
                if di == 0:
                    continue

                # if the neighbor is outside the image, skip it
                if i + di < 0 or i + di >= m:
                    continue

                # if the neighbor is more intense than the current max_val
                # we update max_val and the location of the most intense neighbor
                if data[i + di] > max_val:
                    max_val = data[i + di]
                    max_i = i + di

            # if the most intense neighbor is already labeled, we propagate the label
            # back to the climbing pixel thourgh the path stored in tmpi and stop the climbing.
            if labeled_array[max_i] != 0:
                label_to_propagate = labeled_array[max_i]
                for ki in range(iterations + 1):  # backpropagate the label
                    labeled_array[tmpi[ki]] = label_to_propagate
                climbing = False  # time to stop climbing

            # if the pixel was more intense than its neighbors, it is a local maxima
            # and we assign a new label to the pixel and propagate it back to the climbing
            # pixel through the path stored in tmpi.
            elif max_i == i:
                labeled_array[max_i] = label
                label_to_propagate = labeled_array[max_i]
                label += 1
                for ki in range(iterations + 1):  # backpropagate the label
                    labeled_array[tmpi[ki]] = label_to_propagate
                climbing = False  # time to stop climbing

            # Here we have simply found a more intense neighbor that is unlabeled
            # the climb moves to this neighbor and continues.
            else:
                i = max_i
                iterations += 1

    return labeled_array, label - 1


@numba.njit(cache=True)
def _local_max_label_2D(
    data_block, labeled_array, tmpi, tmpj, kernels, axis, threshold
):
    """Assigns pixels in a 2D image to the closest local maxima.

    The algorithm proceeds as follows:

    1. For a given pixel, find the highest-valued neighbor.
    2. Move the pixel to this neighbor:

        a. If the neighbor is already labeled, propagate the label back to the pixel.
        b. If the pixel is a local maximum, assign it a new label.
        c. Otherwise, repeat step 1 until a label is assigned.

    This process ensures that each pixel is assigned to the nearest local maximum
    through a gradient ascent type climb.

    NOTE: data values of exactly zero are treated as void. When thresholding is used,
        data values less than the threshold are set to zero and are thus treated as void.

    Args:
        data (:obj:`numpy.ndarray`): a 3D data map to process. shape=(m,n)
        labeled_array (:obj:`numpy.ndarray`): shape=(m,n) with integer labels for each pixel.
        tmpi (:obj:`numpy.ndarray`): shape=(m*n, ) with integer indices of the pixels in the data map.
        tmpj (:obj:`numpy.ndarray`): shape=(m*n,) with integer indices of the pixels in the data map.
        tmpk (:obj:`numpy.ndarray`): shape=(m*n,) with integer indices of the pixels in the data map.
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied.
        threshold (:obj:`float`): Data values less than the threshold are set to zero before peak detection.

    Returns:
        labeled_array (:obj:`numpy.ndarray`): shape=(m,n) with integer labels for each pixel.
        number_of_labels (:obj:`int`): the number of labels assigned to the data map. This is the number peaks found.
    """

    data = _apply_filters_2D(data_block, kernels, axis, threshold)
    m, n = data.shape
    max_iterations = m * n
    label = 1

    # The current label to propagate is stored in label_to_propagate.
    # when a pixel reaches a labeled pixel or a local maxima, the label is propagated
    # back to the climbing pixel through the path stored in tmpi and tmpj.
    # thus, for each climb many pixels will be labeled.
    label_to_propagate = -1

    for ii in range(0, m):
        for jj in range(0, n):
            # if the pixel is already labeled or has zero intensity, skip it
            if labeled_array[ii, jj] > 0 or data[ii, jj] == 0:
                continue

            # while climbing is True, the current pixel is climbing to a local maxima
            # it will stop once it reaches a labeled pixel or a local maxima.
            climbing = True
            iterations = 0
            i, j = (
                ii,
                jj,
            )  # current location of the climbing pixel is stored in i and j

            # we limit the number of climb moves of the pixel to m*n
            # i.e. the pixel can move to any pixel in the image.
            while climbing and iterations < max_iterations:
                # max_val is the intensity of the most intense neighbor of the pixel
                # or the intensity of the pixel itself if it is a local maxima.
                max_val = data[i, j]
                tmpi[iterations] = i
                tmpj[iterations] = j

                # max_i and max_j are the location of the most intense neighbor of the pixel
                max_i, max_j = i, j

                # Now we are ready to move the pixel to the most intense neighbor
                # of the pixel. We search through all neighbors of the pixel and
                # find the most intense one.
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # skip the current pixel itself
                        if di == 0 and dj == 0:
                            continue

                        # if the neighbor is outside the image, skip it
                        if i + di < 0 or i + di >= m or j + dj < 0 or j + dj >= n:
                            continue

                        # if the neighbor is more intense than the current max_val
                        # we update max_val and the location of the most intense neighbor
                        if data[i + di, j + dj] > max_val:
                            max_val = data[i + di, j + dj]
                            max_i, max_j = i + di, j + dj

                # if the most intense neighbor is already labeled, we propagate the label
                # back to the climbing pixel thourgh the path stored in tmpi and tmpj
                # and stop the climbing.
                if labeled_array[max_i, max_j] != 0:
                    label_to_propagate = labeled_array[max_i, max_j]
                    for ki in range(iterations + 1):  # backpropagate the label
                        labeled_array[tmpi[ki], tmpj[ki]] = label_to_propagate
                    climbing = False  # time to stop climbing

                # if the pixel was more intense than its neighbors, it is a local maxima
                # and we assign a new label to the pixel and propagate it back to the climbing
                # pixel through the path stored in tmpi and tmpj.
                elif max_i == i and max_j == j:
                    labeled_array[max_i, max_j] = label
                    label_to_propagate = labeled_array[max_i, max_j]
                    label += 1
                    for ki in range(iterations + 1):  # backpropagate the label
                        labeled_array[tmpi[ki], tmpj[ki]] = label_to_propagate
                    climbing = False  # time to stop climbing

                # Here we have simply found a more intense neighbor that is unlabeled
                # the climb moves to this neighbor and continues.
                else:
                    i, j = max_i, max_j
                    iterations += 1

    return labeled_array, label - 1


@numba.njit(cache=True)
def _local_max_label_3D(
    data_block, labeled_array, tmpi, tmpj, tmpk, kernels, axis, threshold
):
    """Assigns pixels in a 3D image to the closest local maxima.

    The algorithm proceeds as follows:

    1. For a given pixel, find the highest-valued neighbor.
    2. Move the pixel to this neighbor:

        a. If the neighbor is already labeled, propagate the label back to the pixel.
        b. If the pixel is a local maximum, assign it a new label.
        c. Otherwise, repeat step 1 until a label is assigned.

    This process ensures that each pixel is assigned to the nearest local maximum
    through a gradient ascent type climb.

    NOTE: data values of exactly zero are treated as void. When thresholding is used,
        data values less than the threshold are set to zero and are thus treated as void.

    Args:
        data (:obj:`numpy.ndarray`): a 3D data map to process. shape=(m,n,o)
        labeled_array (:obj:`numpy.ndarray`): shape=(m,n,o) with integer labels for each pixel.
        tmpi (:obj:`numpy.ndarray`): shape=(m*n*o, ) with integer indices of the pixels in the data map.
        tmpj (:obj:`numpy.ndarray`): shape=(m*n*o,) with integer indices of the pixels in the data map.
        tmpk (:obj:`numpy.ndarray`): shape=(m*n*o,) with integer indices of the pixels in the data map.
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied.
        threshold (:obj:`float`): Data values less than the threshold are set to zero before peak detection.

    Returns:
        labeled_array (:obj:`numpy.ndarray`): shape=(m,n,o) with integer labels for each pixel.
        number_of_labels (:obj:`int`): the number of labels assigned to the data map. This is the number peaks found.
    """

    data = _apply_filters_3D(data_block, kernels, axis, threshold)

    m, n, o = data.shape
    max_iterations = m * n * o
    label = 1

    # # The labeled array is to be filled with labels for each pixel in the data map.
    # # The labels are integers starting from 1 and increasing for each new local maxima.
    # labeled_array = np.zeros((m, n, o), dtype=np.uint16)

    # # tmpi and tmpj are temporary arrays to store the path of the climbing pixel
    # # in the data map. The path is stored in these arrays until the pixel reaches
    # # a labeled pixel or a local maxima.
    # tmpi = np.zeros((max_iterations,), dtype=np.uint16)
    # tmpj = np.zeros((max_iterations,), dtype=np.uint16)
    # tmpk = np.zeros((max_iterations,), dtype=np.uint16)

    # The current label to propagate is stored in label_to_propagate.
    # when a pixel reaches a labeled pixel or a local maxima, the label is propagated
    # back to the climbing pixel through the path stored in tmpi and tmpj.
    # thus, for each climb many pixels will be labeled.
    label_to_propagate = -1

    for ii in range(0, m):
        for jj in range(0, n):
            for kk in range(0, o):
                # if the pixel is already labeled or has less than noise floor, skip it
                if labeled_array[ii, jj, kk] > 0 or data[ii, jj, kk] == 0:
                    continue

                # while climbing is True, the current pixel is climbing to a local maxima
                # it will stop once it reaches a labeled pixel or a local maxima.
                climbing = True
                iterations = 0
                i, j, k = (
                    ii,
                    jj,
                    kk,
                )  # current location of the climbing pixel is stored in i and j and k

                # we limit the number of climb moves of the pixel to m*n
                # i.e. the pixel can move to any pixel in the image.
                while climbing and iterations < max_iterations:
                    # max_val is the intensity of the most intense neighbor of the pixel
                    # or the intensity of the pixel itself if it is a local maxima.
                    max_val = data[i, j, k]
                    tmpi[iterations] = i
                    tmpj[iterations] = j
                    tmpk[iterations] = k

                    # max_i and max_j and max_k are the location of the most intense neighbor of the pixel
                    max_i, max_j, max_k = i, j, k

                    # Now we are ready to move the pixel to the most intense neighbor
                    # of the pixel. We search through all neighbors of the pixel and
                    # find the most intense one.
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                # skip the current pixel itself
                                if di == 0 and dj == 0 and dk == 0:
                                    continue

                                # if the neighbor is outside the image, skip it
                                if (
                                    i + di < 0
                                    or i + di >= m
                                    or j + dj < 0
                                    or j + dj >= n
                                    or k + dk < 0
                                    or k + dk >= o
                                ):
                                    continue

                                # if the neighbor is more intense than the current max_val
                                # we update max_val and the location of the most intense neighbor
                                if data[i + di, j + dj, k + dk] > max_val:
                                    max_val = data[i + di, j + dj, k + dk]
                                    max_i, max_j, max_k = i + di, j + dj, k + dk

                    # if the most intense neighbor is already labeled, we propagate the label
                    # back to the climbing pixel thourgh the path stored in tmpi and tmpj and
                    # and tmpk and stop the climbing.
                    if labeled_array[max_i, max_j, max_k] != 0:
                        label_to_propagate = labeled_array[max_i, max_j, max_k]
                        for ki in range(iterations + 1):  # backpropagate the label
                            labeled_array[tmpi[ki], tmpj[ki], tmpk[ki]] = (
                                label_to_propagate
                            )
                        climbing = False  # time to stop climbing

                    # if the pixel was more intense than its neighbors, it is a local maxima
                    # and we assign a new label to the pixel and propagate it back to the climbing
                    # pixel through the path stored in tmpi and tmpj and tmpk.
                    elif max_i == i and max_j == j and max_k == k:
                        labeled_array[max_i, max_j, max_k] = label
                        label_to_propagate = labeled_array[max_i, max_j, max_k]
                        label += 1
                        for ki in range(iterations + 1):  # backpropagate the label
                            labeled_array[tmpi[ki], tmpj[ki], tmpk[ki]] = (
                                label_to_propagate
                            )
                        climbing = False  # time to stop climbing

                    # Here we have simply found a more intense neighbor that is unlabeled
                    # the climb moves to this neighbor and continues.
                    else:
                        i, j, k = max_i, max_j, max_k
                        iterations += 1

    return labeled_array, label - 1


@numba.njit(cache=True)
def _apply_filters_1D(data_block, kernels, axis, threshold):
    """Applies a Gaussian filter and thresholding to a 1D data map.

    Args:
        data_block (:obj:`numpy.ndarray`): a 1D data map to process. shape=(m,)
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied.
        threshold (:obj:`float`): Data values less than the threshold are set to zero before peak detection.

    Returns:
        data (:obj:`numpy.ndarray`): a 1D data map with the filtered data. shape=(m,) as dtype=np.float64
    """
    data = data_block.astype(np.float64)
    m = data.shape[0]
    for ii in range(0, m):
        if data[ii] <= threshold:
            data[ii] = 0
    if kernels is not None:
        data = _convolve_1d(data, kernels, axis)
    return data


@numba.njit(cache=True)
def _apply_filters_2D(data_block, kernels, axis, threshold):
    """Applies a Gaussian filter and thresholding to a 2D data map.

    Args:
        data_block (:obj:`numpy.ndarray`): a 2D data map to process. shape=(m,n)
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied.
        threshold (:obj:`float`): Data values less than the threshold are set to zero before peak detection.

    Returns:
        data (:obj:`numpy.ndarray`): a 2D data map with the filtered data. shape=(m,n) as dtype=np.float64
    """
    data = data_block.astype(np.float64)
    m, n = data.shape
    for ii in range(0, m):
        for jj in range(0, n):
            if data[ii, jj] <= threshold:
                data[ii, jj] = 0
    if kernels is not None:
        data = _convolve_2d(data, kernels, axis)
    return data


@numba.njit(cache=True)
def _apply_filters_3D(data_block, kernels, axis, threshold):
    """Applies a Gaussian filter and thresholding to a 3D data map.

    Args:
        data_block (:obj:`numpy.ndarray`): a 3D data map to process. shape=(m,n,o)
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied.
        threshold (:obj:`float`): Data values less than the threshold are set to zero before peak detection.

    Returns:
        data (:obj:`numpy.ndarray`): a 3D data map with the filtered data. shape=(m,n,o) as dtype=np.float64
    """
    data = data_block.astype(np.float64)
    m, n, o = data.shape
    for ii in range(0, m):
        for jj in range(0, n):
            for kk in range(0, o):
                if data[ii, jj, kk] <= threshold:
                    data[ii, jj, kk] = 0
    if kernels is not None:
        data = _convolve_3d(data, kernels, axis)
    return data


@numba.njit(cache=True)
def _extract_features(labeled_array, data, coordinates, nlabels, k):
    """Wrapper for _extract_features_3D to extract features from a labeled array."""
    ndim = labeled_array.ndim
    if ndim == 1:
        return _extract_features_1D(labeled_array, data, coordinates, nlabels, k)
    elif ndim == 2:
        return _extract_features_2D(labeled_array, data, coordinates, nlabels, k)
    elif ndim == 3:
        return _extract_features_3D(labeled_array, data, coordinates, nlabels, k)
    else:
        raise ValueError(
            f"Labeled array must be 1D, 2D or 3D but found {ndim} dimensions"
        )


@numba.njit(cache=True)
def _extract_features_1D(labeled_array, data, coordinates, nlabels, k):
    """Extract features from a 1D labeled array.

    Args:
        labeled_array (:obj:`numpy.ndarray`): Label array with ``shape=(m,)``.
        data (:obj:`numpy.ndarray`): The underlying intensity data array with ``shape=(m,)``.
        coordinates (:obj:`numpy.ndarray`): The coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0 of the data array.
            and to have shape=(1, m). (indexing = ij is assumed, see numpy.meshgrid docs for details)
        nlabels (:obj:`int`): number of labels in the labeled array.
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.

    Raises:
        ValueError: if labeled_array contains more than 65535 labels.

    Returns:
        'numpy array': a 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict.
    """

    # properties to extract live at the following indices in the feature table
    sum_intensity = 0
    max_pix_intensity = 1
    number_of_pixels = 2

    max_pix_axis_0 = 3
    # max_pix_axis_1 = 4, this is not used in the 1D case, resulting array will be all zeros
    # max_pix_axis_2 = 5, this is not used in the 2D case, resulting array will be all zeros

    mean_axis_0 = 6
    # mean_axis_1 = 7, this is not used in the 1D case, resulting array will be all zeros
    # mean_axis_2 = 8, this is not used in the 2D case, resulting array will be all zeros

    var_axis_0 = 9
    # var_axis_1 = 10, this is not used in the 1D case, resulting array will be all zeros
    # var_axis_2 = 11, this is not used in the 2D case, resulting array will be all zeros

    # var_axis_0_axis_1 = 12, this is not used in the 1D case, resulting array will be all zeros
    # var_axis_0_axis_2 = 13, this is not used in the 2D case, resulting array will be all zeros
    # var_axis_1_axis_2 = 14, this is not used in the 2D case, resulting array will be all zeros

    num_props = 15

    m = data.shape[0]

    if nlabels > 65535:
        raise ValueError("Found more features than can be assigned with uint16")

    feature_table = np.zeros((num_props, np.maximum(k, nlabels)), dtype=float)

    for ii in range(0, m):
        if labeled_array[ii] == 0:
            continue

        x = coordinates[0, ii]

        index = labeled_array[ii] - 1

        feature_table[sum_intensity, index] += data[ii]
        feature_table[number_of_pixels, index] += 1

        feature_table[mean_axis_0, index] += x * data[ii]

        if data[ii] > feature_table[max_pix_intensity, index]:
            feature_table[max_pix_axis_0, index] = x
            feature_table[max_pix_intensity, index] = data[ii]

    nnz_mask = feature_table[sum_intensity, :] > 1
    divider = feature_table[sum_intensity, nnz_mask]
    feature_table[mean_axis_0, nnz_mask] /= divider

    for ii in range(0, m):
        if labeled_array[ii] == 0:
            continue

        x = coordinates[0, ii]
        index = labeled_array[ii] - 1

        if feature_table[sum_intensity, index] != 0:
            diff_axis_0 = x - feature_table[mean_axis_0, index]

            feature_table[var_axis_0, index] += data[ii] * diff_axis_0 * diff_axis_0

    #  (George R. Price, Ann. Hum. Genet., Lond, pp485-490, Extension of covariance selection mathematics, 1972).
    # https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
    # assuming counts are observations
    unbiased_divider = divider - 1
    feature_table[var_axis_0, nnz_mask] /= unbiased_divider

    idx = np.argsort(-feature_table[sum_intensity], kind="quicksort")[0:k]
    return feature_table[:, idx]


@numba.njit(cache=True)
def _extract_features_2D(labeled_array, data, coordinates, nlabels, k):
    """Extract features from a 2D labeled array.

    Args:
        labeled_array (:obj:`numpy.ndarray`): Label array with ``shape=(m, n)``.
        data (:obj:`numpy.ndarray`): The underlying intensity data array with ``shape=(m, n)``.
        coordinates (:obj:`numpy.ndarray`): The coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0, 1 of the data array.
            and to have shape=(2, m, n). (indexing = ij is assumed, see numpy.meshgrid docs for details)
        nlabels (:obj:`int`): number of labels in the labeled array.
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.

    Raises:
        ValueError: if labeled_array contains more than 65535 labels.

    Returns:
        'numpy array': a 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict.
    """

    # properties to extract live at the following indices in the feature table
    sum_intensity = 0
    max_pix_intensity = 1
    number_of_pixels = 2

    max_pix_axis_0 = 3
    max_pix_axis_1 = 4
    # max_pix_axis_2 = 5, this is not used in the 2D case, resulting array will be all zeros

    mean_axis_0 = 6
    mean_axis_1 = 7
    # mean_axis_2 = 8, this is not used in the 2D case, resulting array will be all zeros

    var_axis_0 = 9
    var_axis_1 = 10
    # var_axis_2 = 11, this is not used in the 2D case, resulting array will be all zeros

    var_axis_0_axis_1 = 12
    # var_axis_0_axis_2 = 13, this is not used in the 2D case, resulting array will be all zeros
    # var_axis_1_axis_2 = 14, this is not used in the 2D case, resulting array will be all zeros

    num_props = 15

    m, n = data.shape

    if nlabels > 65535:
        raise ValueError("Found more features than can be assigned with uint16")

    feature_table = np.zeros((num_props, np.maximum(k, nlabels)), dtype=float)

    for ii in range(0, m):
        for jj in range(0, n):
            if labeled_array[ii, jj] == 0:
                continue

            x = coordinates[0, ii, jj]
            y = coordinates[1, ii, jj]

            index = labeled_array[ii, jj] - 1

            feature_table[sum_intensity, index] += data[ii, jj]
            feature_table[number_of_pixels, index] += 1

            feature_table[mean_axis_0, index] += x * data[ii, jj]
            feature_table[mean_axis_1, index] += y * data[ii, jj]

            if data[ii, jj] > feature_table[max_pix_intensity, index]:
                feature_table[max_pix_axis_0, index] = x
                feature_table[max_pix_axis_1, index] = y
                feature_table[max_pix_intensity, index] = data[ii, jj]

    nnz_mask = feature_table[sum_intensity, :] > 1
    divider = feature_table[sum_intensity, nnz_mask]
    feature_table[mean_axis_0, nnz_mask] /= divider
    feature_table[mean_axis_1, nnz_mask] /= divider

    for ii in range(0, m):
        for jj in range(0, n):
            if labeled_array[ii, jj] == 0:
                continue

            x = coordinates[0, ii, jj]
            y = coordinates[1, ii, jj]

            index = labeled_array[ii, jj] - 1
            if feature_table[sum_intensity, index] != 0:
                diff_axis_0 = x - feature_table[mean_axis_0, index]
                diff_axis_1 = y - feature_table[mean_axis_1, index]

                feature_table[var_axis_0, index] += (
                    data[ii, jj] * diff_axis_0 * diff_axis_0
                )
                feature_table[var_axis_1, index] += (
                    data[ii, jj] * diff_axis_1 * diff_axis_1
                )
                feature_table[var_axis_0_axis_1, index] += (
                    data[ii, jj] * diff_axis_0 * diff_axis_1
                )

    #  (George R. Price, Ann. Hum. Genet., Lond, pp485-490, Extension of covariance selection mathematics, 1972).
    # https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
    # assuming counts are observations
    unbiased_divider = divider - 1
    feature_table[var_axis_0, nnz_mask] /= unbiased_divider
    feature_table[var_axis_1, nnz_mask] /= unbiased_divider
    feature_table[var_axis_0_axis_1, nnz_mask] /= unbiased_divider

    idx = np.argsort(-feature_table[sum_intensity], kind="quicksort")[0:k]
    return feature_table[:, idx]


@numba.njit(cache=True)
def _extract_features_3D(labeled_array, data, coordinates, nlabels, k):
    """Extract features from a 3D labeled array.

    Args:
        labeled_array (:obj:`numpy.ndarray`): Label array with ``shape=(m, n, o)``.
        data (:obj:`numpy.ndarray`): The underlying intensity data array with ``shape=(m, n, o)``.
        X, Y, Z (:obj:`numpy.ndarray`): The coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0, 1, 2 of the data array.
            and to have shape=(m, n, o). (indexing = ij is assumed, see numpy.meshgrid docs for details)
        nlabels (:obj:`int`): number of labels in the labeled array.
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.

    Raises:
        ValueError: if labeled_array contains more than 65535 labels.

    Returns:
        'numpy array': a 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict.
    """

    # properties to extract live at the following indices in the feature table
    sum_intensity = 0
    max_pix_intensity = 1
    number_of_pixels = 2

    max_pix_axis_0 = 3
    max_pix_axis_1 = 4
    max_pix_axis_2 = 5

    mean_axis_0 = 6
    mean_axis_1 = 7
    mean_axis_2 = 8

    var_axis_0 = 9
    var_axis_1 = 10
    var_axis_2 = 11

    var_axis_0_axis_1 = 12
    var_axis_0_axis_2 = 13
    var_axis_1_axis_2 = 14

    num_props = 15

    m, n, o = data.shape

    if nlabels > 65535:
        raise ValueError("Found more features than can be assigned with uint16")

    feature_table = np.zeros((num_props, np.maximum(k, nlabels)), dtype=float)

    for ii in range(0, m):
        for jj in range(0, n):
            for kk in range(0, o):
                if labeled_array[ii, jj, kk] == 0:
                    continue

                x = coordinates[0, ii, jj, kk]
                y = coordinates[1, ii, jj, kk]
                z = coordinates[2, ii, jj, kk]

                index = labeled_array[ii, jj, kk] - 1

                feature_table[sum_intensity, index] += data[ii, jj, kk]
                feature_table[number_of_pixels, index] += 1

                feature_table[mean_axis_0, index] += x * data[ii, jj, kk]
                feature_table[mean_axis_1, index] += y * data[ii, jj, kk]
                feature_table[mean_axis_2, index] += z * data[ii, jj, kk]

                if data[ii, jj, kk] > feature_table[max_pix_intensity, index]:
                    feature_table[max_pix_axis_0, index] = x
                    feature_table[max_pix_axis_1, index] = y
                    feature_table[max_pix_axis_2, index] = z
                    feature_table[max_pix_intensity, index] = data[ii, jj, kk]

    nnz_mask = feature_table[sum_intensity, :] > 1
    divider = feature_table[sum_intensity, nnz_mask]
    feature_table[mean_axis_0, nnz_mask] /= divider
    feature_table[mean_axis_1, nnz_mask] /= divider
    feature_table[mean_axis_2, nnz_mask] /= divider

    for ii in range(0, m):
        for jj in range(0, n):
            for kk in range(0, o):
                if labeled_array[ii, jj, kk] == 0:
                    continue

                x = coordinates[0, ii, jj, kk]
                y = coordinates[1, ii, jj, kk]
                z = coordinates[2, ii, jj, kk]

                index = labeled_array[ii, jj, kk] - 1
                if feature_table[sum_intensity, index] != 0:
                    diff_axis_0 = x - feature_table[mean_axis_0, index]
                    diff_axis_1 = y - feature_table[mean_axis_1, index]
                    diff_axis_2 = z - feature_table[mean_axis_2, index]

                    feature_table[var_axis_0, index] += (
                        data[ii, jj, kk] * diff_axis_0 * diff_axis_0
                    )
                    feature_table[var_axis_1, index] += (
                        data[ii, jj, kk] * diff_axis_1 * diff_axis_1
                    )
                    feature_table[var_axis_2, index] += (
                        data[ii, jj, kk] * diff_axis_2 * diff_axis_2
                    )
                    feature_table[var_axis_0_axis_1, index] += (
                        data[ii, jj, kk] * diff_axis_0 * diff_axis_1
                    )
                    feature_table[var_axis_0_axis_2, index] += (
                        data[ii, jj, kk] * diff_axis_0 * diff_axis_2
                    )
                    feature_table[var_axis_1_axis_2, index] += (
                        data[ii, jj, kk] * diff_axis_1 * diff_axis_2
                    )

    #  (George R. Price, Ann. Hum. Genet., Lond, pp485-490, Extension of covariance selection mathematics, 1972).
    # https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
    # assuming counts are observations
    unbiased_divider = divider - 1
    feature_table[var_axis_0, nnz_mask] /= unbiased_divider
    feature_table[var_axis_1, nnz_mask] /= unbiased_divider
    feature_table[var_axis_2, nnz_mask] /= unbiased_divider
    feature_table[var_axis_0_axis_1, nnz_mask] /= unbiased_divider
    feature_table[var_axis_0_axis_2, nnz_mask] /= unbiased_divider
    feature_table[var_axis_1_axis_2, nnz_mask] /= unbiased_divider

    idx = np.argsort(-feature_table[sum_intensity], kind="quicksort")[0:k]
    return feature_table[:, idx]


@numba.njit(parallel=True)
def _peaksearch_parallel_1D(data, coordinates, k, kernels, axis, threshold):
    """Parallel wrapper for local_max_label and extract_features for 1D data. See these functions for algorithm details.

    Args:
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (a, b, m) where
            a is the number of detector rows, b is the number of detector columns.
        coordinates (:obj:`numpy.ndarray`): The coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0 of the data array.
            and to have shape=(1, m). (indexing = ij is assumed, see numpy.meshgrid docs for details)
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels. Defaults to None,
            i.e no smoothing is applied prior to peak detection. When set, a Gaussian filter with the given
            kernels is applied prior to peak detection.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied. Defaults to None,
            i.e no axis is applied. When set, the data is filtered along the given axis.
        threshold (:obj:`float`): Threshold for peak detection. Defaults to None, i.e no thresholding.
            When set, data values less than the threshold are set to zero before peak detection. Thresholding
            is applied before the kernels are applied.

    Returns:
        features (:obj:`numpy.ndarray`): A 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict. shape=(a, b, 15, k)

    """
    a, b, m = data.shape
    features = np.zeros((a, b, 15, k), dtype=float)

    nthreads = numba.get_num_threads()
    labeled_array, tmpi, _, _ = _allocate_climber_arrays_per_thread((m,), m, nthreads)

    for p in numba.prange(a * b):
        t = numba.get_thread_id()
        i = p // b
        j = p % b

        t = numba.get_thread_id()
        labeled_array[t].fill(0)
        labels_1d, nlabels = _local_max_label_1D(
            data[i, j], labeled_array[t], tmpi[t], kernels, axis, threshold
        )
        features[i, j] = _extract_features_1D(
            labels_1d, data[i, j], coordinates, nlabels, k
        )
    return features


@numba.njit(parallel=True)
def _peaksearch_parallel_2D(data, coordinates, k, kernels, axis, threshold):
    """Parallel wrapper for local_max_label and extract_features for 2D data. See these functions for algorithm details.

    Args:
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (a, b, m, n) where
            a is the number of detector rows, b is the number of detector columns.
        coordinates (:obj:`numpy.ndarray`): The coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0 and 1 of the data array.
            and to have shape=(2, m, n). (indexing = ij is assumed, see numpy.meshgrid docs for details)
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels. Defaults to None,
            i.e no smoothing is applied prior to peak detection. When set, a Gaussian filter with the given
            kernels is applied prior to peak detection.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied. Defaults to None,
            i.e no axis is applied. When set, the data is filtered along the given axis.
        threshold (:obj:`float`): Threshold for peak detection. Defaults to None, i.e no thresholding.
            When set, data values less than the threshold are set to zero before peak detection. Thresholding
            is applied before the kernels are applied.

    Returns:
        features (:obj:`numpy.ndarray`): A 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict. shape=(a, b, 15, k)

    """
    a, b, m, n = data.shape
    features = np.zeros((a, b, 15, k), dtype=float)
    nthreads = numba.get_num_threads()
    labeled_array, tmpi, tmpj, _ = _allocate_climber_arrays_per_thread(
        (m, n), m * n, nthreads
    )

    for p in numba.prange(a * b):
        t = numba.get_thread_id()
        i = p // b
        j = p % b

        t = numba.get_thread_id()
        labeled_array[t].fill(0)
        labels_2d, nlabels = _local_max_label_2D(
            data[i, j], labeled_array[t], tmpi[t], tmpj[t], kernels, axis, threshold
        )
        features[i, j] = _extract_features_2D(
            labels_2d, data[i, j], coordinates, nlabels, k
        )
    return features


@numba.njit(parallel=True)
def _peaksearch_parallel_3D(data, coordinates, k, kernels, axis, threshold):
    """Parallel wrapper for local_max_label and extract_features for 3D data. See these functions for algorithm details.

    Args:
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (a, b, m, n, o) where
            a is the number of detector rows, b is the number of detector columns.
        coordinates (:obj:`numpy.ndarray`): The coordinates of the motor positions.
            The coordinates are expected to be in the same order as axis 0, 1, 2 of the data array.
            and to have shape=(3, m, n, o). (indexing = ij is assumed, see numpy.meshgrid docs for details)
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.
        kernels (:obj:`tuple` of `numpy.ndarray`): Gaussian kernels. Defaults to None,
            i.e no smoothing is applied prior to peak detection. When set, a Gaussian filter with the given
            kernels is applied prior to peak detection.
        axis (:obj:`tuple` of `int`): Axis of the data array along which the kernels are applied. Defaults to None,
            i.e no axis is applied. When set, the data is filtered along the given axis.
        threshold (:obj:`float`): Threshold for peak detection. Defaults to None, i.e no thresholding.
            When set, data values less than the threshold are set to zero before peak detection. Thresholding
            is applied before the kernels are applied.

    Returns:
        features (:obj:`numpy.ndarray`): A 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict. shape=(a, b, 15, k)

    """
    a, b, m, n, o = data.shape
    features = np.zeros((a, b, 15, k), dtype=float)
    nthreads = numba.get_num_threads()
    labeled_array, tmpi, tmpj, tmpk = _allocate_climber_arrays_per_thread(
        (m, n, o), m * n * o, nthreads
    )

    for p in numba.prange(a * b):
        t = numba.get_thread_id()
        i = p // b
        j = p % b

        t = numba.get_thread_id()
        labeled_array[t].fill(0)

        labels_3d, nlabels = _local_max_label_3D(
            data[i, j],
            labeled_array[t],
            tmpi[t],
            tmpj[t],
            tmpk[t],
            kernels,
            axis,
            threshold,
        )
        features[i, j] = _extract_features_3D(
            labels_3d, data[i, j], coordinates, nlabels, k
        )
    return features


if __name__ == "__main__":
    pass
