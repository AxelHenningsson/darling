"""This module defines a discrete peak searcher algorithm for 2D data.

This module enables segmentation of 2D data into domains of interest and allow for extraction
of features from these domains. The algorithm is based on a gradient ascent in the data map
moving in a manhattan-like fashion. The algorithm is implemented in numba and can be run in
parallel.

The final result is a gaussian mixture model of a grid of images. The features extracted
from the segmented domains are stored in a feature table. The feature table is a dictionary
with keys corresponding to the features extracted from the segmented domains. The feature
table can be converted to motor positions if motor positions are provided.

Specifically, for each located peak, the following features are extracted:

    - **sum_intensity**: Sum of the intensity values in the segmented domain.
    - **number_of_pixels**: Number of pixels in the segmented domain.
    - **mean_row**: Mean row position in the segmented domain.
    - **mean_col**: Mean column position in the segmented domain.
    - **var_row**: Variance of the row positions in the segmented domain.
    - **var_col**: Variance of the column positions in the segmented domain.
    - **var_row_col**: Covariance of the row and column positions in the segmented domain.
    - **max_pix_row**: Row position of the pixel with the highest intensity.
    - **max_pix_col**: Column position of the pixel with the highest intensity.
    - **max_pix_intensity**: Intensity of the pixel with the highest intensity.

Additionally, when motor coordinate arrays are provided, the following features are included:

    - **mean_motor1**: Mean motor position for the first motor.
    - **mean_motor2**: Mean motor position for the second motor.
    - **var_motor1**: Variance of the motor positions for the first motor.
    - **var_motor2**: Variance of the motor positions for the second motor.
    - **var_motor1_motor2**: Covariance of the motor positions for the first and second motor.
    - **max_pix_motor1**: Motor position for the first motor of the pixel with the highest intensity.
    - **max_pix_motor2**: Motor position for the second motor of the pixel with the highest intensity.
"""

import numba
import numpy as np

# Here we define the mapping between the features extracted from the segmented domains
# and the indices in the feature table. This is a static mapping which allows for the
# feature table computed in compiled code. The mapping is used to extract the features.
_FEATURE_MAPPING = {
    "sum_intensity": 0,
    "number_of_pixels": 1,
    "mean_row": 2,
    "mean_col": 3,
    "var_row": 4,
    "var_col": 5,
    "var_row_col": 6,
    "max_pix_row": 7,
    "max_pix_col": 8,
    "max_pix_intensity": 9,
}

# When motor positions are provided, we can convert the features extracted from the segmented
# domains to motor positions. These additional features are added to the feature table.
# such that mean_col is converted to mean_motor2, mean_row to mean_motor1 etc. and added to the
# feature table. Thus feature_table["mean_motor1"] will contain the mean motor position for the
# first motor while feature_table["mean_row"] will contain the mean row position in the image.
_MOTOR_KEY_MAPPING = {
    "mean_row": "mean_motor1",
    "mean_col": "mean_motor2",
    "var_row": "var_motor1",
    "var_col": "var_motor2",
    "var_row_col": "var_motor1_motor2",
    "max_pix_row": "max_pix_motor1",
    "max_pix_col": "max_pix_motor2",
}


def _gaussian_mixture(data, k, coordinates):
    """Run the peaksearch algorithm on a grid of images and extract features in parallel.

    Args:
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (m,n)
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.
        coordinates (:obj:`tuple`): a tuple with two arrays of motor positions.

    Returns:
        'numpy array': a 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict and possibly motor positions as specified in
         _MOTOR_KEY_MAPPING.
    """
    p = np.max([_FEATURE_MAPPING[key] for key in _FEATURE_MAPPING]) + 1
    dum1 = np.zeros((p,), dtype=np.uint8)
    dum2 = np.zeros((k,), dtype=np.uint8)
    feature_table = np.zeros((data.shape[0], data.shape[1], p, k), dtype=np.float32)
    _peaksearch_parallel(data, dum1, dum2, feature_table)
    feature_table_dict = {}
    for key in _FEATURE_MAPPING:
        feature_table_dict[key] = feature_table[..., _FEATURE_MAPPING[key], :]
    if coordinates is not None:
        feature_table_dict = _add_motor_dimensions(feature_table_dict, coordinates)
    return feature_table_dict


def _median(x):
    if x.size == 0:
        return np.nan
    return np.median(x)


def _add_motor_dimensions(feature_table_dict, coordinates):
    """Adds motor dimensions to the feature table.

    Args:
        feature_table_dict (:obj:`dict`): a dictionary with keys corresponding to the
            features extracted from the segmented domains.
        coordinates (:obj:`tuple`): a tuple with two arrays of motor positions.

    Returns:
        :obj:`dict`: a dictionary with keys corresponding to the features extracted from the
            segmented domains with motor positions added according to _MOTOR_KEY_MAPPING.
    """
    X, Y = coordinates
    dm1 = _median(np.diff(X, axis=0))
    dm2 = _median(np.diff(Y, axis=1))
    m10 = _median(X)
    m20 = _median(Y)
    for key in ["mean_row", "max_pix_row"]:
        feature_table_dict[_MOTOR_KEY_MAPPING[key]] = (
            feature_table_dict[key] - (X.shape[0] - 1) / 2.0
        ) * dm1 + m10

    for key in ["mean_col", "max_pix_col"]:
        feature_table_dict[_MOTOR_KEY_MAPPING[key]] = (
            feature_table_dict[key] - (Y.shape[1] - 1) / 2.0
        ) * dm2 + m20
    feature_table_dict[_MOTOR_KEY_MAPPING["var_row"]] = (
        feature_table_dict["var_row"] * dm1 * dm1
    )
    feature_table_dict[_MOTOR_KEY_MAPPING["var_col"]] = (
        feature_table_dict["var_col"] * dm2 * dm2
    )
    feature_table_dict[_MOTOR_KEY_MAPPING["var_row_col"]] = (
        feature_table_dict["var_row_col"] * dm1 * dm2
    )
    return feature_table_dict


@numba.njit
def label_sparse(data):
    """Assigns pixels in a 2D image to the closest local maxima.

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

        # Label the image following the local max climber algorithm
        labeled_array, nfeatures = darling.peaksearcher.label_sparse(img)

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
        data (:obj:`numpy.ndarray`): a 2D data map to process. shape=(m,n)

    Returns:
        labeled_array (:obj:`numpy.ndarray`): a 2D array with the same shape as the input data
        number_of_labels (:obj:`int`): the number of labels assigned to the data map
    """
    m, n = data.shape
    max_iterations = m * n
    label = 1

    # The labeled array is to be filled with labels for each pixel in the data map.
    # The labels are integers starting from 1 and increasing for each new local maxima.
    labeled_array = np.zeros((m, n), dtype=np.uint16)

    # tmpi and tmpj are temporary arrays to store the path of the climbing pixel
    # in the data map. The path is stored in these arrays until the pixel reaches
    # a labeled pixel or a local maxima.
    tmpi = np.zeros((max_iterations,), dtype=np.uint16)
    tmpj = np.zeros((max_iterations,), dtype=np.uint16)

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
            i, j = ii, jj  # current location of the climbing pixel is stored in i and j

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
                # back to the climbing pixel thourgh the path stored in tmpi and tmpj and
                # stop the climbing.
                if labeled_array[max_i, max_j] != 0:
                    label_to_propagate = labeled_array[max_i, max_j]
                    for k in range(iterations + 1):  # backpropagate the label
                        labeled_array[tmpi[k], tmpj[k]] = label_to_propagate
                    climbing = False  # time to stop climbing

                # if the pixel was more intense than its neighbors, it is a local maxima
                # and we assign a new label to the pixel and propagate it back to the climbing
                # pixel through the path stored in tmpi and tmpj.
                elif max_i == i and max_j == j:
                    labeled_array[max_i, max_j] = label
                    label_to_propagate = labeled_array[max_i, max_j]
                    label += 1
                    for k in range(iterations + 1):  # backpropagate the label
                        labeled_array[tmpi[k], tmpj[k]] = label_to_propagate
                    climbing = False  # time to stop climbing

                # Here we have simply found a more intense neighbor that is unlabeled
                # the climb moves to this neighbor and continues.
                else:
                    i, j = max_i, max_j
                    iterations += 1

    return labeled_array, label - 1


def extract_features(labeled_array, data, k):
    """Extract features from a labeled array.

    Args:
        labeled_array (:obj:`numpy.ndarray`): Label array with shape (m,n)
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (m,n)
        k (:obj:`int`): number of segmented domains to keep features for
            The domain with the highest sum_intensity will be kept.

    Returns:
        'numpy array': a 2D array with the extracted features with indices following
         a static _FEATURE_MAPPING dict.
    """
    nlabels = np.max(labeled_array)
    p = np.max([_FEATURE_MAPPING[key] for key in _FEATURE_MAPPING]) + 1
    return _extract_features(labeled_array, data, nlabels, p, k)


@numba.njit
def _extract_features(labeled_array, data, nlabels, num_props, k):
    """Extract features from a labeled array.

    Args:
        labeled_array (:obj:`numpy.ndarray`): Label array with shape (m,n)
        data (:obj:`numpy.ndarray`): The underlying intensity data array with shape (m,n)
        nlabels (:obj:`int`): number of labels in the labeled array
        num_props (:obj:`int`): number of properties to extract.
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
    number_of_pixels = 1
    mean_row = 2
    mean_col = 3
    var_row = 4
    var_col = 5
    var_row_col = 6
    max_pix_row = 7
    max_pix_col = 8
    max_pix_intensity = 9

    m, n = data.shape

    if nlabels > 65535:
        raise ValueError("Found more features than can be assigned with uint16")

    feature_table = np.zeros((num_props, np.maximum(k, nlabels)), dtype=float)

    for ii in range(0, m):
        for jj in range(0, n):
            if labeled_array[ii, jj] == 0:
                continue

            index = labeled_array[ii, jj] - 1
            feature_table[sum_intensity, index] += data[ii, jj]
            feature_table[number_of_pixels, index] += 1
            feature_table[mean_row, index] += ii * data[ii, jj]
            feature_table[mean_col, index] += jj * data[ii, jj]

            if data[ii, jj] > feature_table[max_pix_intensity, index]:
                feature_table[max_pix_row, index] = ii
                feature_table[max_pix_col, index] = jj
                feature_table[max_pix_intensity, index] = data[ii, jj]

    nnz_mask = feature_table[sum_intensity, :] > 1
    divider = feature_table[sum_intensity, nnz_mask]
    feature_table[mean_row, nnz_mask] /= divider
    feature_table[mean_col, nnz_mask] /= divider

    for ii in range(0, m):
        for jj in range(0, n):
            if labeled_array[ii, jj] == 0:
                continue
            index = labeled_array[ii, jj] - 1
            if feature_table[sum_intensity, index] != 0:
                diff_row = ii - feature_table[mean_row, index]
                diff_col = jj - feature_table[mean_col, index]
                feature_table[var_row, index] += data[ii, jj] * diff_row * diff_row
                feature_table[var_col, index] += data[ii, jj] * diff_col * diff_col
                feature_table[var_row_col, index] += data[ii, jj] * diff_row * diff_col

    #  (George R. Price, Ann. Hum. Genet., Lond, pp485-490, Extension of covariance selection mathematics, 1972).
    # https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
    # assuming counts are observations
    unbiased_divider = divider - 1
    feature_table[var_row, nnz_mask] /= unbiased_divider
    feature_table[var_col, nnz_mask] /= unbiased_divider
    feature_table[var_row_col, nnz_mask] /= unbiased_divider

    idx = np.argsort(-feature_table[sum_intensity], kind="quicksort")[0:k]
    return feature_table[:, idx]


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.uint8[:],
            numba.uint8[:],
            numba.float32[:, :],
        )
    ],
    "(m,n),(p),(k)->(p,k)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _peaksearch_parallel(data, dum1, dum2, res2):
    """Label a grid of images in parallel and return all features for segmented peak."""
    labeled_array, nlabels = label_sparse(data)
    res2[...] = _extract_features(labeled_array, data, nlabels, len(dum1), len(dum2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import darling

    # import a small data set from assets known to
    # comprise crystalline domains
    _, data, coordinates = darling.assets.domains()

    # compute all the gaussian mixture model features
    features = darling.properties.gaussian_mixture(data, k=3, coordinates=coordinates)

    # this is a dict like structure that can be accessed like this:
    sum_intensity_second_strongest_peak = features["sum_intensity"][..., 1]

    # plot the mean in the first motor direction for the strongest peak
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(features["mean_motor1"][..., 0], cmap="plasma")
    plt.tight_layout()
    plt.show()
