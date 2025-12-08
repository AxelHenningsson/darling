"""Functions module for computation of data features over 4D or 5D fields. I.e computation of moments of
mosa-scans strain-mosa-scans and the like.

As an example, in a DFXM strain-mosaicity-scan setting, using random arrays, the 3D moments
in theta, phi and chi can be retrieved as:

.. code-block:: python

    import numpy as np
    import darling

    # create coordinate arrays
    theta = np.linspace(-1, 1, 9) # crl scan grid
    phi = np.linspace(-1, 1, 8) # motor rocking scan grid
    chi = np.linspace(-1, 1, 16) # motor rolling scan grid
    coordinates = np.array(np.meshgrid(phi, chi, theta, indexing='ij'))

    # create a random data array
    detector_dim = (128, 128) # the number of rows and columns of the detector
    data = 64000 * np.random.rand(*detector_dim, len(phi), len(chi), len(theta))

    # compute the first and second moments such that
    # mean[i,j] is the shape=(3,) array of mean coorindates for pixel i,j.
    # covariance[i,j] is the shape=(3,3) covariance matrix of pixel i,j.
    mean, covariance = darling.properties.moments(data, coordinates)

    assert mean.shape==(128, 128, 3)
    assert covariance.shape==(128, 128, 3, 3)


"""

import numpy as np

import darling._color as color
import darling._jitkernels as kernels
import darling.peaksearcher as peaksearcher
from darling._gaussian_fit import fit_gaussian_with_linear_background_1D


def rgb(property_2d, norm="dynamic", coordinates=None):
    """Compute a m, n, 3 rgb array from a 2d property map, e.g from a first moment map.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import darling

        # create some phantom data
        phi = np.linspace(-1, 1, 64)
        chi = np.linspace(-1, 1, 128)
        coord = np.meshgrid(phi, chi, indexing="ij")
        property_2d = np.zeros((len(phi), len(chi), 2))
        property_2d[..., 0] = np.cos(np.outer(phi, chi))
        property_2d[..., 1] = np.sin(np.outer(phi, chi))

        # compute the rgb map normalising to the coordinates array
        rgb_map, colorkey, colorgrid = darling.properties.rgb(property_2d, norm="full", coordinates=coord)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(rgb_map)
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/rgbmapfull.png

    alternatively; normalize to the dynamic range of the property_2d array

    .. code-block:: python

        rgb_map, colorkey, colorgrid = darling.properties.rgb(property_2d, norm="dynamic")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(rgb_map)
        plt.tight_layout()
        plt.show()

    .. image:: ../../docs/source/images/rgbmapdynamic.png

    Args:
        property_2d (:obj:`numpy array`): The property map to colorize, shape=(a, b, 2),
            the last two dimensions will be mapped to rgb colors.
        coordinates (:obj:`numpy array`): Coordinate grid assocated to the
            property map, shape=(m, n), optional for norm="full". Defaults to None.
        norm (:obj:`numpy array` or :obj:`str`): array of shape=(2, 2) of the normalization
            range of the colormapping. Defaults to 'dynamic', in which case the range is computed
            from the property_2d array max and min.
            (norm[i,0] is min value for property_2d[:,:,i] and norm[i,1] is max value
            for property_2d[:,:,i].). If the string 'full' is passed, the range is
            computed from the coordinates as the max and min of the coordinates. This
            requires the coordinates to be passed as well.

    Returns:
        :obj:`tuple` of :obj:`numpy array` : RGB map of shape=(a, b, 3) and
            the colorkey of shape (m, n, 3) and the grid of the colorkey
            of shape=(m, n).
    """

    _property_2d = property_2d.copy()

    if isinstance(norm, str) and norm == "full":
        norm = np.zeros((2, 2))

        # handles rounding errors in property map.
        width_0 = np.max(coordinates[0]) - np.min(coordinates[0])
        pad_0 = width_0 * 0.001
        width_1 = np.max(coordinates[1]) - np.min(coordinates[1])
        pad_1 = width_1 * 0.001

        norm[0] = np.min(coordinates[0]) - pad_0, np.max(coordinates[0]) + pad_0
        norm[1] = np.min(coordinates[1]) - pad_1, np.max(coordinates[1]) + pad_1

    elif isinstance(norm, str) and norm == "dynamic":
        norm = np.zeros((2, 2))
        norm[0] = np.nanmin(_property_2d[..., 0]), np.nanmax(_property_2d[..., 0])
        norm[1] = np.nanmin(_property_2d[..., 1]), np.nanmax(_property_2d[..., 1])

    elif isinstance(norm, np.ndarray) and norm.shape == (2, 2):
        if norm[0, 0] > norm[0, 1] or norm[1, 0] > norm[1, 1]:
            raise ValueError("norm[i,0] must be less than norm[i,1] for i=0,1")
        _property_2d[..., 0][_property_2d[..., 0] <= norm[0, 0]] = np.nan
        _property_2d[..., 0][_property_2d[..., 0] >= norm[0, 1]] = np.nan
        _property_2d[..., 1][_property_2d[..., 1] <= norm[1, 0]] = np.nan
        _property_2d[..., 1][_property_2d[..., 1] >= norm[1, 1]] = np.nan
    else:
        raise ValueError(
            "norm must be a string ('full' or 'dynamic') or a numpy array of shape (2, 2)"
        )

    for i in range(2):
        assert np.nanmin(_property_2d[..., i]) >= norm[i, 0], (
            "property_2d values exceed norm, please select a feasible normalization range"
        )
        assert np.nanmin(_property_2d[..., i]) <= norm[i, 1], (
            "property_2d values exceed norm, please select a feasible normalization range"
        )

    x, y = color.normalize(_property_2d, norm)
    rgb_map = color.rgb(x, y)
    colorkey, colorgrid = color.colorkey(norm)

    return rgb_map, colorkey, colorgrid


def kam(property_2d, size=(3, 3)):
    """Compute the KAM (Kernel Average Misorientation) map of a 2D property map.

    KAM is computed by sliding a kernel across the image and for each voxel computing
    the average misorientation between the central voxel and the surrounding voxels.
    Here the misorientation is defined as the L2 euclidean distance between the
    (potentially vectorial) property map and the central voxel such that scalars formed
    as for instance np.linalg.norm( property_2d[i + 1, j] - property_2d[i, j] ) are
    computed and averaged over the kernel.

    NOTE: This is a projected KAM in the sense that the rotation the full rotation
    matrix of the voxels are unknown. I.e this is a computation of the misorientation
    between diffraction vectors Q and not orientation elements of SO(3). For 1D rocking
    scans this is further reduced due to the fact that the roling angle is unknown.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter

        import darling

        # create some phantom data
        phi = np.linspace(-1, 1, 64)
        chi = np.linspace(-1, 1, 128)
        coord = np.meshgrid(phi, chi, indexing="ij")
        property_2d = np.random.rand(len(phi), len(chi), 2)
        property_2d[property_2d > 0.9] = 1
        property_2d -= 0.5
        property_2d = gaussian_filter(property_2d, sigma=2)

        # compute the KAM map
        kam = darling.properties.kam(property_2d, size=(3, 3))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(kam, cmap="plasma")
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/kam.png

    Args:
        property_2d (:obj:`numpy array`): The property map to compute the KAM from,
            shape=(a, b, m) or (a, b). This is assumed to be the angular coordinates of
            diffraction such that np.linalg.norm( property_2d[i,j]) gives the mismatch
            in degrees between the reference diffraction vector and the local mean
            diffraction vector.
        size (:obj:`tuple`): The size of the kernel to use for the KAM computation.
            Defaults to (3, 3).

    Returns:
        :obj:`numpy array` : The KAM map of shape=(a, b). (same units as input.)
    """
    km, kn = size
    assert km > 1 and kn > 1, "size must be larger than 1"
    assert km % 2 == 1 and kn % 2 == 1, "size must be odd"
    kam_map = np.zeros((property_2d.shape[0], property_2d.shape[1], (km * kn) - 1))
    counts_map = np.zeros((property_2d.shape[0], property_2d.shape[1]), dtype=int)
    if property_2d.ndim == 2:
        kernels._kam(property_2d[..., None], km, kn, kam_map, counts_map)
    else:
        kernels._kam(property_2d, km, kn, kam_map, counts_map)
    counts_map[counts_map == 0] = 1
    return np.sum(kam_map, axis=-1) / counts_map


def moments(data, coordinates):
    """Compute the sample mean amd covariance of a 3D, 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 1, 2 or 3 degrees of freedom.
    These could be mu, phi and chi, or phi and energy, etc. The total data array
    is therefore either 3d, 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just
        in time compiling.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)
        coordinates = np.array(np.meshgrid(phi, chi, indexing='ij'))

        # create a random data array
        detector_dim = (128, 128)
        data = 64000 * np.random.rand(*detector_dim, len(phi), len(chi))

        data = data.astype(np.uint16)

        # compute the first and second moments
        mean, covariance = darling.properties.moments(data, coordinates)


    Args:
        data (:obj:`numpy array`): Array of shape=(a, b, m) or shape=(a, b, m, n)
            or shape=(a, b, m, n, o) where the maps over which the mean will be
            calculated are of shape=(m) or shape=(m, n) or shape=(m, n, o) respectively
            and the detector field dimensions are of shape=(a, b).
            I.e data[i, j, ...] is a distribution for pixel i, j.
        coordinates (:obj:`numpy array`): numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid. Shape=(ndim, m, n, ...). where ndim=1 for a rocking scan,
            ndim=2 for a mosaicity scan, etc.

    Returns:
        :obj:`tuple` of :obj:`numpy array` : The mean map of shape=(a,b,...) and the
            covariance map of shape=(a,b,...).
    """
    _check_data(data, coordinates)
    first_moments = kernels._first_moments_ND(data, coordinates)
    second_moments = kernels._second_moments_ND(data, coordinates, first_moments)
    mu = np.squeeze(first_moments)
    cov = np.squeeze(second_moments)
    return mu, cov


def mean(data, coordinates):
    """Compute the sample mean of a 3D, 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 1, 2 or 3 degrees of freedom.
    These could be mu, phi and chi, or phi and energy, etc. The total data array
    is therefore either 3d, 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just
        in time compiling.

    Example in a DFXM energy-mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        theta = np.linspace(-1, 1, 7)
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)
        coordinates = np.array(np.meshgrid(phi, chi, theta, indexing='ij'))

        # create a random data array
        detector_dim = (128, 128)
        data = 64000 * np.random.rand(*detector_dim, len(phi), len(chi), len(theta))

        data = data.astype(np.uint16)

        # compute the first moments
        first_moment = darling.properties.mean(data, coordinates)


    Args:
        data (:obj:`numpy array`): Array of shape=(a, b, m) or shape=(a, b, m, n)
            or shape=(a, b, m, n, o) where the maps over which the mean will be
            calculated are of shape=(m) or shape=(m, n) or shape=(m, n, o) respectively
            and the detector field dimensions are of shape=(a, b).
            I.e data[i, j, ...] is a distribution for pixel i, j.
        coordinates (:obj:`numpy array`): numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid. Shape=(ndim, m, n, ...). where ndim=1 for a rocking scan,
            ndim=2 for a mosaicity scan, etc.

    Returns:
        :obj:`numpy array` : The mean map of shape=(a,b,k) where k=data.ndim - 2.
    """
    _check_data(data, coordinates)
    first_moments = kernels._first_moments_ND(data, coordinates)
    return np.squeeze(first_moments)


def covariance(data, coordinates, first_moments=None):
    """Compute the sample mean of a 3D, 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 1, 2 or 3 degrees of freedom.
    These could be mu, phi and chi, or phi and energy, etc. The total data array
    is therefore either 3d, 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just
        in time compiling.

    Example in a DFXM energy-mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)
        coordinates = np.array(np.meshgrid(phi, chi, indexing='ij'))

        # create a random data array
        detector_dim = (128, 128)
        data = 64000 * np.random.rand(*detector_dim, len(phi), len(chi))

        data = data.astype(np.uint16)

        # compute the first moments
        first_moment = darling.properties.mean(data, coordinates)

        # compute the second moments
        covariance = darling.properties.covariance(data, coordinates, first_moments=first_moment)


    Args:
        data (:obj:`numpy array`): Array of shape=(a, b, m) or shape=(a, b, m, n)
            or shape=(a, b, m, n, o) where the maps over which the mean will be
            calculated are of shape=(m) or shape=(m, n) or shape=(m, n, o) respectively
            and the detector field dimensions are of shape=(a, b).
            I.e data[i, j, ...] is a distribution for pixel i, j.
        coordinates (:obj:`numpy array`): numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid. Shape=(ndim, m, n, ...). where ndim=1 for a rocking scan,
            ndim=2 for a mosaicity scan, etc.
        first_moments (:obj:`numpy array`): Array of shape=(a, b, ...) of the first
            moments as described in darling.properties.mean(). Defaults to None, in
            which case the first moments are recomputed on the fly.

    Returns:
        :obj:`numpy array` : The covariance map of shape=(a,b,...).
    """
    _check_data(data, coordinates)
    if first_moments is None:
        first_moments = mean(data, coordinates)

    if first_moments.ndim == 2:
        second_moments = kernels._second_moments_ND(
            data, coordinates, first_moments[..., None]
        )
    else:
        second_moments = kernels._second_moments_ND(data, coordinates, first_moments)
    return np.squeeze(second_moments)


def _check_data(data, coordinates):
    if data.shape[0] == 1 or data.shape[1] == 1:
        raise ValueError(
            "First two detector row-column dimensions of data array must be greater than 1"
        )

    if len(coordinates) == 1:
        assert len(data.shape) == 3, "1D scan data array must be of shape=(a, b, m)"
    elif len(coordinates) == 2:
        assert len(data.shape) == 4, "2D scan data array must be of shape=(a, b, n, m)"
    elif len(coordinates) == 3:
        assert len(data.shape) == 5, (
            "3D scan data array must be of shape=(a, b, n, m, o)"
        )
    else:
        raise ValueError("The coordinate array must have 1, 2 or 3 motors")
    for c in coordinates:
        if not isinstance(c, np.ndarray):
            raise ValueError("Coordinate array must be a numpy array")
    assert np.allclose(list(c.shape), list(data.shape)[2:]), (
        "coordinate array do not match data shape"
    )


def fit_1d_gaussian(
    data,
    coordinates,
    number_of_gauss_newton_iterations=7,
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
        number_of_gauss_newton_iterations (:obj:`int`): Number of Gauss-Newton iterations to use for the fit. Defaults to 7.
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
        number_of_gauss_newton_iterations,
        mask,
    )


def gaussian_mixture(data, k=8, coordinates=None):
    """Model a 2D grid of 2D images with a 2D grid of gaussian mixtures.

    For a data array of shape (m, n, a, b), each primary pixel (i, j) contains a
    (a, b) sub-array that is analyzed as a 2D image. Local maxima are identified
    within this sub-array, and segmentation is performed to assign a label to
    each secondary pixel.

    Each segmented region is treated as a Gaussian, with mean and covariance
    extracted for each label. Additionally, a set of features is computed
    for each label.

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

    Example:

    .. code-block:: python

        import darling
        import matplotlib.pyplot as plt

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


    .. image:: ../../docs/source/images/domains.png

    Args:
        data (:obj:`numpy array`):  Array of shape=(a, b, m, n) or shape=(a, b, m)
            where the maps over which the mean will be calculated are of shape=(m, n)
            or shape=(m,) and the field is of shape=(a, b) such that data[i, j, :, :] or
            data[i, j, :] is a 2D or 1D intensity distribution for pixel i,j.
        k (:obj:`int`): The number of gaussians to fit to the data. Defaults to 8.
            this means that the k strongest peaks, with respect to intensity, will be
            fitted with gaussians. The remaining peaks will be ignored.
        coordinates (:obj:`numpy array`): array specifying the coordinates of
            shape=(2, m, n) or shape=(1, m). I.e, as an example, these could be the
            phi and chi angular coordinates or the mu rocking angle.

    Returns:
        :obj:`dict` : features as a dictionary containing the extracted features for
            each peak with keys as specified above. i.e features["sum_intensity"][..., i]
            is a 2D image where each pixel holds the summed intensity of the i-th
            strongest peak. Likewise features["mean_row"][..., i] is the mean row
            position of the i-th strongest peak etc.

    """

    assert k > 0, "k must be larger than 0"
    assert (
        data.dtype == np.uint16
        or data.dtype == np.float32
        or data.dtype == np.float64
        or data.dtype == np.int32
    ), "data must be of type uint16, float32, float64 or int32"
    assert (len(data.shape) == 4) or (len(data.shape) == 3), (
        "data array must be 3D or 4D"
    )

    if coordinates is not None:
        if len(coordinates) == 1:
            # Here we handle rocking scans with 1D coordinates i.e
            # motor cooridnate arrays that are shape=(1, N), this
            # requires some reshaping to interface with the peaksearcher
            # numba functions.
            assert len(coordinates.shape) == 2, "coordinate array shape not reckognized"
            assert len(coordinates[0]) == data.shape[2], (
                "1d scan shape is a mismatch with the data array"
            )
            _coordinates = np.zeros((2, coordinates.shape[1], 1))
            _coordinates[0] = coordinates.T
            _coordinates[1] = 1
            props = peaksearcher._gaussian_mixture(data[..., :, None], k, _coordinates)

            # Remove the motor2 and col keys from the dictionary
            # as they are not relevant for 1D scans
            for k in [k for k in props.keys() if "_col" in k or "_motor2" in k]:
                props.pop(k, None)

        elif len(coordinates) == 2:
            assert len(coordinates[0].shape) == 2, "2D scan coordinates must be 2D"
            assert len(coordinates[1].shape) == 2, "2D scan coordinates must be 2D"
            assert coordinates[0].shape[0] == data.shape[2], (
                "2D scan coordinates must match data shape"
            )
            assert coordinates[0].shape[1] == data.shape[3], (
                "2D scan coordinates must match data shape"
            )
            assert coordinates[1].shape[0] == data.shape[2], (
                "2D scan coordinates must match data shape"
            )
            assert coordinates[1].shape[1] == data.shape[3], (
                "2D scan coordinates must match data shape"
            )
            props = peaksearcher._gaussian_mixture(data, k, coordinates)
        else:
            raise NotImplementedError("coordinates must be a tuple of length 1 or 2")
    else:
        if len(data.shape) == 3:  # 1D rocking scan needs special handling
            props = peaksearcher._gaussian_mixture(
                data[..., :, None], k, coordinates=None
            )
            # Remove the motor2 and col keys from the dictionary
            # as they are not relevant for 1D scans
            for k in [k for k in props.keys() if "_col" in k or "_motor2" in k]:
                props.pop(k, None)
        else:
            props = peaksearcher._gaussian_mixture(data, k, coordinates=None)

    return props


if __name__ == "__main__":
    pass
