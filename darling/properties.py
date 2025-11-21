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
    coordinates = np.meshgrid(phi, chi, theta, indexing='ij')

    # create a random data array
    detector_dim = (128, 128) # the number of rows and columns of the detector
    data = 64000 * np.random.rand(*detector_dim, len(phi), len(chi), len(theta))

    data = data.astype(np.uint16) # the collected intensity data for the entire scan

    # compute the first and second moments such that
    # mean[i,j] is the shape=(3,) array of mean coorindates for pixel i,j.
    # covariance[i,j] is the shape=(3,3) covariance matrix of pixel i,j.
    mean, covariance = darling.properties.moments(data, coordinates)

    assert mean.shape==(128, 128, 3)
    assert covariance.shape==(128, 128, 3, 3)


"""

import numba
import numpy as np

import darling._color as color
import darling.peaksearcher as peaksearcher


def rgb(property_2d, norm="dynamic", coordinates=None):
    """Compute a m, n, 3 rgb array from a 2d property map, e.g from a first moment map.

    NOTE: Only normalization ranges that covers the full range of the property_2d are
    accepted here. Consider marking values outside range by np.nan before calling in
    case such normalization is needed.

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
    if norm == "full":
        norm = np.zeros((2, 2))

        # handles rounding errors in property map.
        width_0 = np.max(coordinates[0]) - np.min(coordinates[0])
        pad_0 = width_0 * 0.001
        width_1 = np.max(coordinates[1]) - np.min(coordinates[1])
        pad_1 = width_1 * 0.001

        norm[0] = np.min(coordinates[0]) - pad_0, np.max(coordinates[0]) + pad_0
        norm[1] = np.min(coordinates[1]) - pad_1, np.max(coordinates[1]) + pad_1

    elif norm == "dynamic":
        norm = np.zeros((2, 2))
        norm[0] = np.nanmin(property_2d[..., 0]), np.nanmax(property_2d[..., 0])
        norm[1] = np.nanmin(property_2d[..., 1]), np.nanmax(property_2d[..., 1])
    else:
        assert norm.shape == (2, 2), "scale must be of shape (2, 2)"

    for i in range(2):
        assert np.nanmin(property_2d[..., i]) >= norm[i, 0], (
            "property_2d values exceed norm, please select a feasible normalization range"
        )
        assert np.nanmin(property_2d[..., i]) <= norm[i, 1], (
            "property_2d values exceed norm, please select a feasible normalization range"
        )

    x, y = color.normalize(property_2d, norm)
    rgb_map = color.rgb(x, y)
    colorkey, colorgrid = color.colorkey(norm)

    return rgb_map, colorkey, colorgrid


def kam(vector_field, ndim=None, size=3):

    #vector_field lets do ndim and than size
    """Compute the KAM (Kernel Average Misorientation) map on a data input for 2D or 3D data with C input channels.

    KAM is computed by sliding a kernel across the image and for each voxel computing
    the average misorientation between the central voxel and the surrounding voxels.
    Here the misorientation is defined as the L2 euclidean distance between the
    (potentially vectorial) property map and the central voxel such that scalars formed
    as for instance np.linalg.norm( data[i + 1, j] - data[i, j] ) are
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
        data = np.random.rand(len(phi), len(chi), 2)
        data[data > 0.9] = 1
        data -= 0.5
        data = gaussian_filter(data, sigma=2)

        # compute the KAM map
        kam = darling.properties.kam(data, ndim=2, size=3)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(kam, cmap="plasma")
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/kam.png

    Args:
        vector_field (:obj:`numpy.ndarray`): Input data array used for KAM computation,
            with shape (Y, X), (Y, X, C), (Z, Y, X, C), or (Z, Y, X). Each element is assumed
            to represent a vector-valued diffraction property.
        ndim (:obj:`int`): Needs to be specified, dimensionality of the data, must be 2 or 3.
        size (:obj:`int` or :obj:`tuple` or :obj:`numpy.ndarray`, optional): Kernel size used for neighborhood evaluation.
            Defaults to 3 . For 2D input use (ky, kx); for 3D input use (kz, ky, kx).

    Returns:
        :obj:`numpy.ndarray`: KAM map of the same spatial shape as the input (without
        the vector channel), in the same units as the input.
    """

    if ndim is None:
        raise ValueError("ndim must be specified")
    if ndim not in [2, 3]:
        raise ValueError("ndim must be 2 or 3")

    # --- fit the size of the kernel to the input and ndim ---
    if isinstance(size, int):
        size = np.array([size] * ndim, dtype=int)
    elif isinstance(size, (tuple, np.ndarray)):
        size = np.array(size, dtype=int)
        if size.size != ndim:
            raise ValueError(
                f"size length {size.size} does not match ndim={ndim}."
            )
    else:
        raise TypeError("size must be int, tuple, or numpy.ndarray if the size is defined for each axis")

    # --- fit the vector_field to the ndim if needed ---
    if ndim == 2:
        if vector_field.ndim == 2:
            vector_field = vector_field[..., None]
        elif vector_field.ndim   == 3:
            pass
        else:
            raise ValueError("For a 2D kernel, property must be 2D or 3D")
    elif ndim == 3:
        if vector_field.ndim == 3:
            vector_field = vector_field[..., None]
        elif vector_field.ndim == 4:
            pass
        else:
            raise ValueError("For a 3D kernel, property must be 3D or 4D")
    else:
        raise ValueError("Kernel size must be 2D 3D")


    assert all(s % 2 == 1 for s in size), "size must be odd"
    assert all(s > 1 for s in size), "size must be larger than 1"

    # --- compute the shape of the kam map ---
    shape = vector_field.shape[:-1] + (np.prod(size) - 1,)
    kam_map = np.zeros(shape)
    counts_map = np.zeros(shape[:-1], dtype=int)

    if ndim == 2:
        _kam3D(vector_field[None,...], 1, size[0], size[1], kam_map[None,...], counts_map[None,...])
    elif ndim == 3:
        _kam3D(vector_field, size[0], size[1], size[2], kam_map, counts_map)

    else:
        raise ValueError("Kernel size must be 2D or 3D")

    counts_map[counts_map == 0] = 1
    return np.sum(kam_map, axis=-1) / counts_map


def moments(data, coordinates):
    """Compute the sample mean amd covariance of a 3D, 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 1, 2 or 3 degrees of freedom.
    These could be mu, phi and chi, or phi and energy, etc. The total data array
    is therefore either 3d, 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just
        in time compiling. For this reason the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)
        coordinates = np.meshgrid(phi, chi, indexing='ij')

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
            and the detector field dimensions are of shape=(a, b). Must be numpy uint16.
            I.e data[i, j, ...] is a distribution for pixel i, j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=1, len=2 or len=3
            containing numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid.

    Returns:
        :obj:`tuple` of :obj:`numpy array` : The mean map of shape=(a,b,...) and the
            covariance map of shape=(a,b,...).
    """
    mu = mean(data, coordinates)
    cov = covariance(data, coordinates, first_moments=mu)
    return mu, cov


def mean(data, coordinates):
    """Compute the sample mean of a 3D, 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 1, 2 or 3 degrees of freedom.
    These could be mu, phi and chi, or phi and energy, etc. The total data array
    is therefore either 3d, 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just
        in time compiling. For this reason the data array must be of type numpy uint16.

    Example in a DFXM energy-mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        theta = np.linspace(-1, 1, 7)
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)
        coordinates = np.meshgrid(phi, chi, theta, indexing='ij')

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
            and the detector field dimensions are of shape=(a, b). Must be numpy uint16.
            I.e data[i, j, ...] is a distribution for pixel i, j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=1, len=2 or len=3
            containing numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid.

    Returns:
        :obj:`numpy array` : The mean map of shape=(a,b,k) where k=data.ndim - 2.
    """
    _check_data(data, coordinates)
    dum = np.arange(len(coordinates)).astype(np.uint8)
    res = np.zeros((*data.shape[0:2], len(dum)), dtype=np.float32)
    if len(coordinates) == 1:
        X = np.array(coordinates).astype(np.float32).squeeze()
        _first_moments1D(data, X, dum, res)
        res = res.squeeze()
    elif len(coordinates) == 2:
        X, Y = np.array(coordinates).astype(np.float32)
        _first_moments2D(data, X, Y, dum, res)
    elif len(coordinates) == 3:
        X, Y, Z = np.array(coordinates).astype(np.float32)
        _first_moments3D(data, X, Y, Z, dum, res)
    else:
        raise ValueError("Coordinate array must be one of 1, 2 or 3-dimensional.")
    return res


def covariance(data, coordinates, first_moments=None):
    """Compute the sample mean of a 3D, 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 1, 2 or 3 degrees of freedom.
    These could be mu, phi and chi, or phi and energy, etc. The total data array
    is therefore either 3d, 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just
        in time compiling. For this reason the data array must be of type numpy uint16.

    Example in a DFXM energy-mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)
        coordinates = np.meshgrid(phi, chi, indexing='ij')

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
            and the detector field dimensions are of shape=(a, b). Must be numpy uint16.
            I.e data[i, j, ...] is a distribution for pixel i, j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=1, len=2 or len=3
            containing numpy nd arrays specifying the coordinates in each dimension
            respectively. I.e, as an example, these could be the phi and chi angular
            cooridnates as a meshgrid.
        first_moments (:obj:`numpy array`): Array of shape=(a, b, ...) of the first
            moments as described in darling.properties.mean(). Defaults to None, in
            which case the first moments are recomputed on the fly.

    Returns:
        :obj:`numpy array` : The covariance map of shape=(a,b,...).
    """
    _check_data(data, coordinates)
    dim = len(coordinates)
    dum = np.arange(dim).astype(np.uint8)
    points = np.array([c.flatten() for c in coordinates]).astype(np.float32)
    if first_moments is None:
        first_moments = mean(data, coordinates)
    if dim == 1:
        res = np.zeros((data.shape[0], data.shape[1], dim), dtype=np.float32)
        _second_moments1D(data, first_moments[..., None], points, dum, res)
        res = res.squeeze()
    if dim == 2:
        res = np.zeros((data.shape[0], data.shape[1], dim, dim), dtype=np.float32)
        _second_moments2D(data, first_moments, points, dum, res)
    elif dim == 3:
        res = np.zeros((data.shape[0], data.shape[1], dim, dim), dtype=np.float32)
        _second_moments3D(data, first_moments, points, dum, res)
    return res


def _check_data(data, coordinates):
    assert data.dtype == np.uint16, "data must be of type uint16"
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


@numba.guvectorize(
    [
        (
            numba.uint16[:],
            numba.float32[:],
            numba.uint8[:],
            numba.float32[:],
        )
    ],
    "(m),(m),(p)->(p)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _first_moments1D(data, x, dum, res):
    """Compute the sample mean of a 1D map.

    Args:
        data (:obj:`numpy array`): a 2d data map to proccess.
        x (:obj:`numpy array`): the first coordinate array
        dum (:obj:`numpy array`): dummpy variable for numba shapes. (len=2)
        res (:obj:`numpy array`): array in which to store output.
    """
    total_intensity = np.sum(data)
    if total_intensity == 0:
        res[...] = 0
    else:
        res[...] = np.sum(data * x) / total_intensity


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:, :],
            numba.float32[:, :],
            numba.uint8[:],
            numba.float32[:],
        )
    ],
    "(m,n),(m,n),(m,n),(p)->(p)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _first_moments2D(data, x, y, dum, res):
    """Compute the sample mean of a 2D map.

    Args:
        data (:obj:`numpy array`): a 2d data map to proccess.
        x (:obj:`numpy array`): the first coordinate array
        y (:obj:`numpy array`): the second coordinate array
        dum (:obj:`numpy array`): dummpy variable for numba shapes. (len=2)
        res (:obj:`numpy array`): array in which to store output.
    """
    total_intensity = np.sum(data)
    if total_intensity == 0:
        res[...] = np.zeros((2,))
    else:
        com_x = np.sum(data * x) / total_intensity
        com_y = np.sum(data * y) / total_intensity
        res[...] = [com_x, com_y]


@numba.guvectorize(
    [
        (
            numba.uint16[:, :, :],
            numba.float32[:, :, :],
            numba.float32[:, :, :],
            numba.float32[:, :, :],
            numba.uint8[:],
            numba.float32[:],
        )
    ],
    "(m,n,o),(m,n,o),(m,n,o),(m,n,o),(p)->(p)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _first_moments3D(data, x, y, z, dum, res):
    """Compute the sample mean of a 3D map.

    Args:
        data (:obj:`numpy array`): a 3d data map to proccess.
        x (:obj:`numpy array`): the first coordinate array.
        y (:obj:`numpy array`): the second coordinate array.
        z (:obj:`numpy array`): the third coordinate array.
        dum (:obj:`numpy array`): dummy variable for numba shapes. (of shape 3)
        res (:obj:`numpy array`): array in which to store output.
    """
    total_intensity = np.sum(data)
    if total_intensity == 0:
        res[...] = np.zeros((3,))
    else:
        com_x = np.sum(data * x) / total_intensity
        com_y = np.sum(data * y) / total_intensity
        com_z = np.sum(data * z) / total_intensity
        res[...] = [com_x, com_y, com_z]


@numba.guvectorize(
    [
        (
            numba.uint16[:],
            numba.float32[:],
            numba.float32[:],
            numba.uint8[:],
            numba.float32[:],
        )
    ],
    "(m),(p),(m),(p)->(p)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _second_moments1D(data, first_moments, points, dum, res):
    """Compute the sample variance of a 1D map.

    Args:
        data (:obj:`numpy array`): a 1d data map to proccess.
        first_moments (:obj:`numpy array`): the first moment of the 1d data map to proccess.
        points (:obj:`numpy array`): array of the angular coordinates.
        dum (:obj:`numpy array`): dummuy variable for numba shapes.
        res (:obj:`numpy array`): array in which to store output.
    """
    total_intensity = np.sum(data)
    if total_intensity == 0:
        res[...] = 0
    else:
        # Equivalent of the numpy.cov function setting the chi_phi intesnity as aweights,
        # see also https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        diff = points - first_moments
        res[...] = np.sum(diff * diff * data) / total_intensity


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.uint8[:],
            numba.float32[:, :],
        )
    ],
    "(n,m),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _second_moments2D(chi_phi, first_moments, points, dum, res):
    """Compute the sample covariance of a 2D map.

    Args:
        chi_phi (:obj:`numpy array`): a 2d data map to proccess.
        first_moments (:obj:`numpy array`): the first moment of the 2d data map to proccess.
        points (:obj:`numpy array`): 2,n flattened array of the coordinates.
        dum (:obj:`numpy array`): dummpy variable for numba shapes.
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(chi_phi)
    if I == 0:
        res[...] = np.zeros((2, 2))
    else:
        # Equivalent of the numpy.cov function setting the chi_phi intesnity as aweights,
        # see also https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        m = points.copy()
        m[0] -= first_moments[0]
        m[1] -= first_moments[1]
        a = chi_phi.flatten()
        cov = np.dot(m * a, m.T) / I
        res[...] = cov


@numba.guvectorize(
    [
        (
            numba.uint16[:, :, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.uint8[:],
            numba.float32[:, :],
        )
    ],
    "(n,m,o),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
    cache=True,
)
def _second_moments3D(data, first_moments, points, dum, res):
    """Compute the sample covariance of a 3D map.

    Args:
        data (:obj:`numpy array`): a 3d data map to proccess.
        first_moments (:obj:`numpy array`): the first moment of the 2d data map to proccess.
        points (:obj:`numpy array`): 2,n flattened array of the coordinates.
        dum (:obj:`numpy array`): dummy variable for numba shapes. (of shape 3)
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(data)
    if I == 0:
        res[...] = np.zeros((3, 3))
    else:
        # Equivalent of the numpy.cov function setting the chi_phi intesnity as aweights,
        # see also https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        m = points.copy()
        m[0] -= first_moments[0]
        m[1] -= first_moments[1]
        m[2] -= first_moments[2]
        a = data.flatten()
        cov = np.dot(m * a, m.T) / I
        res[...] = cov


@numba.jit(nopython=True, parallel=True, cache=True)
def _kam3D(vector_field, kz, ky, kx, kam_map, counts_map):
    """
    Fills the KAM and count maps in place.

    Args:
        data (:obj:`numpy.ndarray`): The input map used for the KAM computation,
            shape=(Z, Y, X, C), where Z is the slice dimension and C the number
            of vector components.
        kz (:obj:`int`): Kernel size along the slices (Z-axis).
        ky (:obj:`int`): Kernel size along the rows (Y-axis).
        kx (:obj:`int`): Kernel size along the columns (X-axis).
        kam_map (:obj:`numpy.ndarray`): Empty array to store the KAM values,
            shape=(Z, Y, X, (kz*ky*kx)-1).
        counts_map (:obj:`numpy.ndarray`): Empty array to store the valid
            neighbor counts, shape=(Z, Y, X).

    Notes:
        This function computes the Kernel Average Misorientation (KAM) for
        each voxel by evaluating the Euclidean distance between the local
        vector `c` and its valid neighbors `n` within the defined kernel.
        The results are stored directly in the provided `kam_map` and
        `counts_map` arrays.

        Technically we choose to prange over the x dimension, as it it is the largest and gives biggest performance boot.
    """
    Z, Y, X, C = vector_field.shape

    for x in numba.prange(kx // 2, X - kx // 2):
        for y in range(ky // 2, Y - ky // 2):
            for z in range(kz // 2, Z - kz // 2):
                c = vector_field[z, y, x]
                if not np.isnan(c[0]):
                    count = 0
                    for dz in range(-(kz // 2), kz // 2 + 1):
                        for dy in range(-(ky // 2), ky // 2 + 1):
                            for dx in range(-(kx // 2), kx // 2 + 1):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
                                n = vector_field[z + dz, y + dy, x + dx]
                                if not np.isnan(n[0]):
                                    dist = 0.0
                                    for d in range(C):
                                        dist += (n[d] - c[d]) ** 2
                                    kam_map[z, y, x, count] = np.sqrt(dist)
                                    count += 1
                    counts_map[z, y, x] = count


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
    assert data.dtype == np.uint16, "data must be of type uint16"
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
