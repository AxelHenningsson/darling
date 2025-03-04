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
import darling._gaussian as gfit


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
        norm[0] = np.min(coordinates[0]), np.max(coordinates[0])
        norm[1] = np.min(coordinates[1]), np.max(coordinates[1])
    elif norm == "dynamic":
        norm = np.zeros((2, 2))
        norm[0] = np.nanmin(property_2d[..., 0]), np.nanmax(property_2d[..., 0])
        norm[1] = np.nanmin(property_2d[..., 1]), np.nanmax(property_2d[..., 1])
    else:
        assert norm.shape == (2, 2), "scale must be of shape (2, 2)"

    for i in range(2):
        assert (
            np.nanmin(property_2d[..., i]) >= norm[i, 0]
        ), "property_2d values exceed norm, please select a feasible normalization range"
        assert (
            np.nanmin(property_2d[..., i]) <= norm[i, 1]
        ), "property_2d values exceed norm, please select a feasible normalization range"

    x, y = color.normalize(property_2d, norm)
    rgb_map = color.rgb(x, y)
    colorkey, colorgrid = color.colorkey(norm)

    return rgb_map, colorkey, colorgrid


def kam(property_2d, size=(3, 3)):
    """Compute the KAM (Kernel Average Misorientation) map of a 2D property map.

    KAM is compute dby sliding a kernel across the image and for each voxel computing
    the average misorientation between the central voxel and the surrounding voxels.

    NOTE: This is a projected KAM in the sense that the rotation the full rotation
    matrix of the voxels are unknown. I.e this is a computation of the misorientation
    between diffraction vectors Q and not orientation elements of SO(3).

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
        plt.show()

    .. image:: ../../docs/source/images/kam.png

    Args:
        property_2d (:obj:`numpy array`): The property map to compute the KAM from,
            shape=(a, b, 2). This is assumed to be the angular coordinates of diffraction.
            such that np.linalg.norm( property_2d[i,j]) gives the mismatch in degrees
            between the reference diffraction vector and the local mean diffraction vector.
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
    _kam(property_2d, km, kn, kam_map, counts_map)
    counts_map[counts_map == 0] = 1
    return np.sum(kam_map, axis=-1) / counts_map


def moments(data, coordinates, method="statistical"):
    """Compute the sample mean and covariance of a 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 2 or 3 degrees of freedom. These could be phi and chi or phi and energy, etc.
    The total data array is therefore either 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

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
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) or shape=(n, m, o, a, b) where the maps over which the mean will
            be calculated are of shape=(n, m) or shape=(m, n, o) and the field is of shape=(a, b). Must be numpy uint16. I.e the
            detector roi is of shape=(a,b) while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution
            for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 or len==3 continaning numpy 1d arrays specifying the
            coordinates in the n and m dimensions respectively. I.e, as an example, as an example these could be the phi and chi
            angular cooridnates.

    Returns:
        :obj:`tuple` of :obj:`numpy array` : The mean map of shape=(k,l) and the covariance map of shape=(k,l,2,2).
    """
    mu = mean(data, coordinates)
    cov = covariance(data, coordinates, first_moments=mu)

    if method == "statistical":
        return mu, cov
    elif method == "gaussian":
        mu_gauss = mean_gaussian(data, coordinates, mu, cov)
        cov_gauss = covariance_gaussian(data, coordinates, mu_gauss, cov)
        return mu_gauss, cov_gauss
    else:
        raise ValueError("method must be either 'statistical' or 'gaussian'")


def mean_gaussian(data, coordinates, stat_mean, stat_cov):
    """Compute first moments by fitting Gaussians using statistical initialization."""
    _check_data(data, coordinates)
    dum = np.arange(len(coordinates)).astype(np.float32)

    if len(coordinates) == 2:
        X, Y = np.array(coordinates).astype(np.float32)
        a, b, m, n = data.shape
        res = np.zeros((a, b, 2), dtype=np.float32)
        gfit.moments_2d_gaussian(data, X, Y, dum, stat_mean, stat_cov, res)
    else:
        raise NotImplementedError("3D Gaussian fitting not yet implemented")

    return res


def covariance_gaussian(data, coordinates, first_moments, stat_cov):
    """Compute covariance by fitting Gaussians using statistical initialization."""
    _check_data(data, coordinates)
    dim = len(coordinates)
    dum = np.arange(dim).astype(np.float32)
    res = np.zeros((data.shape[0], data.shape[1], dim, dim), dtype=np.float32)
    points = np.array([c.flatten() for c in coordinates]).astype(np.float32)

    if dim == 2:
        gfit.covariance_2d_gaussian(data, first_moments, points, dum, stat_cov, res)
    else:
        raise NotImplementedError("3D Gaussian fitting not yet implemented")

    return res


def mean(data, coordinates):
    """Compute the sample mean of a 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 2 or 3 degrees of freedom. These could be phi and chi or phi and energy, etc.
    The total data array is therefore either 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

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
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) or shape=(n, m, o, a, b) where the maps over which the mean will be calculated are
            of shape=(n, m) or shape=(m, n, o) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector roi is of shape=(a,b)
            while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 or len=3 continaning numpy 1d arrays specifying the coordinates
            in the n and m dimensions respectively. I.e, as an example, these could be the
            phi and chi angular cooridnates.

    Returns:
        :obj:`numpy array` : The mean map of shape=(k,l).
    """
    _check_data(data, coordinates)
    dum = np.arange(len(coordinates)).astype(np.float32)
    if len(coordinates) == 2:
        X, Y = np.array(coordinates).astype(np.float32)
        a, b, m, n = data.shape
        res = np.zeros((a, b, 2), dtype=np.float32)
        _first_moments2D(data, X, Y, dum, res)
    else:
        X, Y, Z = np.array(coordinates).astype(np.float32)
        a, b, m, n, o = data.shape
        res = np.zeros((a, b, 3), dtype=np.float32)
        _first_moments3D(data, X, Y, Z, dum, res)
    return res


def covariance(data, coordinates, first_moments=None):
    """Compute the sample covariance of a 4D or 5D DFXM data-set.

    The data-set represents a DFXM scan with 2 or 3 degrees of freedom. These could be phi and chi or phi and energy, etc.
    The total data array is therefore either 4d or 5d.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

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
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) or shape=(n, m, o, a, b) where the maps over which the covariance will
            be calculated are of shape=(n, m) or shape=(n, m, o) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector
            roi is of shape=(a,b) while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 or len=3 continaning numpy 1d arrays specifying the coordinates
            in the n and m (and o) dimensions respectively. I.e, as an example, these could be the phi and chi angular cooridnates.
        first_moments (:obj:`numpy array`):  Array of shape=(a, b) of the first moments. Defaults to None, in which case the first moments
            are recomputed on the fly.

    Returns:
        :obj:`numpy array` : The covariance map of shape=(k,l,2,2).
    """
    _check_data(data, coordinates)
    dim = len(coordinates)
    dum = np.arange(dim).astype(np.float32)
    res = np.zeros((data.shape[0], data.shape[1], dim, dim), dtype=np.float32)
    points = np.array([c.flatten() for c in coordinates]).astype(np.float32)
    if first_moments is None:
        first_moments = mean(data, coordinates)
    if dim == 2:
        _second_moments2D(data, first_moments, points, dum, res)
    elif dim == 3:
        _second_moments3D(data, first_moments, points, dum, res)
    return res


def _check_data(data, coordinates):
    assert data.dtype == np.uint16, "data must be of type uint16"
    if len(coordinates) == 2:
        assert len(data.shape) == 4, "2D scan data array must be of shape=(a, b, n, m)"
    elif len(coordinates) == 3:
        assert (
            len(data.shape) == 5
        ), "3D scan data array must be of shape=(a, b, n, m, o)"
    else:
        raise ValueError("The coordinate array must have 2 or 3 motors")
    for c in coordinates:
        if not isinstance(c, np.ndarray):
            raise ValueError("Coordinate array must be a numpy array")
    assert np.allclose(
        list(c.shape), list(data.shape)[2:]
    ), "coordinate array do not match data shape"


@numba.guvectorize(
    [
        (
            numba.uint16[:, :],
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(m,n),(m,n),(m,n),(p)->(p)",
    nopython=True,
    target="parallel",
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
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(m,n,o),(m,n,o),(m,n,o),(m,n,o),(p)->(p)",
    nopython=True,
    target="parallel",
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
            numba.uint16[:, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
        )
    ],
    "(n,m),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
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
        m -= np.sum(m * a) / I
        cov = np.dot(m * a, m.T) / I
        res[...] = cov


@numba.guvectorize(
    [
        (
            numba.uint16[:, :, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
        )
    ],
    "(n,m,o),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
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
        m -= np.sum(m * a) / I
        cov = np.dot(m * a, m.T) / I
        res[...] = cov


@numba.jit(nopython=True, cache=True, parallel=True)
def _kam(property_2d, km, kn, kam_map, counts_map):
    """Fills the KAM and count maps in place.

    Args:
        property_2d (:obj:`numpy.ndarray`): The shape=(a,b,2) map to
            be used for the KAM computation.
        km (:obj:`int`): kernel size in rows
        kn (:obj:`int`): kernel size in columns
        kam_map (:obj:`numpy.ndarray`): empty array to store the KAM
            values of shape=(a,b, (km*kn)-1)
        counts_map (:obj:`numpy.ndarray`): empty array to store the counts
            of shape=(a,b)
    """
    for i in numba.prange(km // 2, property_2d.shape[0] - (km // 2)):
        for j in range(kn // 2, property_2d.shape[1] - (kn // 2)):
            if ~np.isnan(property_2d[i, j, 0]):
                c = property_2d[i, j]
                for ii in range(-(km // 2), (km // 2) + 1):
                    for jj in range(-(kn // 2), (kn // 2) + 1):
                        if ii == 0 and jj == 0:
                            continue
                        else:
                            n = property_2d[i + ii, j + jj]
                            if ~np.isnan(n[0]):
                                kam_map[i, j, counts_map[i, j]] = np.linalg.norm(n - c)
                                counts_map[i, j] += 1


if __name__ == "__main__":
    pass
