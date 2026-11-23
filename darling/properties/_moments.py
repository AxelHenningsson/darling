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

import numba
import numpy as np


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
    first_moments = _first_moments_ND(data, coordinates)
    second_moments = _second_moments_ND(data, coordinates, first_moments)
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
    first_moments = _first_moments_ND(data, coordinates)
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
        second_moments = _second_moments_ND(data, coordinates, first_moments[..., None])
    else:
        second_moments = _second_moments_ND(data, coordinates, first_moments)
    return np.squeeze(second_moments)


@numba.njit(parallel=True, cache=True)
def _first_moments_ND(data, coordinates):
    """Compute the first moments of a ND data array.

    NOTE: Computation is done in parallel using shared memory with numba. This kernel supports arbitrary dimensions
        and arbitrary dtypes for the data and coordinates arrays.

    Args:
        data (:obj:`numpy array`): The data array to compute the first moments of. Arbitrary dtype and shape.
        coordinates (:obj:`numpy array`): The coordinates of the data array. Arbitrary dtype and shape=(ndim, m, n, ...).

    Returns:
        :obj:`numpy array` : The first moments of the data array of shape (a, b, ndim).
    """
    ndim = len(coordinates)
    res = np.zeros((data.shape[0], data.shape[1], ndim), dtype=coordinates.dtype)
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            total_intensity = np.sum(data[i, j])
            if total_intensity != 0:
                for p in range(ndim):
                    res[i, j, p] = np.sum(data[i, j] * coordinates[p])
                res[i, j, :] /= total_intensity
    return res


@numba.njit(parallel=True, cache=True)
def _second_moments_ND(data, coordinates, first_moments):
    """Compute the second moments of a ND data array.

    NOTE: Computation is done in parallel using shared memory with numba. This kernel supports arbitrary dimensions
        and arbitrary dtypes for the data and coordinates arrays.

    Args:
        data (:obj:`numpy array`): The data array to compute the second moments of. Arbitrary dtype and shape.
        coordinates (:obj:`numpy array`): The coordinates of the data array. Arbitrary dtype and shape=(ndim, m, n, ...).

    Returns:
        :obj:`numpy array` : The second moments of the data array of shape (a, b, ndim, ndim).
    """
    ndim = len(coordinates)
    res = np.zeros((data.shape[0], data.shape[1], ndim, ndim), dtype=coordinates.dtype)
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            total_intensity = np.sum(data[i, j])
            mu = first_moments[i, j]
            if total_intensity != 0:
                for p in range(ndim):
                    diff_p = (coordinates[p] - mu[p]) * data[i, j]
                    for q in range(ndim):
                        res[i, j, p, q] = np.sum(diff_p * (coordinates[q] - mu[q]))
                res[i, j, :, :] /= total_intensity
    return res


def _check_data(data, coordinates):
    if not isinstance(coordinates, np.ndarray):
        raise ValueError(
            f"coordinates must be a numpy array but got coordinates oftype {type(coordinates)}"
        )

    if not coordinates.shape[1:] == data.shape[2:]:
        raise ValueError(
            f"trailing dimensions of coordinates shape {coordinates.shape[1:]} do not match trailing dimensions of data shape {data.shape[2:]}"
        )

    if data.shape[0] == 1 or data.shape[1] == 1:
        raise ValueError(
            "First two detector row-column dimensions of data array must be greater than 1"
        )


if __name__ == "__main__":
    pass
