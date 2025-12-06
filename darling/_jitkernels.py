import numba
import numpy as np


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


@numba.jit(nopython=True, cache=True, parallel=True)
def _kam(property_2d, km, kn, kam_map, counts_map):
    """Fills the KAM and count maps in place.

    Args:
        property_2d (:obj:`numpy.ndarray`): The shape=(a,b,m) map to
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
