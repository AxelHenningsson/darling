import numba
import numpy as np


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
        kam = darling.transforms.kam(property_2d, size=(3, 3))

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
        _kam(property_2d[..., None], km, kn, kam_map, counts_map)
    else:
        _kam(property_2d, km, kn, kam_map, counts_map)
    counts_map[counts_map == 0] = 1
    return np.sum(kam_map, axis=-1) / counts_map
