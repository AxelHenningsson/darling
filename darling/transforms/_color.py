import numpy as np
from matplotlib.colors import hsv_to_rgb


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
        rgb_map, colorkey, colorgrid = darling.transforms.rgb(property_2d, norm="full", coordinates=coord)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(rgb_map)
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/rgbmapfull.png

    alternatively; normalize to the dynamic range of the property_2d array

    .. code-block:: python

        rgb_map, colorkey, colorgrid = darling.transforms.rgb(property_2d, norm="dynamic")

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

    mask = ~np.isnan(_property_2d[..., 0]) & ~np.isnan(_property_2d[..., 1])
    x, y = normalize(_property_2d, norm, mask)
    rgb_map = xy_to_rgb(x, y, mask)
    colorkey, colorgrid = get_colorkey(norm)

    return rgb_map, colorkey, colorgrid


def normalize(property_2d, norm, mask):
    """Normalize a property map to a given range.

    Args:
        property_2d (:obj:`numpy array`): The property map to normalize, shape=(a, b, 2),
            the last two dimensions will be mapped by the norm array range.
        norm (:obj:`numpy array` or :obj:`str`): array of shape=(2, 2), norm[i,0] is min
            value for property_2d[:,:,i] and norm[i,1] is max value for property_2d[:,:,i]
            the property_2d values will thus be normalised into this range.
        mask (:obj:`numpy array`): a mask of shape=(a, b) indicating which values are not nan
    Returns:
        :obj:`numpy array`: a normalized property map of shape=(a, b, 2)
    """
    norm_property_2d = np.full(
        (2, property_2d.shape[0], property_2d.shape[1]), fill_value=np.nan
    )

    for i in range(2):
        # move "leftmost-bottommost" datapoint to the origin, 0
        norm_property_2d[i][mask] = property_2d[..., i][mask] - norm[i, 0]

        # stretch all data to the [0, 1] box
        norm_property_2d[i][mask] = norm_property_2d[i][mask] / (
            norm[i, 1] - norm[i, 0]
        )

        # center the data around the origin, 0
        norm_property_2d[i][mask] = norm_property_2d[i][mask] - 0.5

        # stretch the data to a [-1, 1] box
        norm_property_2d[i][mask] = 2 * norm_property_2d[i][mask]

        # stretch the data to fit inside a unit circle
        norm_property_2d[i][mask] = norm_property_2d[i][mask] / (np.sqrt(2) + 1e-8)

    return norm_property_2d


def xy_to_rgb(x, y, mask=None):
    """Map 2D data to RGB color space by converting to HSV.

    The 2d points are assumed to lie on the top of the hsv color
    cylinder, with the angle of the point mapping to the hue, and the
    distance from the origin mapping to the saturation. The value is
    set to 1 for all points (brightest color).

    Args:
        x (:obj:`numpy array`): x-values cound by the unit circle, shape=(a,b)
        y (:obj:`numpy array`): y-values cound by the unit circle, shape=(a,b)
        mask (:obj:`numpy array`): a mask of shape=(a, b) indicating which values are not nan. Defaults to None.
            If None, all values are considered.

    Returns:
        :obj:`numpy array`: rgb values of shape (a, b, 3)
    """
    # angle of the point in the plane parameterised by 0,1

    if mask is None:
        mask = np.ones(x.shape, dtype=bool)

    angles = np.full(x.shape, fill_value=np.nan)
    angles[mask] = (
        (np.arctan2(-y[mask], -x[mask]) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
    )

    # radius of the point in the plane
    radius = np.full(x.shape, fill_value=np.nan)
    radius[mask] = np.sqrt(x[mask] ** 2 + y[mask] ** 2)

    hsv = np.stack(
        (
            angles,  # HUE (the actual color)
            radius,  # SATURATION (how saturated the color is)
            np.ones(x.shape),  # VALUE. (white to black)
        ),
        axis=2,
    )
    hsv[~mask, :] = 0

    rgb = hsv_to_rgb(hsv)

    rgb[~mask, :] = 0

    return rgb


def get_colorkey(norm, resolution=512):
    """Create a colorkey for a given normalization range.

    Args:
        norm (:obj:`numpy array` or :obj:`str`): array of shape=(2, 2).
            (norm[i,0] is min value and norm[i,1] is max value in
            dimension i of the property map)
        resolution (:obj:`int`, optional): The resolution of the colorkey.
            Defaults to 512. Higher resolution will give a smoother colorkey
            with more array points.

    Returns:
        :obj:`tuple` of :obj:`numpy array`: colorkey, (X, Y) the colorkey and the
            corresponding meshgrid of the colorkey specifying the numerical value
            of each point in the colorkey.
            X.shape=Y.shape=colorkey.shape=(resolution, resolution).
    """
    ang_grid = np.linspace(-1, 1, resolution) / (np.sqrt(2) + 1e-8)
    ang1, ang2 = np.meshgrid(ang_grid, ang_grid, indexing="ij")
    colorkey = xy_to_rgb(ang1, ang2)
    x = np.linspace(norm[0, 0], norm[0, 1], colorkey.shape[0])
    y = np.linspace(norm[1, 0], norm[1, 1], colorkey.shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    return colorkey, (X, Y)


if __name__ == "__main__":
    pass
