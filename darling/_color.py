"""Transforms to colorize property maps mapping between hsv and rgb spaces.

These are interal darling utility functions, see darling.properties.rgb for
an interface to the rgb function.
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb


def normalize(property_2d, norm):
    """Normalize a property map to a given range.

    Args:
        property_2d (:obj:`numpy array`): The property map to normalize, shape=(a, b, 2),
            the last two dimensions will be mapped by the norm array range.
        norm (:obj:`numpy array` or :obj:`str`): array of shape=(2, 2), norm[i,0] is min
            value for property_2d[:,:,i] and norm[i,1] is max value for property_2d[:,:,i]
            the property_2d values will thus be normalised into this range.

    Returns:
        :obj:`numpy array`: a normalized property map of shape=(a, b, 2)
    """
    norm_property_2d = np.zeros((2, property_2d.shape[0], property_2d.shape[1]))

    for i in range(2):
        # move "leftmost-bottommost" datapoint to the origin, 0
        norm_property_2d[i] = property_2d[..., i] - norm[i, 0]

        # stretch all data to the [0, 1] box
        norm_property_2d[i] = norm_property_2d[i] / (norm[i, 1] - norm[i, 0])

        # center the data around the origin, 0
        norm_property_2d[i] = norm_property_2d[i] - 0.5

        # stretch the data to a [-1, 1] box
        norm_property_2d[i] = 2 * norm_property_2d[i]

        # stretch the data to fit inside a unit circle
        norm_property_2d[i] = norm_property_2d[i] / (np.sqrt(2) + 1e-8)

    return norm_property_2d


def rgb(x, y):
    """Map 2D data to RGB color space by converting to HSV.

    The 2d points are assumed to lie on the top of the hsv color
    cylinder, with the angle of the point mapping to the hue, and the
    distance from the origin mapping to the saturation. The value is
    set to 1 for all points (brightest color).

    Args:
        x (:obj:`numpy array`): x-values cound by the unit circle, shape=(a,b)
        y (:obj:`numpy array`): y-values cound by the unit circle, shape=(a,b)

    Returns:
        :obj:`numpy array`: rgb values of shape (a, b, 3)
    """
    # angle of the point in the plane parameterised by 0,1
    angles = (np.arctan2(-y, -x) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

    # radius of the point in the plane
    radius = np.sqrt(x**2 + y**2)

    hsv = np.stack(
        (
            angles,  # HUE (the actual color)
            radius,  # SATURATION (how saturated the color is)
            np.ones(angles.shape),  # VALUE. (white to black)
        ),
        axis=2,
    )
    hsv[np.isnan(x), :] = 0

    return hsv_to_rgb(hsv)


def colorkey(norm, resolution=512):
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
    ang1, ang2 = np.meshgrid(ang_grid, ang_grid, indexing='ij')
    colorkey = rgb(ang1, ang2)
    x = np.linspace(norm[0, 0], norm[0, 1], colorkey.shape[0])
    y = np.linspace(norm[1, 0], norm[1, 1], colorkey.shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    return colorkey, (X, Y)


if __name__ == "__main__":
    pass
