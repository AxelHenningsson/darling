import numpy as np
from scipy.spatial.transform import Rotation


def field_rotation(
    reference_vector,
    rotation_axes,
    rotation_angle_fields,
    degrees=False,
):
    """Rotate a reference vector by an n-dimensional field of rotation angles.

    The rotational sequence is defined by the input rotation axes which defines
    the ordering of rotations.

    Example:

    .. code-block:: python

        import numpy as np
        from darling import transforms

        reference_vector = np.array([1, 0, 0])
        rotation_axes = np.array([[0, 1, 0]])
        rotation_angle_fields = np.array([[1, -2], [3, 4.3]])
        rotated_vectors = transforms.field_rotation(
            reference_vector,
            rotation_axes,
            rotation_angle_fields,
            degrees=True,
        )


    Args:
        reference_vector (:obj:`numpy array`): shape=(3,) array
        rotation_axes (:obj:`numpy array`): shape=(n, 3) array. Each row is a
            rotation axis. rotation_axes[0] is the first rotation axis to be applied
            and so on. I.e when representing a sequence of motors, the bottom motor
            is at rotation_axes[-1] and the top motor is at rotation_axes[0].
        rotation_angle_fields (:obj:`numpy array`): shape=(m,n,...,p) or shape=(m,n,...) array.
            Each element is a rotation angle.
        degrees (:obj:`bool`): If True, the rotation angles are in degrees,
            otherwise they are in radians. Defaults to False.

    Returns:
        rotated_vector (:obj:`numpy array`): n-dimensional field of rotated vectors, shape=(m,n,...,3).
    """
    assert reference_vector.shape == (3,), "reference_vector must be a shape=(3,) array"
    assert rotation_axes.ndim == 2 or rotation_axes.ndim == 1, (
        "rotation_axes must be a shape=(n, 3) or shape=(3, ) array"
    )

    p = 1 if rotation_axes.shape == (3,) else rotation_axes.shape[0]

    if p == 1 and rotation_angle_fields.shape[-1] != 1:
        field_shape = rotation_angle_fields.shape
    else:
        field_shape = rotation_angle_fields.shape[0:-1]

    # we start with a constant reference vector for each pixel, these will be mutated
    # in place to produce the final rotated vector field following the sequence of rotations
    # defined by the rotation axes and the angular fields.
    v = np.full((np.prod(field_shape), 3), fill_value=reference_vector)

    # for each rotation, we apply the rotation to the vector field, by:
    for i in range(p):
        # 1: Fetching the angular field for the current rotation, one angle per pixel
        angle_field_2d = rotation_angle_fields.reshape(*field_shape, p)[..., i]

        # 2: Fetching the axis for the current rotation, one axis for all pixels
        axis = rotation_axes.reshape(p, 3)[i]
        normalized_axis = axis / np.linalg.norm(axis)

        # 3: Computing the rotation to be applied to the vector field at each pixel
        rotation_vector = angle_field_2d.reshape(-1, 1) * normalized_axis
        R = Rotation.from_rotvec(rotation_vector, degrees=degrees)

        # 4: Finally, we mutate the vector field with the per-pixel rotations
        v = R.apply(v)

    return v.reshape(*field_shape, 3)


if __name__ == "__main__":
    pass
