import numpy as np

_UNIT_TO_RAD = {
    "rad": 1.0,
    "deg": np.pi / 180.0,
    "mrad": 1e-3,
    "urad": 1e-6,
}


def stepsize(motors, method="median"):
    """Calculate the mean or median or max or min step-size along each motor axis.

    Args:
        motors (:obj:`numpy.ndarray`):
            Motor values. Shape ``(d, m)``, ``(d, m, n)`` or ``(d, m, n, o)``,
            where ``d`` is the number of motors and the remaining dimensions
            are scan axes.
        method (str, optional):
            The method to use to calculate the step-size. Defaults to ``"median"``.
            Must be one of ``"median"`` or ``"mean"`` or ``"max"`` or ``"min"``.
            These correspond to ``numpy.median``, ``numpy.mean``, ``numpy.max``
            and ``numpy.min`` respectively.

    Returns:
        :obj:`numpy.ndarray`:
            The step-sizes. Shape ``(d,)``.
    """
    if method not in ["median", "mean", "max", "min"]:
        raise ValueError(
            f"Invalid method: {method}, must be 'median' or 'mean' or 'max' or 'min'"
        )
    method_dict = {
        "median": np.median,
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
    }
    dx = []
    numpy_method = method_dict[method]
    for ax in range(len(motors.shape[1:])):
        steps = np.diff(motors[ax], axis=ax)
        dx.append(numpy_method(steps))
    return np.array(dx)


def urad(motors, input_unit="deg"):
    """Convert motor values to micro-radians and center them on zero.

    Args:
        motors (:obj:`numpy.ndarray`):
            The motor values to convert. Shape ``(d, m)``, ``(d, m, n)`` or
            ``(d, m, o)``, where ``d`` is the number of motors and the
            remaining dimensions are arbitrary sampling axes.
        input_unit (str, optional):
            The input unit of the motor values. Defaults to ``"deg"``.
    """
    _motors = scale(motors, unit="urad", input_unit=input_unit)
    _motors = center(_motors)
    return _motors


def mrad(motors, input_unit="deg"):
    """Convert motor values to milli-radians and center them on zero.

    Args:
        motors (:obj:`numpy.ndarray`):
            The motor values to convert. Shape ``(d, m)``, ``(d, m, n)`` or
            ``(d, m, o)``, where ``d`` is the number of motors and the
            remaining dimensions are arbitrary sampling axes.
        input_unit (str, optional):
            The input unit of the motor values. Defaults to ``"deg"``.
    """
    _motors = scale(motors, unit="mrad", input_unit=input_unit)
    _motors = center(_motors)
    return _motors


def rad(motors, input_unit="deg"):
    """Convert motor values to radians and center them on zero.

    Args:
        motors (:obj:`numpy.ndarray`):
            The motor values to convert. Shape ``(d, m)``, ``(d, m, n)`` or
            ``(d, m, o)``, where ``d`` is the number of motors and the
            remaining dimensions are arbitrary sampling axes.
        input_unit (str, optional):
            The input unit of the motor values. Defaults to ``"deg"``.
    """
    _motors = scale(motors, unit="rad", input_unit=input_unit)
    _motors = center(_motors)
    return _motors


def bounds(motors):
    """Get per motor axis min and max values.

    Args:
        motors (:obj:`numpy.ndarray`):
            Motor values. Shape ``(d, m)``, ``(d, m, n)`` or ``(d, m, n, o)``,
            where ``d`` is the number of motors and the remaining dimensions
            are scan axes.

    Returns:
        :obj:`numpy.ndarray`:
            The min and max values per motor axis. Shape ``(d, 2)``,
            where ``d`` is the number of motors.
    """
    bb = np.zeros((motors.shape[0], 2), dtype=motors.dtype)
    for i in range(motors.shape[0]):
        bb[i, 0] = motors[i].min()
        bb[i, 1] = motors[i].max()
    return bb


def scale(motors, unit="mrad", input_unit="deg"):
    """Scale motor values between angular units.

    The input values are interpreted in ``input_unit`` and converted to the
    requested output ``unit``. Supported units are ``"rad"``, ``"deg"``,
    ``"mrad"``, and ``"urad"``.

    Args:
        motors (:obj:`numpy.ndarray`):
            The motor values to scale. Shape ``(d, m)``, ``(d, m, n)`` or
            ``(d, m, o)``, where ``d`` is the number of motors and the
            remaining dimensions are arbitrary sampling axes.
        unit (str, optional):
            Output angular unit, one of ``{"rad", "deg", "mrad", "urad"}``.
            Defaults to ``"mrad"``.
        input_unit (str, optional):
            Input angular unit, one of ``{"rad", "deg", "mrad", "urad"}``.
            Defaults to ``"deg"``.

    Raises:
        ValueError:
            If ``input_unit`` is not one of the supported units.
        ValueError:
            If ``unit`` is not one of the supported units.

    Returns:
        :obj:`numpy.ndarray`:
            The scaled motor values with the same shape as ``motors``.
    """
    try:
        in_to_rad = _UNIT_TO_RAD[input_unit]
    except KeyError:
        raise ValueError(f"Invalid input unit: {input_unit}")
    try:
        out_to_rad = _UNIT_TO_RAD[unit]
    except KeyError:
        raise ValueError(f"Invalid unit: {unit}")
    return motors * (in_to_rad / out_to_rad)


def center(motors):
    """Center motor values by subtracting the median per motor.

    For each motor (indexed along the first axis), the global median over its
    remaining dimensions is subtracted in place so that each motor block has
    zero median.

    Args:
        motors (:obj:`numpy.ndarray`):
            Motor values to center. Shape ``(d, m)``, ``(d, m, n)`` or
            ``(d, m, o)``, where ``d`` is the number of motors and the
            remaining dimensions are arbitrary sampling axes.

    Returns:
        :obj:`numpy.ndarray`:
            The centered motor values. (copy of the input array)
    """
    assert motors.ndim in [2, 3, 4], (
        f"motors must have 2, 3 or 4 dimensions, but got {motors.ndim}"
    )
    assert motors.shape[0] >= 1 and motors.shape[0] <= 3, (
        f"motors must have between 1 and 3 motors, but got {motors.shape[0]}"
    )
    _motors = motors.copy()
    for i in range(_motors.shape[0]):
        _motors[i] = _motors[i] - np.median(_motors[i])
    return _motors


def reduce(motors, axis):
    """Reduce the number of motor dimensions by keeping a subset of axes.

    This creates a new motor array as if the data was lower dimensional.
    The motors corresponding to the kept scan axes are returned and any
    removed scan dimensions are averaged out y using ``numpy.mean()``.

    Note that axis must be ordered monotonically increasing (0, 1, 2, ... or -2, -1, 0 etc),
    use darling.transforms.motor.reorder to reorder the motors as a secondary step if needed.

    Example:
        If ``motors.shape == (3, m, n, o)`` and ``axis = (0, 1)``, the result
        has shape ``(2, m, n)``. The third scan dimension (o) is removed by
        averaging over it, and the remaining two motors are returned.

    Args:
        motors (:obj:`numpy.ndarray`):
            Motor values. Shape ``(d, m)``, ``(d, m, n)`` or ``(d, m, n, o)``,
            where ``d`` is the number of motors and the remaining dimensions
            are scan axes. It is assumed that motor ``i`` corresponds to scan
            axis ``i``.
        axis (int or tuple of int):
            Indices of scan axes to keep. These indices refer both to the scan
            axes and to the corresponding motors. Negative indices are allowed
            and follow NumPy semantics with respect to the scan axes
            (excluding the motor axis).

    Returns:
        :obj:`numpy.ndarray`:
            Reduced motor values. Shape ``(k, ...)`` where ``k`` is the number
            of kept axes and the trailing dimensions are the kept scan axes
            in their original order.

    Raises:
        ValueError:
            If any axis is out of range or if duplicate axes are given.
    """
    assert motors.ndim in [2, 3, 4], (
        f"motors must have 2, 3 or 4 dimensions, but got {motors.ndim}"
    )
    if motors.shape[0] == 1:
        raise ValueError(
            "Cannot reduce single motor, must have at least 2 motors for reduction to be meaningful"
        )
    assert motors.shape[0] >= 2 and motors.shape[0] <= 3, (
        f"motors must have between 2 and 3 motors, but got {motors.shape[0]}"
    )

    axes_keep = np.atleast_1d(axis).astype(int)
    grid_ndim = motors.ndim - 1
    if grid_ndim < 1:
        raise ValueError("motors must have at least one scan dimension")

    axes_keep[axes_keep < 0] += grid_ndim
    if np.any((axes_keep < 0) | (axes_keep >= grid_ndim)):
        raise ValueError("axis out of range for motor grid dimensions")
    if np.unique(axes_keep).size != axes_keep.size:
        raise ValueError("duplicate axes are not allowed")
    if np.any(np.diff(axes_keep) <= 0):
        raise ValueError(
            "axes must be ordered monotonically increasing (0, 1, 2, ... or -2, -1, 0 etc), use darling.transforms.motor.reorder to reorder the motors as a secondary step"
        )

    all_axes = np.arange(grid_ndim)
    axes_reduce = np.setdiff1d(all_axes, axes_keep, assume_unique=True)

    if axes_reduce.size > 0:
        reduce_axes_full = tuple(a + 1 for a in axes_reduce)
        reduced = motors.mean(axis=reduce_axes_full)
    else:
        reduced = motors

    reduced = reduced[axes_keep]

    for ax in range(len(reduced.shape[1:])):
        assert np.median(np.diff(reduced[ax], axis=ax)) > 0, (
            "Reduced motor is not indexing=ij"
        )

    return reduced


def reorder(motors, axis, data=None):
    """Reorder motors and scan dimensions while preserving motor monotonicity.

    It is assumed that motor ``i`` varies along scan axis ``i``, that is,
    ``motors[i]`` is increasing along index ``i`` in the trailing dimensions.
    This function reorders both the motors and the scan axes so that this
    property is preserved in the new ordering.

    Example:
        If ``motors.shape == (3, m, n, o)`` and ``axis == (1, 0, 2)``, then
        the output has shape ``(3, n, m, o)``. The first motor in the result
        corresponds to the original second scan dimension, the second motor
        to the original first, and the third stays the same. In all cases,
        motor ``i`` still increases along axis ``i`` of the trailing
        dimensions. i.e. ``motors[0, :, 0, 0]`` is monotonically increasing,
        while ``motors[1, :, 0, 0]`` is constant (up to noise).

    Args:
        motors (:obj:`numpy.ndarray`):
            Motor values to reorder. Shape ``(d, m)``, ``(d, m, n)`` or
            ``(d, m, n, o)``, where ``d`` is the number of motors.
        axis (tuple or list of int):
            Permutation of the motor/scan indices ``0, 1, ..., d-1``. For
            example, ``(1, 0, 2)`` means that the first motor in the output
            corresponds to the original second scan dimension, the second
            motor to the original first, and so on.
        data (:obj:`numpy.ndarray`, optional):
            Data to reorder to match the new motor dimension. Defaults to None.
            When given it is assumed that the data shape is ``(a, b, ...)``
            where ``a, b`` are the detector dimensions and ``...`` are the
            scan dimensions which must match the input motor shape, i.e.
            ``(m,)`` or ``(m, n)`` or ``(m, n, o)``.

    Returns:
        :obj:`numpy.ndarray` or tuple:
            If ``data is None``, returns the reordered motor values with shape
            compatible with the new dimension ordering. The number of motors
            is unchanged and motor ``i`` varies along scan axis ``i``.

            If ``data`` is given, returns a tuple
            ``(reordered_motors, reordered_data)`` where ``reordered_data``
            has scan dimensions reordered consistently with ``motors``.

    Raises:
        ValueError:
            If ``axis`` is not a permutation of ``range(d)`` or if
            ``d != motors.ndim - 1`` or if shapes are inconsistent.
    """
    assert motors.ndim in [2, 3, 4], (
        f"motors must have 2, 3 or 4 dimensions, but got {motors.ndim}"
    )
    if motors.shape[0] == 1:
        raise ValueError(
            "Cannot reorder single motor, must have at least 2 motors for reordering to be meaningful"
        )
    assert motors.shape[0] >= 2 and motors.shape[0] <= 3, (
        f"motors must have between 2 and 3 motors, but got {motors.shape[0]}"
    )

    d = motors.shape[0]
    grid_ndim = motors.ndim - 1
    if d != grid_ndim:
        raise ValueError("Expected number of motors d == motors.ndim - 1")

    axes = np.atleast_1d(axis).astype(int)
    if axes.size != d:
        raise ValueError("axis must contain exactly one entry for each motor")

    axes[axes < 0] += d
    if np.any((axes < 0) | (axes >= d)):
        raise ValueError("axis indices out of range")

    if np.unique(axes).size != d:
        raise ValueError("axis must be a permutation of range(d)")

    motor_perm = axes
    dim_perm_motors = [0] + [a + 1 for a in axes]

    reordered = np.transpose(motors[motor_perm], axes=dim_perm_motors)

    if reordered.shape[0] != motors.shape[0]:
        raise ValueError("Reordered motor dimension mismatch")
    if set(reordered.shape) != set(motors.shape):
        raise ValueError("Reordered motor shape mismatch")

    for ax in range(len(motors.shape[1:])):
        if np.median(np.diff(reordered[ax], axis=ax)) <= 0:
            raise ValueError("Reordered motor is not indexing=ij")

    if data is not None:
        if data.shape[2:] != motors.shape[1:]:
            raise ValueError(
                f"Data shape mismatch, data.shape[2:] = {data.shape[2:]}, "
                f"motors.shape[1:] = {motors.shape[1:]}"
            )
        dim_perm_data = [0, 1] + [a + 2 for a in axes]
        reordered_data = np.transpose(data, axes=dim_perm_data)
        return reordered, reordered_data

    return reordered


if __name__ == "__main__":
    pass
