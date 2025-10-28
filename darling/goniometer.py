import numpy as np
from scipy.spatial.transform import Rotation

from darling import transforms


def total_rotation(mu, omega, chi, phi, degrees=True, rotation_representation="object"):
    """Goniometer rotation for the ID03 DFXM microscope (september 2025).

    This class is currently implemneted for the case of Dark Field X-ray Microscopy (DFXM) where
    the gonimeter/hexapod has 4 degrees of freedom (mu, omega, chi, phi). Stacked as:

        (1) base : mu
        (2) bottom : omega
        (3) top 1    : chi
        (4) top 2    : phi

    Here mu is a rotation about the negative y-axis, omega is a positive rotation about the
    z-axis, chi is a positive rotation about the x-axis, and phi is a positive rotation about
    the y-axis.

    Args:
        mu (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for the base cradle motor, mu. shape=(n,) or float.
        omega (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for omega motor. shape=(n,) or float.
        chi (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for chi motor. shape=(n,) or float.
        phi (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for the top motor, phi. shape=(n,) or float.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.
        rotation_representation (str, optional): The representation of the rotation. Defaults to "object" in which case the rotation is
        returned as a scipy.spatial.transform.Rotation object. Other options are "quat", "matrix", "rotvec". Additoinally, "euler-seq"
        can be passed where seq are 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'}
        for extrinsic rotations. Adjacent axes cannot be the same. Extrinsic and intrinsic rotations cannot be mixed in one
        function call. This is a wrapper around scipy.spatial.transform.Rotation, for more details see the documentation for that class:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The total rotation of the stage. Dimensions will match the largest input array.
    """
    rotation = (
        mu_rotation(mu, degrees=degrees)
        * omega_rotation(omega, degrees=degrees)
        * chi_rotation(chi, degrees=degrees)
        * phi_rotation(phi, degrees=degrees)
    )
    return transforms.as_rotation_representation(
        rotation, rotation_representation, degrees=degrees
    )


def mu_rotation(mu, degrees=True):
    """Rotation about the negative y-axis.

    Args:
        mu (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for the base cradle motor, mu.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The rotation about the negative y-axis. Dimensions will match the input array.
    """
    yhat = np.array([0, 1, 0])
    return _broadcast(mu, -yhat, degrees=degrees)


def omega_rotation(omega, degrees=True):
    """Rotation about the z-axis.

    Args:
        omega (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for omega motor.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The rotation about the z-axis. Dimensions will match the input array.
    """
    zhat = np.array([0, 0, 1])
    return _broadcast(omega, zhat, degrees=degrees)


def chi_rotation(chi, degrees=True):
    """Rotation about the x-axis.

    Args:
        chi (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for chi motor.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The rotation about the x-axis. Dimensions will match the input array.
    """
    xhat = np.array([1, 0, 0])
    return _broadcast(chi, xhat, degrees=degrees)


def phi_rotation(phi, degrees=True):
    """Rotation about the y-axis.

    Args:
        phi (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for the top motor.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The rotation about the y-axis. Dimensions will match the input array.
    """
    yhat = np.array([0, 1, 0])
    return _broadcast(phi, yhat, degrees=degrees)


def _broadcast(angle, axis, degrees):
    """This is a helper function to broadcast the angle to the axis and return a rotation object when angle is an array.

    Args:
        angle (:obj:`float` or :obj:`numpy.ndarray`): The angle to broadcast. shape=(n,) or float.
        axis (:obj:`numpy.ndarray`): The axis to broadcast the angle to. shape=(3,)
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians.

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The broadcasted rotation. Dimensions will match the input array.
    """
    rotax = (
        axis.reshape(3, 1)
        if (isinstance(angle, np.ndarray) and angle.size > 1)
        else axis
    )
    return Rotation.from_rotvec((angle * rotax).T, degrees=degrees)


def median_rotation(
    mu, omega, chi, phi, degrees=True, rotation_representation="object"
):
    """Median rotation for the goniometer. Usefull when mu, omega, chi or phi are arrays of size>1 and a single
    reference rotation is needed. The median is taken over the angles for each motor.

    Input angles can be either float or np.ndarray of arbitrary dimensions. angular medians are computed over
    all elements of the array that are not numpy.nans.

    Args:
        mu (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for the base cradle motor, mu.
        omega (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for omega motor.
        chi (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for chi motor.
        phi (:obj:`float` or :obj:`numpy.ndarray`): Goniometer angle for the top motor.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.
        rotation_representation (str, optional): The representation of the rotation. Defaults to "object" in which case the rotation is
        returned as a scipy.spatial.transform.Rotation object. Other options are "quat", "matrix", "rotvec". Additoinally, "euler-seq"
        can be passed where seq are 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'}
        for extrinsic rotations. Adjacent axes cannot be the same. Extrinsic and intrinsic rotations cannot be mixed in one
        function call. This is a wrapper around scipy.spatial.transform.Rotation, for more details see the documentation for that class:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        ValueError: Expected input goniometer angles (mu, omega, chi, phi) to be either float or np.ndarray of size>1

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The median rotation. Dimensions will match the input array.
    """
    for ang in [mu, omega, chi, phi]:
        if not _is_scalar(ang) and not ang.size > 1:
            raise ValueError(
                "Expected input goniometer angles (mu, omega, chi, phi) to be either float or np.ndarray of size>1"
            )
    mean_mu = mu if _is_scalar(mu) else np.median(mu[~np.isnan(mu)])
    mean_omega = omega if _is_scalar(omega) else np.median(omega[~np.isnan(omega)])
    mean_chi = chi if _is_scalar(chi) else np.median(chi[~np.isnan(chi)])
    mean_phi = phi if _is_scalar(phi) else np.median(phi[~np.isnan(phi)])
    rotation = total_rotation(mean_mu, mean_omega, mean_chi, mean_phi, degrees=degrees)
    return transforms.as_rotation_representation(
        rotation, rotation_representation, degrees=degrees
    )


def _is_scalar(x):
    return isinstance(x, float) or isinstance(x, int)


if __name__ == "__main__":
    pass
