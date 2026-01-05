import numpy as np
from scipy.spatial.transform import Rotation

import darling


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
    return as_rotation_representation(
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
    return as_rotation_representation(
        rotation, rotation_representation, degrees=degrees
    )


def _is_scalar(x):
    return isinstance(x, float) or isinstance(x, int)


def _as_scipy_rotation(rotation):
    """Convert the rotation to a scipy.spatial.transform.Rotation object."""
    return (
        rotation if isinstance(rotation, Rotation) else Rotation.from_matrix(rotation)
    )


def ccmth_to_wavelength(ccmth):
    """Convert monochromator position (ccmth) to wavelength.

    NOTE: we use d_Si_111 = 3.1384245 as the calibrated value for the Si(111) reflection, which
    is the monochromator material at ID03 ESRF.

    This was derived from a 3DXRD measurement assuming no offsets in ccmth using a calibration
    crystal that was known to be strain-free.

    Args:
        ccmth (:obj:`numpy array` or :obj:`float`): The ccmth in degrees. shape=(n,) or float.

    Returns:
        :obj:`numpy array` or :obj:`float`: The wavelength in Angstroms. shape=(n,) or float.
    """
    d_Si_111 = 3.1384245
    return d_Si_111 * np.sin(np.radians(ccmth)) * 2


def ccmth_to_energy(ccmth):
    """Convert monochromator position (ccmth) to energy.

    NOTE: we use d_Si_111 = 3.1384245 as the calibrated value for the Si(111) reflection, which
    is the monochromator material at ID03 ESRF.

    This was derived from a 3DXRD measurement assuming no offsets in ccmth using a calibration
    crystal that was known to be strain-free.

    Args:
        ccmth (:obj:`numpy array` or :obj:`float`): The ccmth in degrees. shape=(n,) or float.

    Returns:
        :obj:`numpy array` or :obj:`float`: The energy in keV. shape=(n,) or float.
    """
    return wavelength_to_energy(ccmth_to_wavelength(ccmth))


def wavelength_to_energy(wavelength):
    """Convert wavelength to energy.

    Args:
        wavelength (:obj:`float`): The wavelength in Angstroms.

    Returns:
        :obj:`float`: The energy in keV.
    """
    return 12.398419874273968 / wavelength


def energy_to_wavelength(energy):
    """Convert energy to wavelength.

    Args:
        energy (:obj:`float`): The energy in keV.

    Returns:
        :obj:`float`: The wavelength in Angstroms.
    """
    return 12.398419874273968 / energy


def _as_scipy_rotation(rotation):
    """Convert the rotation to a scipy.spatial.transform.Rotation object."""
    return (
        rotation if isinstance(rotation, Rotation) else Rotation.from_matrix(rotation)
    )


def from_lab_to_frame(G_lab, frame, mu, omega, chi, phi, U0, degrees=True):
    """Convert the diffraction vectors from the lab frame to the requested frame.

    Args:
        G_lab (:obj:`numpy.ndarray`): The diffraction vectors in the lab frame. shape=(n,3)
        frame (str): The frame to convert to. Options are "lab", "sample" or "crystal".
        mu (:obj:`float` or :obj:`numpy.ndarray`): The mu angle. shape=(n,) or float.
        omega (:obj:`float` or :obj:`numpy.ndarray`): The omega angle. shape=(n,) or float.
        chi (:obj:`float` or :obj:`numpy.ndarray`): The chi angle. shape=(n,) or float.
        phi (:obj:`float` or :obj:`numpy.ndarray`): The phi angle. shape=(n,) or float.
        U0 (:obj:`scipy.spatial.transform.Rotation`): The grain orientation.
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`numpy.ndarray`: The diffraction vectors in the requested frame. shape=(n,3)
    """
    if frame == "lab":
        return G_lab
    elif frame == "sample" or frame == "crystal":
        gamma = median_rotation(mu, omega, chi, phi, degrees=degrees)
        if frame == "sample":
            return gamma.inv().apply(G_lab)
        elif frame == "crystal":
            return (U0.inv() * gamma.inv()).apply(G_lab)
    else:
        raise ValueError(
            f"Expected input frame to be one of lab, sample or crystal but got frame={frame}"
        )


def _cross(a, b):
    return np.stack(
        [
            a[1] * b[:, 2] - a[2] * b[:, 1],
            a[2] * b[:, 0] - a[0] * b[:, 2],
            a[0] * b[:, 1] - a[1] * b[:, 0],
        ],
        axis=1,
    )


def align_vector_bundle(target_vector, vector_bundle):
    """
    Align a target vector with a vector bundle.

    Args:
        target_vector (:obj:`numpy.ndarray`): The target vector. shape=(3,)
        vector_bundle (:obj:`numpy.ndarray`): The vector bundle. shape=(n,3)

    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The minimal rotations needed to align the target vector with each
            vector in the vector bundle. len=n
    """
    v1_normalised = target_vector / np.linalg.norm(target_vector)
    v2_normalised = vector_bundle / np.linalg.norm(vector_bundle, axis=1, keepdims=True)
    axes = _cross(v1_normalised, v2_normalised)
    norms = np.linalg.norm(axes, axis=1)[:, None]
    axes = np.divide(axes, norms, out=np.zeros_like(axes), where=norms > 1e-12)
    angles = np.arccos(np.clip(v2_normalised @ v1_normalised, -1, 1))
    rotvecs = axes * angles[:, None]
    rotations = Rotation.from_rotvec(rotvecs)
    return rotations, angles


def minimal_norm_rotation(
    diffraction_vector_field,
    lattice_parameters,
    hkl,
    grain_orientation=None,
    mask=None,
    rotation_representation="object",
    degrees=True,
    difference_rotation=False,
):
    """Find the smallest rotation elements in SO3 that aligns a mean diffraction vector with a field of target diffraction vectors.

    This function is usefull when estimating local grain orientations form an angular DFXM scan.

    Given that the input ``diffraction_vector_field`` are defomred versions of a fixed diffraction vector,

        G0_sample = ``grain_orientation`` @ B @ hkl

    This function finds orientation elements that can perturb ``grain_orientation`` such that G0_sample is aligned with the
    input ``diffraction_vector_field`` at each point. I.e the output rotations represent the local grain orientation, U=U(x)
    computed as a sequential transform, first rotating to the ``grain_orientation``, and then applying the incremental rotations
    to reach each diffraction vector in the ``diffraction_vector_field``. (when ``difference_rotation`` is True, only the
    incremental part of the rotation is returned)

    NOTE: When the ``grain_orientation`` is the identity matrix, the input ``diffraction_vector_field`` is expected to be in crystal coordinates.
    Otherwise it is expected to be in sample coordinates.

    Example use-case for a mosaicity-scan:

    .. code-block:: python

        import numpy as np
        import darling

        U0 = np.array(
            [
                [0.83550027, 0.33932153, -0.43220389],
                [0.01167249, 0.77541728, 0.63134127],
                [0.54936605, -0.53253069, 0.64390062],
            ]
        )
        hkl = np.array([1, -1, 1])
        lattice_parameters = [4.05, 4.05, 4.05, 90, 90, 90]
        mosa = darling.io.assets.mosa_field()

        diff_vecs = darling.diffraction.diffraction_vectors(
            grain_orientation=U0,
            lattice_parameters=lattice_parameters,
            hkl=hkl,
            mu=mosa[..., 1],
            omega=17.83,
            chi=mosa[..., 0],
            phi=0.61,
            frame="sample",
            mask=None,
            degrees=True,
        )

        rotation_field = darling.geometry.minimal_norm_rotation(
            diff_vecs,
            lattice_parameters,
            hkl,
            grain_orientation=U0,
            mask=None,
            rotation_representation="object",
            degrees=True,
            difference_rotation=False,
        )


    Args:
        diffraction_vector_field (:obj:`numpy.ndarray`): The diffraction vector field. shape=(n,3) or shape=(m,n,3) etc. When a `grain_orientation` is provided,
            the `diffraction_vector_field` is expected to be in sample frame. Otherwise it is expected to be in crystal coordinates.
        lattice_parameters (:obj:`numpy array`): The lattice parameters. shape=(6,)
        hkl (:obj:`numpy array`): The hkl indices. shape=(3,)
        grain_orientation (:obj:`numpy array` or :obj:`scipy.spatial.transform.Rotation`, optional): The grain orientation as shape=(3,3) matrix or rotation object. Defaults to `np.eye(3)`
            in which case the input `diffraction_vector_field` is expected to be in crystal coordinates. Otherwise `diffraction_vector_field` is expected to be in sample coordinates.
        mask (:obj:`bool`, optional): The mask to apply to the diffraction vectors. Defaults to None.
        rotation_representation (:obj:`str`, optional): The representation of the rotation. Defaults to "object". Options are "object", "quat", "matrix", "rotvec". Additionally, "euler-seq"
            can be passed where seq are 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'}
            for extrinsic rotations. Adjacent axes cannot be the same. Extrinsic and intrinsic rotations cannot be mixed in one
            function call. This is a wrapper around scipy.spatial.transform.Rotation, for more details see the documentation for that class:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        degrees (:obj:`bool`, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.
        difference_rotation (:obj:`bool`, optional): If True, an incremental rotation is returned representing the difference rotation needed to bring the reference diffraction vector
            to the input `diffraction_vector_field`. If False, the full rotation is returned i.e the rotations represents the local grain orientation, U=U(x). Defaults to False.

    Returns:
        :obj:`numpy.ndarray`: The minimal norm rotations. shape=(n,3) or shape=(m,n,3) etc.
    """
    grain_orientation = np.eye(3) if grain_orientation is None else grain_orientation
    angular_mask = ~np.isnan(diffraction_vector_field[..., 0]) if mask is None else mask
    U0 = _as_scipy_rotation(grain_orientation)
    B0 = darling.crystal.reciprocal_basis(lattice_parameters)
    G0_crystal = B0 @ hkl
    G0_sample = U0.apply(G0_crystal)
    rotations, angles = align_vector_bundle(
        G0_sample, diffraction_vector_field[angular_mask]
    )

    if np.any(angles > np.radians(10)):
        frame = "crystal" if np.allclose(grain_orientation, np.eye(3)) else "sample"
        raise ValueError(
            f"Large rotations detected, note that minimal_rotations does not take symmetry groups into account. This is not safe. Expected `diffraction_vector_field` to be in {frame} coordinates."
        )

    rotations = rotations if difference_rotation else rotations * U0

    rotations = as_rotation_representation(
        rotations, rotation_representation, degrees=degrees
    )

    if rotation_representation == "object":
        rotation_field = np.full(angular_mask.shape, dtype=object, fill_value=np.nan)
    else:
        rotation_field = np.full(
            (*angular_mask.shape, rotations.shape[1]), fill_value=np.nan
        )

    rotation_field[angular_mask] = rotations

    return rotation_field


def as_rotation_representation(rotation, rotation_representation, degrees=True):
    """Convert the rotation to the requested representation.

    Args:
        rotation (:obj:`scipy.spatial.transform.Rotation`): The rotation to convert.
        rotation_representation (str): The representation of the rotation. Options are "object", "quat", "matrix", "rotvec". Additionally, "euler-seq"
        can be passed where seq are 3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'}
        for extrinsic rotations. Adjacent axes cannot be the same. Extrinsic and intrinsic rotations cannot be mixed in one
        function call. This is a wrapper around scipy.spatial.transform.Rotation, for more details see the documentation for that class:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        degrees (bool, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True. Only used for euler-seq.


    Returns:
        :obj:`scipy.spatial.transform.Rotation`: The rotation in the requested representation. Dimensions will match the input array.
    """
    if rotation_representation == "object":
        return rotation
    elif rotation_representation.startswith("euler"):
        seq = rotation_representation.split("-")[1]
        return rotation.as_euler(seq, degrees=degrees)
    elif rotation_representation == "quat":
        return rotation.as_quat()
    elif rotation_representation == "matrix":
        return rotation.as_matrix()
    elif rotation_representation == "rotvec":
        return rotation.as_rotvec()
    else:
        raise ValueError(
            f"no such rotation representation implemented : {rotation_representation}"
        )


if __name__ == "__main__":
    pass
if __name__ == "__main__":
    pass
