import numpy as np
from scipy.spatial.transform import Rotation

from darling import goniometer


def reciprocal_basis(lattice_parameters, degrees=True):
    """Calculate the reciprocal basis vectors.

    Calculate B matrix such that B^-T contains the reals space lattice vectors as columns.

    Args:
        lattice_parameters (:obj:`numpy array` or :obj:`list`): unit cell parameters [a,b,c,alpha,beta,gamma].
        degrees (:obj:`bool`, optional): If True, the angles (alpha, beta, gamma) are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        B (:obj:`numpy array`): The B matrix. ``shape=(3,3)``
    """
    a, b, c = lattice_parameters[0:3]
    alpha, beta, gamma = (
        np.radians(lattice_parameters[3:]) if degrees else lattice_parameters[3:]
    )
    calp = np.cos(alpha)
    cbet = np.cos(beta)
    cgam = np.cos(gamma)
    salp = np.sin(alpha)
    sbet = np.sin(beta)
    sgam = np.sin(gamma)
    V = (
        a
        * b
        * c
        * np.sqrt(1 - calp * calp - cbet * cbet - cgam * cgam + 2 * calp * cbet * cgam)
    )
    astar = 2 * np.pi * b * c * salp / V
    bstar = 2 * np.pi * a * c * sbet / V
    cstar = 2 * np.pi * a * b * sgam / V
    sbetstar = V / (a * b * c * salp * sgam)
    sgamstar = V / (a * b * c * salp * sbet)
    cbetstar = (calp * cgam - cbet) / (salp * sgam)
    cgamstar = (calp * cbet - cgam) / (salp * sbet)
    B = np.array(
        [
            [astar, bstar * cgamstar, cstar * cbetstar],
            [0, bstar * sgamstar, -cstar * sbetstar * calp],
            [0, 0, cstar * sbetstar * salp],
        ]
    )
    return B / 2 / np.pi


def _as_scipy_rotation(rotation):
    """Convert the rotation to a scipy.spatial.transform.Rotation object."""
    return (
        rotation if isinstance(rotation, Rotation) else Rotation.from_matrix(rotation)
    )


def _compute_diffraction_vectors(
    grain_orientation,
    lattice_parameters,
    hkl,
    mu,
    omega,
    chi,
    phi,
    frame="lab",
    degrees=True,
):
    """Compute the diffraction vectors.

    NOTE: Here the arrays cannot contain nans, i.e these are the masked angular arrays.

    Computation is done in the following order:
        1. Convert hkl into the crystal frame via reciprocal_basis.
        2. Convert crystal diffraction vectors into the sample frame via grain_orientation.
        3. Convert sample diffraction vectors into the lab frame via goniometer.total_rotation (using the mean angular position of goniometer).
        4. Convert lab diffraction vectors into the requested frame via _from_lab_to_frame.

    Args:
        grain_orientation (:obj:`numpy array` or :obj:`scipy.spatial.transform.Rotation`): The grain orientation as shape=(3,3) matrix or rotation object.
        lattice_parameters (:obj:`numpy array`): The lattice parameters. shape=(6,)
        hkl (:obj:`numpy array`): The hkl indices. shape=(3,)
        mu (:obj:`float` or :obj:`numpy.ndarray`): The mu angle. shape=(n,) or float. If array, must not contain nans.
        omega (:obj:`float` or :obj:`numpy.ndarray`): The omega angle. shape=(n,) or float. If array, must not contain nans.
        chi (:obj:`float` or :obj:`numpy.ndarray`): The chi angle. shape=(n,) or float. If array, must not contain nans.
        phi (:obj:`float` or :obj:`numpy.ndarray`): The phi angle. shape=(n,) or float. If array, must not contain nans.
        frame (:obj:`str`, optional): The frame of the diffraction vectors. Defaults to "lab". Options are "lab", "sample" or "crystal".
        degrees (:obj:`bool`, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`numpy.ndarray`: The diffraction vectors. shape=(n,3)
    """
    U0 = _as_scipy_rotation(grain_orientation)
    goniometer_rotations = goniometer.total_rotation(mu, omega, chi, phi)
    B0 = reciprocal_basis(lattice_parameters)
    G0_crystal = B0 @ hkl
    G0_sample = U0.apply(G0_crystal)
    G_lab = goniometer_rotations.apply(G0_sample)
    G_frame = _from_lab_to_frame(G_lab, frame, mu, omega, chi, phi, U0, degrees=degrees)
    return G_frame


def _from_lab_to_frame(G_lab, frame, mu, omega, chi, phi, U0, degrees=True):
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
        gamma = goniometer.median_rotation(mu, omega, chi, phi, degrees=degrees)
        if frame == "sample":
            return gamma.inv().apply(G_lab)
        elif frame == "crystal":
            return (U0.inv() * gamma.inv()).apply(G_lab)
    else:
        raise ValueError(
            f"Expected input frame to be one of lab, sample or crystal but got frame={frame}"
        )


def diffraction_vectors(
    grain_orientation,
    lattice_parameters,
    hkl,
    mu,
    omega,
    chi,
    phi,
    frame="lab",
    mask=None,
    degrees=True,
):
    """Compute the diffraction vectors over a spatial field.

    Diffraction vectors, G, are defined from the Laue equation:

        G_lab = O @ U @ B @ hkl

    where O is the goniometer rotation, U is the grain orientation, B is the reciprocal basis
    and hkl are the integer Miller indices. The vector

        G_crystal = B @ hkl

    is the diffraction vector in the crystal frame and the vector

        G_sample = U @ G_crystal

    is the diffraction vector in the sample frame. The rotation O is the goniometer rotation which maps
    from sample space to lab space as

        G_lab = O @ G_sample

    The lab reference frame is defined with x along the x-ray beam propagation direction, z towards the
    roof and y traverse, such that x, y and z form a right-handed coordinate system. The definition of
    U and B follows 3DXRD conventions.

    See also Poulsen 2004: https://orbit.dtu.dk/en/publications/3dxrd-a-new-probe-for-materials-science

    NOTE: Sometimes G is denoted as Q in DFXM.

    Here, any of the input angles can be a scalar or an array. The largest dimension of the input angles will be used
    to determine the shape of the output diffraction vector field. For instace if mu is mu.shape=(m,n) then the output
    diffraction vector field will have shape=(m,n,3) etc.

    Computation is done in the following order:
        1. Convert hkl into the crystal frame via reciprocal_basis.
        2. Convert crystal diffraction vectors into the sample frame via grain_orientation.
        3. Convert sample diffraction vectors into the lab frame via goniometer.total_rotation (using the mean angular position of goniometer).
        4. Convert lab diffraction vectors into the requested frame via _from_lab_to_frame.

    NOTE: if no mask is provided a mask will be impliclty created as ~np.isnan(angle) where angle is one of mu, omega, chi or phi,
    depending on which one/ones of the input angles have a size>1

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
        mosa = darling.assets.mosa_field()

        diff_vecs = darling.transforms.diffraction_vectors(
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


    Args:
        grain_orientation (:obj:`numpy array` or :obj:`scipy.spatial.transform.Rotation`): The grain orientation as shape=(3,3) matrix or rotation object.
        lattice_parameters (:obj:`numpy array`): The lattice parameters. shape=(6,)
        hkl (:obj:`numpy array`): The hkl indices. shape=(3,)
        mu (:obj:`float` or :obj:`numpy.ndarray`): The mu angle. shape=(n,) or float
        omega (:obj:`float` or :obj:`numpy.ndarray`): The omega angle. shape=(n,) or float.
        chi (:obj:`float` or :obj:`numpy.ndarray`): The chi angle. shape=(n,) or float.
        phi (:obj:`float` or :obj:`numpy.ndarray`): The phi angle. shape=(n,) or float..
        frame (:obj:`str`, optional): The frame of the diffraction vectors. Defaults to "lab". Options are "lab", "sample" or "crystal".
        mask (:obj:`bool`, optional): The mask to apply to the diffraction vectors. Defaults to None.
        degrees (:obj:`bool`, optional): If True, the angles are assumed to be in degrees, otherwise they are in radians. Defaults to True.

    Returns:
        :obj:`numpy.ndarray`: The diffraction vectors as a field. shape=(n,3) or shape=(m,n,3) etc. depending on input angles. The reference
            frame of the diffraction vectors is the one specified by the frame argument.
    """
    angles_flat_and_filtered = []
    angular_mask = None if mask is None else mask
    for ang in [mu, omega, chi, phi]:
        if not isinstance(ang, float):
            angular_mask = ~np.isnan(ang) if angular_mask is None else angular_mask
            angles_flat_and_filtered.append(ang[angular_mask])
        else:
            angles_flat_and_filtered.append(ang)

    diff_vecs = _compute_diffraction_vectors(
        grain_orientation,
        lattice_parameters,
        hkl,
        *angles_flat_and_filtered,
        frame,
        degrees,
    )

    if diff_vecs.size == 3:  # input was scalar, no array ranges of angles.
        return diff_vecs

    diff_vec_field = np.full((*angular_mask.shape, 3), fill_value=np.nan)
    diff_vec_field[angular_mask] = diff_vecs

    return diff_vec_field


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
        mosa = darling.assets.mosa_field()

        diff_vecs = darling.transforms.diffraction_vectors(
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

        rotation_field = darling.transforms.minimal_norm_rotation(
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
    B0 = reciprocal_basis(lattice_parameters)
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
