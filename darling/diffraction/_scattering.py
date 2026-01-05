import numpy as np
from scipy.spatial.transform import Rotation

import darling


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
    goniometer_rotations = darling.geometry.total_rotation(mu, omega, chi, phi)
    B0 = darling.crystal.reciprocal_basis(lattice_parameters)
    G0_crystal = B0 @ hkl
    G0_sample = U0.apply(G0_crystal)
    G_lab = goniometer_rotations.apply(G0_sample)
    G_frame = darling.geometry.from_lab_to_frame(
        G_lab, frame, mu, omega, chi, phi, U0, degrees=degrees
    )
    return G_frame


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
