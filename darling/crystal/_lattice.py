import numpy as np


def lattice_spacing(unit_cell, hkl):
    """Get the lattice spacing for a given unit cell and hkl."""
    B = reciprocal_basis(unit_cell)
    G = B @ hkl
    return 1.0 / np.linalg.norm(G)


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
