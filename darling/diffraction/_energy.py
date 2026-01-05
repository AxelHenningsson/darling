import numpy as np

from darling.crystal import lattice_spacing


def ccmth_to_strain(ccmth, unit_cell, hkl, reference_bragg_angle):
    """Convert monochromator position (ccmth) to strain given a reference unit cell parameter.

    Args:
        ccmth (:obj:`numpy array` or :obj:`float`): The ccmth in degrees. shape=(n,) or float.
        unit_cell (:obj:`numpy array`): The unit cell parameters. shape=(6,) these are in Angstroms and degrees, ordered as [a,b,c,alpha,beta,gamma].
        hkl (:obj:`numpy array`): The hkl indices. shape=(3,) these are integers.
        reference_bragg_angle (:obj:`float`): The reference Bragg angle in degrees. This is the Bragg angle for the reference unit cell and the reference hkl.
        i.e it is the angle at which diffraction will be observed for a strain free unit cell.

    Returns:
        :obj:`numpy array` or :obj:`float`: The strain. shape=(n,) or float.
    """
    wavelength = ccmth_to_wavelength(ccmth)
    d0 = lattice_spacing(unit_cell, hkl)
    d = wavelength / (2 * np.sin(np.radians(reference_bragg_angle)))
    return (d - d0) / d0


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
