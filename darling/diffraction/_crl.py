import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

import darling

from ._constants import materials, required_material_keys


class CompoundRefractiveLens(object):
    """A Compound Refractive Lens.

    .. note::

        The math and notation in this class are partly based on Simons 2017:

        *Simulating and optimizing compound refractive lens-based X-ray microscopes*,
        Journal of Synchrotron Radiation, 24, 392-401.
        doi: https://doi.org/10.1107/S160057751602049X

    :param int number_of_lenses: Number of lenses.
    :param float lens_space: Space between the lens centers, in microns.
    :param float lens_radius: Lens radius of curvature at the apex, in microns.
    :param float energy: X-ray energy, in keV.
    :param float magnification: Desired magnification. When specified, the
        detector distance is computed to achieve this magnification. Exactly one
        of ``magnification`` or ``sample_to_detector_distance`` must be specified.
    :param float sample_to_detector_distance: Desired total sample-to-detector
        distance, in mm. When specified, the magnification is derived from this
        distance. This parameter is sometimes referred to as ``L`` and is the
        length of the optical path from the sample image plane to the detector.
        Exactly one of ``magnification`` or ``sample_to_detector_distance`` must
        be specified.
    :param lens_material: Lens material properties, or a key specifying a
        predefined lens material.
    :type lens_material: dict or str

    ``lens_material`` may be one of the following strings:

    - ``"beryllium"`` or ``"Be"`` for beryllium
    - ``"diamond"`` or ``"C"`` for diamond

    If ``lens_material`` is a dictionary, it must contain:

    - ``"atomic_number"``: atomic number
    - ``"density"``: density in g/cm³
    - ``"atomic_mass"``: atomic mass in g/mol

    The default is ``"Be"``.
    """

    def __init__(
        self,
        number_of_lenses,
        lens_space,
        lens_radius,
        energy,
        magnification=None,
        sample_to_detector_distance=None,
        lens_material="Be",
    ):
        self.number_of_lenses = number_of_lenses
        self.lens_space = lens_space
        self.lens_radius = lens_radius
        self.material = self._unpack_material(lens_material)

        self.energy = energy

        self._xhat_lab = np.array([1.0, 0.0, 0.0])
        self._yhat_lab = np.array([0.0, 1.0, 0.0])
        self._zhat_lab = np.array([0.0, 0.0, 1.0])

        self._imaging_system_0 = np.eye(3, 3)

        self.theta = 0
        self.eta = 0

        if magnification is not None and sample_to_detector_distance is not None:
            raise ValueError(
                f"Exactly one of `magnification` or `sample_to_detector_distance` should be specified but got both `magnification={magnification}` and `sample_to_detector_distance={sample_to_detector_distance}`."
            )
        if magnification is None and sample_to_detector_distance is None:
            raise ValueError(   
                "Exactly one of `magnification` or `sample_to_detector_distance` should be specified."
            )

        self.magnification = (
            self._find_magnification(sample_to_detector_distance)
            if sample_to_detector_distance is not None
            else magnification
        )

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = value
        self._refractive_decrement = darling.diffraction.refractive_decrement(
            self.material["atomic_number"],
            self.material["density"],
            self.material["atomic_mass"],
            self._energy,
        )

    def _find_magnification(self, sample_to_detector_distance):
        """Find the magnification that correspond to a given sample-to-detector-distance in mm.

        Args:
            detector_distance (:obj:`float`): Detector distance in mm.
        """

        L_target = sample_to_detector_distance * 1e3  # convert to microns
        if L_target < self.length:
            raise ValueError(
                f"`sample_to_detector_distance` must be greater than the lens-stack length which is {self.length} microns"
            )

        def cost(magnification):
            self.magnification = magnification
            return self.L - L_target

        magnification = brentq(cost, 1 + 1e-8, 1e16, xtol=1e-8, maxiter=99)

        return magnification

    def _unpack_material(self, lens_material):
        if isinstance(lens_material, str):
            if lens_material not in materials:
                raise ValueError(
                    f"Invalid lens material key: {lens_material}. Must be one of {list(materials.keys())}"
                )
            material = materials[lens_material]
        elif isinstance(lens_material, dict):
            keys = set(lens_material.keys())
            if not keys.issubset(required_material_keys):
                raise ValueError(
                    f"lens_material dictionary must contain the following keys: {required_material_keys}, but got {keys}"
                )
            material = lens_material
        else:
            raise ValueError(
                f"lens_material must be a string or a dictionary but got {type(lens_material)}"
            )
        return material

    def goto(self, theta, eta):
        """Go to a fixed theta eta setting, rotates imaging system.

        Args:
            theta (:obj:`float`): Bragg angle in radians.
            eta (:obj:`float`): Azimuth angle in radians.
        """
        self.theta = theta
        self.eta = eta

    @property
    def optical_axis(self):
        """optical axis for the current angles (as given in lab coordinates).

        Returns:
            (:obj:`numpy array`): optical axis. shape=(3,).
        """
        return self.imaging_system[:, 0]

    @property
    def imaging_system(self):
        """imaging coordinate system for the current angles (as given in lab coordinates).

        Returns:
            (:obj:`numpy array`): imaging coordinate system. shape=(3,3).
        """
        rotation_th = Rotation.from_rotvec(self._yhat_lab * (-2 * self.theta))
        rotation_eta = Rotation.from_rotvec(self._xhat_lab * (self.eta))
        rot = (rotation_eta * rotation_th).as_matrix()
        return rot @ self._imaging_system_0

    @property
    def T(self):
        return self.lens_space

    @property
    def R(self):
        return self.lens_radius

    @property
    def N(self):
        return self.number_of_lenses

    @property
    def delta(self):
        return self._refractive_decrement

    @property
    def f(self):
        return self.R / (2 * self.delta)

    @property
    def f_N(self):
        return self.f * self.phi * (1.0 / np.tan(self.N * self.phi))

    @property
    def phi(self):
        t2 = 1 - (self.T / (2 * self.f))
        t1 = np.sqrt(1 - (t2) ** 2)
        return np.arctan(t1 / t2)

    @property
    def M_N(self):
        Nc = np.cos(self.N * self.phi)
        Ns = np.sin(self.N * self.phi)
        s = np.sin(self.phi)
        return np.array([[Nc, self.f * Ns * s], [-Ns / (s * self.f), Nc]])

    @property
    def K(self):
        M = self.M_N
        M11, M12 = M[0]
        M21, M22 = M[1]
        d1, d2 = self.d1, self.d2

        K11 = M11 + d2 * M21
        K12 = M12 + d1 * (M11 + d2 * M21) + d2 * M22
        K21 = M21
        K22 = d1 * M21 + M22

        return np.array([[K11, K12], [K21, K22]])

    @property
    def lens_focal_length(self):
        return self.f

    @property
    def crl_focal_length(self):
        return self.f_N

    @property
    def d2(self):
        M = self.M_N
        return -(self.magnification + M[0, 0]) / M[1, 0]

    @property
    def d1(self):
        M = self.M_N
        d2 = self.d2
        return -(d2 * M[1, 1] + M[0, 1]) / (M[0, 0] + d2 * M[1, 0])

    @property
    def source_to_detector_distance(self):
        return self.L

    @property
    def length(self):
        return self.N * self.T

    @property
    def L(self):
        return self.d1 + self.d2 + self.length

    @L.setter
    def L(self, value):
        self.magnification = self._find_magnification(value)

    @property
    def sample_to_detector_distance(self):
        return self.L

    @sample_to_detector_distance.setter
    def sample_to_detector_distance(self, value):
        self.L = value

    @property
    def info(self):
        print("------------------------------------------------------------")
        print("CRL information in units of [m]")
        print("------------------------------------------------------------")
        print("Sample to crl distance (d1)     : ", self.d1 / 1e6)
        print("CRL to detector distance (d2)   : ", self.d2 / 1e6)
        print("CRL focal length (f_N)          : ", self.crl_focal_length / 1e6)
        print("single lens focal length (f)    : ", self.f / 1e6)
        print(
            "Source to detector distance (L) : ", self.source_to_detector_distance / 1e6
        )
        print("Lens spacing (T)                : ", self.T / 1e6)
        print("Number of lenses (N)            : ", self.N)
        print("Lens radius (R)                 : ", self.R / 1e6)
        print("Refractive Decrement (delta)    : ", self._refractive_decrement)
        print("------------------------------------------------------------")
        print("Refractive Decrement (delta)    : ", self._refractive_decrement)
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
