import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import darling


class TestMinimalNormRotation(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_minimal_norm_rotation(self):
        diff_vecs, mosa, lattice_parameters, hkl, U0 = self._get_diffraction_vectors()
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
        mask = ~np.isnan(diff_vecs[..., 0])
        rots = Rotation.concatenate(rotation_field[mask])
        diffrots = Rotation.from_matrix(U0.T) * rots
        np.testing.assert_equal(np.degrees(diffrots.magnitude()) < 3, True)

        B0 = darling.crystal.reciprocal_basis(lattice_parameters)
        G_sample = rots.apply(B0 @ hkl)
        G_sample_hat = G_sample / np.linalg.norm(G_sample, axis=1, keepdims=True)
        angle_to_tensile_axis = np.degrees(np.arccos(G_sample_hat[:, 2]))

        G_sample_tdxrd = U0 @ B0 @ hkl
        G_sample_hat_tdxrd = G_sample_tdxrd / np.linalg.norm(G_sample_tdxrd)

        tdxrd_angle_to_tensile_axis = np.degrees(np.arccos(G_sample_hat_tdxrd[2]))

        diff = angle_to_tensile_axis - tdxrd_angle_to_tensile_axis

        np.testing.assert_equal(np.abs(diff) < 3, True)

    def _get_diffraction_vectors(self):
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
            omega=17.834999944120554,
            chi=mosa[..., 0],
            phi=0.6099709562089967,
            frame="sample",
            mask=None,
            degrees=True,
        )
        return diff_vecs, mosa, lattice_parameters, hkl, U0


if __name__ == "__main__":
    unittest.main()
