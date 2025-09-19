import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import darling


class TestTransforms(unittest.TestCase):
    # Tests for the darling.transforms module.

    def setUp(self):
        self.debug = False

    def test_reciprocal_basis(self):
        lattice_parameters = [2, 3, 4, 90, 90, 90]
        B = darling.transforms.reciprocal_basis(lattice_parameters)
        np.testing.assert_allclose(
            np.diag(np.linalg.inv(B)), np.array([2, 3, 4]), atol=1e-10, rtol=1e-10
        )
        for i in range(3):
            for j in range(3):
                if i != j:
                    np.testing.assert_allclose(B[i, j], 0, atol=1e-10, rtol=1e-10)

    def test_diffraction_vectors(self):
        diff_vecs, mosa, lattice_parameters, hkl, U0 = self._get_diffraction_vectors()

        np.testing.assert_allclose(diff_vecs.shape[0], mosa.shape[0])
        np.testing.assert_allclose(diff_vecs.shape[1], mosa.shape[1])
        np.testing.assert_allclose(diff_vecs.shape[2], 3)
        np.testing.assert_allclose(np.isnan(diff_vecs[..., 0]), np.isnan(mosa[..., 1]))

        Q = diff_vecs[~np.isnan(diff_vecs[..., 0])]
        Qhat = Q / np.linalg.norm(Q, axis=1, keepdims=True)

        angle_to_tensile_axis = np.degrees(np.arccos(Qhat[:, 2]))

        np.testing.assert_equal(angle_to_tensile_axis < 10, True)

        B0 = darling.transforms.reciprocal_basis(lattice_parameters)
        G0_crystal = B0 @ hkl
        G0_sample = U0 @ G0_crystal
        G0_sample_hat = G0_sample / np.linalg.norm(G0_sample)
        tdxrd_angle_to_tensile_axis = np.degrees(np.arccos(G0_sample_hat[2]))

        intragranular_mosaicity = angle_to_tensile_axis - tdxrd_angle_to_tensile_axis
        np.testing.assert_equal(np.abs(intragranular_mosaicity) < 3, True)

        if self.debug:
            fontsize = 22
            ticksize = 22
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["xtick.labelsize"] = ticksize
            plt.rcParams["ytick.labelsize"] = ticksize
            plt.rcParams["font.family"] = "Times New Roman"
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            xmin, xmax = np.percentile(angle_to_tensile_axis, [0.01, 99.99])
            bins = np.arange(xmin, xmax + 1e-8, 0.01)
            vals, bins, _ = ax.hist(
                angle_to_tensile_axis,
                bins=bins,
                color="pink",
                label="mosa-map angle to tensile axis",
            )
            ax.vlines(
                tdxrd_angle_to_tensile_axis,
                0,
                np.max(vals),
                color="blue",
                label="3DXRD angle to tensile axis",
                linestyle="--",
                linewidth=3,
            )
            ax.legend(fontsize=fontsize - 2)
            ax.set_xlabel("Angle to tensile axis (degrees)")
            ax.set_ylabel("Count")
            ax.set_title(
                "Angle to tensile (1-11) axis distribution derived from mosa-map (1st maxima)",
                fontsize=fontsize,
            )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(xmin, xmax)
            plt.tight_layout()
            plt.show()

    def test_minimal_norm_rotation(self):
        diff_vecs, mosa, lattice_parameters, hkl, U0 = self._get_diffraction_vectors()
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
        mask = ~np.isnan(diff_vecs[..., 0])
        rots = Rotation.concatenate(rotation_field[mask])
        diffrots = Rotation.from_matrix(U0.T) * rots
        np.testing.assert_equal(np.degrees(diffrots.magnitude()) < 3, True)

        B0 = darling.transforms.reciprocal_basis(lattice_parameters)
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
        mosa = darling.assets.mosa_field()

        diff_vecs = darling.transforms.diffraction_vectors(
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
