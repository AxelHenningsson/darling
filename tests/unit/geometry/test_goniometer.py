import unittest

import numpy as np

import darling


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_total_rotation_zero(self):
        rotation = darling.geometry.total_rotation(
            0, 0, 0, 0, degrees=True, rotation_representation="euler-xyz"
        )
        np.testing.assert_allclose(rotation, np.zeros(3))

    def test_total_rotation_non_zero(self):
        rotation = darling.geometry.total_rotation(
            1, 2, 3, 4, degrees=True, rotation_representation="euler-XYX"
        )
        np.testing.assert_allclose(
            rotation, np.array([36.35639315, 3.72013452, -33.44540495])
        )

    def test_median_rotation_zero(self):
        rotation = darling.geometry.median_rotation(
            0, 0, 0, 0, degrees=True, rotation_representation="euler-xyz"
        )
        np.testing.assert_allclose(rotation, np.zeros(3))

    def test_median_rotation_array(self):
        mu = np.array([0, 1, 1, 1])
        rotation = darling.geometry.median_rotation(
            mu, 0, 0, 0, degrees=True, rotation_representation="euler-xyz"
        )
        np.testing.assert_allclose(rotation, np.array([0, -1, 0]))

    def test_median_rotation_double_arrays(self):
        mu = np.array([0, 1, 1, 1])
        omega = np.array([0, 2, 2, 2])
        rotation = darling.geometry.median_rotation(
            mu, omega, 0, 0, degrees=True, rotation_representation="quat"
        )
        np.testing.assert_allclose(
            rotation,
            np.array([-1.522990e-04, -8.725206e-03, 1.745174e-02, 9.998096e-01]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_mosa_map(self):
        path_to_data_1, _, _ = darling.io.assets.mosaicity_scan()
        self.dset = darling.DataSet(path_to_data_1, scan_id="1.1")

        mean, covariance = darling.properties.moments(self.dset.data, self.dset.motors)

        rotation = darling.geometry.total_rotation(
            mean[..., 1].flatten(),
            0,
            mean[..., 0].flatten(),
            0,
            degrees=True,
            rotation_representation="euler-xyz",
        )


if __name__ == "__main__":
    unittest.main()
