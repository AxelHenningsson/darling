import unittest

import numpy as np

import darling


class TestAssets(unittest.TestCase):
    # Tests for the darling.io.assets module.

    def setUp(self):
        self.debug = False

    def test_mosaicity_scan(self):
        path, data, coordinates = darling.io.assets.mosaicity_scan()

        self.assertEqual(coordinates.shape[0], 2)
        self.assertEqual(coordinates.shape[1], data.shape[2])
        self.assertEqual(coordinates.shape[2], data.shape[3])
        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(coordinates.dtype == np.float32)

    def test_energy_mu_scan(self):
        path, data, coordinates = darling.io.assets.energy_mu_scan()

        self.assertEqual(coordinates.shape[0], 2)
        self.assertEqual(coordinates.shape[1], data.shape[2])
        self.assertEqual(coordinates.shape[2], data.shape[3])
        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(coordinates.dtype == np.float32)

    def test_energy_mosa_scan(self):
        path, data, coordinates = darling.io.assets.energy_mosa_scan()

        self.assertEqual(coordinates.shape[0], 3)
        self.assertEqual(coordinates.shape[1], data.shape[2])
        self.assertEqual(coordinates.shape[2], data.shape[3])
        self.assertEqual(coordinates.shape[3], data.shape[4])
        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 5)

    def test_rocking_scan(self):
        path, data, coordinates = darling.io.assets.rocking_scan()

        self.assertEqual(coordinates.shape[0], 1)
        self.assertEqual(coordinates.shape[1], data.shape[2])
        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(coordinates.dtype == np.float32)

    def test_motor_drift(self):
        path, data, coordinates = darling.io.assets.motor_drift()

        self.assertEqual(coordinates.shape[0], 2)
        self.assertEqual(coordinates.shape[1], data.shape[2])
        self.assertEqual(coordinates.shape[2], data.shape[3])
        self.assertTrue(isinstance(path, str))
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(coordinates.dtype == np.float32)

    def test_gaussian_blobs(self):
        m = 5
        N = 19
        data, coordinates = darling.io.assets.gaussian_blobs(N=N, m=m)

        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(data.shape[2] == m)
        self.assertTrue(data.shape[3] == m)
        self.assertEqual(coordinates.shape[1], data.shape[2])
        self.assertEqual(coordinates.shape[2], data.shape[3])


if __name__ == "__main__":
    unittest.main()
    unittest.main()
