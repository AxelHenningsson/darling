import unittest

import numpy as np

import darling


class TestReciprocalBasis(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_reciprocal_basis(self):
        lattice_parameters = [2, 3, 4, 90, 90, 90]
        B = darling.crystal.reciprocal_basis(lattice_parameters)
        np.testing.assert_allclose(
            np.diag(np.linalg.inv(B)), np.array([2, 3, 4]), atol=1e-10, rtol=1e-10
        )
        for i in range(3):
            for j in range(3):
                if i != j:
                    np.testing.assert_allclose(B[i, j], 0, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
