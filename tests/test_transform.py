import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from darling import transforms


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_field_rotation_multiple_axes(self):
        reference_vector = np.array([1, 0, 0])
        rotation_axes = np.array([[0, 1, 0], [0, 0, 1]])
        rotation_angle_fields = np.array([[1, 2], [3, 4]])
        rotated_vector = transforms.field_rotation(
            reference_vector,
            rotation_axes,
            rotation_angle_fields,
            degrees=True,
        )

        self.assertEqual(rotated_vector.shape, (2, 3))

        for i in range(rotation_angle_fields.shape[0]):
            R1 = Rotation.from_rotvec(
                rotation_axes[0] * rotation_angle_fields[i, 0], degrees=True
            )
            R2 = Rotation.from_rotvec(
                rotation_axes[1] * rotation_angle_fields[i, 1], degrees=True
            )
            expected_vector = R2.apply(R1.apply(reference_vector))
            np.testing.assert_allclose(rotated_vector[i], expected_vector)

    def test_field_rotation_single_axis(self):
        reference_vector = np.array([1, 0, 0])
        rotation_axes = np.array([[0, 1, 0]])
        rotation_angle_fields = np.array([[1, -2], [3, 4.3]])
        rotated_vector = transforms.field_rotation(
            reference_vector,
            rotation_axes,
            rotation_angle_fields,
            degrees=True,
        )

        self.assertEqual(rotated_vector.shape, (2, 2, 3))

        for i in range(rotation_angle_fields.shape[0]):
            for j in range(rotation_angle_fields.shape[1]):
                R = Rotation.from_rotvec(
                    rotation_axes[0] * rotation_angle_fields[i, j], degrees=True
                )
                expected_vector = R.apply(reference_vector)
                np.testing.assert_allclose(rotated_vector[i, j], expected_vector)


if __name__ == "__main__":
    unittest.main()
