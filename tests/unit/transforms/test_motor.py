import unittest

import numpy as np

import darling


class TestMotor(unittest.TestCase):
    def setUp(self):
        self.debug = False

        _, self.data, self.motors = darling.io.assets.domains()

        self.motors1D = (np.arange(5) - 1) * 0.1234
        self.motors1D = self.motors1D.reshape((1, 5))
        self.motors2D = (np.indices((5, 7)) - 1) * 0.1234
        self.motors2D = self.motors2D.reshape((2, 5, 7))
        self.motors3D = (np.indices((5, 7, 9)) - 1) * 0.1234
        self.motors3D = self.motors3D.reshape((3, 5, 7, 9))

        self.motorsND = (self.motors1D, self.motors2D, self.motors3D)

    def test_bounds(self):
        for motors in self.motorsND:
            bb = darling.transforms.motor.bounds(motors)
            self.assertEqual(bb.shape, (motors.shape[0], 2))
            self.assertEqual(bb.max(), motors.max())
            self.assertEqual(bb.min(), motors.min())

    def test_stepsize(self):
        expected = np.array([0.08, 0.06])
        for method in ("mean", "median", "max", "min"):
            dx = darling.transforms.motor.stepsize(self.motors, method=method)
            self.assertEqual(dx.shape, (2,))
            self.assertEqual(np.round(np.abs(dx[0] - expected[0]), 3), 0)
            self.assertEqual(np.round(np.abs(dx[1] - expected[1]), 3), 0)

            for motors in self.motorsND:
                dx = darling.transforms.motor.stepsize(motors, method=method)
                self.assertEqual(motors.ndim - 1, len(dx))
                for i, val in enumerate(dx):
                    self.assertAlmostEqual(val, 0.1234)

    def test_scale(self):
        # run all cases
        for unit in ("rad", "deg", "mrad", "urad"):
            for input_unit in ("rad", "deg", "mrad", "urad"):
                for motors in self.motorsND:
                    _ = darling.transforms.motor.scale(
                        motors, unit=unit, input_unit=input_unit
                    )
        # check units for one specific:

        out = darling.transforms.motor.scale(self.motors, unit="rad", input_unit="deg")
        np.testing.assert_allclose(out, np.radians(self.motors))
        out = darling.transforms.motor.scale(self.motors, unit="mrad", input_unit="deg")
        np.testing.assert_allclose(
            out, np.radians(self.motors) * 1000, rtol=1e-6, atol=1e-6
        )

    def test_center(self):
        out = darling.transforms.motor.center(self.motors)
        self.assertEqual(np.min(np.abs(out)), 0)
        for motors in self.motorsND:
            out = darling.transforms.motor.center(motors)
            self.assertEqual(np.min(np.abs(out)), 0)

    def test_mrad(self):
        out = darling.transforms.motor.mrad(self.motors)
        self.assertEqual(np.min(np.abs(out)), 0)
        np.testing.assert_array_less(np.abs(self.motors).max(), np.abs(out).max())

        for motors in self.motorsND:
            out = darling.transforms.motor.mrad(motors)
            self.assertEqual(np.min(np.abs(out)), 0)
            np.testing.assert_array_less(np.abs(motors).max(), np.abs(out).max())

    def test_rad(self):
        out = darling.transforms.motor.rad(self.motors)
        self.assertEqual(np.min(np.abs(out)), 0)
        np.testing.assert_array_less(np.abs(out).max(), np.abs(self.motors).max())

        for motors in self.motorsND:
            out = darling.transforms.motor.rad(motors)
            self.assertEqual(np.min(np.abs(out)), 0)
            np.testing.assert_array_less(np.abs(out).max(), np.abs(motors).max())

    def test_urad(self):
        out = darling.transforms.motor.urad(self.motors)
        self.assertEqual(np.min(np.abs(out)), 0)
        np.testing.assert_array_less(np.abs(self.motors).max(), np.abs(out).max())

        for motors in self.motorsND:
            out = darling.transforms.motor.urad(motors)
            self.assertEqual(np.min(np.abs(out)), 0)
            np.testing.assert_array_less(np.abs(motors).max(), np.abs(out).max())

    def test_reduce_3d_to_1d_negative(self):
        # allow for motor values to be indexing=ij but with decreasing values along axis
        motors = -self.motors3D
        rm = darling.transforms.motor.reduce(motors, axis=0)
        self.assertEqual(rm.shape, (1, 5))
        self.assertAlmostEqual(np.abs(np.diff(rm[0, :])).min(), 0.1234)

    def test_reduce(self):
        for motors in self.motorsND:
            if motors.shape[0] > 1:
                rm = darling.transforms.motor.reduce(motors, axis=0)
                self.assertEqual(rm.shape, (1, 5))

                rm = darling.transforms.motor.reduce(motors, axis=(0, 1))
                self.assertEqual(rm.shape, (2, 5, 7))

                rm = darling.transforms.motor.reduce(motors, axis=-1)
                if motors.shape[0] == 2:
                    self.assertEqual(rm.shape, (1, 7))
                if motors.shape[0] == 3:
                    self.assertEqual(rm.shape, (1, 9))

                rm = darling.transforms.motor.reduce(motors, axis=(0, -1))
                if motors.shape[0] == 2:
                    self.assertEqual(rm.shape, (2, 5, 7))
                if motors.shape[0] == 3:
                    self.assertEqual(rm.shape, (2, 5, 9))

    def test_reorder(self):
        for motors in self.motorsND:
            if motors.shape[0] == 2:
                out = darling.transforms.motor.reorder(motors, axis=(0, 1))
                self.assertEqual(out.shape, (2, 5, 7))
                self.assertAlmostEqual(np.abs(np.diff(out[0, :, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[1, 0, :])).min(), 0.1234)
                out = darling.transforms.motor.reorder(motors, axis=(1, 0))
                self.assertEqual(out.shape, (2, 7, 5))
                self.assertAlmostEqual(np.abs(np.diff(out[0, :, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[1, 0, :])).min(), 0.1234)
            if motors.shape[0] == 3:
                out = darling.transforms.motor.reorder(motors, axis=(0, 1, 2))
                self.assertEqual(out.shape, (3, 5, 7, 9))
                self.assertAlmostEqual(np.abs(np.diff(out[0, :, 0, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[1, 0, :, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[2, 0, 0, :])).min(), 0.1234)
                out = darling.transforms.motor.reorder(motors, axis=(1, 0, 2))
                self.assertEqual(out.shape, (3, 7, 5, 9))
                self.assertAlmostEqual(np.abs(np.diff(out[0, :, 0, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[1, 0, :, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[2, 0, 0, :])).min(), 0.1234)
                out = darling.transforms.motor.reorder(motors, axis=(2, 0, 1))
                self.assertEqual(out.shape, (3, 9, 5, 7))
                self.assertAlmostEqual(np.abs(np.diff(out[0, :, 0, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[1, 0, :, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[2, 0, 0, :])).min(), 0.1234)
                out = darling.transforms.motor.reorder(motors, axis=(2, 1, 0))
                self.assertEqual(out.shape, (3, 9, 7, 5))
                self.assertAlmostEqual(np.abs(np.diff(out[0, :, 0, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[1, 0, :, 0])).min(), 0.1234)
                self.assertAlmostEqual(np.abs(np.diff(out[2, 0, 0, :])).min(), 0.1234)

                self.assertAlmostEqual(np.abs(np.diff(out[0, 0, :, 0])).max(), 0)
                self.assertAlmostEqual(np.abs(np.diff(out[0, 0, 0, :])).max(), 0)


if __name__ == "__main__":
    unittest.main()
