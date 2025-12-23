import unittest

import numpy as np

from darling import filters


class TestSnrThreshold(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_snr_threshold_single_pixel(self):
        mean = 2
        sigma = 1
        data = mean + np.random.uniform(2, 10, (8, 8, 8)) * sigma
        data[0, 0, 0] = mean + 20 * sigma
        primary_threshold = 1

        data1 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold=None,
            copy=True,
            loop_outer_dims=False,
        )

        np.testing.assert_allclose(data1, data)

        primary_threshold = 20.001
        data2 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold=None,
            copy=True,
            loop_outer_dims=False,
        )

        np.testing.assert_allclose(data2, 0)

        primary_threshold = 19.99
        data3 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold=None,
            copy=True,
            loop_outer_dims=False,
        )

        np.testing.assert_allclose(np.sum(data3 != 0), 1)
        np.testing.assert_allclose(data3[0, 0, 0], data[0, 0, 0])

        if self.debug:
            import matplotlib.pyplot as plt

            primary_threshold = 4.0
            data4 = filters.snr_threshold(
                data,
                mean,
                sigma,
                primary_threshold,
                secondary_threshold=None,
                copy=True,
                loop_outer_dims=False,
            )

            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 4, figsize=(15, 6))
            im = ax[0].imshow(data[:, 0, :])
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(data1[:, 0, :])
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            im = ax[2].imshow(data3[:, 0, :])
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            im = ax[3].imshow(data4[:, 0, :])
            fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
            plt.show()

    def test_snr_threshold_parallel(self):
        shape = (10, 10, 8, 8, 8)
        mean = np.ones((shape[0], shape[1])) * 2
        sigma = np.ones((shape[0], shape[1]))
        data = (
            mean[..., None, None, None]
            + np.random.uniform(2, 10, shape) * sigma[..., None, None, None]
        )
        data[..., 0, 0, 0] = mean[0, 0] + 20 * sigma[0, 0]
        primary_threshold = 1.0

        data1 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold=None,
            copy=True,
            loop_outer_dims=True,
        )
        np.testing.assert_allclose(data1, data)

        primary_threshold = 20.001

        data2 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold=None,
            copy=True,
            loop_outer_dims=True,
        )

        np.testing.assert_allclose(data2, 0)

    def test_snr_neighbourhood_behaviour_3d(self):
        shape = (10, 10, 8, 8, 8)
        mean = np.ones((shape[0], shape[1]))
        sigma = np.ones((shape[0], shape[1]))
        data = 6 * np.ones(shape)
        data[5, 5, 4, 4, 4] = 11
        primary_threshold = 8
        secondary_threshold = 4

        data1 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )

        np.testing.assert_allclose(data1[5, 5, 4, 4, 4], data[5, 5, 4, 4, 4])
        np.testing.assert_allclose(
            data1[5, 5, 3:4, 3:4, 3:4], data[5, 5, 3:4, 3:4, 3:4]
        )
        self.assertAlmostEqual(data1[5, 5, 3, 4, 4], 6)
        self.assertAlmostEqual(data1[5, 5, 4, 4, 4], 11)

        self.assertEqual(
            np.sum(data1 == 0), np.prod(shape) - 3 ** len(data[0, 0].shape)
        )

        secondary_threshold = None
        data2 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )
        self.assertAlmostEqual(data2[5, 5, 3, 4, 4], 0)
        self.assertAlmostEqual(data2[5, 5, 4, 4, 4], 11)
        self.assertEqual(np.sum(data2 == 0), np.prod(shape) - 1)

        secondary_threshold = 7
        data3 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )
        self.assertAlmostEqual(data3[5, 5, 3, 4, 4], 0)
        self.assertAlmostEqual(data3[5, 5, 4, 4, 4], 11)
        self.assertEqual(np.sum(data3 == 0), np.prod(shape) - 1)

    def test_snr_neighbourhood_behaviour_2d(self):
        shape = (10, 10, 8, 8)
        mean = np.ones((shape[0], shape[1]))
        sigma = np.ones((shape[0], shape[1]))
        data = 6 * np.ones(shape)
        data[5, 5, 4, 4] = 11
        primary_threshold = 8
        secondary_threshold = 4

        data1 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )

        np.testing.assert_allclose(data1[5, 5, 4, 4], data[5, 5, 4, 4])
        np.testing.assert_allclose(data1[5, 5, 3:4, 3:4], data[5, 5, 3:4, 3:4])
        self.assertAlmostEqual(data1[5, 5, 3, 4], 6)
        self.assertAlmostEqual(data1[5, 5, 4, 4], 11)

        self.assertEqual(
            np.sum(data1 == 0), np.prod(shape) - 3 ** len(data[0, 0].shape)
        )

        secondary_threshold = None
        data2 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )
        self.assertAlmostEqual(data2[5, 5, 3, 4], 0)
        self.assertAlmostEqual(data2[5, 5, 4, 4], 11)
        self.assertEqual(np.sum(data2 == 0), np.prod(shape) - 1)

        secondary_threshold = 7
        data3 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )
        self.assertAlmostEqual(data3[5, 5, 3, 4], 0)
        self.assertAlmostEqual(data3[5, 5, 4, 4], 11)
        self.assertEqual(np.sum(data3 == 0), np.prod(shape) - 1)

    def test_snr_neighbourhood_behaviour_1d(self):
        shape = (10, 10, 8)
        mean = np.ones((shape[0], shape[1]))
        sigma = np.ones((shape[0], shape[1]))
        data = 6 * np.ones(shape)
        data[5, 5, 4] = 11
        primary_threshold = 8
        secondary_threshold = 4

        data1 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )

        np.testing.assert_allclose(data1[5, 5, 4], data[5, 5, 4])
        np.testing.assert_allclose(data1[5, 5, 3:4], data[5, 5, 3:4])
        self.assertAlmostEqual(data1[5, 5, 3], 6)
        self.assertAlmostEqual(data1[5, 5, 4], 11)

        self.assertEqual(
            np.sum(data1 == 0), np.prod(shape) - 3 ** len(data[0, 0].shape)
        )

        secondary_threshold = None
        data2 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )
        self.assertAlmostEqual(data2[5, 5, 3], 0)
        self.assertAlmostEqual(data2[5, 5, 4], 11)
        self.assertEqual(np.sum(data2 == 0), np.prod(shape) - 1)

        secondary_threshold = 7
        data3 = filters.snr_threshold(
            data,
            mean,
            sigma,
            primary_threshold,
            secondary_threshold,
            copy=True,
            loop_outer_dims=True,
        )
        self.assertAlmostEqual(data3[5, 5, 3], 0)
        self.assertAlmostEqual(data3[5, 5, 4], 11)
        self.assertEqual(np.sum(data3 == 0), np.prod(shape) - 1)


class TestGaussianFilter(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_convolve_1d(self):
        data = self._get_dirac_delta_ND(shape=(32,), location=(16,))
        kernel = np.array([0, 1, 0])
        data_convolved = filters._convolve_1d(data, kernel)
        self.assertTrue(np.allclose(data_convolved, data))

        kernel = np.array([0, -1.4, 2, 1.1, 0])
        data_convolved = filters._convolve_1d(data, kernel)
        expected = data.copy()
        expected[16] = 2
        expected[17] = -1.4
        expected[15] = 1.1

        self.assertTrue(np.allclose(data_convolved, expected))

    def test_convolve_nd(self):
        shape = (32, 32, 32)
        location = (16, 12, 11)
        data = self._get_dirac_delta_ND(shape, location)
        kernels = (np.ones((3,)), np.ones((3,)), np.ones((3,)))
        data = filters._convolve_nd(data, kernels, axis=(0, 1, 2))
        expected = data.copy()
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    expected[location[0] + i, location[1] + j, location[2] + k] = 1
        self.assertTrue(np.allclose(data, expected))

        shape = (32,)
        location = (16,)
        data = self._get_dirac_delta_ND(shape, location)
        kernels = (np.ones((3,)),)
        data = filters._convolve_nd(data, kernels, axis=(0,))
        expected = data.copy()
        for i in [-1, 0, 1]:
            expected[location[0] + i] = 1
        self.assertTrue(np.allclose(data, expected))

    def test_gaussian_kernel_1d(self):
        sigma = 1.0
        radius = 3
        kernel = filters._gaussian_kernel_1d(sigma, radius)
        self.assertTrue(kernel.size == 1 + 2 * radius)
        np.testing.assert_allclose(np.sum(kernel), 1.0)
        np.testing.assert_allclose(kernel[radius], np.max(kernel))

    def test_get_kernels_and_axis_target_axis(self):
        shape = (32, 32, 8, 8, 8)
        data = np.zeros(shape)
        sigma = (1.0, 2.0)
        truncate = 4.0
        radius = 5
        axis = (1, 2)
        kernels, axis = filters._get_kernels_and_axis(
            data, sigma, truncate, radius, axis
        )

        self.assertTrue(kernels[0] == np.array([0.0], dtype=np.float64))
        self.assertTrue(isinstance(kernels[1], np.ndarray))
        self.assertTrue(isinstance(kernels[2], np.ndarray))

        shape = (32,)
        location = (16,)
        sigma = 1.0
        radius = 3
        truncate = 4.0
        data = self._get_dirac_delta_ND(shape, location)
        kernels, axis = filters._get_kernels_and_axis(
            data, sigma, truncate, radius, 0, trailing_dims=1
        )
        self.assertTrue(len(kernels) == 1)
        self.assertTrue(len(axis) == 1)
        self.assertTrue(axis[0] == 0)

    def test_get_kernels_and_axis(self):
        shapes = [(32, 32, 8), (32, 32, 8, 8), (32, 32, 8, 8, 8)]
        for shape in shapes:
            data = np.zeros(shape)
            sigma = 2.0
            truncate = 4.0
            radius = 5
            axis = None
            kernels, axis = filters._get_kernels_and_axis(
                data, sigma, truncate, radius, axis
            )
            for k in kernels:
                self.assertTrue(k.size == 1 + 2 * radius)
                np.testing.assert_allclose(np.sum(k), 1.0)
                np.testing.assert_allclose(k[radius], np.max(k))

            self.assertTrue(len(kernels) == data.ndim - 2)
            self.assertTrue(np.allclose(axis, list(range(data.ndim - 2))))

        for shape in shapes:
            data = np.zeros(shape)
            sigma = 2.0
            truncate = 4.0
            radius = 5
            axis = 0
            kernels, axis = filters._get_kernels_and_axis(
                data, sigma, truncate, radius, axis
            )
            self.assertTrue(len(kernels) == data.ndim - 2)
            self.assertTrue(len(axis) == 1)

        for shape in shapes:
            data = np.zeros(shape)
            sigma = 2.0
            truncate = 4.0
            radius = None
            axis = (0, 1)
            kernels, axis = filters._get_kernels_and_axis(
                data, sigma, truncate, radius, axis
            )
            self.assertTrue(len(kernels) == data.ndim - 2)
            self.assertTrue(len(axis) == 2)
            for k in kernels:
                expected_radius = 8
                self.assertTrue(k.size == 1 + 2 * expected_radius)
                np.testing.assert_allclose(np.sum(k), 1.0)
                np.testing.assert_allclose(k[expected_radius], np.max(k))

    def test_gaussian_filter_single_pixel(self):
        shape = (32, 32, 32)
        location = (11, 13, 16)
        dirac = self._get_dirac_delta_ND(shape, location)
        data = dirac.copy()
        sigma = (1.0, 2.0)
        truncate = 4.0
        radius = 6
        axis = (-1, 0)
        data1 = filters.gaussian_filter(
            data,
            sigma,
            truncate,
            radius,
            axis,
            loop_outer_dims=False,
            copy=True,
        )
        data2 = filters.gaussian_filter(
            data,
            (sigma[1], sigma[0]),
            truncate,
            radius,
            axis=(axis[1], axis[0]),
            loop_outer_dims=False,
            copy=True,
        )

        np.testing.assert_allclose(dirac, data)
        np.testing.assert_allclose(data1, data2)

        i, j, k = location
        np.testing.assert_allclose(data1[:, j + 1, :], 0)
        np.testing.assert_allclose(data1[:, j - 1, :], 0)
        np.testing.assert_allclose(np.sum(data1[:, j, :]), 1)
        np.testing.assert_allclose(np.sum(data1), 1)
        np.testing.assert_allclose(np.sum(data1[:, j, :] > 0), (2 * radius + 1) ** 2)

        if self.debug:
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 3, figsize=(15, 6))
            i, j, k = location
            im = ax[0].imshow(data[:, :, k])
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(data[i, :, :])
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            im = ax[2].imshow(data[:, j, :])
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_gaussian_filter_parallel(self):
        shape = (32, 32, 32)
        data = np.zeros((8, 8, *shape))

        sigma = (1.0, 2.0)
        truncate = 4.0
        radius = 6
        axis = (-1, 0)

        location = np.empty((data.shape[0], data.shape[1]), dtype=tuple)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                location[i, j] = tuple(
                    np.random.randint(radius + 1, 32 - radius - 1, size=3)
                )

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, ...] = self._get_dirac_delta_ND(shape, location[i, j])
        dirac = data.copy()

        data1 = filters.gaussian_filter(
            data,
            sigma,
            truncate,
            radius,
            axis,
            loop_outer_dims=True,
            copy=True,
        )
        data2 = filters.gaussian_filter(
            data,
            (sigma[1], sigma[0]),
            truncate,
            radius,
            axis=(axis[1], axis[0]),
            loop_outer_dims=True,
            copy=True,
        )

        np.testing.assert_allclose(data1, data2)

        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                i, j, k = location[ii, jj]
                np.testing.assert_allclose(dirac[ii, jj], data[ii, jj])
                np.testing.assert_allclose(data1[ii, jj, :, j + 1, :], 0)
                np.testing.assert_allclose(data1[ii, jj, :, j - 1, :], 0)
                np.testing.assert_allclose(np.sum(data1[ii, jj, :, j, :]), 1)
                np.testing.assert_allclose(np.sum(data1[ii, jj]), 1)
                np.testing.assert_allclose(
                    np.sum(data1[ii, jj, :, j, :] > 0), (2 * radius + 1) ** 2
                )

    def test_gaussian_filter_parallel(self):
        shape = (32, 32, 32)
        data = np.zeros((8, 8, *shape))

        sigma = 2
        radius = 4 * sigma

        location = np.empty((data.shape[0], data.shape[1]), dtype=tuple)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                location[i, j] = tuple(np.random.randint(15, 22, size=3))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, ...] = self._get_dirac_delta_ND(shape, location[i, j])
        dirac = data.copy()

        data1 = filters.gaussian_filter(
            data,
            sigma,
            loop_outer_dims=True,
            copy=True,
        )

        for ii in range(data.shape[0]):
            for jj in range(data.shape[1]):
                i, j, k = location[ii, jj]
                np.testing.assert_allclose(dirac[ii, jj], data[ii, jj])
                np.testing.assert_allclose(np.sum(data1[ii, jj]), 1)
                np.testing.assert_allclose(
                    np.sum(data1[ii, jj] > 0), (2 * radius + 1) ** 3
                )
                idx = np.unravel_index(np.argmax(data1[ii, jj]), data1[ii, jj].shape)
                np.testing.assert_allclose(idx, (i, j, k))

        if self.debug:
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 3, figsize=(15, 6))
            i, j, k = location[0, 0]
            im = ax[0].imshow(data1[0, 0, :, :, k])
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(data1[0, 0, i, :, :])
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            im = ax[2].imshow(data1[0, 0, :, j, :])
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_break_gaussian_filter(self):
        shape = (16, 16, 16)
        data = np.zeros(shape)
        with self.assertRaises(ValueError):
            filters.gaussian_filter(data, sigma=-1, loop_outer_dims=True, copy=False)
        with self.assertRaises(ValueError):
            filters.gaussian_filter(data, sigma=-1, loop_outer_dims=False, copy=False)

        with self.assertRaises(ValueError):
            filters.gaussian_filter(
                data,
                sigma=(1, 2, 3),
                radius=(6, 10),
                axis=(0, 1, 2),
                loop_outer_dims=False,
                copy=False,
            )

        with self.assertRaises(ValueError):
            filters.gaussian_filter(
                data,
                sigma=(1, 3),
                radius=(6, 6, 10),
                axis=(0, 1, 2),
                loop_outer_dims=False,
                copy=False,
            )

        with self.assertRaises(ValueError):
            filters.gaussian_filter(
                data,
                sigma=(1, 2, 3),
                radius=(6, 6, 10),
                axis=(0, 2),
                loop_outer_dims=False,
                copy=False,
            )

        with self.assertRaises(ValueError):
            filters.gaussian_filter(
                data,
                sigma=(1, 2, 3),
                radius=(6, 6, -1),
                axis=(0, 2, 1),
                loop_outer_dims=False,
                copy=False,
            )

        shape = (4, 4, 8, 8, 8)
        data = np.zeros(shape)
        with self.assertRaises(ValueError):
            filters.gaussian_filter(data, sigma=1, loop_outer_dims=False, copy=False)
        filters.gaussian_filter(
            data, sigma=1, loop_outer_dims=True, copy=False
        )  # should not raise an error

        shape = (8, 8, 8, 3)
        data = np.zeros(shape)
        with self.assertRaises(ValueError):
            filters.gaussian_filter(data, sigma=1, loop_outer_dims=False, copy=False)
        filters.gaussian_filter(
            data, sigma=1, loop_outer_dims=True, copy=False
        )  # should not raise an error

        shape = (16, 16)
        data = np.zeros(shape)
        filters.gaussian_filter(
            data, sigma=1, loop_outer_dims=False, copy=False
        )  # should not raise an error

        shape = (16,)
        data = np.zeros(shape)
        filters.gaussian_filter(
            data, sigma=1, loop_outer_dims=False, copy=False
        )  # should not raise an error

    def _get_dirac_delta_ND(self, shape, location):
        data = np.zeros(shape)
        data[location] = 1
        return data


if __name__ == "__main__":
    unittest.main()
