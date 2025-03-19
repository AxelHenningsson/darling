import unittest

import matplotlib.pyplot as plt
import numpy as np

import darling
from darling import assets, properties


class TestMoments(unittest.TestCase):
    # Tests for the darling.properties module.

    def setUp(self):
        self.debug = False
        _, self.data, self.coordinates = assets.mosaicity_scan()

    def test_mean(self):
        # Test that a series of displaced gaussians gives back the input mean
        # coordinates with precision better than the cooridinate resolution.

        # Data creation
        x = y = np.linspace(-1, 1, 9, dtype=np.float32)
        sigma = x[2] - x[0]
        X, Y = np.meshgrid(x, y, indexing="ij")
        N = 32
        data = np.zeros((N, N, len(x), len(y)))
        true_mean = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0, y0 = sigma * i / N, sigma * j / N - 0.5 * sigma * i / N
                data[i, j] = (
                    np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)) * 64000
                )
                true_mean.append([x0, y0])
        true_mean = np.array(true_mean).reshape(N, N, 2)
        data = data.round().astype(np.uint16)

        # Compute mean values
        mu = properties.mean(data, coordinates=np.array([X, Y]))

        # Check that error is within the x,y resolution
        resolution = x[1] - x[0]
        relative_error = (true_mean - mu) / resolution
        np.testing.assert_array_less(relative_error, np.ones_like(relative_error))

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(3, 2, figsize=(6, 8))
            for i in range(2):  # computed mean
                im = ax[0, i].imshow(mu[:, :, i])
                fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
            for i in range(2):  # true mean
                im = ax[1, i].imshow(true_mean[:, :, i])
                fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)
            for i in range(2):  # relative error
                im = ax[2, i].imshow(relative_error[:, :, i], cmap="jet")
                fig.colorbar(im, ax=ax[2, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_mean_3d(self):
        x = y = z = np.linspace(-1, 1, 9, dtype=np.float32)

        sigma = x[2] - x[0]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        N = 32
        M = 37
        data = np.zeros((N, M, len(x), len(y), len(z)))
        true_mean = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0, y0, z0 = (
                    sigma * i / N,
                    sigma * j / N - 0.5 * sigma * i / N,
                    sigma * np.sqrt(i * j) / N,
                )
                data[i, j] = (
                    np.exp(
                        -((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
                        / (2 * sigma**2)
                    )
                    * 64000
                )
                true_mean.append([x0, y0, z0])
        true_mean = np.array(true_mean).reshape(N, M, 3)
        data = data.round().astype(np.uint16)

        mu = properties.mean(data, coordinates=np.array([X, Y, Z]))

        resolution = x[1] - x[0]
        relative_error = (true_mean - mu) / resolution

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(3, 3, figsize=(6, 8))
            for i in range(3):  # computed mean
                im = ax[0, i].imshow(mu[:, :, i])
                fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
            for i in range(3):  # true mean
                im = ax[1, i].imshow(true_mean[:, :, i])
                fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)
            for i in range(3):  # relative error
                im = ax[2, i].imshow(relative_error[:, :, i], cmap="jet")
                fig.colorbar(im, ax=ax[2, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_mean_noisy(self):
        # Simply assert that the mean function runs on real noisy data from id03.
        mu = properties.mean(self.data, self.coordinates)
        self.assertEqual(mu.shape[0], self.data.shape[0])
        self.assertEqual(mu.shape[1], self.data.shape[1])
        self.assertEqual(mu.shape[2], 2)
        self.assertEqual(mu.dtype, np.float32)

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            for i in range(2):
                im = ax[i].imshow(mu[:, :, i])
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_covariance(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance with precision better than the cooridinate
        # resolution, given that the gaussian blob fits the coordinate range.

        # Data creation
        x = y = np.linspace(-1, 1, 9, dtype=np.float32)
        sigma0 = (x[1] - x[0]) / 3.0
        X, Y = np.meshgrid(x, y, indexing="ij")
        N = 32
        data = np.zeros((N, N, len(x), len(y)))
        true_variance = []
        S = np.eye(2)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0, y0 = sigma0 * i / N, sigma0 * j / N - 0.5 * sigma0 * i / N
                S[0, 0] = sigma0 + 0.5 * sigma0 * i / N
                S[1, 1] = sigma0 + 0.5 * sigma0 * j / N - 0.25 * sigma0 * i / N
                Si = 1.0 / np.diag(S)
                data[i, j] = (
                    np.exp(-0.5 * (Si[0] * (X - x0) ** 2 + Si[1] * (Y - y0) ** 2))
                    * 64000
                )
                true_variance.append(S.copy())
        true_variance = np.array(true_variance).reshape(N, N, 2, 2)
        data = data.round().astype(np.uint16)

        # Compute covariance values
        cov = properties.covariance(data, coordinates=np.array([X, Y]))

        # Check that error is within the x,y resolution
        resolution = x[1] - x[0]
        relative_error = (true_variance - cov) / resolution**2
        np.testing.assert_array_less(
            relative_error[:, :, 0, 0], np.ones_like(relative_error[:, :, 0, 0])
        )
        np.testing.assert_array_less(
            relative_error[:, :, 1, 1], np.ones_like(relative_error[:, :, 1, 1])
        )
        np.testing.assert_allclose(cov[:, :, 0, 1], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 1, 0], 0, atol=sigma0 * 1e-3)

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(3, 2, figsize=(6, 8))
            for i in range(2):  # computed covariance
                im = ax[0, i].imshow(np.sqrt(cov[:, :, i, i]))
                fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
            for i in range(2):  # true covariance
                im = ax[1, i].imshow(np.sqrt(true_variance[:, :, i, i]))
                fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)
            for i in range(2):  # relative error
                im = ax[2, i].imshow(relative_error[:, :, i, i], cmap="jet")
                fig.colorbar(im, ax=ax[2, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_covariance_3d(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance with precision better than the cooridinate
        # resolution, given that the gaussian blob fits the coordinate range.

        # Data creation
        x = y = z = np.linspace(-1, 1, 9, dtype=np.float32)
        sigma0 = (x[1] - x[0]) / 3.0
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        N = 32
        data = np.zeros((N, N, len(x), len(y), len(z)))
        true_variance = []
        S = np.eye(3)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0, y0, z0 = (
                    sigma0 * i / N,
                    sigma0 * j / N - 0.5 * sigma0 * i / N,
                    sigma0 * np.sqrt(i * j) / N,
                )
                S[0, 0] = sigma0 + 0.5 * sigma0 * i / N
                S[1, 1] = sigma0 + 0.5 * sigma0 * j / N - 0.25 * sigma0 * i / N
                S[2, 2] = sigma0 + 0.5 * sigma0 * np.sqrt(i * j) / N

                Si = 1.0 / np.diag(S)
                data[i, j] = (
                    np.exp(
                        -0.5
                        * (
                            Si[0] * (X - x0) ** 2
                            + Si[1] * (Y - y0) ** 2
                            + Si[2] * (Z - z0) ** 2
                        )
                    )
                    * 64000
                )
                true_variance.append(S.copy())
        true_variance = np.array(true_variance).reshape(N, N, 3, 3)
        data = data.round().astype(np.uint16)

        # Compute covariance values
        cov = properties.covariance(data, coordinates=np.array([X, Y, Z]))

        # Check that error is within the x,y resolution
        resolution = x[1] - x[0]
        relative_error = (true_variance - cov) / resolution**2
        np.testing.assert_array_less(
            relative_error[:, :, 0, 0], np.ones_like(relative_error[:, :, 0, 0])
        )
        np.testing.assert_array_less(
            relative_error[:, :, 1, 1], np.ones_like(relative_error[:, :, 1, 1])
        )
        np.testing.assert_array_less(
            relative_error[:, :, 2, 2], np.ones_like(relative_error[:, :, 2, 2])
        )
        np.testing.assert_allclose(cov[:, :, 0, 1], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 1, 0], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 0, 2], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 2, 0], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 1, 2], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 2, 1], 0, atol=sigma0 * 1e-3)

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(3, 3, figsize=(6, 8))
            for i in range(3):  # computed covariance
                im = ax[0, i].imshow(np.sqrt(cov[:, :, i, i]))
                fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
            for i in range(3):  # true covariance
                im = ax[1, i].imshow(np.sqrt(true_variance[:, :, i, i]))
                fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)
            for i in range(3):  # relative error
                im = ax[2, i].imshow(relative_error[:, :, i, i], cmap="jet")
                fig.colorbar(im, ax=ax[2, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_covariance_noisy(self):
        # Simply assert that the covariance function runs on real noisy data from id03.
        cov = properties.covariance(self.data, self.coordinates)
        self.assertEqual(cov.shape[0], self.data.shape[0])
        self.assertEqual(cov.shape[1], self.data.shape[1])
        self.assertEqual(cov.shape[2], 2)
        self.assertEqual(cov.shape[3], 2)
        self.assertEqual(cov.dtype, np.float32)

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            for i in range(2):
                im = ax[i].imshow(cov[:, :, i, i])
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_moments(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance & mean with precision better than the
        # cooridinate resolution, given that the gaussian blob fits the coordinate
        # range.

        # Data creation
        x = y = np.linspace(-1, 1, 9, dtype=np.float32)
        sigma0 = (x[1] - x[0]) / 3.0
        X, Y = np.meshgrid(x, y, indexing="ij")
        N = 32
        data = np.zeros((N, N, len(x), len(y)))
        true_variance, true_mean = [], []
        S = np.eye(2)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0, y0 = sigma0 * i / N, sigma0 * j / N - 0.5 * sigma0 * i / N
                S[0, 0] = sigma0 + 0.5 * sigma0 * i / N
                S[1, 1] = sigma0 + 0.5 * sigma0 * j / N - 0.25 * sigma0 * i / N
                Si = 1.0 / np.diag(S)
                data[i, j] = (
                    np.exp(-0.5 * (Si[0] * (X - x0) ** 2 + Si[1] * (Y - y0) ** 2))
                    * 64000
                )

                true_variance.append(S.copy())
                true_mean.append([x0, y0])
        true_mean = np.array(true_mean).reshape(N, N, 2)
        true_variance = np.array(true_variance).reshape(N, N, 2, 2)
        data = data.round().astype(np.uint16)

        # Compute covariance and mean values
        mu, cov = properties.moments(data, coordinates=np.array([X, Y]))

        # Check that errors are within the x,y resolution
        resolution = x[1] - x[0]
        relative_cov_error = (true_variance - cov) / resolution**2
        np.testing.assert_array_less(
            relative_cov_error[:, :, 0, 0], np.ones_like(relative_cov_error[:, :, 0, 0])
        )
        np.testing.assert_array_less(
            relative_cov_error[:, :, 1, 1], np.ones_like(relative_cov_error[:, :, 1, 1])
        )
        np.testing.assert_allclose(cov[:, :, 0, 1], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 1, 0], 0, atol=sigma0 * 1e-3)

        relative_mean_error = (true_mean - mu) / resolution
        np.testing.assert_array_less(
            relative_mean_error, np.ones_like(relative_mean_error)
        )

    def test_moments_3d(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance & mean with precision better than the
        # cooridinate resolution, given that the gaussian blob fits the coordinate
        # range.

        # Data creation
        true_variance, true_mean = [], []
        # Data creation
        x = y = z = np.linspace(-1, 1, 9, dtype=np.float32)
        sigma0 = (x[1] - x[0]) / 3.0
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        N = 32
        data = np.zeros((N, N, len(x), len(y), len(z)))
        true_variance = []
        S = np.eye(3)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0, y0, z0 = (
                    sigma0 * i / N,
                    sigma0 * j / N - 0.5 * sigma0 * i / N,
                    sigma0 * np.sqrt(i * j) / N,
                )
                S[0, 0] = sigma0 + 0.5 * sigma0 * i / N
                S[1, 1] = sigma0 + 0.5 * sigma0 * j / N - 0.25 * sigma0 * i / N
                S[2, 2] = sigma0 + 0.5 * sigma0 * np.sqrt(i * j) / N

                Si = 1.0 / np.diag(S)
                data[i, j] = (
                    np.exp(
                        -0.5
                        * (
                            Si[0] * (X - x0) ** 2
                            + Si[1] * (Y - y0) ** 2
                            + Si[2] * (Z - z0) ** 2
                        )
                    )
                    * 64000
                )
                true_variance.append(S.copy())
                true_mean.append([x0, y0, z0])
        true_mean = np.array(true_mean).reshape(N, N, 3)
        true_variance = np.array(true_variance).reshape(N, N, 3, 3)
        data = data.round().astype(np.uint16)

        # Compute covariance and mean values
        mu, cov = properties.moments(data, coordinates=np.array([X, Y, Z]))

        # Check that errors are within the x,y resolution
        resolution = x[1] - x[0]
        relative_cov_error = (true_variance - cov) / resolution**2
        np.testing.assert_array_less(
            relative_cov_error[:, :, 0, 0], np.ones_like(relative_cov_error[:, :, 0, 0])
        )
        np.testing.assert_array_less(
            relative_cov_error[:, :, 1, 1], np.ones_like(relative_cov_error[:, :, 1, 1])
        )
        np.testing.assert_array_less(
            relative_cov_error[:, :, 2, 2], np.ones_like(relative_cov_error[:, :, 2, 2])
        )
        np.testing.assert_allclose(cov[:, :, 0, 1], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 1, 0], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 1, 2], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 2, 1], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 0, 2], 0, atol=sigma0 * 1e-3)
        np.testing.assert_allclose(cov[:, :, 2, 0], 0, atol=sigma0 * 1e-3)

        relative_mean_error = (true_mean - mu) / resolution
        np.testing.assert_array_less(
            relative_mean_error, np.ones_like(relative_mean_error)
        )

    def test_moments_noisy(self):
        # Simply assert that the covariance and mean functions runs on real
        # noisy data from id03.
        mu, cov = properties.moments(self.data, self.coordinates)
        self.assertEqual(cov.shape[0], self.data.shape[0])
        self.assertEqual(cov.shape[1], self.data.shape[1])
        self.assertEqual(cov.shape[2], 2)
        self.assertEqual(cov.shape[3], 2)
        self.assertEqual(cov.dtype, np.float32)
        self.assertEqual(mu.shape[0], self.data.shape[0])
        self.assertEqual(mu.shape[1], self.data.shape[1])
        self.assertEqual(mu.shape[2], 2)
        self.assertEqual(mu.dtype, np.float32)

    def test_kam(self):
        mu, _ = properties.moments(self.data, self.coordinates)
        kam = properties.kam(mu, size=(3, 3))
        self.assertEqual(kam.shape[0], self.data.shape[0])
        self.assertEqual(kam.shape[1], self.data.shape[1])

        mu = np.zeros_like(mu)
        mu[5:8, 5:8, 0] = 1
        kam = properties.kam(mu, size=(3, 3))
        self.assertEqual(kam[6, 6], 0)
        self.assertEqual(kam[6 - 2, 6 - 2], 1 / 8.0)
        self.assertEqual(kam[6 + 2, 6 - 2], 1 / 8.0)
        self.assertEqual(kam[6 + 2, 6 + 2], 1 / 8.0)
        self.assertEqual(kam[6 - 2, 6 + 2], 1 / 8.0)
        self.assertEqual(kam[6 - 1, 6 - 1], 5 / 8.0)
        self.assertEqual(kam[6 + 1, 6 - 1], 5 / 8.0)
        self.assertEqual(kam[6 + 1, 6 + 1], 5 / 8.0)
        self.assertEqual(kam[6 - 1, 6 + 1], 5 / 8.0)
        self.assertEqual(kam[6, 6 + 1], 3 / 8.0)
        self.assertEqual(kam[6, 6 + 2], 3 / 8.0)
        self.assertEqual(kam[6, 6 - 1], 3 / 8.0)
        self.assertEqual(kam[6, 6 - 2], 3 / 8.0)
        self.assertEqual(kam[6 + 1, 6], 3 / 8.0)
        self.assertEqual(kam[6 + 2, 6], 3 / 8.0)
        self.assertEqual(kam[6 - 1, 6], 3 / 8.0)
        self.assertEqual(kam[6 - 2, 6], 3 / 8.0)
        self.assertEqual(np.sum(kam), 8)

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            im = ax.imshow(kam)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_rgb(self):
        mu, _ = properties.moments(self.data, self.coordinates)
        rgb_map, colorkey, colorgrid = properties.rgb(
            mu, norm="dynamic", coordinates=None
        )

        self.assertEqual(rgb_map.shape[0], self.data.shape[0])
        self.assertEqual(rgb_map.shape[1], self.data.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid
        self.assertEqual(X.shape[0], colorkey.shape[0])
        self.assertEqual(X.shape[1], colorkey.shape[1])
        self.assertEqual(Y.shape[0], colorkey.shape[0])
        self.assertEqual(Y.shape[1], colorkey.shape[1])

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title("RGB map dynamic")
            ax.imshow(rgb_map)
            plt.tight_layout()

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.pcolormesh(X, Y, colorkey)
            ax.set_title("RGB map dynamic")
            plt.tight_layout()
            plt.show()

        rgb_map, colorkey, colorgrid = properties.rgb(
            mu, norm="full", coordinates=self.coordinates
        )

        self.assertEqual(rgb_map.shape[0], self.data.shape[0])
        self.assertEqual(rgb_map.shape[1], self.data.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid
        self.assertEqual(X.shape[0], colorkey.shape[0])
        self.assertEqual(X.shape[1], colorkey.shape[1])
        self.assertEqual(Y.shape[0], colorkey.shape[0])
        self.assertEqual(Y.shape[1], colorkey.shape[1])

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title("RGB map full")
            ax.imshow(rgb_map)
            plt.tight_layout()

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.pcolormesh(X, Y, colorkey)
            ax.set_title("RGB map full")
            plt.tight_layout()
            plt.show()

        # check that we can handle nan values
        mu[0:10, 0:10, 0] = np.nan
        rgb_map, colorkey, colorgrid = properties.rgb(
            mu, norm="full", coordinates=self.coordinates
        )

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title("Handle nans")
            ax.imshow(rgb_map)
            plt.tight_layout()


class TestGaussianMixture(unittest.TestCase):
    def setUp(self):
        self.debug = False
        _, self.data, self.coordinates = assets.domains()

    def test_gaussian_mixture_on_domains_data(self):
        features = properties.gaussian_mixture(
            self.data,
            k=16,
            coordinates=self.coordinates,
        )

        for key in darling.peaksearcher._FEATURE_MAPPING:
            self.assertTrue(key in features)

        for key in darling.peaksearcher._MOTOR_KEY_MAPPING:
            self.assertTrue(darling.peaksearcher._MOTOR_KEY_MAPPING[key] in features)

        for key in features:
            self.assertEqual(features[key].shape[0], self.data.shape[0])
            self.assertEqual(features[key].shape[1], self.data.shape[1])
            self.assertEqual(features[key].shape[2], 16)
            self.assertTrue(np.all(np.isfinite(features[key])))
            self.assertTrue(np.all(np.isreal(features[key])))

        self.assertTrue(np.all(features["sum_intensity"] >= 0))
        self.assertTrue(np.all(features["number_of_pixels"] >= 0))

        if self.debug:
            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
            im1 = ax1.imshow(features["mean_motor1"][..., 0])
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            im2 = ax2.imshow(features["mean_motor2"][..., 0])
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_gaussian_mixture(self):
        data = np.zeros((29, 32, 4, 7), dtype=np.uint16)
        x = np.linspace(-0.81, 1.00465, 4, dtype=np.float32)
        y = np.linspace(-0.6, 1.1, 7, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data[..., 0, 0] = 1
        data[..., 0, 1] = 2
        data[..., 3, 3] = 4

        features = properties.gaussian_mixture(
            data,
            k=3,
            coordinates=(X, Y),
        )
        rtol = 1e-6
        atol = 1e-6
        np.testing.assert_allclose(
            features["sum_intensity"][..., 1], 3, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["sum_intensity"][..., 0], 4, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["number_of_pixels"][..., 1], 2, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["number_of_pixels"][..., 0], 1, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(features["var_row"][..., 0], 0, atol=atol, rtol=rtol)
        np.testing.assert_allclose(features["var_row"][..., 1], 0, atol=atol, rtol=rtol)

        np.testing.assert_allclose(features["var_col"][..., 0], 0, atol=atol, rtol=rtol)
        np.testing.assert_allclose(
            features["var_col"][..., 1], 1 / 3.0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["mean_row"][..., 1], 0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_row"][..., 0], 3, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["mean_motor1"][..., 1], x[0], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_motor1"][..., 0], x[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["mean_col"][..., 1], 2 / 3.0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_col"][..., 0], 3, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["mean_motor2"][..., 1], (y[0] + 2 * y[1]) / 3, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_motor2"][..., 0], y[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["var_motor1"][..., 0], 0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["var_motor1"][..., 1], 0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["var_motor2"][..., 0], 0, atol=atol, rtol=rtol
        )

        mean_motor2 = (y[0] + 2 * y[1]) / 3
        var_motor2 = ((mean_motor2 - y[0]) ** 2 + 2 * (mean_motor2 - y[1]) ** 2) / 2
        np.testing.assert_allclose(
            features["var_motor2"][..., 1], var_motor2, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["var_motor1_motor2"][..., 1], 0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["var_row_col"][..., 1], 0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["var_motor1_motor2"][..., 0], 0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["var_row_col"][..., 0], 0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["max_pix_row"][..., 1], 0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["max_pix_row"][..., 0], 3, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["max_pix_motor1"][..., 1], x[0], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["max_pix_motor1"][..., 0], x[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["max_pix_motor2"][..., 1], y[1], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["max_pix_motor2"][..., 0], y[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["max_pix_intensity"][..., 1], 2, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["max_pix_intensity"][..., 0], 4, atol=atol, rtol=rtol
        )


if __name__ == "__main__":
    unittest.main()
