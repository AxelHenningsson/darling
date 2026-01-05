import unittest

import numpy as np

import darling


class TestMoments(unittest.TestCase):
    def setUp(self):
        self.debug = False
        _, self.data, self.coordinates = darling.io.assets.mosaicity_scan()

    def test_mean_1D(self):
        # Test that a series of displaced gaussians gives back the input mean
        # coordinates with precision better than the cooridinate resolution.

        # Data creation
        x = np.linspace(-1, 1, 32, dtype=np.float32)
        sigma = x[2] - x[0]
        N = 32
        data = np.zeros((N, N, len(x)))
        true_mean = np.zeros((N, N))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0 = -sigma * (i / N) + 2 * sigma * (j / N)
                data[i, j] = np.exp(-((x - x0) ** 2) / (2 * sigma**2)) * 65535
                true_mean[i, j] = x0
        data = data.round().astype(np.uint16)

        # Compute mean values
        mu = darling.properties.mean(data, coordinates=x[None, :])

        # Check that error is within the x,y resolution
        resolution = x[1] - x[0]
        relative_error = (true_mean - mu) / resolution

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 3, figsize=(6, 8))
            im = ax[0].imshow(mu[:, :])
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(true_mean[:, :])
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            im = ax[2].imshow(relative_error[:, :], cmap="jet")
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

        np.testing.assert_array_less(
            np.abs(relative_error), np.ones_like(relative_error)
        )

    def test_mean_2D(self):
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
                    np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)) * 65535
                )
                true_mean.append([x0, y0])
        true_mean = np.array(true_mean).reshape(N, N, 2)
        data = data.round().astype(np.uint16)

        # Compute mean values
        mu = darling.properties.mean(data, coordinates=np.array([X, Y]))

        # Check that error is within the x,y resolution
        resolution = x[1] - x[0]
        relative_error = (true_mean - mu) / resolution
        np.testing.assert_array_less(
            np.abs(relative_error), np.ones_like(relative_error)
        )

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
                    * 65535
                )
                true_mean.append([x0, y0, z0])
        true_mean = np.array(true_mean).reshape(N, M, 3)
        data = data.round().astype(np.uint16)

        mu = darling.properties.mean(data, coordinates=np.array([X, Y, Z]))

        resolution = x[1] - x[0]
        relative_error = (true_mean - mu) / resolution

        np.testing.assert_array_less(
            np.abs(relative_error), np.ones_like(relative_error)
        )

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

    def test_mean_3d_arbitrary_dtypes(self):
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
                    * 65535
                )
                true_mean.append([x0, y0, z0])
        true_mean = np.array(true_mean).reshape(N, M, 3)
        data = data.round().astype(np.uint16)

        for dtype_data in [np.float64, np.float32, np.uint16, np.int32]:
            for dtype_coords in [np.float64, np.float32]:
                mu = darling.properties.mean(
                    data.astype(dtype_data),
                    coordinates=np.array([X, Y, Z]).astype(dtype_coords),
                )

                resolution = x[1] - x[0]
                relative_error = (true_mean - mu) / resolution

                np.testing.assert_array_less(
                    np.abs(relative_error), np.ones_like(relative_error)
                )

    def test_mean_2d_noisy(self):
        # Simply assert that the mean function runs on real noisy data from id03.
        mu = darling.properties.mean(self.data, self.coordinates)
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

    def test_covariance_1d(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance with precision better than 0.1%

        # Data creation
        x = np.linspace(-1, 1, 32, dtype=np.float32)
        sigma0 = 2 * (x[1] - x[0])  # can not be too low due to uint16 clip
        N = 32
        data = np.zeros((N, N, len(x)))
        true_variance = np.zeros((N, N))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x0 = sigma0 * i / (N)
                S = sigma0 + 0.5 * sigma0 * i / N
                data[i, j] = np.exp(-0.5 * ((x - x0) ** 2 / (S**2))) * 65535
                true_variance[i, j] = S**2
        data = data.round().astype(np.uint16)

        # Compute covariance values
        cov = darling.properties.covariance(data, coordinates=x[None, :])

        # Check that error is within the resolution
        relative_error = (np.sqrt(true_variance) - np.sqrt(cov)) / sigma0
        np.testing.assert_array_less(
            np.abs(relative_error), np.ones_like(relative_error) * 0.001
        )

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 3, figsize=(8, 6))
            im = ax[0].imshow(np.sqrt(cov) / sigma0)
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(np.sqrt(true_variance) / sigma0)
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            im = ax[2].imshow(np.abs(relative_error), cmap="jet")
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_covariance_2d(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance with precision better than 1%

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
                    * 65535
                )
                true_variance.append(S.copy())
        true_variance = np.array(true_variance).reshape(N, N, 2, 2)
        data = data.round().astype(np.uint16)

        cov = darling.properties.covariance(data, coordinates=np.array([X, Y]))

        for i in range(cov.shape[2]):
            np.testing.assert_allclose(
                np.sqrt(true_variance[..., i, i]),
                np.sqrt(cov[..., i, i]),
                atol=sigma0 * 1e-3,
                rtol=0.01,
            )

        for i in range(cov.shape[2]):
            for j in range(cov.shape[3]):
                if i != j:
                    np.testing.assert_allclose(cov[..., i, j], 0, atol=sigma0 * 1e-3)

        if self.debug:
            error = np.sqrt(true_variance) - np.sqrt(cov)
            plt.style.use("dark_background")
            fig, ax = plt.subplots(3, cov.shape[2], figsize=(6, 8))
            for i in range(cov.shape[2]):  # computed covariance
                im = ax[0, i].imshow(np.sqrt(cov[..., i, i]))
                fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
            for i in range(cov.shape[2]):  # true covariance
                im = ax[1, i].imshow(np.sqrt(true_variance[..., i, i]))
                fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)
            for i in range(cov.shape[2]):  # relative error
                im = ax[2, i].imshow(error[..., i, i], cmap="jet")
                fig.colorbar(im, ax=ax[2, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_covariance_3d(self):
        # Test that a series of displaced gaussians with different covariance
        # gives back the input covariance with precision better than 1%

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
                    * 65535
                )
                true_variance.append(S.copy())
        true_variance = np.array(true_variance).reshape(N, N, 3, 3)
        data = data.round().astype(np.uint16)

        cov = darling.properties.covariance(data, coordinates=np.array([X, Y, Z]))

        for i in range(cov.shape[2]):
            np.testing.assert_allclose(
                np.sqrt(true_variance[..., i, i]),
                np.sqrt(cov[..., i, i]),
                atol=sigma0 * 1e-3,
                rtol=0.01,
            )

        for i in range(cov.shape[2]):
            for j in range(cov.shape[3]):
                if i != j:
                    np.testing.assert_allclose(cov[..., i, j], 0, atol=sigma0 * 1e-3)

        if self.debug:
            error = np.sqrt(true_variance) - np.sqrt(cov)
            plt.style.use("dark_background")
            fig, ax = plt.subplots(3, cov.shape[2], figsize=(6, 8))
            for i in range(cov.shape[2]):  # computed covariance
                im = ax[0, i].imshow(np.sqrt(cov[..., i, i]))
                fig.colorbar(im, ax=ax[0, i], fraction=0.046, pad=0.04)
            for i in range(cov.shape[2]):  # true covariance
                im = ax[1, i].imshow(np.sqrt(true_variance[..., i, i]))
                fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)
            for i in range(cov.shape[2]):  # relative error
                im = ax[2, i].imshow(error[..., i, i], cmap="jet")
                fig.colorbar(im, ax=ax[2, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_covariance_3d_arbitrary_dtypes(self):
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
                    * 65535
                )
                true_variance.append(S.copy())
        true_variance = np.array(true_variance).reshape(N, N, 3, 3)
        data = data.round().astype(np.uint16)

        for dtype_data in [np.float64, np.float32, np.uint16, np.int32]:
            for dtype_coords in [np.float64, np.float32]:
                cov = darling.properties.covariance(
                    data.astype(dtype_data),
                    coordinates=np.array([X, Y, Z]).astype(dtype_coords),
                )

                for i in range(cov.shape[2]):
                    np.testing.assert_allclose(
                        np.sqrt(true_variance[..., i, i]),
                        np.sqrt(cov[..., i, i]),
                        atol=sigma0 * 1e-3,
                        rtol=0.01,
                    )

                for i in range(cov.shape[2]):
                    for j in range(cov.shape[3]):
                        if i != j:
                            np.testing.assert_allclose(
                                cov[..., i, j], 0, atol=sigma0 * 1e-3
                            )

    def test_covariance_noisy(self):
        # Simply assert that the covariance function runs on real noisy data from id03.
        cov = darling.properties.covariance(self.data, self.coordinates)
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
                    * 65535
                )

                true_variance.append(S.copy())
                true_mean.append([x0, y0])
        true_mean = np.array(true_mean).reshape(N, N, 2)
        true_variance = np.array(true_variance).reshape(N, N, 2, 2)
        data = data.round().astype(np.uint16)

        # Compute covariance and mean values
        mu, cov = darling.properties.moments(data, coordinates=np.array([X, Y]))

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
                    * 65535
                )
                true_variance.append(S.copy())
                true_mean.append([x0, y0, z0])
        true_mean = np.array(true_mean).reshape(N, N, 3)
        true_variance = np.array(true_variance).reshape(N, N, 3, 3)
        data = data.round().astype(np.uint16)

        # Compute covariance and mean values
        mu, cov = darling.properties.moments(data, coordinates=np.array([X, Y, Z]))

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
        mu, cov = darling.properties.moments(self.data, self.coordinates)
        self.assertEqual(cov.shape[0], self.data.shape[0])
        self.assertEqual(cov.shape[1], self.data.shape[1])
        self.assertEqual(cov.shape[2], 2)
        self.assertEqual(cov.shape[3], 2)
        self.assertEqual(cov.dtype, np.float32)
        self.assertEqual(mu.shape[0], self.data.shape[0])
        self.assertEqual(mu.shape[1], self.data.shape[1])
        self.assertEqual(mu.shape[2], 2)
        self.assertEqual(mu.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
