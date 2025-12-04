import unittest

import numpy as np

from darling.properties import fit_1d_gaussian


class TestGaussianFit(unittest.TestCase):
    def setUp(self):
        self.debug = False
        self.rng = np.random.default_rng(42)

    def gaussian_with_linear_bg(self, A, mu, sigma, k, m, x):
        """Gaussian peak on top of a linear background.

        f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m
        """
        return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2)) + k * x + m

    def make_fake_data(self, ny=64, nx=64, m=32):
        """Generate synthetic 3D data with a Gaussian + linear background per pixel.

        Args:
            ny (:obj:`int`): Number of pixels in first image dimension.
            nx (:obj:`int`): Number of pixels in second image dimension.
            m (:obj:`int`): Number of points along the 1D coordinate axis.
            seed (:obj:`int`): Random seed for reproducibility.

        Returns:
            tuple:
                x (:obj:`numpy.ndarray`): 1D coordinate array of length m.
                data (:obj:`numpy.ndarray`): 3D array of shape (ny, nx, m)
                    containing noisy Gaussian + linear background traces.
                true_params (:obj:`numpy.ndarray`): 4D array of shape
                    (ny, nx, 5) with the true parameters
                    [A, mu, sigma, k, m_bg] for each pixel.
        """
        x = np.arange(m, dtype=np.float32)

        data = np.zeros((ny, nx, m), dtype=np.float32)
        true_params = np.zeros((ny, nx, 5), dtype=np.float64)

        for i in range(ny):
            for j in range(nx):
                mu = (m // 2) + self.rng.uniform(-2.0, 2.0)
                sigma = self.rng.uniform(5.3, 8.3)
                A = self.rng.uniform(800.0, 2000.0)
                k = self.rng.uniform(2.5, 3.89)
                m_bg = self.rng.uniform(-44.0, 44.0)

                y = self.gaussian_with_linear_bg(A, mu, sigma, k, m_bg, x)
                noise = self.rng.normal(0.0, 10 + A / 15.0, size=m)
                y_noisy = y + noise

                data[i, j, :] = y_noisy.astype(np.float32)
                true_params[i, j, 0] = A
                true_params[i, j, 1] = mu
                true_params[i, j, 2] = sigma
                true_params[i, j, 3] = k
                true_params[i, j, 4] = m_bg

        return x, data, true_params

    def test_fit_1d_gaussian_with_linear_background(self):
        """Test that fit_1d_gaussian recovers Gaussian + linear parameters on synthetic data."""
        ny, nx, m = 64, 64, 64
        x, data, true_params = self.make_fake_data(ny=ny, nx=nx, m=m)

        mask = np.ones((ny, nx), dtype=bool)
        mask[10:20, 10:20] = False

        true_params[~mask] = 0

        params = fit_1d_gaussian(
            data=data,
            coordinates=(x,),
            mask=mask,
        )

        if self.debug:
            import matplotlib.pyplot as plt

            # show single pixel fit
            i, j = 47, 30
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title(
                f"pixel {i}, {j} residual = {np.abs(params[i, j, 2] - true_params[i, j, 2]) / true_params[i, j, 2]}"
            )
            ax.plot(x, data[i, j], label="data")
            ax.plot(
                x,
                self.gaussian_with_linear_bg(
                    params[i, j, 0],
                    params[i, j, 1],
                    params[i, j, 2],
                    params[i, j, 3],
                    params[i, j, 4],
                    x,
                ),
                label="fitted",
            )

            kest = params[i, j, 3]
            mest = params[i, j, 4]
            ktrue = true_params[i, j, 3]
            mtrue = true_params[i, j, 4]
            ax.plot(x, ktrue * x + mtrue, label="true", linestyle="--")
            ax.plot(x, kest * x + mest, label="fitted")

            ax.legend()
            plt.tight_layout()
            plt.show()

            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 5, figsize=(14, 7), sharex=True, sharey=True)

            titles = ["amplitude", "mean", "sigma", "slope", "intercept"]
            for i in range(0, 5):
                if i == 4 or i == 3:
                    residual = np.abs(params[..., i] - true_params[..., i])
                    im = ax[i].imshow(residual, cmap="viridis")
                else:
                    residual = np.abs(params[..., i] - true_params[..., i]) / (
                        true_params[..., i] + 1e-8
                    )
                    im = ax[i].imshow(residual, cmap="jet", vmin=0, vmax=0.2)
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
                ax[i].set_title(f"residual {titles[i]}")

            plt.tight_layout()
            plt.show()

        np.testing.assert_array_equal(params.shape, (ny, nx, 6))
        np.testing.assert_array_equal(params.dtype, np.float64)
        np.testing.assert_allclose(
            params[mask, 0], true_params[mask, 0], atol=50, rtol=0.1
        )

        np.testing.assert_allclose(
            params[mask, 1], true_params[mask, 1], atol=1.0, rtol=0.1
        )

        np.testing.assert_allclose(
            params[mask, 2], true_params[mask, 2], atol=0.5, rtol=0.2
        )

        np.testing.assert_allclose(
            np.abs(params[mask, 3] - true_params[mask, 3]) < 5.0,
            np.ones_like(mask)[mask],
        )

        np.testing.assert_allclose(
            np.abs(params[mask, 4] - true_params[mask, 4])
            < true_params[mask, 0] / 10.0,
            np.ones_like(mask)[mask],
        )


if __name__ == "__main__":
    unittest.main()
