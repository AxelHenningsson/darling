import unittest

import numpy as np

from darling import io, properties


class TestMosaicGaussian(unittest.TestCase):
    def test_full_domains_scan(self):
        _, data, grid = io.assets.domains("1.1")
        fit = properties.curvefit.fit_2D_gaussian(data, grid)
        self.assertEqual(fit.shape, (200, 200, 7))
        np.testing.assert_array_equal(fit[..., 6], 1)
        # On real domains pixels, independently score loose and tight fits
        # against the same motor samples and weights.
        rows, cols = [4, 69, 100], [124, 166, 100]
        pixels = data[rows, cols][:, None]
        for power in (1, 4, 6):
            loose = properties.curvefit.fit_2D_gaussian(pixels, grid, power)
            tight = properties.curvefit.fit_2D_gaussian(pixels, grid, power, tolerance=1e-6)
            if power == 1:
                np.testing.assert_array_equal(loose[:, 0], fit[rows, cols])
            np.testing.assert_array_equal(tight[..., 6], 1)

            def weighted_mse(params):
                a, mx, my, sx, sy, rho = params[:, 0, :6].T[:, :, None, None]
                u, v = (grid[0] - mx) / sx, (grid[1] - my) / sy
                pred = a * np.exp(-(u**2 + v**2 - 2*rho*u*v) / (2*(1-rho**2)))
                obs = pixels[:, 0]
                peak = obs.max(axis=(1, 2), keepdims=True)
                residual = (pred - obs) / peak * (obs / peak)**power
                return np.nanmean(np.where(obs >= 0.2*peak, residual**2, np.nan), axis=(1, 2))

            loose_mse, tight_mse = weighted_mse(loose), weighted_mse(tight)
            self.assertTrue(np.all(tight_mse <= loose_mse + 1e-12))
            self.assertTrue(np.any(tight_mse < loose_mse - 1e-8))

    def test_parameters_and_moment_seeds(self):
        x, y = np.meshgrid(
            np.linspace(-2, 2, 25), np.linspace(-2, 2, 29), indexing="ij"
        )
        grid = np.stack([x, y])
        sx, sy, rho = 0.32, 0.4, -0.3
        z = (x / sx) ** 2 + (y / sy) ** 2 - 2 * rho * x * y / (sx * sy)
        data = (1000 * np.exp(-z / (2 * (1 - rho**2))))[None, None]
        # Seed at the analytical solution to check the model convention.
        mu = np.zeros((1, 1, 2), dtype=np.float32)
        seed = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])[None, None]
        kwargs = dict(weight_power=6, stat_mean=mu, stat_cov=seed)
        params = properties.curvefit.fit_2D_gaussian(data, grid, **kwargs)[0, 0]
        self.assertEqual(params[6], 1)
        np.testing.assert_allclose(
            params[:6], [1000, 0, 0, sx, sy, rho], atol=0.012, rtol=0.04
        )

    def test_mask_and_unusable_pixels(self):
        grid = np.indices((3, 4), dtype=np.float32)
        data = np.zeros((1, 6, 3, 4), dtype=np.float32)
        data[0, 1] = 2
        data[0, 2, 1, 2] = np.nan
        data[0, 3, 1, 2] = -1
        data[0, 4:] = 20 - (grid[0] - 1)**2 - (grid[1] - 2)**2
        mask = np.array([[True, True, True, True, False, True]])
        fit = properties.curvefit.fit_2D_gaussian(data, grid, mask=mask)
        np.testing.assert_array_equal(fit[0, :, 6], [0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(fit[0, :5], 0)
        for tolerance in (-1, np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, "tolerance"):
                properties.curvefit.fit_2D_gaussian(data, grid, tolerance=tolerance)
        with self.assertRaisesRegex(ValueError, "two motor grids"):
            properties.curvefit.fit_2D_gaussian(data, grid[:1])
