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
            loose, bg_loose = properties.curvefit.fit_2D_gaussian(pixels, grid, power, return_background=True)
            tight, bg_tight = properties.curvefit.fit_2D_gaussian(pixels, grid, power, tolerance=1e-6, return_background=True)
            if power == 1:
                np.testing.assert_array_equal(loose[:, 0], fit[rows, cols])
            np.testing.assert_array_equal(tight[..., 6], 1)

            def weighted_mse(params, background):
                a, mx, my, sx, sy, rho = params[:, 0, :6].T[:, :, None, None]
                u, v = (grid[0] - mx) / sx, (grid[1] - my) / sy
                pred = a * np.exp(-(u**2 + v**2 - 2*rho*u*v) / (2*(1-rho**2)))
                bx, by, offset = background[:, 0].T[:, :, None, None]
                pred += bx * grid[0] + by * grid[1] + offset
                obs = pixels[:, 0]
                design = np.column_stack((grid[0].ravel(), grid[1].ravel(), np.ones(grid[0].size)))
                coefficients = np.linalg.lstsq(design, obs.reshape(len(obs), -1).T, rcond=None)[0]
                signal = obs - (design @ coefficients).T.reshape(obs.shape)
                height = signal.max(axis=(1, 2), keepdims=True)
                weights = np.maximum(signal / height, 0.2)**power
                return (
                    np.sum((weights * (pred - obs) / height)**2, axis=(1, 2))
                    / np.sum(weights**2, axis=(1, 2))
                )

            loose_mse = weighted_mse(loose, bg_loose)
            tight_mse = weighted_mse(tight, bg_tight)
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

    def test_narrow_peak_on_constant_background(self):
        x, y = np.meshgrid(np.linspace(-5, 5, 71), np.linspace(-4, 4, 75), indexing="ij")
        u, v = (x - 0.4) / 0.42, (y + 0.35) / 0.29
        z = (u**2 + v**2 - 0.6*u*v) / (1 - 0.3**2)
        peak = 180 * np.exp(-0.5*z)
        baseline = np.full_like(x, 80)
        noise = np.random.default_rng(27).normal(0, 0.6, x.shape)
        data = (baseline + peak + noise)[None, None]
        fit, background = properties.curvefit.fit_2D_gaussian(
            data, (x, y), tolerance=1e-6, return_background=True
        )
        self.assertEqual(fit[0, 0, 6], 1)
        np.testing.assert_allclose(fit[0, 0, :6], [180, .4, -.35, .42, .29, .3], rtol=.035, atol=.012)
        np.testing.assert_array_equal(background[0, 0, :2], 0)
        self.assertAlmostEqual(background[0, 0, 2], 80, delta=.15)
        empty = properties.curvefit.fit_2D_gaussian(baseline[None, None], (x,y))
        self.assertEqual(empty[0, 0, 6], 0)

    def test_correct_kernel_broadening(self):
        from darling import filters
        x, y = np.meshgrid(np.arange(-30,31)*.08, np.arange(-35,36)*.05, indexing="ij")
        sx, sy, rho = .24, .17, .4
        u, v = x/sx, y/sy
        peak = 130*np.exp(-0.5*(u*u + v*v - 2*rho*u*v)/(1-rho*rho))
        smoothed = filters.gaussian_filter(peak[None,None], sigma=1.5, copy=True)
        fit = properties.curvefit.fit_2D_gaussian(smoothed, (x,y), tolerance=1e-5)
        observed = properties.curvefit.combined_fwhm(fit)
        intrinsic = properties.curvefit.combined_fwhm(fit, (x,y), smoothing_sigma=1.5)
        expected = np.sqrt(8*np.log(2))*np.hypot(sx,sy)
        self.assertGreater(observed.item(), expected*1.06)
        np.testing.assert_allclose(intrinsic.item(), expected, rtol=.015)
        fit[...,5] = .9999  # Positive marginal variances alone are insufficient.
        self.assertTrue(np.isnan(properties.curvefit.combined_fwhm(fit, (x,y), smoothing_sigma=1.5)).all())

    def test_kernel_covariance_in_motor_units(self):
        i, j = np.indices((5, 7), dtype=float)
        grid = np.stack([3 + 0.1*i + 0.03*j, -1 - 0.08*j])
        # For radius one the actual kernel variance differs materially from sigma**2.
        var = 2*np.exp(-0.5) / (1 + 2*np.exp(-0.5))
        steps = np.array([[0.1, 0.03], [0, -0.08]])
        intrinsic = np.array([[0.04, 0.009], [0.009, 0.016]])
        observed = intrinsic + var * steps @ steps.T
        sx, sy = np.sqrt(observed.diagonal())
        fit = np.array([80, 0, 0, sx, sy, observed[0, 1] / (sx*sy), 1])
        width = properties.curvefit.combined_fwhm(fit, grid, smoothing_sigma=1, radius=1)
        np.testing.assert_allclose(width, np.sqrt(8*np.log(2)*np.trace(intrinsic)))
        grid[1, 0, 3] += 0.1
        with self.assertRaisesRegex(ValueError, "affine"):
            properties.curvefit.combined_fwhm(fit, grid, smoothing_sigma=1)
