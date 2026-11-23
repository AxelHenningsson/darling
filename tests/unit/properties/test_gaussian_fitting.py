import unittest

import numpy as np

from darling import properties
from darling.properties import _gaussian


class TestGaussianFitting(unittest.TestCase):
    """Tests for the Gaussian fitting module."""

    def setUp(self):
        self.debug = False  # Set to True to show debug plots

    def moments(self, x, y, data):
        total = data.sum()
        mean = np.array([(data * x).sum(), (data * y).sum()]) / total
        dx, dy = x - mean[0], y - mean[1]
        cov_xy = (data * dx * dy).sum()
        cov = (
            np.array([[(data * dx**2).sum(), cov_xy], [cov_xy, (data * dy**2).sum()]])
            / total
        )
        return mean, cov

    def plot_fit(self, x, y, data, params):
        import matplotlib.pyplot as plt

        fitted = _gaussian.gaussian_2d(x, y, *params[:6])
        if len(params) == 7:
            fitted += params[6]
        fig, axes = plt.subplots(1, 3)
        for ax, image, title in zip(
            axes, (data, fitted, data - fitted), ("Data", "Fit", "Residual")
        ):
            fig.colorbar(ax.imshow(image), ax=ax)
            ax.set_title(title)
        plt.show()

    def test_clip_value(self):
        """Test the clip_value function."""
        self.assertEqual(_gaussian.clip_value(5, 0, 10), 5)
        self.assertEqual(_gaussian.clip_value(-5, 0, 10), 0)
        self.assertEqual(_gaussian.clip_value(15, 0, 10), 10)
        self.assertEqual(_gaussian.clip_value(0, 0, 10), 0)
        self.assertEqual(_gaussian.clip_value(10, 0, 10), 10)

    def test_gaussian_2d(self):
        """Test the 2D Gaussian function."""
        amplitude = 1.0
        x0, y0 = 0.0, 0.0
        sigma_x, sigma_y = 1.0, 1.0
        rho = 0.0

        value = _gaussian.gaussian_2d(0, 0, amplitude, x0, y0, sigma_x, sigma_y, rho)
        self.assertTrue(np.isclose(value, amplitude, rtol=1e-6))

        value = _gaussian.gaussian_2d(1, 0, amplitude, x0, y0, sigma_x, sigma_y, rho)
        self.assertTrue(np.isclose(value, amplitude * np.exp(-0.5), rtol=1e-6))

        value = _gaussian.gaussian_2d(0, 1, amplitude, x0, y0, sigma_x, sigma_y, rho)
        self.assertTrue(np.isclose(value, amplitude * np.exp(-0.5), rtol=1e-6))

        rho = 0.5
        value = _gaussian.gaussian_2d(1, 1, amplitude, x0, y0, sigma_x, sigma_y, rho)
        expected = amplitude * np.exp(-0.5 * (1 + 1 - 2 * 0.5) / (1 - 0.5**2))
        self.assertTrue(np.isclose(value, expected, rtol=1e-6))

        value = _gaussian.gaussian_2d(0, 0, amplitude, x0, y0, 1e-11, 1e-11, rho)
        self.assertTrue(value > 0)

    def test_gaussian_2d_vectorized_consistency(self):
        """Vectorized inputs should match elementwise scalar evaluations."""
        amplitude = 2.5
        x0, y0 = -0.7, 1.3
        sigma_x, sigma_y = 0.8, 1.7
        rho = -0.2
        x = np.linspace(-2, 2, 11)
        y = np.linspace(-3, 3, 13)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z_vec = _gaussian.gaussian_2d(X, Y, amplitude, x0, y0, sigma_x, sigma_y, rho)
        Z_ref = np.zeros_like(Z_vec)
        for j in range(Y.shape[0]):
            for i in range(X.shape[1]):
                Z_ref[j, i] = _gaussian.gaussian_2d(
                    X[j, i], Y[j, i], amplitude, x0, y0, sigma_x, sigma_y, rho
                )
        self.assertTrue(np.allclose(Z_vec, Z_ref, rtol=1e-10, atol=0.0))

    def test_params_from_moments(self):
        """Test initialization of Gaussian parameters from statistical moments."""
        x = y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        x0, y0 = 1.0, -1.0
        sigma_x, sigma_y = 1.5, 0.8
        rho = 0.3

        Z = _gaussian.gaussian_2d(X, Y, 100.0, x0, y0, sigma_x, sigma_y, rho)

        stat_mean, stat_cov = self.moments(X, Y, Z)
        mean_x, mean_y = stat_mean
        var_x, var_y = stat_cov.diagonal()
        cov_xy = stat_cov[0, 1]
        data_flat = Z.flatten()

        params = _gaussian.params_from_moments(data_flat, stat_mean, stat_cov)

        self.assertAlmostEqual(params[0], np.max(Z), places=2)  # amplitude
        self.assertAlmostEqual(params[1], mean_x, places=2)  # x0
        self.assertAlmostEqual(params[2], mean_y, places=2)  # y0
        self.assertAlmostEqual(params[3], np.sqrt(var_x), places=2)  # sigma_x
        self.assertAlmostEqual(params[4], np.sqrt(var_y), places=2)  # sigma_y
        self.assertAlmostEqual(
            params[5], cov_xy / (np.sqrt(var_x) * np.sqrt(var_y)), places=2
        )  # rho

        params = _gaussian.params_from_moments(
            np.zeros(10), np.zeros(2), np.zeros((2, 2))
        )
        self.assertTrue(np.all(params == 0))

    def test_params_from_moments_rho_clipping(self):
        """rho derived from covariance must be clipped to [-0.99, 0.99]."""
        data_flat = np.array([1.0, 2.0, 3.0])
        stat_mean = np.array([0.0, 0.0])
        # Construct impossible covariance with |rho| > 1
        var_x = 1e-4
        var_y = 1e-4
        cov_xy = 10.0 * np.sqrt(var_x * var_y)
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])

        params = _gaussian.params_from_moments(data_flat, stat_mean, stat_cov)
        self.assertLessEqual(params[5], 0.99)
        # Now negative extreme
        cov_xy = -10.0 * np.sqrt(var_x * var_y)
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        params = _gaussian.params_from_moments(data_flat, stat_mean, stat_cov)
        self.assertGreaterEqual(params[5], -0.99)

    def test_adam_update(self):
        """Test the Adam optimizer update step."""
        param = 2.0
        m = 0.0
        v = 0.0
        grad = -1.0
        t = 1

        new_param, new_m, new_v = _gaussian.adam_update(param, m, v, grad, t)

        self.assertGreater(new_param, param)

        param, m, v = new_param, new_m, new_v
        t += 1
        new_param, new_m, new_v = _gaussian.adam_update(param, m, v, grad, t)

        self.assertGreater(new_param, param)

        param, m, v = new_param, new_m, new_v
        t += 1
        new_param, new_m, new_v = _gaussian.adam_update(param, m, v, 0.0, t)

        self.assertAlmostEqual(new_param, param, delta=0.01)

    def test_weighted_mse_gradient(self):
        x = np.array([-0.8, 0.4, 1.2, 0.3])
        y = np.array([0.1, -0.2, 0.8, 1.0])
        data = np.array([0.55, 1.0, 0.78, 0.21])
        weights = data**4
        weights /= np.linalg.norm(weights)

        def mse(p):
            u, v = (x - p[1]) / p[3], (y - p[2]) / p[4]
            pred = p[0] * np.exp(-(u*u + v*v - 2*p[5]*u*v) / (2*(1 - p[5]**2)))
            pred += p[6]
            return np.sum((weights * (pred - data))**2)

        for rho in (-0.6, 0.0, 0.45):
            p = np.array([1.2, 0.25, -0.1, 1.1, 0.7, rho, 0.07])
            loss, grad = _gaussian._weighted_mse_and_grad(p, x, y, data, weights)
            self.assertAlmostEqual(loss, mse(p))
            steps = np.eye(7) * 1e-6
            numerical = [(mse(p + h) - mse(p - h)) / 2e-6 for h in steps]
            np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-9)

    def test_returns_scored_parameters(self):
        x, y = np.indices((4, 5), dtype=float).reshape(2, -1)
        data = np.linspace(0.25, 1.0, x.size)
        mean, cov = np.array([1.5, 2.0]), np.eye(2)

        def fit(steps, tol):
            return _gaussian.fit_gaussian_2d(data, x, y, mean, cov, 4, steps, tol)

        # The initial guess already satisfies this loose absolute threshold.
        np.testing.assert_array_equal(fit(100, 1.0), fit(0, 0.0))
        # Extending the step budget must never lose a better evaluated iterate.
        def loss(p):
            pred = _gaussian.gaussian_2d(x, y, *p[:6]) + p[6]
            return np.mean((data**4 * (pred - data))**2)

        losses = [loss(fit(n, 0.0)) for n in (0, 1, 5, 50, 100)]
        self.assertTrue(np.all(np.diff(losses) <= 1e-12))

    def test_fit_gaussian_2d_simple(self):
        """Test fitting a simple 2D Gaussian without noise."""
        x = y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()

        true_amplitude = 100.0
        true_x0, true_y0 = 1.5, -1.2
        true_sigma_x, true_sigma_y = 1.2, 0.9
        true_rho = 0.3

        data = _gaussian.gaussian_2d(
            X, Y, true_amplitude, true_x0, true_y0, true_sigma_x, true_sigma_y, true_rho
        )

        stat_mean, stat_cov = self.moments(X, Y, data)

        params = _gaussian.fit_gaussian_2d(
            data.flatten(), x_flat, y_flat, stat_mean, stat_cov, 1.0
        )

        self.assertAlmostEqual(params[0], true_amplitude, delta=true_amplitude * 0.05)
        self.assertAlmostEqual(params[1], true_x0, delta=0.05)  # x0
        self.assertAlmostEqual(params[2], true_y0, delta=0.05)  # y0
        self.assertAlmostEqual(params[3], true_sigma_x, delta=0.05)  # sigma_x
        self.assertAlmostEqual(params[4], true_sigma_y, delta=0.05)  # sigma_y
        self.assertAlmostEqual(params[5], true_rho, delta=0.05)  # rho

        if self.debug:
            self.plot_fit(X, Y, data, params)

    def test_fit_gaussian_2d_noisy(self):
        """Test fitting a 2D Gaussian with noise."""
        x = y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()

        true_amplitude = 100.0
        true_x0, true_y0 = 1.5, -1.2
        true_sigma_x, true_sigma_y = 1.2, 0.9
        true_rho = 0.3

        data = _gaussian.gaussian_2d(
            X, Y, true_amplitude, true_x0, true_y0, true_sigma_x, true_sigma_y, true_rho
        )

        noise = np.random.RandomState(42).normal(0, 5.0, data.shape)
        noisy_data = data + noise

        stat_mean, stat_cov = self.moments(X, Y, noisy_data)

        params = _gaussian.fit_gaussian_2d(
            noisy_data.flatten(), x_flat, y_flat, stat_mean, stat_cov, 1.0
        )

        self.assertAlmostEqual(params[0], true_amplitude, delta=true_amplitude * 0.1)
        self.assertAlmostEqual(params[1], true_x0, delta=0.2)
        self.assertAlmostEqual(params[2], true_y0, delta=0.2)
        self.assertAlmostEqual(params[3], true_sigma_x, delta=0.3)
        self.assertAlmostEqual(params[4], true_sigma_y, delta=0.3)
        self.assertAlmostEqual(params[5], true_rho, delta=0.3)

        if self.debug:
            self.plot_fit(X, Y, noisy_data, params)

    def test_fit_reduces_mse_vs_moment_init(self):
        """Fitting should reduce MSE compared to raw moment-based initialization."""
        x = y = np.linspace(-5, 5, 40)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()

        amp, x0, y0, sx, sy, rho = 120.0, 0.9, -0.6, 0.9, 1.4, 0.25
        clean = _gaussian.gaussian_2d(X, Y, amp, x0, y0, sx, sy, rho)
        rng = np.random.default_rng(123)
        noisy = clean + rng.normal(0, 6.0, clean.shape)

        stat_mean, stat_cov = self.moments(X, Y, noisy)

        init_params = _gaussian.params_from_moments(
            noisy.flatten(), stat_mean, stat_cov
        )

        def mse_from_params(p):
            pred = _gaussian.gaussian_2d(X, Y, *p[:6])
            if len(p) == 7:
                pred += p[6]
            return np.mean((pred - noisy) ** 2)

        mse_init = mse_from_params(init_params)
        fit_params = _gaussian.fit_gaussian_2d(
            noisy.flatten(), x_flat, y_flat, stat_mean, stat_cov, 1.0
        )
        mse_fit = mse_from_params(fit_params)

        self.assertLess(mse_fit, mse_init)

    def test_edge_cases(self):
        """Test edge cases like zero data or single point data."""
        x = y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        zero_data = np.zeros_like(X).flatten()

        stat_mean = np.array([0.0, 0.0])
        stat_cov = np.array([[1.0, 0.0], [0.0, 1.0]])

        params = _gaussian.fit_gaussian_2d(
            zero_data, x_flat, y_flat, stat_mean, stat_cov, 1.0
        )

        self.assertTrue(np.all(params == 0))

        single_point_data = np.zeros_like(X)
        single_point_data[15, 15] = 100.0

        stat_mean, stat_cov = self.moments(X, Y, single_point_data)

        params = _gaussian.fit_gaussian_2d(
            single_point_data.flatten(), x_flat, y_flat, stat_mean, stat_cov, 1.0
        )

        self.assertAlmostEqual(params[0], 100.0, delta=10.0)  # amplitude
        self.assertAlmostEqual(params[1], x[15], delta=0.5)  # x0
        self.assertAlmostEqual(params[2], y[15], delta=0.5)  # y0

    def test_gaussian_pred_map(self):
        """Test gaussian_pred_map function produces correct predictions."""
        x = np.linspace(-3, 3, 20, dtype=np.float32)
        y = np.linspace(-3, 3, 20, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")

        params = np.array([100.0, 0.5, -0.5, 1.2, 0.8, 0.3], dtype=np.float32)
        pred = _gaussian.gaussian_pred_map(X, Y, params)

        self.assertEqual(pred.shape, X.shape)
        self.assertEqual(pred.dtype, np.float32)

        center_pred = _gaussian.gaussian_2d(0.5, -0.5, 100.0, 0.5, -0.5, 1.2, 0.8, 0.3)
        center_idx = np.unravel_index(
            np.argmin(np.abs(X - 0.5) + np.abs(Y + 0.5)), X.shape
        )
        self.assertAlmostEqual(pred[center_idx], center_pred, delta=1.0)

        self.assertTrue(np.all(pred >= 0))

        for i in [5, 10, 15]:
            for j in [5, 10, 15]:
                expected = _gaussian.gaussian_2d(
                    X[i, j],
                    Y[i, j],
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                    params[5],
                )
                self.assertAlmostEqual(pred[i, j], expected, delta=1e-5)

    def test_residual_map(self):
        """Test residual_map function computes correct residuals."""
        x = np.linspace(-2, 2, 15, dtype=np.float32)
        y = np.linspace(-2, 2, 15, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")

        true_params = np.array([50.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        data = _gaussian.gaussian_pred_map(X, Y, true_params).astype(np.float32)

        residual = _gaussian.residual_map(data, X, Y, true_params)
        self.assertEqual(residual.shape, data.shape)
        self.assertEqual(residual.dtype, np.float32)
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-5))

        wrong_params = true_params.copy()
        wrong_params[0] = 45.0
        residual2 = _gaussian.residual_map(data, X, Y, wrong_params)
        self.assertGreater(np.abs(residual2).max(), np.abs(residual).max())

    def test_nrmse(self):
        """Test nrmse function computes normalized RMSE correctly."""
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        residual = np.zeros_like(data, dtype=np.float32)
        nrmse_val = _gaussian.nrmse(data, residual)
        self.assertAlmostEqual(nrmse_val, 0.0, places=6)

        residual2 = np.array([1.0, -2.0, 3.0, -1.0], dtype=np.float32)
        nrmse_val2 = _gaussian.nrmse(data, residual2)

        rmse_manual = np.sqrt(np.mean(residual2**2))
        max_data = np.max(data)
        expected_nrmse = rmse_manual / max_data
        self.assertAlmostEqual(nrmse_val2, expected_nrmse, places=5)

        tiny_data = np.array([1e-13, 2e-13], dtype=np.float32)
        tiny_residual = np.array([1e-14, -1e-14], dtype=np.float32)
        nrmse_val3 = _gaussian.nrmse(tiny_data, tiny_residual)
        self.assertTrue(np.isfinite(nrmse_val3))

    def test_fit_gaussian_improves_over_stat_moments(self):
        """Test that fit_gaussian improves estimates compared to raw statistical moments."""
        a, b, m, n = 5, 5, 20, 20
        x = np.linspace(-2, 2, m, dtype=np.float32)
        y = np.linspace(-2, 2, n, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        true_x0, true_y0 = 0.3, -0.2
        true_sigma_x, true_sigma_y = 0.8, 0.6
        true_rho = 0.4

        peak = _gaussian.gaussian_2d(
            X, Y, 100.0, true_x0, true_y0, true_sigma_x, true_sigma_y, true_rho
        )
        noise = np.random.default_rng(42).normal(0, 2.0, (a, b, m, n))
        data = np.clip(peak + noise, 0, 65535).astype(np.uint16)
        grid = np.stack((X, Y))
        stat_mean, stat_cov = properties.moments(data, grid)
        fit = properties.curvefit.fit_2D_gaussian(
            data, grid, weight_power=1, stat_mean=stat_mean, stat_cov=stat_cov, tolerance=1e-6
        )

        fitted_mean = fit[..., 1:3]
        center = np.array([true_x0, true_y0])
        self.assertLess(
            np.abs(fitted_mean - center).mean(), np.abs(stat_mean - center).mean()
        )
        np.testing.assert_array_equal(fit[..., 6], 1)

    def test_fit_gaussian_shape_validation(self):
        """Test that fit_gaussian validates input shapes correctly."""
        x = np.linspace(-1, 1, 5, dtype=np.float32)
        y = np.linspace(-1, 1, 5, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data_wrong = np.zeros((10, 10, 5), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "two motor grids"):
            properties.curvefit.fit_2D_gaussian(data_wrong, coordinates=(X, Y))

        data_4d = np.zeros((5, 5, 5, 5), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "two motor grids"):
            properties.curvefit.fit_2D_gaussian(data_4d, coordinates=(X, Y, X))


if __name__ == "__main__":
    unittest.main()
