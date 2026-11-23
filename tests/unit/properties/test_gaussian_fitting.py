import unittest

import numpy as np

from darling import _gaussian, properties


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

        fitted = _gaussian.gaussian_2d(x, y, *params)
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
            pred = _gaussian.gaussian_2d(X, Y, *p)
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

    def test_moments_2d_gaussian_ufunc(self):
        x = np.linspace(-5, 5, 30, dtype=np.float32)
        y = np.linspace(-3, 6, 20, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        amp, x0, y0, sigma_x, sigma_y, rho = 100.0, 1.5, -1.2, 1.2, 0.9, 0.3
        Z = _gaussian.gaussian_2d(X, Y, amp, x0, y0, sigma_x, sigma_y, rho)
        data_u16 = np.clip(Z, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        dum = np.zeros(2, dtype=np.float32)
        stat_mean, stat_cov = self.moments(X, Y, Z)
        stat_mean = stat_mean.astype(np.float32)
        stat_cov = stat_cov.astype(np.float32)

        # Prepare output arrays for guvectorize function
        res = np.zeros(2, dtype=np.float32)
        rmse = np.zeros(1, dtype=np.float32)

        _gaussian.moments_2d_gaussian(
            data_u16,
            X.astype(np.float32),
            Y.astype(np.float32),
            dum,
            stat_mean,
            stat_cov,
            1.0,  # weight_power
            res,
            rmse,
        )
        self.assertEqual(res.shape, (2,))
        self.assertTrue(np.isclose(res[0], x0, rtol=5e-2))
        self.assertTrue(np.isclose(res[1], y0, rtol=5e-2))

    def test_ufuncs_zero_input(self):
        """Ufuncs should return zeros for empty or zero data."""
        x = np.linspace(-1, 1, 8, dtype=np.float32)
        y = np.linspace(-1, 1, 6, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        zeros_u16 = np.zeros_like(X, dtype=np.uint16)
        dum = np.zeros(2, dtype=np.float32)
        stat_mean = np.zeros(2, dtype=np.float32)
        stat_cov = np.zeros((2, 2), dtype=np.float32)

        res_m = np.zeros(2, dtype=np.float32)
        rmse_m = np.zeros(1, dtype=np.float32)

        _gaussian.moments_2d_gaussian(
            zeros_u16,
            X.astype(np.float32),
            Y.astype(np.float32),
            dum,
            stat_mean,
            stat_cov,
            1.0,  # weight_power
            res_m,
            rmse_m,
        )
        self.assertEqual(res_m.shape, (2,))
        self.assertTrue(np.allclose(res_m, 0.0))
        self.assertTrue(np.allclose(rmse_m, 0.0))

        res_mean = np.zeros(2, dtype=np.float32)
        res_cov = np.zeros((2, 2), dtype=np.float32)
        rmse = np.zeros(1, dtype=np.float32)
        _gaussian.fit_2d_gaussian(
            zeros_u16,
            X.astype(np.float32),
            Y.astype(np.float32),
            dum,
            stat_mean,
            stat_cov,
            1.0,
            res_mean,
            rmse,
            res_cov,
        )
        self.assertTrue(np.allclose(res_mean, 0.0))
        self.assertTrue(np.allclose(res_cov, 0.0))
        self.assertTrue(np.allclose(rmse, 0.0))

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

    def test_fit_2d_gaussian_ufunc(self):
        """Test fit_2d_gaussian ufunc returns mean, rmse, and covariance."""
        x = np.linspace(-5, 5, 30, dtype=np.float32)
        y = np.linspace(-3, 6, 20, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        amp, x0, y0, sigma_x, sigma_y, rho = 100.0, 1.5, -1.2, 1.2, 0.9, 0.3
        Z = _gaussian.gaussian_2d(X, Y, amp, x0, y0, sigma_x, sigma_y, rho)
        data_u16 = np.clip(Z, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        dum = np.arange(2, dtype=np.float32)
        stat_mean, stat_cov = self.moments(X, Y, Z)
        stat_mean = stat_mean.astype(np.float32)
        stat_cov = stat_cov.astype(np.float32)

        res_mean = np.zeros(2, dtype=np.float32)
        res_cov = np.zeros((2, 2), dtype=np.float32)
        rmse = np.zeros(1, dtype=np.float32)

        _gaussian.fit_2d_gaussian(
            data_u16,
            X.astype(np.float32),
            Y.astype(np.float32),
            dum,
            stat_mean,
            stat_cov,
            1.0,
            res_mean,
            rmse,
            res_cov,
        )

        self.assertEqual(res_mean.shape, (2,))
        self.assertTrue(np.isclose(res_mean[0], x0, rtol=5e-2))
        self.assertTrue(np.isclose(res_mean[1], y0, rtol=5e-2))

        self.assertEqual(res_cov.shape, (2, 2))
        self.assertTrue(np.isclose(res_cov[0, 0], sigma_x**2, rtol=5e-2))
        self.assertTrue(np.isclose(res_cov[1, 1], sigma_y**2, rtol=5e-2))
        expected_offdiag = rho * (sigma_x * sigma_y)
        self.assertTrue(np.isclose(res_cov[0, 1], expected_offdiag, rtol=5e-2))
        self.assertTrue(np.isclose(res_cov[1, 0], expected_offdiag, rtol=5e-2))

        self.assertEqual(rmse.shape, (1,))
        self.assertTrue(np.isfinite(rmse[0]))
        self.assertTrue(rmse[0] >= 0)

    def test_fit_2d_gaussian_ufunc_zero_input(self):
        """Test fit_2d_gaussian ufunc handles zero input correctly."""
        x = np.linspace(-1, 1, 8, dtype=np.float32)
        y = np.linspace(-1, 1, 6, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        zeros_u16 = np.zeros_like(X, dtype=np.uint16)
        dum = np.arange(2, dtype=np.float32)
        stat_mean = np.zeros(2, dtype=np.float32)
        stat_cov = np.zeros((2, 2), dtype=np.float32)

        res_mean = np.ones(2, dtype=np.float32) * np.nan
        res_cov = np.ones((2, 2), dtype=np.float32) * np.nan
        rmse = np.ones(1, dtype=np.float32) * np.nan

        _gaussian.fit_2d_gaussian(
            zeros_u16,
            X.astype(np.float32),
            Y.astype(np.float32),
            dum,
            stat_mean,
            stat_cov,
            1.0,
            res_mean,
            rmse,
            res_cov,
        )

        self.assertTrue(np.allclose(res_mean, 0.0))
        self.assertTrue(np.allclose(res_cov, 0.0))
        self.assertTrue(np.allclose(rmse, 0.0))


class TestFitGaussian(unittest.TestCase):
    """Tests for the fit_gaussian function in properties module."""

    def test_fit_gaussian_basic(self):
        """Test fit_gaussian with simple synthetic data."""
        a, b, m, n = 10, 10, 15, 15
        x = np.linspace(-2, 2, m, dtype=np.float32)
        y = np.linspace(-2, 2, n, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data = np.zeros((a, b, m, n), dtype=np.uint16)

        for i in range(a):
            for j in range(b):
                x0 = 0.1 * (i - a / 2) / (a / 2)
                y0 = 0.1 * (j - b / 2) / (b / 2)
                amp = 100.0
                sigma = 0.5

                gauss = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)) * amp
                data[i, j] = np.clip(gauss, 0, 65535).astype(np.uint16)

        fitted_mean, fitted_cov, rmse = properties.fit_gaussian(
            data, coordinates=(X, Y)
        )

        self.assertEqual(fitted_mean.shape, (a, b, 2))
        self.assertEqual(fitted_cov.shape, (a, b, 2, 2))
        self.assertEqual(rmse.shape, (a, b))

        self.assertEqual(fitted_mean.dtype, np.float32)
        self.assertEqual(fitted_cov.dtype, np.float32)
        self.assertEqual(rmse.dtype, np.float32)

        self.assertTrue(np.all(np.isfinite(fitted_mean)))
        self.assertTrue(np.all(np.isfinite(fitted_cov)))
        self.assertTrue(np.all(np.isfinite(rmse)))

        self.assertTrue(np.all(rmse >= 0))

        np.testing.assert_array_equal(fitted_cov[..., 0, 1], fitted_cov[..., 1, 0])

    def test_fit_gaussian_with_stat_moments(self):
        """Test fit_gaussian with provided statistical moments."""
        a, b, m, n = 5, 5, 10, 10
        x = np.linspace(-1, 1, m, dtype=np.float32)
        y = np.linspace(-1, 1, n, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data = np.zeros((a, b, m, n), dtype=np.uint16)
        center_m, center_n = m // 2, n // 2
        data[:, :, center_m, center_n] = 1000

        grid = np.stack((X, Y))
        stat_mean, stat_cov = properties.moments(data, grid)

        fitted_mean, fitted_cov, rmse = properties.fit_gaussian(
            data, coordinates=(X, Y), stat_mean=stat_mean, stat_cov=stat_cov
        )

        self.assertEqual(fitted_mean.shape, (a, b, 2))
        self.assertEqual(fitted_cov.shape, (a, b, 2, 2))
        self.assertEqual(rmse.shape, (a, b))

    def test_fit_gaussian_weight_power(self):
        """Test fit_gaussian with different weight_power values."""
        a, b, m, n = 3, 3, 10, 10
        x = np.linspace(-1, 1, m, dtype=np.float32)
        y = np.linspace(-1, 1, n, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data = np.zeros((a, b, m, n), dtype=np.uint16)
        for i in range(a):
            for j in range(b):
                gauss = np.exp(-(X**2 + Y**2) / 0.5) * 200
                data[i, j] = np.clip(gauss, 0, 65535).astype(np.uint16)

        for weight_power in [0.5, 1.0, 2.0]:
            fitted_mean, fitted_cov, rmse = properties.fit_gaussian(
                data, coordinates=(X, Y), weight_power=weight_power
            )

            self.assertEqual(fitted_mean.shape, (a, b, 2))
            self.assertTrue(np.all(np.isfinite(fitted_mean)))
            self.assertTrue(np.all(np.isfinite(rmse)))

    def test_fit_gaussian_zero_data(self):
        """Test fit_gaussian handles zero data correctly."""
        a, b, m, n = 5, 5, 8, 8
        x = np.linspace(-1, 1, m, dtype=np.float32)
        y = np.linspace(-1, 1, n, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data = np.zeros((a, b, m, n), dtype=np.uint16)

        fitted_mean, fitted_cov, rmse = properties.fit_gaussian(
            data, coordinates=(X, Y)
        )

        self.assertEqual(fitted_mean.shape, (a, b, 2))
        self.assertEqual(fitted_cov.shape, (a, b, 2, 2))
        self.assertEqual(rmse.shape, (a, b))

        self.assertTrue(np.all(np.isfinite(fitted_mean)))
        self.assertTrue(np.all(np.isfinite(fitted_cov)))
        self.assertTrue(np.all(np.isfinite(rmse)))

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
        fitted_mean, _, rmse = properties.fit_gaussian(
            data, grid, stat_mean=stat_mean, stat_cov=stat_cov
        )

        center = np.array([true_x0, true_y0])
        self.assertLess(
            np.abs(fitted_mean - center).mean(), np.abs(stat_mean - center).mean()
        )
        self.assertTrue(np.isfinite(rmse).all())
        self.assertTrue((rmse >= 0).all())

    def test_fit_gaussian_shape_validation(self):
        """Test that fit_gaussian validates input shapes correctly."""
        x = np.linspace(-1, 1, 5, dtype=np.float32)
        y = np.linspace(-1, 1, 5, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data_wrong = np.zeros((10, 10, 5), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "two motor grids"):
            properties.fit_gaussian(data_wrong, coordinates=(X, Y))

        data_4d = np.zeros((5, 5, 5, 5), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "two motor grids"):
            properties.fit_gaussian(data_4d, coordinates=(X, Y, X))


if __name__ == "__main__":
    unittest.main()
