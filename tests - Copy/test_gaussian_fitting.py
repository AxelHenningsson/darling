import unittest

import matplotlib.pyplot as plt
import numpy as np

from darling import _gaussian


class TestGaussianFitting(unittest.TestCase):
    """Tests for the Gaussian fitting module."""

    def setUp(self):
        self.debug = False  # Set to True to show debug plots
        np.random.seed(42)

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

        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = _gaussian.gaussian_2d(X[j, i], Y[j, i], 100.0, x0, y0, sigma_x, sigma_y, rho)
        
        total = np.sum(Z)
        mean_x = np.sum(Z * X) / total
        mean_y = np.sum(Z * Y) / total
        var_x = np.sum(Z * (X - mean_x)**2) / total
        var_y = np.sum(Z * (Y - mean_y)**2) / total
        cov_xy = np.sum(Z * (X - mean_x) * (Y - mean_y)) / total

        data_flat = Z.flatten()
        stat_mean = np.array([mean_x, mean_y])
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        
        params = _gaussian.params_from_moments(data_flat, stat_mean, stat_cov)
        
        self.assertAlmostEqual(params[0], np.max(Z), places=2)  # amplitude
        self.assertAlmostEqual(params[1], mean_x, places=2)      # x0
        self.assertAlmostEqual(params[2], mean_y, places=2)      # y0
        self.assertAlmostEqual(params[3], np.sqrt(var_x), places=2)  # sigma_x
        self.assertAlmostEqual(params[4], np.sqrt(var_y), places=2)  # sigma_y
        self.assertAlmostEqual(params[5], cov_xy / (np.sqrt(var_x) * np.sqrt(var_y)), places=2)  # rho
        
        params = _gaussian.params_from_moments(np.zeros(10), np.zeros(2), np.zeros((2, 2)))
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

        data = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                data[j, i] = _gaussian.gaussian_2d(
                    X[j, i], Y[j, i], 
                    true_amplitude, true_x0, true_y0,
                    true_sigma_x, true_sigma_y, true_rho
                )

        total = np.sum(data)
        mean_x = np.sum(data * X) / total
        mean_y = np.sum(data * Y) / total
        var_x = np.sum(data * (X - mean_x)**2) / total
        var_y = np.sum(data * (Y - mean_y)**2) / total
        cov_xy = np.sum(data * (X - mean_x) * (Y - mean_y)) / total
        
        stat_mean = np.array([mean_x, mean_y])
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])

        params = _gaussian.fit_gaussian_2d(
            data.flatten(), x_flat, y_flat, stat_mean, stat_cov
        )

        self.assertAlmostEqual(params[0], true_amplitude, delta=true_amplitude*0.05)  # amplitude within 5%
        self.assertAlmostEqual(params[1], true_x0, delta=0.05)        # x0
        self.assertAlmostEqual(params[2], true_y0, delta=0.05)        # y0
        self.assertAlmostEqual(params[3], true_sigma_x, delta=0.05)   # sigma_x
        self.assertAlmostEqual(params[4], true_sigma_y, delta=0.05)   # sigma_y
        self.assertAlmostEqual(params[5], true_rho, delta=0.05)       # rho
        
        if self.debug:
            fitted_data = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    fitted_data[j, i] = _gaussian.gaussian_2d(
                        X[j, i], Y[j, i], 
                        params[0], params[1], params[2],
                        params[3], params[4], params[5]
                    )
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            im0 = ax[0].imshow(data, cmap='viridis')
            fig.colorbar(im0, ax=ax[0])
            ax[0].set_title('Original Gaussian')
            
            im1 = ax[1].imshow(fitted_data, cmap='viridis')
            fig.colorbar(im1, ax=ax[1])
            ax[1].set_title('Fitted Gaussian')
            
            im2 = ax[2].imshow(np.abs(data - fitted_data), cmap='jet')
            fig.colorbar(im2, ax=ax[2])
            ax[2].set_title('Residual')
            
            plt.tight_layout()
            plt.show()
            
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

        data = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                data[j, i] = _gaussian.gaussian_2d(
                    X[j, i], Y[j, i], 
                    true_amplitude, true_x0, true_y0,
                    true_sigma_x, true_sigma_y, true_rho
                )

        np.random.seed(42)
        noise = np.random.normal(0, 5.0, data.shape)
        noisy_data = data + noise

        total = np.sum(noisy_data)
        mean_x = np.sum(noisy_data * X) / total
        mean_y = np.sum(noisy_data * Y) / total
        var_x = np.sum(noisy_data * (X - mean_x)**2) / total
        var_y = np.sum(noisy_data * (Y - mean_y)**2) / total
        cov_xy = np.sum(noisy_data * (X - mean_x) * (Y - mean_y)) / total
        
        stat_mean = np.array([mean_x, mean_y])
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])

        params = _gaussian.fit_gaussian_2d(
            noisy_data.flatten(), x_flat, y_flat, stat_mean, stat_cov
        )

        self.assertAlmostEqual(params[0], true_amplitude, delta=true_amplitude*0.1)  # amplitude within 10%
        self.assertAlmostEqual(params[1], true_x0, delta=0.2)        # x0
        self.assertAlmostEqual(params[2], true_y0, delta=0.2)        # y0
        self.assertAlmostEqual(params[3], true_sigma_x, delta=0.3)   # sigma_x
        self.assertAlmostEqual(params[4], true_sigma_y, delta=0.3)   # sigma_y
        self.assertAlmostEqual(params[5], true_rho, delta=0.3)       # rho
        
        if self.debug:
            fitted_data = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    fitted_data[j, i] = _gaussian.gaussian_2d(
                        X[j, i], Y[j, i], 
                        params[0], params[1], params[2],
                        params[3], params[4], params[5]
                    )
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            
            im0 = ax[0, 0].imshow(data, cmap='viridis')
            fig.colorbar(im0, ax=ax[0, 0])
            ax[0, 0].set_title('Original Gaussian')
            
            im1 = ax[0, 1].imshow(noisy_data, cmap='viridis')
            fig.colorbar(im1, ax=ax[0, 1])
            ax[0, 1].set_title('Noisy Gaussian')
            
            im2 = ax[1, 0].imshow(fitted_data, cmap='viridis')
            fig.colorbar(im2, ax=ax[1, 0])
            ax[1, 0].set_title('Fitted Gaussian')
            
            im3 = ax[1, 1].imshow(np.abs(noisy_data - fitted_data), cmap='jet')
            fig.colorbar(im3, ax=ax[1, 1])
            ax[1, 1].set_title('Residual')
            
            plt.tight_layout()
            plt.show()
            

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

        total = noisy.sum()
        mean_x = (noisy * X).sum() / total
        mean_y = (noisy * Y).sum() / total
        var_x = (noisy * (X - mean_x)**2).sum() / total
        var_y = (noisy * (Y - mean_y)**2).sum() / total
        cov_xy = (noisy * (X - mean_x) * (Y - mean_y)).sum() / total

        stat_mean = np.array([mean_x, mean_y])
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])

        init_params = _gaussian.params_from_moments(noisy.flatten(), stat_mean, stat_cov)

        def mse_from_params(p):
            pred = _gaussian.gaussian_2d(X, Y, p[0], p[1], p[2], p[3], p[4], p[5])
            return np.mean((pred - noisy)**2)

        mse_init = mse_from_params(init_params)
        fit_params = _gaussian.fit_gaussian_2d(noisy.flatten(), x_flat, y_flat, stat_mean, stat_cov)
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
            zero_data, x_flat, y_flat, stat_mean, stat_cov
        )

        self.assertTrue(np.all(params == 0))
        
        single_point_data = np.zeros_like(X)
        single_point_data[15, 15] = 100.0

        total = np.sum(single_point_data)
        mean_x = np.sum(single_point_data * X) / total
        mean_y = np.sum(single_point_data * Y) / total
        var_x = np.sum(single_point_data * (X - mean_x)**2) / total
        var_y = np.sum(single_point_data * (Y - mean_y)**2) / total
        cov_xy = np.sum(single_point_data * (X - mean_x) * (Y - mean_y)) / total
        
        stat_mean = np.array([mean_x, mean_y])
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])

        params = _gaussian.fit_gaussian_2d(
            single_point_data.flatten(), x_flat, y_flat, stat_mean, stat_cov
        )
        
        self.assertAlmostEqual(params[0], 100.0, delta=10.0)  # amplitude
        self.assertAlmostEqual(params[1], x[15], delta=0.5)   # x0
        self.assertAlmostEqual(params[2], y[15], delta=0.5)   # y0
        
    def test_moments_2d_gaussian_ufunc(self):
        # Non-square grid, float32 inputs as required by numba
        x = np.linspace(-5, 5, 30, dtype=np.float32)
        y = np.linspace(-3, 6, 20, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        amp, x0, y0, sigma_x, sigma_y, rho = 100.0, 1.5, -1.2, 1.2, 0.9, 0.3
        Z = _gaussian.gaussian_2d(X, Y, amp, x0, y0, sigma_x, sigma_y, rho)
        # Ensure dtype conversion happens after computing the clip bounds
        data_u16 = np.clip(Z, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        total = Z.sum()
        mean_x = (Z * X).sum() / total
        mean_y = (Z * Y).sum() / total
        var_x = (Z * (X - mean_x)**2).sum() / total
        var_y = (Z * (Y - mean_y)**2).sum() / total
        cov_xy = (Z * (X - mean_x) * (Y - mean_y)).sum() / total

        dum = np.zeros(2, dtype=np.float32)
        stat_mean = np.array([mean_x, mean_y], np.float32)
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], np.float32)
        
        res = _gaussian.moments_2d_gaussian(data_u16, X.astype(np.float32), Y.astype(np.float32), dum, stat_mean, stat_cov)
        self.assertEqual(res.shape, (2,))
        self.assertTrue(np.isclose(res[0], x0, rtol=5e-2))
        self.assertTrue(np.isclose(res[1], y0, rtol=5e-2))

    def test_covariance_2d_gaussian_ufunc(self):
        # Non-square grid, float32 inputs as required by numba
        x = np.linspace(-5, 5, 30, dtype=np.float32)
        y = np.linspace(-3, 6, 20, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        amp, x0, y0, sigma_x, sigma_y, rho = 100.0, 1.5, -1.2, 1.2, 0.9, 0.3
        Z = _gaussian.gaussian_2d(X, Y, amp, x0, y0, sigma_x, sigma_y, rho)
        # Ensure dtype conversion happens after computing the clip bounds
        data_u16 = np.clip(Z, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        
        total = Z.sum()
        mean_x = (Z * X).sum() / total
        mean_y = (Z * Y).sum() / total
        var_x = (Z * (X - mean_x)**2).sum() / total
        var_y = (Z * (Y - mean_y)**2).sum() / total
        cov_xy = (Z * (X - mean_x) * (Y - mean_y)).sum() / total
        
        dum = np.zeros(2, dtype=np.float32)
        first_moments = np.array([mean_x, mean_y], np.float32)
        stat_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], np.float32)
        points = np.stack([X.astype(np.float32).ravel(), Y.astype(np.float32).ravel()])
        
        res = _gaussian.covariance_2d_gaussian(data_u16, first_moments, points, dum, stat_cov)
        self.assertEqual(res.shape, (2, 2))
        self.assertTrue(np.isclose(res[0, 0], sigma_x**2, rtol=5e-2))
        self.assertTrue(np.isclose(res[1, 1], sigma_y**2, rtol=5e-2))
        expected_offdiag = rho * (sigma_x * sigma_y)
        self.assertTrue(np.isclose(res[0, 1], expected_offdiag, rtol=5e-2))
        self.assertTrue(np.isclose(res[1, 0], expected_offdiag, rtol=5e-2))

    def test_ufuncs_zero_input(self):
        """Ufuncs should return zeros for empty or zero data."""
        x = np.linspace(-1, 1, 8, dtype=np.float32)
        y = np.linspace(-1, 1, 6, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        zeros_u16 = np.zeros_like(X, dtype=np.uint16)
        dum = np.zeros(2, dtype=np.float32)
        stat_mean = np.zeros(2, dtype=np.float32)
        stat_cov = np.zeros((2, 2), dtype=np.float32)

        res_m = _gaussian.moments_2d_gaussian(zeros_u16, X.astype(np.float32), Y.astype(np.float32), dum, stat_mean, stat_cov)
        self.assertEqual(res_m.shape, (2,))
        self.assertTrue(np.allclose(res_m, 0.0))

        pts = np.stack([X.astype(np.float32).ravel(), Y.astype(np.float32).ravel()])
        res_c = _gaussian.covariance_2d_gaussian(zeros_u16, stat_mean, pts, dum, stat_cov)
        self.assertEqual(res_c.shape, (2, 2))
        self.assertTrue(np.allclose(res_c, 0.0))

if __name__ == "__main__":
    unittest.main()