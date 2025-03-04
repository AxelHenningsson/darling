import unittest

import matplotlib.pyplot as plt
import numpy as np

from darling import _gaussian


class TestGaussianFitting(unittest.TestCase):
    """Tests for the Gaussian fitting module."""

    def setUp(self):
        self.debug = False  # Set to True to show debug plots

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
        self.assertAlmostEqual(value, amplitude, places=6)

        value = _gaussian.gaussian_2d(1, 0, amplitude, x0, y0, sigma_x, sigma_y, rho)
        self.assertAlmostEqual(value, amplitude * np.exp(-0.5), places=6)

        value = _gaussian.gaussian_2d(0, 1, amplitude, x0, y0, sigma_x, sigma_y, rho)
        self.assertAlmostEqual(value, amplitude * np.exp(-0.5), places=6)

        rho = 0.5
        value = _gaussian.gaussian_2d(1, 1, amplitude, x0, y0, sigma_x, sigma_y, rho)
        expected = amplitude * np.exp(-0.5 * (1 + 1 - 2 * 0.5) / (1 - 0.5**2))
        self.assertAlmostEqual(value, expected, places=6)

        value = _gaussian.gaussian_2d(0, 0, amplitude, x0, y0, 1e-11, 1e-11, rho)
        self.assertGreater(value, 0)
        
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


if __name__ == "__main__":
    unittest.main()