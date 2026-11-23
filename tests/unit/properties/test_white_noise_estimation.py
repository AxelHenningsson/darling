import unittest

import numpy as np

import darling
from darling.properties._noise import (
    _curve_fit_white_noise_1D,
    _curve_fit_white_noise_parallel,
    _histogram_white_noise_1d,
    _iterate_white_noise_tail_statistics,
)


class TestWhiteNoiseEstimation(unittest.TestCase):
    def setUp(self):
        self.debug = False

        self.rng = np.random.default_rng(9)

        shape = (2, 2, 20, 20, 20)
        self.mean = 109
        self.std = 9
        self.data = self.rng.normal(self.mean, self.std, size=shape)
        self.data = self.data.flatten()
        index = self.rng.permutation(self.data.size)[:1000]
        self.data[index] += self.rng.normal(900, 10, size=len(index))
        self.data = np.round(self.data).astype(np.uint16)
        self.data = self.data.reshape(shape)

    def test_estimate_white_noise(self):
        true_mean = 109.4
        true_std = 9.2
        shape = (16, 16, 21, 21, 21)
        k, l, m = np.array(shape)[2:] // 2
        data = self.rng.normal(true_mean, true_std, size=shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if self.rng.random() < 0.5:
                    data[i, j, k - 1 : k + 2, l - 1 : l + 3, m - 2 : m + 3] = (
                        350 + np.random.normal(0, 100)
                    )
                if self.rng.random() > 0.5:
                    data[i, j, k - 3 : k + 5, l - 3 : l + 2, m - 3 : m + 2] = (
                        900 + np.random.normal(0, 100)
                    )

        # A really poor initial guess
        mean_guess = 145
        std_guess = 5

        mean, std = darling.properties.estimate_white_noise(
            data,
            inital_guess=(mean_guess, std_guess),
        )

        average_mean_estimate = mean.mean()
        average_std_estimate = std.mean()

        np.testing.assert_array_less(
            np.abs(average_mean_estimate - true_mean) / true_mean, 0.02
        )
        np.testing.assert_array_less(
            np.abs(average_std_estimate - true_std) / true_std, 0.02
        )

        lowest_mean_estimate = mean.min()
        lowest_std_estimate = std.min()

        np.testing.assert_array_less(
            np.abs(lowest_mean_estimate - true_mean) / true_mean, 0.05
        )
        np.testing.assert_array_less(
            np.abs(lowest_std_estimate - true_std) / true_std, 0.05
        )

        highest_mean_estimate = mean.max()
        highest_std_estimate = std.max()

        np.testing.assert_array_less(
            np.abs(highest_mean_estimate - true_mean) / true_mean, 0.05
        )
        np.testing.assert_array_less(
            np.abs(highest_std_estimate - true_std) / true_std, 0.05
        )

        if self.debug:
            import matplotlib.pyplot as plt

            print(
                f"Max Mean: {mean.max()}, Min Mean: {mean.min()}, Mean Mean: {mean.mean()}"
            )
            print(f"Max Std: {std.max()}, Min Std: {std.min()}, Mean Std: {std.mean()}")
            fontsize = ticksize = 22
            plt.style.use("dark_background")
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["xtick.labelsize"] = ticksize
            plt.rcParams["ytick.labelsize"] = ticksize
            plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            im = ax[0].imshow(mean)
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(std)
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            plt.show()

    def test_iterate_white_noise_tail_statistics(self):
        new_mean, new_std = _iterate_white_noise_tail_statistics(
            self.data[0, 0].ravel().astype(np.float64),
            3.5,
            self.mean,
            self.std,
        )
        self.assertLessEqual(np.abs(new_mean - self.mean), 1)
        self.assertLessEqual(np.abs(new_std - self.std), 0.5)

    def test_histogram_1d(self):
        truncate = 4.1
        hist, bin_edges_start, bin_size = _histogram_white_noise_1d(
            self.data[0, 0].ravel(), truncate, self.mean, self.std
        )

        bin_edges = np.array(
            [bin_edges_start + i * bin_size for i in range(len(hist) + 1)]
        )
        y = self.data[0, 0].copy().ravel()
        y = y[
            (y < self.mean + truncate * self.std)
            & (y > self.mean - truncate * self.std)
        ]
        numpy_hist, bedges = np.histogram(y, bins=bin_edges)

        np.testing.assert_array_equal(hist, numpy_hist)

        new_mean, new_std, success = _curve_fit_white_noise_1D(
            y, truncate, self.mean, self.std, n_iter_gauss_newton=5
        )

        mean_array = np.ones(self.data.shape[:2]) * self.mean
        std_array = np.ones(self.data.shape[:2]) * self.std

        new_mean, new_std = _curve_fit_white_noise_parallel(
            self.data,
            truncate,
            mean_array,
            std_array,
            n_iter_gauss_newton=5,
        )

        np.testing.assert_array_less(np.abs(new_mean - self.mean), 1)
        np.testing.assert_array_less(np.abs(new_std - self.std), 0.5)

        if self.debug:
            import matplotlib.pyplot as plt

            fontsize = 22
            ticksize = 22
            plt.style.use("dark_background")
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["xtick.labelsize"] = ticksize
            plt.rcParams["ytick.labelsize"] = ticksize
            plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            bc = (bedges[:-1] + bedges[1:]) / 2.0
            ax.plot(bc, numpy_hist, "bo-", label="numpy")
            ax.plot(bc, hist, "ro--", label="darling")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(True, alpha=0.25)
            ax.legend()
            plt.show()


if __name__ == "__main__":
    unittest.main()
