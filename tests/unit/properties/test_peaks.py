import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import darling


class TestPeakSearcher(unittest.TestCase):
    def setUp(self):
        self.debug = False

    def test_extract_features(self):
        img = np.zeros((256, 256), dtype=np.float32)
        targetindices = [(0, 0), (100, 100), (150, 150), (200, 200), (255, 255)]
        intensities = [1, 2, 3, 1, 1236]
        for i, (x, y) in enumerate(targetindices):
            img[x, y] = intensities[i]

        img[x - 1, y - 1] = 1
        img[x, y - 1] = 1

        labeled_array, n_features = darling.properties.local_max_label(
            img, loop_outer_dims=False
        )
        features = darling.properties.extract_features(
            labeled_array, img, k=len(intensities) + 1
        )
        self.assertEqual(n_features, len(targetindices))
        self.assertEqual(features.shape[1], len(intensities) + 1)
        self.assertEqual(features[0, -1], 0)

        exp_intensities = np.sort(intensities)[::-1]
        exp_intensities[0] += 2
        exp_intensities_n_pix = [3, 1, 1, 1, 1]

        for i, ei in enumerate(exp_intensities):
            self.assertTrue(ei == features[0, i])
            self.assertTrue(exp_intensities_n_pix[i] == features[2, i])

    def test_local_max_label_2D_simple(self):
        img = np.zeros((256, 256), dtype=np.float32)
        targetindices = [(0, 0), (100, 100), (150, 150), (200, 200), (255, 255)]
        intensities = [1, 2, 3, 1, 1236]
        for i, (x, y) in enumerate(targetindices):
            img[x, y] = intensities[i]

        img[x - 1, y - 1] = 1
        img[x, y - 1] = 1

        labeled_array, n_features = darling.properties.local_max_label(
            img, loop_outer_dims=False
        )

        np.testing.assert_equal(n_features, len(intensities))
        np.testing.assert_equal(labeled_array.max(), len(intensities))

        i, j = np.where(labeled_array != 0)

        for x, y in targetindices:
            self.assertTrue(x in i)
            self.assertTrue(y in j)

    def test_local_max_label_2D(self):
        h, w = 256, 256
        x, y = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        img = np.zeros((h, w), dtype=np.float32)

        n_expected_features = 0
        for i in range(0, h, 21):
            for j in range(0, w, 21):
                sigma = np.random.uniform(1, 4)
                img += np.exp(-((x - i) ** 2 + (y - j) ** 2) / (2 * sigma**2))
                n_expected_features += 1

        img /= img.max()
        img[img < 0.1] = 0
        img *= 65535
        img = img.astype(np.uint16)
        img = img.clip(0, 65535)

        labeled_array, n_features = darling.properties.local_max_label(
            img, loop_outer_dims=False
        )

        np.testing.assert_equal(n_features, n_expected_features)
        np.testing.assert_equal(labeled_array.max(), n_expected_features)

        for i in range(1, n_features + 1):
            _labeled_array = labeled_array.copy()
            _labeled_array[_labeled_array != i] = 0
            _labeled_array[_labeled_array == i] = 1
            cx, cy = np.unravel_index(
                np.argmax(_labeled_array * img), _labeled_array.shape
            )
            self.assertTrue(img[cx, cy] > 0)
            self.assertTrue(cx in range(0, h, 21))
            self.assertTrue(cy in range(0, w, 21))

        if self.debug:
            import matplotlib.pyplot as plt
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            im = ax[0].imshow(img)
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(labeled_array, cmap="hsv")
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_peaksearch(self):
        a, b, m, n, o = 2, 3, 12, 12, 12
        x, y, z = np.meshgrid(np.arange(m), np.arange(n), np.arange(o), indexing="ij")
        coordinates = np.array([x, y, z])
        data = np.zeros((a, b, m, n, o))
        data[0, 0, 3, 6, 6] = 1
        data[0, 0, 9, 6, 6] = 1
        data[0, 0] = ndimage.gaussian_filter(data[0, 0], 1)
        data[data < np.max(data) * 0.03] = 0
        data *= 65535
        features = darling.properties.peaks(data, k=2, coordinates=coordinates)

        self.assertTrue(isinstance(features, darling.properties.PeakMap))
        self.assertTrue("sum_intensity" in features)
        self.assertTrue("max_intensity" in features)
        self.assertTrue("number_of_pixels" in features)
        self.assertTrue("maxima_axis_0" in features)
        self.assertTrue("maxima_axis_1" in features)
        self.assertTrue("maxima_axis_2" in features)
        self.assertTrue("mean_axis_0" in features)
        self.assertTrue("mean_axis_1" in features)
        self.assertTrue("mean_axis_2" in features)
        self.assertTrue("variance_axis_0" in features)
        self.assertTrue("variance_axis_1" in features)
        self.assertTrue("variance_axis_2" in features)
        self.assertTrue("variance_axis_0_axis_1" in features)
        self.assertTrue("variance_axis_0_axis_2" in features)
        self.assertTrue("variance_axis_1_axis_2" in features)
        for key in features:
            self.assertTrue(features[key].shape == (a, b, 2))

        np.testing.assert_allclose(
            np.sort(features["maxima_axis_0"][0, 0]), np.array([3, 9])
        )
        np.testing.assert_allclose(
            np.sort(features["maxima_axis_1"][0, 0]), np.array([6, 6])
        )
        np.testing.assert_allclose(
            np.sort(features["maxima_axis_2"][0, 0]), np.array([6, 6])
        )

        sig1 = np.sqrt(features["variance_axis_0"][0, 0])
        sig2 = np.sqrt(features["variance_axis_1"][0, 0])
        sig3 = np.sqrt(features["variance_axis_2"][0, 0])
        cov12 = features["variance_axis_0_axis_1"][0, 0]
        cov13 = features["variance_axis_0_axis_2"][0, 0]
        cov23 = features["variance_axis_1_axis_2"][0, 0]

        np.testing.assert_allclose(sig1, 1, atol=0.1, rtol=0.1)
        np.testing.assert_allclose(sig2, 1, atol=0.1, rtol=0.1)
        np.testing.assert_allclose(sig3, 1, atol=0.1, rtol=0.1)
        np.testing.assert_allclose(cov12, 0, atol=0.01, rtol=0.01)
        np.testing.assert_allclose(cov13, 0, atol=0.01, rtol=0.01)
        np.testing.assert_allclose(cov23, 0, atol=0.01, rtol=0.01)

        a, b, m, n = 2, 3, 12, 12
        x, y = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
        coordinates = np.array([x, y])
        data = np.zeros((a, b, m, n))
        data[0, 0, 3, 6] = 1
        data[0, 0, 9, 6] = 1
        data[0, 0] = ndimage.gaussian_filter(data[0, 0], 1)
        data[data < np.max(data) * 0.03] = 0
        data *= 65535
        features = darling.properties.peaks(data, k=2, coordinates=coordinates)

        self.assertTrue(isinstance(features, darling.properties.PeakMap))
        self.assertTrue("sum_intensity" in features)
        self.assertTrue("max_intensity" in features)
        self.assertTrue("number_of_pixels" in features)
        self.assertTrue("maxima_axis_0" in features)
        self.assertTrue("maxima_axis_1" in features)
        self.assertTrue("mean_axis_0" in features)
        self.assertTrue("mean_axis_1" in features)
        self.assertTrue("variance_axis_0" in features)
        self.assertTrue("variance_axis_1" in features)
        for key in features:
            self.assertTrue(features[key].shape == (a, b, 2))

        a, b, m = 2, 3, 12
        coordinates = np.arange(m).reshape(1, -1)
        data = np.zeros((a, b, m))
        data[0, 0, 3] = 1
        data[0, 0, 9] = 1
        data[0, 0] = ndimage.gaussian_filter(data[0, 0], 1)
        features = darling.properties.peaks(data, k=2, coordinates=coordinates)

        self.assertTrue(isinstance(features, darling.properties.PeakMap))
        self.assertTrue("sum_intensity" in features)
        self.assertTrue("max_intensity" in features)
        self.assertTrue("number_of_pixels" in features)
        self.assertTrue("maxima_axis_0" in features)
        self.assertTrue("mean_axis_0" in features)
        self.assertTrue("variance_axis_0" in features)
        for key in features:
            self.assertTrue(features[key].shape == (a, b, 2))

    def test_local_max_label_3D_parallel(self):
        a, b, m, n, o = 2, 3, 12, 12, 12
        data = np.zeros((a, b, m, n, o))
        data[0, 0, 3, 6, 6] = 1
        data[0, 0, 9, 6, 6] = 1
        data[0, 0] = ndimage.gaussian_filter(data[0, 0], 1)
        data[data < np.max(data) * 0.03] = 0
        data *= 65535
        labeled_array, nlabels = darling.properties.local_max_label(data)
        ndimage_labeled_array, ndimage_n_features = ndimage.label(data[0, 0, :, :, :])
        np.testing.assert_equal(nlabels[0, 0], ndimage_n_features)
        np.testing.assert_equal(labeled_array[0, 0] > 0, ndimage_labeled_array > 0)
        np.testing.assert_allclose(nlabels[1:, 1:], 0)

        if 0:
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            fig, ax = plt.subplots(2, 3, figsize=(14, 7))
            im = ax[0, 0].imshow(data[0, 0, :, :, 6])
            fig.colorbar(im, ax=ax[0, 0], fraction=0.026, pad=0.04)
            im = ax[0, 1].imshow(labeled_array[0, 0, :, :, 6])
            fig.colorbar(im, ax=ax[0, 1], fraction=0.026, pad=0.04)
            im = ax[0, 2].imshow(ndimage_labeled_array[:, :, 6])
            fig.colorbar(im, ax=ax[0, 2], fraction=0.026, pad=0.04)
            im = ax[1, 0].imshow(data[0, 0, :, 6, :])
            fig.colorbar(im, ax=ax[1, 0], fraction=0.026, pad=0.04)
            im = ax[1, 1].imshow(labeled_array[0, 0, :, 6, :])
            fig.colorbar(im, ax=ax[1, 1], fraction=0.026, pad=0.04)
            im = ax[1, 2].imshow(ndimage_labeled_array[:, 6, :])
            fig.colorbar(im, ax=ax[1, 2], fraction=0.026, pad=0.04)
            ax[0, 0].set_title("Original image")
            ax[0, 1].set_title("Labeled image")
            ax[0, 2].set_title("NDImage labeled image")
            ax[1, 0].set_title("Original image")
            ax[1, 1].set_title("Labeled image")
            ax[1, 2].set_title("NDImage labeled image")
            plt.tight_layout()
            plt.show()

        data = np.zeros((a, b, m, n, o))
        data[0, 0, 3, 6, 6] = 1
        data[0, 0, 9, 6, 6] = 1
        data[0, 1, 3, 6, 6] = 1
        data[0, 1, 8, 5, 6] = 2
        data[0, 0] = ndimage.gaussian_filter(data[0, 0], 1)
        data *= 65535
        labeled_array, nlabels = darling.properties.local_max_label(data)
        np.testing.assert_equal(nlabels[0, 0], 2)
        np.testing.assert_equal(nlabels[0, 1], 2)
        np.testing.assert_allclose(nlabels[1:, 2:], 0)

    def test_local_max_label_1D_parallel(self):
        a, b, m = 2, 3, 32
        data = np.zeros((a, b, m))
        data[0, 0, 3] = 1
        data[0, 0, 9] = 1
        data[0, 1, 12] = 1
        data[0, 1, 17] = 2
        data[0, 0] = ndimage.gaussian_filter(data[0, 0], 1)
        data *= 65534
        labeled_array, nlabels = darling.properties.local_max_label(
            data.astype(np.uint16), loop_outer_dims=True
        )
        self.assertEqual(nlabels[0, 0], 2)
        self.assertEqual(nlabels[0, 1], 2)
        np.testing.assert_allclose(nlabels[1:, 2:], 0)
        self.assertEqual(set(labeled_array[0, 0, [3, 9]]), {1, 2})
        self.assertEqual(set(labeled_array[0, 1, [12, 17]]), {1, 2})
        labeled_array, nlabels = darling.properties.local_max_label(
            data.astype(np.float32), loop_outer_dims=True
        )
        self.assertEqual(nlabels[0, 0], 2)
        self.assertEqual(nlabels[0, 1], 2)
        np.testing.assert_allclose(nlabels[1:, 2:], 0)
        self.assertEqual(set(labeled_array[0, 0, [3, 9]]), {1, 2})
        self.assertEqual(set(labeled_array[0, 1, [12, 17]]), {1, 2})


class TestPeaks(unittest.TestCase):
    def setUp(self):
        self.debug = False
        _, self.data, self.coordinates = darling.io.assets.domains()

    def test_peaks_on_domains_data(self):
        features = darling.properties.peaks(
            self.data,
            k=16,
            coordinates=self.coordinates,
        )

        trailing_dims = len(self.data.shape) - 2

        for key in darling.properties._peaks._FEATURE_MAPPING[trailing_dims]:
            self.assertTrue(key in features)

        for key in features:
            self.assertEqual(features[key].shape[0], self.data.shape[0])
            self.assertEqual(features[key].shape[1], self.data.shape[1])
            self.assertEqual(features[key].shape[2], 16)
            self.assertTrue(np.all(np.isfinite(features[key])))
            self.assertTrue(np.all(np.isreal(features[key])))

        self.assertTrue(np.all(features["sum_intensity"] >= 0))
        self.assertTrue(np.all(features["number_of_pixels"] >= 0))

        if self.debug:
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
            im1 = ax1.imshow(features["mean_motor1"][..., 0])
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            im2 = ax2.imshow(features["mean_motor2"][..., 0])
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    def test_peaks_different_dtypes(self):
        for dtype in [np.uint16, np.float32, np.float64, np.int32]:
            features = darling.properties.peaks(
                self.data.astype(dtype),
                k=16,
                coordinates=self.coordinates,
            )

            for key in features:
                self.assertEqual(features[key].shape[0], self.data.shape[0])
                self.assertEqual(features[key].shape[1], self.data.shape[1])
                self.assertEqual(features[key].shape[2], 16)
                self.assertTrue(np.all(np.isfinite(features[key])))
                self.assertTrue(np.all(np.isreal(features[key])))

    def test_peaks(self):
        data = np.zeros((29, 32, 4, 7), dtype=np.uint16)
        x = np.linspace(-0.81, 1.00465, 4, dtype=np.float32)
        y = np.linspace(-0.6, 1.1, 7, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")

        data[..., 0, 0] = 1
        data[..., 0, 1] = 2
        data[..., 3, 3] = 4

        features = darling.properties.peaks(
            data,
            k=3,
            coordinates=np.array([X, Y]),
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

        np.testing.assert_allclose(
            features["variance_axis_0"][..., 0], 0, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["variance_axis_0"][..., 1], 0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["variance_axis_1"][..., 0], 0, atol=atol, rtol=rtol
        )

        mean_motor2 = (y[0] + 2 * y[1]) / 3
        var_motor2 = ((mean_motor2 - y[0]) ** 2 + 2 * (mean_motor2 - y[1]) ** 2) / 2
        np.testing.assert_allclose(
            features["variance_axis_1"][..., 1], var_motor2, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["mean_axis_0"][..., 1], x[0], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_axis_0"][..., 0], x[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["mean_axis_1"][..., 1], (y[0] + 2 * y[1]) / 3, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_axis_1"][..., 0], y[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["variance_axis_0_axis_1"][..., 1], 0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["variance_axis_0_axis_1"][..., 0], 0, atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["maxima_axis_0"][..., 1], x[0], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["maxima_axis_0"][..., 0], x[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["maxima_axis_1"][..., 1], y[1], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["maxima_axis_1"][..., 0], y[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["max_intensity"][..., 1], 2, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["max_intensity"][..., 0], 4, atol=atol, rtol=rtol
        )

    def test_peaks_rocking_scan(self):
        data = np.zeros((29, 32, 9), dtype=np.uint16)
        x = np.linspace(-0.81, 1.00465, 9, dtype=np.float32)

        data[..., 0] = 1
        data[..., 1] = 2
        data[..., 3] = 4

        features = darling.properties.peaks(
            data,
            k=3,
            coordinates=x.reshape(1, -1),
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

        np.testing.assert_allclose(
            features["variance_axis_0"][..., 0], 0, atol=atol, rtol=rtol
        )

        mean_motor1 = (x[0] + 2 * x[1]) / 3
        var_motor1 = ((mean_motor1 - x[0]) ** 2 + 2 * (mean_motor1 - x[1]) ** 2) / 2
        np.testing.assert_allclose(
            features["variance_axis_0"][..., 1], var_motor1, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["mean_axis_0"][..., 1],
            (x[0] * 1 + 2 * x[1]) / (2 + 1),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            features["mean_axis_0"][..., 0], x[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["maxima_axis_0"][..., 1], x[1], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["maxima_axis_0"][..., 0], x[3], atol=atol, rtol=rtol
        )

        np.testing.assert_allclose(
            features["max_intensity"][..., 1], 2, atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            features["max_intensity"][..., 0], 4, atol=atol, rtol=rtol
        )
        for key in features.keys():
            self.assertTrue("_col" not in key)
            self.assertTrue("_motor2" not in key)


if __name__ == "__main__":
    unittest.main()
