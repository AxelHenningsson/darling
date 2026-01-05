import unittest

import numpy as np

import darling


class TestPeakMap(unittest.TestCase):
    def setUp(self):
        self.debug = False
        _, self.data, self.motors = darling.io.assets.domains()
        self.peakmap = darling.properties.peaks(self.data, k=4, coordinates=self.motors)
        self.expected_keys = darling.properties._peaks._FEATURE_MAPPING[2]

    def test_instantiation(self):
        self.assertEqual(
            self.peakmap.shape, (self.data.shape[0], self.data.shape[1], 4)
        )
        self.assertEqual(self.peakmap.ndim, len(self.motors))
        self.assertEqual(self.peakmap.k, 4)
        self.assertEqual(self.peakmap.sorting, "sum_intensity")

        # test that attribute access works
        sum_intensity_1 = self.peakmap.sum_intensity[..., 0]
        sum_intensity_2 = self.peakmap["sum_intensity"][..., 0]
        sum_intensity_3 = self.peakmap.feature_table["sum_intensity"][..., 0]
        np.testing.assert_allclose(sum_intensity_1, sum_intensity_2)
        np.testing.assert_allclose(sum_intensity_1, sum_intensity_3)

        number_of_pixels_1 = self.peakmap.number_of_pixels[..., 1]
        number_of_pixels_2 = self.peakmap["number_of_pixels"][..., 1]
        number_of_pixels_3 = self.peakmap.feature_table["number_of_pixels"][..., 1]
        np.testing.assert_allclose(number_of_pixels_1, number_of_pixels_2)
        np.testing.assert_allclose(number_of_pixels_1, number_of_pixels_3)

        self.assertTrue(
            self.expected_keys.keys() == self.peakmap.keys(),
            msg="feature table keys do not match",
        )

    def test_sorting(self):
        # check that the array is sorted by sum_intensity in descending order
        expected = np.sort(self.peakmap.sum_intensity, axis=-1)[..., ::-1]
        np.testing.assert_allclose(self.peakmap.sum_intensity, expected)
        pixel_before_sorting_si = self.peakmap.sum_intensity[20, 20, :]
        pixel_before_sorting_n_pix = self.peakmap.number_of_pixels[20, 20, :]
        pixel_before_sorting_maxima_axis_0 = self.peakmap.maxima_axis_0[20, 20, :]
        self.peakmap.sort(key="number_of_pixels")

        for j in range(self.peakmap.k - 1):
            self.assertTrue(
                np.all(
                    self.peakmap.number_of_pixels[:, :, j]
                    >= self.peakmap.number_of_pixels[:, :, j + 1]
                )
            )

        expected = np.sort(self.peakmap.number_of_pixels, axis=-1)[..., ::-1]
        np.testing.assert_allclose(self.peakmap.number_of_pixels, expected)
        pixel_after_sorting_si = self.peakmap.sum_intensity[20, 20, :]
        pixel_after_sorting_n_pix = self.peakmap.number_of_pixels[20, 20, :]
        pixel_after_sorting_maxima_axis_0 = self.peakmap.maxima_axis_0[20, 20, :]

        indices = np.argsort(pixel_before_sorting_n_pix, kind="quicksort")[::-1]

        for i, index in enumerate(indices):
            self.assertEqual(pixel_before_sorting_si[index], pixel_after_sorting_si[i])
            self.assertEqual(
                pixel_before_sorting_n_pix[index], pixel_after_sorting_n_pix[i]
            )
            self.assertEqual(
                pixel_before_sorting_maxima_axis_0[index],
                pixel_after_sorting_maxima_axis_0[i],
            )

    def test_variance(self):
        variance = self.peakmap.variance
        self.assertEqual(variance.shape, (self.data.shape[0], self.data.shape[1], 2, 2))
        np.testing.assert_allclose(variance[..., 0, 1], variance[..., 1, 0])

        nnz = np.count_nonzero(variance[..., 0, 1])
        self.assertTrue(nnz > 0)

        nnz = np.count_nonzero(variance[..., 0, 0])
        self.assertTrue(nnz > 0)

        nnz = np.count_nonzero(variance[..., 1, 1])
        self.assertTrue(nnz > 0)

        np.testing.assert_allclose(
            variance[..., 0, 1], self.peakmap.variance_axis_0_axis_1[..., 0]
        )
        np.testing.assert_allclose(
            variance[..., 1, 0], self.peakmap.variance_axis_0_axis_1[..., 0]
        )
        np.testing.assert_allclose(
            variance[..., 0, 0], self.peakmap.variance_axis_0[..., 0]
        )
        np.testing.assert_allclose(
            variance[..., 1, 1], self.peakmap.variance_axis_1[..., 0]
        )

        var2 = self.peakmap.get_variance(k=1)
        self.assertEqual(var2.shape, (self.data.shape[0], self.data.shape[1], 2, 2))
        np.testing.assert_allclose(var2[..., 0, 1], var2[..., 1, 0])
        np.testing.assert_allclose(
            var2[..., 0, 0], self.peakmap.variance_axis_0[..., 1]
        )
        np.testing.assert_allclose(
            var2[..., 1, 1], self.peakmap.variance_axis_1[..., 1]
        )

    def test_max(self):
        max = self.peakmap.max
        self.assertEqual(max.shape, (self.data.shape[0], self.data.shape[1], 2))
        np.testing.assert_allclose(max[..., 0], self.peakmap.maxima_axis_0[..., 0])
        np.testing.assert_allclose(max[..., 1], self.peakmap.maxima_axis_1[..., 0])

        max2 = self.peakmap.get_max(k=1)
        self.assertEqual(max2.shape, (self.data.shape[0], self.data.shape[1], 2))
        np.testing.assert_allclose(max2[..., 0], self.peakmap.maxima_axis_0[..., 1])
        np.testing.assert_allclose(max2[..., 1], self.peakmap.maxima_axis_1[..., 1])

    def test_mean(self):
        mean = self.peakmap.mean
        self.assertEqual(mean.shape, (self.data.shape[0], self.data.shape[1], 2))
        np.testing.assert_allclose(mean[..., 0], self.peakmap.mean_axis_0[..., 0])
        np.testing.assert_allclose(mean[..., 1], self.peakmap.mean_axis_1[..., 0])

        mean2 = self.peakmap.get_mean(k=1)
        self.assertEqual(mean2.shape, (self.data.shape[0], self.data.shape[1], 2))
        np.testing.assert_allclose(mean2[..., 0], self.peakmap.mean_axis_0[..., 1])
        np.testing.assert_allclose(mean2[..., 1], self.peakmap.mean_axis_1[..., 1])
