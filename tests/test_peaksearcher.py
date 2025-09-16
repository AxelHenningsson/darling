import unittest

import matplotlib.pyplot as plt
import numpy as np

import darling


class TestGaussianMixture(unittest.TestCase):
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

        labeled_array, n_features = darling.peaksearcher.label_sparse(img)
        features = darling.peaksearcher.extract_features(
            labeled_array, img, k=len(intensities) + 1
        )
        self.assertEqual(features.shape[1], len(intensities) + 1)
        self.assertEqual(features[0, -1], 0)

        exp_intensities = np.sort(intensities)[::-1]
        exp_intensities[0] += 2
        exp_intensities_n_pix = [3, 1, 1, 1, 1]

        for i, ei in enumerate(exp_intensities):
            self.assertTrue(ei == features[0, i])
            self.assertTrue(exp_intensities_n_pix[i] == features[1, i])

    def test_label_sparse_simple(self):
        img = np.zeros((256, 256), dtype=np.float32)
        targetindices = [(0, 0), (100, 100), (150, 150), (200, 200), (255, 255)]
        intensities = [1, 2, 3, 1, 1236]
        for i, (x, y) in enumerate(targetindices):
            img[x, y] = intensities[i]

        img[x - 1, y - 1] = 1
        img[x, y - 1] = 1

        labeled_array, n_features = darling.peaksearcher.label_sparse(img)

        np.testing.assert_equal(n_features, len(intensities))
        np.testing.assert_equal(labeled_array.max(), len(intensities))

        i, j = np.where(labeled_array != 0)

        for x, y in targetindices:
            self.assertTrue(x in i)
            self.assertTrue(y in j)

    def test_label_sparse(self):
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

        labeled_array, n_features = darling.peaksearcher.label_sparse(img)

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
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            im = ax[0].imshow(img)
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(labeled_array, cmap="hsv")
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    unittest.main()
