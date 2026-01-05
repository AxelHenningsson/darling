import unittest

import numpy as np

import darling


class TestRGB(unittest.TestCase):
    def setUp(self):
        self.debug = False
        _, self.data, self.coordinates = darling.io.assets.domains()

    def test_rgb_full(self):
        mu = darling.properties.mean(self.data, self.coordinates)
        rgb_map, colorkey, colorgrid = darling.transforms.rgb(
            mu, norm="full", coordinates=self.coordinates
        )
        self.assertEqual(rgb_map.shape[0], mu.shape[0])
        self.assertEqual(rgb_map.shape[1], mu.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid

        self.assertEqual(colorkey.shape[0], X.shape[0])
        self.assertEqual(colorkey.shape[1], X.shape[1])

    def test_rgb_dynamic(self):
        mu = darling.properties.mean(self.data, self.coordinates)
        rgb_map, colorkey, colorgrid = darling.transforms.rgb(
            mu, norm="dynamic", coordinates=None
        )
        self.assertEqual(rgb_map.shape[0], mu.shape[0])
        self.assertEqual(rgb_map.shape[1], mu.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid

        self.assertEqual(colorkey.shape[0], X.shape[0])
        self.assertEqual(colorkey.shape[1], X.shape[1])

    def test_rgb_norm(self):
        mu = darling.properties.mean(self.data, self.coordinates)
        print(mu[..., 0].max(), mu[..., 0].min())
        print(mu[..., 1].max(), mu[..., 1].min())
        norm = np.array([[-0.39293402, 1.1919645], [7.192096, 8.859955]])
        rgb_map, colorkey, colorgrid = darling.transforms.rgb(
            mu, norm=norm, coordinates=None
        )
        self.assertEqual(rgb_map.shape[0], mu.shape[0])
        self.assertEqual(rgb_map.shape[1], mu.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid

        self.assertEqual(colorkey.shape[0], X.shape[0])
        self.assertEqual(colorkey.shape[1], X.shape[1])

    def test_rgb(self):
        mu, _ = darling.properties.moments(self.data, self.coordinates)
        rgb_map, colorkey, colorgrid = darling.transforms.rgb(
            mu, norm="dynamic", coordinates=None
        )

        self.assertEqual(rgb_map.shape[0], self.data.shape[0])
        self.assertEqual(rgb_map.shape[1], self.data.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid
        self.assertEqual(X.shape[0], colorkey.shape[0])
        self.assertEqual(X.shape[1], colorkey.shape[1])
        self.assertEqual(Y.shape[0], colorkey.shape[0])
        self.assertEqual(Y.shape[1], colorkey.shape[1])

        if self.debug:
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title("RGB map dynamic")
            ax.imshow(rgb_map)
            plt.tight_layout()

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.pcolormesh(X, Y, colorkey)
            ax.set_title("RGB map dynamic")
            plt.tight_layout()
            plt.show()

        rgb_map, colorkey, colorgrid = darling.transforms.rgb(
            mu, norm="full", coordinates=self.coordinates
        )

        self.assertEqual(rgb_map.shape[0], self.data.shape[0])
        self.assertEqual(rgb_map.shape[1], self.data.shape[1])
        self.assertEqual(rgb_map.shape[2], 3)

        X, Y = colorgrid
        self.assertEqual(X.shape[0], colorkey.shape[0])
        self.assertEqual(X.shape[1], colorkey.shape[1])
        self.assertEqual(Y.shape[0], colorkey.shape[0])
        self.assertEqual(Y.shape[1], colorkey.shape[1])

        if self.debug:
            import matplotlib.pyplot as plt

            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title("RGB map full")
            ax.imshow(rgb_map)
            plt.tight_layout()

            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.pcolormesh(X, Y, colorkey)
            ax.set_title("RGB map full")
            plt.tight_layout()
            plt.show()

        # check that we can handle nan values
        mu[0:10, 0:10, 0] = np.nan
        rgb_map, colorkey, colorgrid = darling.transforms.rgb(
            mu, norm="full", coordinates=self.coordinates
        )

        if self.debug:
            plt.style.use("dark_background")
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.set_title("Handle nans")
            ax.imshow(rgb_map)
            plt.tight_layout()


if __name__ == "__main__":
    unittest.main()
