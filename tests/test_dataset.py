import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import darling


class TestDataSet(unittest.TestCase):
    # Tests for the darling.assets module.

    def setUp(self):
        self.debug = False

        # we test for the mosa scan reader
        path_to_data_1, _, _ = darling.assets.mosaicity_scan()
        self.reader_1 = darling.reader.MosaScan(path_to_data_1)

        # as well as the rocking scan reader
        path_to_data_3, _, _ = darling.assets.rocking_scan()
        self.reader_3 = darling.reader.RockingScan(path_to_data_3)

        self.readers = [self.reader_1, self.reader_3]
        self.scan_ids = [["1.1", "2.1"], ["1.1"]]
        self.checks = [self.check_data_2d, self.check_data_1d]

        self.names = ["mosa", "rocking"]

    def test_init(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)

    def test_load_scan(self):
        for i, reader in enumerate(self.readers):
            dset = darling.DataSet(reader)

            # test that a scan can be loaded.
            dset.load_scan(scan_id="1.1", roi=None)
            self.checks[i](dset)

            # test the tuple args option
            dset.load_scan(scan_id="1.1", roi=None)
            self.checks[i](dset)
            data_layer_1 = dset.data.copy()

            # test to load a diffrent layer
            if "2.1" in self.scan_ids[i]:
                dset.load_scan(scan_id="2.1", roi=None)
                self.checks[i](dset)

                # ensure the data shape is consistent between layers
                self.assertEqual(data_layer_1.shape, dset.data.shape)

                # ensure the data is actually different between layers.
                residual = data_layer_1 - dset.data
                self.assertNotEqual(np.max(np.abs(residual)), 0)

                # test that a roi can be loaded and that the resulting shape is ok.
                dset.load_scan(scan_id="2.1", roi=(0, 9, 3, 19))
                self.checks[i](dset)
                self.assertTrue(dset.data.shape[0] == 9)
                self.assertTrue(dset.data.shape[1] == 16)

    def test_subtract(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            mm = np.max(dset.data)
            dset.subtract(value=200)
            self.assertEqual(np.max(dset.data), mm - 200)

    def test_moments(self):
        for i, reader in enumerate(self.readers):
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            mean, covariance = dset.moments()
            self.assertEqual(mean.shape[0], dset.data.shape[0])
            self.assertEqual(mean.shape[1], dset.data.shape[1])
            self.assertEqual(covariance.shape[0], dset.data.shape[0])
            self.assertEqual(covariance.shape[1], dset.data.shape[1])
            if len(mean.shape) == 3:
                self.assertEqual(mean.shape[2], 2)
                self.assertEqual(covariance.shape[2], 2)
                self.assertEqual(covariance.shape[3], 2)
            if len(mean.shape) == 2:
                self.assertEqual(covariance.shape, mean.shape)

    def test_estimate_mask(self):
        for reader in self.readers:
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            mask = dset.estimate_mask()
            self.assertEqual(mask.shape[0], dset.data.shape[0])
            self.assertEqual(mask.shape[1], dset.data.shape[1])
            self.assertEqual(mask.dtype, bool)

            if self.debug:
                plt.style.use("dark_background")
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                im = ax.imshow(mask)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()

    def test_integrate(self):
        for i, reader in enumerate(self.readers):
            dset = darling.DataSet(reader)
            dset.load_scan(scan_id="1.1", roi=None)
            int_frames = dset.integrate()
            self.assertEqual(int_frames.shape[0], dset.data.shape[0])
            self.assertEqual(int_frames.shape[1], dset.data.shape[1])
            self.assertEqual(int_frames.dtype, np.float32)

            int_frames = dset.integrate(dtype=np.uint16)
            self.assertEqual(int_frames.shape[0], dset.data.shape[0])
            self.assertEqual(int_frames.shape[1], dset.data.shape[1])
            self.assertEqual(int_frames.dtype, np.uint16)

            int_frames = dset.integrate(dtype=np.uint64)
            self.assertEqual(int_frames.shape[0], dset.data.shape[0])
            self.assertEqual(int_frames.shape[1], dset.data.shape[1])
            self.assertEqual(int_frames.dtype, np.uint64)

            int_frames = dset.integrate(axis=len(dset.data.shape) - 1, dtype=np.uint64)
            self.assertEqual(int_frames.dtype, np.uint64)

            self.assertEqual(int_frames.shape[0], dset.data.shape[0])
            self.assertEqual(int_frames.shape[1], dset.data.shape[1])
            if self.names[i] == "rocking":
                self.assertEqual(len(int_frames.shape), 2)

            int_frames = dset.integrate(axis=(0, 1), dtype=np.uint64)
            self.assertEqual(int_frames.dtype, np.uint64)
            self.assertNotEqual(int_frames.shape[0], dset.data.shape[0])
            if self.names[i] == "rocking":
                self.assertEqual(len(int_frames.shape), 1)

            if self.debug:
                int_frames = dset.integrate()
                plt.style.use("dark_background")
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                im = ax.imshow(int_frames)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()

    def test_compile_layers(self):
        # test that the mosa reader will stack layers
        path_to_data, _, _ = darling.assets.mosaicity_scan()
        reader = darling.reader.MosaScan(path_to_data)

        dset_mosa = darling.DataSet(reader)

        mean_3d, cov_3d = dset_mosa.compile_layers(
            scan_ids=["1.1", "2.1"], verbose=False
        )

        self.assertEqual(mean_3d.shape[0], 2)
        self.assertEqual(len(mean_3d.shape), 4)
        self.assertEqual(cov_3d.shape[0], 2)
        self.assertEqual(len(cov_3d.shape), 5)

    def test_as_paraview(self):
        path_to_data, _, _ = darling.assets.mosaicity_scan()
        reader = darling.reader.MosaScan(path_to_data)

        dset_mosa = darling.DataSet(reader)

        mean_3d, cov_3d = dset_mosa.compile_layers(
            scan_ids=["1.1", "2.1"], verbose=False
        )

        filename = os.path.join(darling.assets.path(), "saves", "mosa_stack")
        dset_mosa.to_paraview(filename)

    def check_data_2d(self, dset):
        self.assertTrue(dset.data.dtype == np.uint16)
        self.assertTrue(len(dset.data.shape) == 4)
        self.assertTrue(dset.data.shape[2] == dset.motors.shape[1])
        self.assertTrue(dset.data.shape[3] == dset.motors.shape[2])
        self.assertTrue(dset.motors.dtype == np.float32)

    def check_data_1d(self, dset):
        self.assertTrue(dset.data.dtype == np.uint16)
        self.assertTrue(len(dset.data.shape) == 3)
        self.assertTrue(len(dset.motors.shape) == 2)
        self.assertTrue(dset.data.shape[2] == dset.motors.shape[1])
        self.assertTrue(dset.motors.dtype == np.float32)


if __name__ == "__main__":
    unittest.main()
    unittest.main()
    unittest.main()
