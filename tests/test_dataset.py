import unittest

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

    def test_read_from_constructor_with_roi(self):
        path_to_data, _, _ = darling.assets.energy_mu_scan()
        dset = darling.DataSet(path_to_data, scan_id="1.1", roi=(2, 19, 0, 10))
        print(dset.data.shape, dset.motors.shape)
        self.assertEqual(dset.data.shape[0], 17)
        self.assertEqual(dset.data.shape[1], 10)
        self.assertEqual(dset.motors.shape[0], 2)
        self.assertEqual(dset.motors.shape[1], 31)
        self.assertEqual(dset.motors.shape[2], 20)

    def test_read_overloaded_motor_from_constructor_with_suffix(self):
        path_to_data, _, _ = darling.assets.energy_mosa_scan()
        dset = darling.DataSet(
            path_to_data, suffix=".1", scan_motor="instrument/positioners/ccmth"
        )
        self.assertEqual(dset.data.shape[0], 5)
        self.assertEqual(dset.data.shape[1], 5)
        self.assertEqual(dset.motors.ndim, 4)
        self.assertEqual(dset.motors.shape[0], 3)
        self.assertEqual(dset.motors.shape[1], 12)
        self.assertEqual(dset.motors.shape[2], 6)
        self.assertEqual(dset.motors.shape[3], 15)

    def test_suffix_sorting(self):
        path_to_data, _, _ = darling.assets.energy_mosa_scan()
        dset = darling.DataSet(
            path_to_data, suffix=".1", scan_motor="instrument/positioners/ccmth"
        )
        dx = dset.motors[-1][0, 0, :][1] - dset.motors[-1][0, 0, :][0]
        self.assertGreater(dx, 0)
        np.testing.assert_allclose(np.diff(dset.motors[-1][0, 0, :]), dx)

    def test_read_overloaded_motor_from_constructor_with_scan_id_list(self):
        path_to_data, _, _ = darling.assets.energy_mosa_scan()
        scan_id = ["15.1", "16.1", "17.1"]
        dset = darling.DataSet(
            path_to_data, scan_id=scan_id, scan_motor="instrument/positioners/ccmth"
        )
        self.assertEqual(dset.data.shape[0], 5)
        self.assertEqual(dset.data.shape[1], 5)
        self.assertEqual(dset.motors.ndim, 4)
        self.assertEqual(dset.motors.shape[0], 3)
        self.assertEqual(dset.motors.shape[1], 12)
        self.assertEqual(dset.motors.shape[2], 6)
        self.assertEqual(dset.motors.shape[3], len(scan_id))

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
            dset.subtract(200)
            self.assertEqual(np.max(dset.data), mm - 200)

            dset.subtract(np.full(dset.data.shape[:2], 200))
            self.assertEqual(np.max(dset.data), mm - 400)

            dset.subtract(
                np.full(dset.data.shape[:2], np.min(dset.data) + 1), dtype=np.int32
            )
            self.assertTrue(dset.dtype == np.int32)
            self.assertTrue(np.min(dset.data) == -1)

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

    def test_fetch(self):
        path = darling.assets.energy_mu_scan()[0]
        dset = darling.DataSet(path, scan_id="1.1")
        pico4 = dset.reader.fetch(key="1.2/instrument/pico4/data")
        self.assertEqual(pico4.size, 1117)


if __name__ == "__main__":
    unittest.main()
