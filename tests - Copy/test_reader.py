import unittest

import numpy as np

import darling


class TestMosaScan(unittest.TestCase):
    # Tests for the darling.MosaScan class.

    def setUp(self):
        self.debug = False
        self.path_to_data, _, _ = darling.assets.mosaicity_scan()

    def test_init(self):
        # Assert that the reader can be instantiated.
        reader = darling.reader.MosaScan(self.path_to_data)
        self.assertTrue(isinstance(reader, darling.reader.MosaScan))

    def test_read(self):
        # assert that the reader will read data and coordinates of the correct type and
        # expected shapes.
        reader = darling.reader.MosaScan(self.path_to_data)

        data, motors = reader(
            scan_id="1.1",
        )

        self.check_data(data, motors)

    def test_roi_read(self):
        # assert that the reader will read a roi
        reader = darling.reader.MosaScan(
            self.path_to_data,
        )

        data, motors = reader(scan_id="1.1", roi=(10, 20, 0, 7))

        self.check_data(data, motors)

        self.assertTrue(data.shape[0] == 10)
        self.assertTrue(data.shape[1] == 7)

    def test_drift(self):
        path_to_data, data, motors = darling.assets.motor_drift()
        self.check_data(data, motors)

    def test_scan_id(self):
        # ensure that another scan id can be read
        # assert that the reader will read a roi
        reader = darling.reader.MosaScan(self.path_to_data)

        data, motors = reader(scan_id="2.1")

        data_layer_2 = data.copy()

        self.check_data(data_layer_2, motors)

        data, motors = reader(scan_id="1.1")

        data_layer_1 = data.copy()

        # ensure the data shape is consistent between layers
        self.assertEqual(data_layer_1.shape, data_layer_2.shape)

        # ensure the data is actually different between layers.
        residual = data_layer_1 - data_layer_2
        self.assertNotEqual(np.max(np.abs(residual)), 0)

    def check_data(self, data, motors):
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 4)
        self.assertTrue(len(motors.shape) == 3)
        self.assertTrue(data.shape[2] == motors.shape[1])
        self.assertTrue(data.shape[3] == motors.shape[2])
        self.assertTrue(motors.dtype == np.float32)


class TestRockingScan(unittest.TestCase):
    # Tests for the darling.RockingScan class.

    def setUp(self):
        self.debug = False
        self.path_to_data, _, _ = darling.assets.rocking_scan()

    def test_init(self):
        # Assert that the reader can be instantiated.
        reader = darling.reader.RockingScan(self.path_to_data)
        self.assertTrue(isinstance(reader, darling.reader.RockingScan))

    def test_read(self):
        # assert that the reader will read data and coordinates of the correct type and
        # expected shapes.
        reader = darling.reader.RockingScan(self.path_to_data)

        data, motors = reader(
            scan_id="1.1",
        )

        self.check_data(data, motors)

    def test_roi_read(self):
        # assert that the reader will read a roi
        reader = darling.reader.RockingScan(
            self.path_to_data,
        )

        data, motors = reader(scan_id="1.1", roi=(10, 20, 0, 7))

        self.check_data(data, motors)

        self.assertTrue(data.shape[0] == 10)
        self.assertTrue(data.shape[1] == 7)

    def check_data(self, data, motors):
        self.assertTrue(data.dtype == np.uint16)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(len(motors.shape) == 2)
        self.assertTrue(data.shape[2] == motors.shape[1])
        self.assertTrue(motors.dtype == np.float32)


if __name__ == "__main__":
    unittest.main()
