import unittest

import numpy as np

import darling


class TestMetadata(unittest.TestCase):
    # Tests for the darling.io.metadata module.

    def setUp(self):
        self.debug = False

    def test_invariant_motors_energy_mu_scan(self):
        path, _, _ = darling.io.assets.energy_mu_scan()
        dset = darling.DataSet(path, scan_id="1.1")

        expected_result = {
            "scan_command": "fscan2d ccmth 5.86565 0.001 31 mu 9.228 0.0015 20 0.8 0.880509",
            "scan_shape": np.array([31, 20]),
            "motor_names": ["instrument/positioners/ccmth", "instrument/mu/data"],
            "integrated_motors": [False, True],
            "data_name": "measurement/pco_ff",
            "scan_id": "1.1",
            "invariant_motors": {
                "s8vg": 1.0,
                "s8vo": 0.0,
                "s8ho": 0.0,
                "s8hg": 1.0,
                "chi": 0.39100917566042864,
                "phi": 0.16254463935658822,
                "mu": 9.242999047010525,
                "omega": 0.0,
                "ux": 1.23596,
                "uy": 1.58931,
                "uz": 0.7069999999999999,
                "mainx": -5000.0,
                "obx": 347.5000000000001,
                "oby": 0.28700000000003456,
                "obz": 155.32799999999952,
                "obz3": -38.623999999999995,
                "obpitch": 18.5833,
                "obyaw": 0.34343749999999984,
                "sovg": 0.04999999999999993,
                "sovo": 2.53323786858134,
                "soho": -1.176576176838369,
                "sohg": 5.0,
                "cdx": -900.8,
                "dcx": 527.6018125,
                "dcz": -145.60500000000002,
                "ffz": 1675.040099009901,
                "ffy": 0.0,
                "ffsel": -60.0,
                "x_pixel_size": 6.5,
                "y_pixel_size": 6.5,
            },
        }
        scan_params = dset.reader.scan_params
        sensors = dset.reader.sensors

        for key in expected_result["invariant_motors"]:
            self.assertTrue(
                key in scan_params["invariant_motors"],
                msg=f"{key} not in {scan_params['invariant_motors']}",
            )
            self.assertTrue(
                np.isclose(
                    scan_params["invariant_motors"][key],
                    expected_result["invariant_motors"][key],
                ),
                msg=f"{key} not close to {expected_result['invariant_motors'][key]}",
            )

        self.assertTrue(sensors["pico4"].size == 1117)
        keys = ["pico4", "pico3", "current", "elapsed_time"]
        for key in keys:
            self.assertTrue(key in sensors, msg=f"{key} not in {sensors}")


if __name__ == "__main__":
    unittest.main()
