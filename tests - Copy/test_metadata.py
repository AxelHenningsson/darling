import unittest

import numpy as np

import darling


class TestMetadata(unittest.TestCase):
    # Tests for the darling.metadata module.

    def setUp(self):
        self.debug = False

    def test_invariant_motors_rocking_scan(self):
        path, _, _ = darling.assets.rocking_scan()
        dset = darling.DataSet(path, scan_id="1.1")

        expected_result = {
            "ccmth": 6.679671220242654,
            "s8vg": 0.19999999999999996,
            "s8vo": 0.0,
            "s8ho": 0.0,
            "s8hg": 0.5,
            "chi": -1.62008645520433,
            "phi": -9.072763305084665e-07,
            "omega": 5.029150443647268e-06,
            "ux": -0.4200334815385251,
            "uy": 0.23766689950436912,
            "uz": -7.702405530031179,
            "mainx": -5011.7,
            "obx": 262.0000000000001,
            "oby": 0.0,
            "obz": 83.9864000000016,
            "obz3": 0.06380000000000052,
            "obpitch": 18.0036,
            "obyaw": -0.005812499999999776,
            "cdx": -718.0,
            "dcx": 449.9999375,
            "dcz": -100.0,
            "ffz": 1623.1,
            "ffy": 0.0,
            "ffsel": 0.0,
            "x_pixel_size": 6.5,
            "y_pixel_size": 6.5,
        }

        scan_params = dset.reader.scan_params

        for key in expected_result:
            self.assertTrue(
                key in scan_params["invariant_motors"],
                msg=f"{key} not in {scan_params['invariant_motors']}",
            )
            self.assertTrue(
                np.isclose(scan_params["invariant_motors"][key], expected_result[key]),
                msg=f"{key} not close to {expected_result[key]}",
            )


if __name__ == "__main__":
    unittest.main()
    unittest.main()
