"""The metadata module is intended to contain easy configuration for the different beamlines/ changes
in beamline data storage formatting.

The ID03 object is used internally by darling to interface the ESRF ID03 beamline data storage.
It contains specific information on how to fetch motor names, scan shapes and other meta-data from
a h5 file.
"""

import h5py
import numpy as np


class ID03(object):
    """The configuration object parses & fetches meta-data on the used motors and
    scan configuration from the h5 file. This one is specific to the ID03 beamline
    at the ESRF following the bliss configuration.

    Args:
        abs_path_to_h5_file (str): The absolute path to the h5 file.
    """

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file

        # These are the places to fetch the motor steps and names from the scan commands
        # for instance argfetch[]["motor_steps"]["ascan"] will return that the argument
        # at the 3rd place in ascan motor(0) start(1) stop(2) steps(3) ... is the number of steps
        # and argfetch[]["motor_names"]["ascan"] will return that the motor name is at the
        # 0th place in ascan motor(0) start(1) stop(2) steps(3)... etc. This is a bliss
        # configuration specific to the macro implementation at ID03.

        self.scan_arg_pos = {
            "motor_steps": {
                "ascan": [3],
                "fscan": [3],
                "a2scan": [6, 6],
                "d2scan": [6, 6],
                "fscan2d": [3, 7],
            },
            "motor_names": {
                "ascan": [0],
                "fscan": [0],
                "a2scan": [0, 3],
                "d2scan": [0, 3],
                "fscan2d": [0, 4],
            },
        }

        # tells us which motors are integrated in the scan command
        self.is_integrated = {
            "ascan": [False],
            "fscan": [True],
            "a2scan": [False, False],
            "d2scan": [False, False],
            "fscan2d": [False, True],
        }

        # These are the expected h5 mappings between the scan command motor names and the
        # corresponding motor values storage location in the h5 file.
        self.motor_map = {
            "chi": "instrument/chi/value",
            "phi": "instrument/phi/value",
            "diffrz": "instrument/diffrz/data",
            "diffry": "instrument/diffry/data",
            "diffrx": "instrument/diffrx/data",
            "ccmth": "instrument/positioners/ccmth",
            "mu": "instrument/positioners/mu",
        }

    def __call__(self, scan_id):
        """Return a dictionary of scan parameters, including scan shape, motor names etc.

        The possible scan_command options for ID03 are

        .. code-block:: bash

            ascan motor start stop intervals ...
            fscan motor start stop steps ...
            a2scan motor1 start1 stop1 motor2 start2 stop2 intervals ...
            d2scan motor1 start1 stop1 motor2 start2 stop2 intervals ...
            fscan2d motor1 start1 stop1 steps1 motor2 start2 stop2 steps2 ...

        The used command should be located under title in the h5 file. This
        command is parsed to extract the scan shape, motor names and other
        meta-data parameters.

        Args:
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...

        Returns:
            :obj:`dict`: scan_params, dictionary of scan parameters;
                scan_params["scan_command"], scan_params["scan_shape"],
                scan_params["motor_names"], scan_params["integrated_motors"],
                scan_params["data_name"]
        """
        scan_params = {}
        scan_params["scan_command"] = self._get_scan_command(scan_id)
        scan_params["scan_shape"] = self._get_scan_shape(scan_params)
        scan_params["motor_names"] = self._get_motor_names(scan_params)
        scan_params["integrated_motors"] = self._get_integrated_motors(scan_params)
        scan_params["data_name"] = self._get_data_name(
            scan_id, scan_params["scan_shape"]
        )
        return scan_params

    def _get_scan_command(self, scan_id):
        """The string representation of the scan command.

        the possible scan_command options are
            ascan motor start stop intervals ...
            fscan motor start stop steps ...
            a2scan motor1 start1 stop1 motor2 start2 stop2 intervals ...
            d2scan motor1 start1 stop1 motor2 start2 stop2 intervals ...
            fscan2d motor1 start1 stop1 steps1 motor2 start2 stop2 steps2 ...

        Args:
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...

        Returns:
            :obj:`str`: The used scan command, e.g fscan2d motor1 start1 stop1 ....
        """
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            scan_command = h5f[scan_id]["title"][()].decode("utf-8")
        return scan_command

    def _get_scan_shape(self, scan_params):
        """Fetch the shape of the scan from the scan command.

        Returns:
            :obj:`tuple` of :obj:`int`: The shape of the scan.
        """
        command = scan_params["scan_command"].split(" ")[0]
        params = np.array(scan_params["scan_command"].split(" ")[1:])
        scan_shape = params[self.scan_arg_pos["motor_steps"][command]].astype(int)
        if command in ["ascan2d", "ascan"]:
            for i in range(len(scan_shape)):
                scan_shape[i] += 1
        return scan_shape

    def _get_motor_names(self, scan_params):
        """Fetch motor names from the scan command.

        Returns:
            :obj:`list` of :obj:`str`: The names of the motors in the scan.
        """
        command = scan_params["scan_command"].split(" ")[0]
        params = scan_params["scan_command"].split(" ")[1:]
        motor_names = [
            self.motor_map[params[i]] for i in self.scan_arg_pos["motor_names"][command]
        ]
        return motor_names

    def _get_integrated_motors(self, scan_params):
        """Mark the motor names that are integrated in the scan command.

        Returns:
            :obj:`list` of :obj:`bool`: A list of booleans marking which motors are integrated.
        """
        command = scan_params["scan_command"].split(" ")[0]
        return self.is_integrated[command]

    def _get_data_name(self, scan_id, scan_shape):
        """Find the h5 key to the stack of DFXM images.

        This is done by iterating all keys and checking for a
        key that is (1) a dataset (2) 3-dimensional, i.e a stack
        of 2D images, and (3) has first dimension that matches the
        shape of the scan, i. if a 10x8 mosa scan was made the dataset
        should have shape[0]==80 etc.

        Args:
            scan_id (:obj:`str`): scan id to load from, e.g 1.1, 2.1 etc...
            scan_shape (:obj:``tuple): The number of scan steps in each motor dimension

        Returns:
            :obj:`str`: h5 key to the stack of DFXM images.
        """
        leafs = []

        def get_leaf(name, obj):
            leafs.append(name)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            h5f[scan_id].visititems(get_leaf)

            while len(leafs) > 0:
                leaf = leafs.pop()
                if (
                    isinstance(h5f[scan_id][leaf], h5py.Dataset)
                    and len(h5f[scan_id][leaf].shape) == 3
                    and h5f[scan_id][leaf].shape[0] == np.prod(scan_shape)
                ):
                    return leaf
        raise ValueError("No dataset found in h5 file")
