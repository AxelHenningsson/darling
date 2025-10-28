import h5py
import numpy as np

import darling


class DataSet(object):
    """A DFXM data-set.

    This is the master data class of darling. Given a data source the DataSet class will read data from
    arbitrary layers, process, threshold, compute moments, visualize results, and compile 3D feature maps.

    Args:
        data_source (:obj: `string` or `darling.reader.Reader`): A string to the absolute h5 file path
            location of the data, or a reader object implementing the darling.reader.Reader() interface.

    Attributes:
        reader (:obj: `darling.reader.Reader`): A file reader implementing, at least, the functionallity
            specified in darling.reader.Reader().
        data (:obj: `numpy.ndarray`): The data array of shape (a,b,m,n,(o)) where a,b are the detector
            dimensions and m,n,(o) are the motor dimensions.
        motors (:obj: `numpy.ndarray`): The motor grids of shape (k, m,n,(o)) where k is the number of
            motors and m,n,(o) are the motor dimensions.
        h5file (:obj: `string`): The absolute path to the h5 file in which all data resides.

    """

    def __init__(self, data_source, scan_id=None):
        if isinstance(data_source, darling.reader.Reader):
            self.reader = data_source
            self.h5file = self.reader.abs_path_to_h5_file
        elif isinstance(data_source, str):
            self.reader = None
            self.h5file = data_source
        else:
            raise ValueError(
                "reader should be a darling.reader.Reader or a string to the h5 file."
            )

        self.data = None
        self.motors = None

        if scan_id is not None:
            self.load_scan(scan_id, roi=None)

    @property
    def dtype(self):
        return self.data.dtype

    def info(self):
        if self.data is not None:
            for k in self.reader.scan_params:
                print(f"{k:<20} :  {str(self.reader.scan_params[k]):<30}")
        else:
            print("No data loaded, use load_scan() to load data.")

    @property
    def scan_params(self):
        """The scan parameters for the loaded data in a dictionary.

        Example output:

        .. code-block:: python

            out = {
                'scan_command': 'fscan2d chi -0.5 0.08 26 diffry 7 0.06 37 0.5 0.500417',
                'scan_shape': [26, 37],
                'motor_names': ['instrument/chi/value', 'instrument/diffry/data'],
                'integrated_motors': [False, True],
                'data_name': 'instrument/pco_ff/image',
                'scan_id': '1.1',
                'invariant_motors': {
                    'ccmth': 6.679671220242654,
                    's8vg': 0.09999999999999998,
                    's8vo': 0.0,
                    's8ho': 0.0,
                    's8hg': 0.5,
                    'phi': 0.0,
                    'mainx': -5000.0,
                    'obx': 263.1299999999999,
                    'oby': 0.0,
                    'obz': 85.35999999999876,
                    'obz3': 0.0693999999999999,
                    'obpitch': 17.979400000000002,
                    'obyaw': -0.06589062499999998,
                    'cdx': -11.8,
                    'dcx': 545.0,
                    'dcz': -150.0,
                    'ffz': 1621.611386138614,
                    'ffy': 0.0,
                    'ffsel': -60.0,
                    'x_pixel_size': 6.5,
                    'y_pixel_size': 6.5
                }
            }


        Returns:
            :obj:`dict`: The scan parameters.

        """
        if self.reader is None:
            raise ValueError("No data has been loaded, use load_scan() to load data.")
        else:
            return self.reader.scan_params

    @property
    def sensors(self):
        """The sensor data for the loaded data in a dictionary.

        this contains things like the pico4 current data that monitors the direct beam current.

        Returns:
            :obj:`dict`: The sensor data.
        """
        if self.reader is None:
            raise ValueError("No data has been loaded, use load_scan() to load data.")
        else:
            return self.reader.sensors

    def load_scan(self, scan_id, scan_motor=None, roi=None):
        """Load a scan into RAM.

        Args:
            scan_id (:obj:`str` or :obj:`list` or :obj:`str`): scan id or scan ids to load.
            scan_motor (:obj:`str`): path in h5file to the motor that is changing with the scan_id.
                Defaults to None. Must be set when scan_id is not a single string.
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded. The roi refers to the detector
                dimensions.

        """
        if not (isinstance(scan_id, list) or isinstance(scan_id, str)):
            raise ValueError(
                "When scan_id must be a list of strings or a single string"
            )
        if isinstance(scan_id, list) and not isinstance(scan_motor, str):
            raise ValueError(
                "When scan_id is a list of keys scan_motor path must be set."
            )
        if isinstance(scan_id, list) and len(scan_id) == 1:
            raise ValueError(
                "When scan_id is a list of keys len(scan_id) must be > than 1."
            )

        if self.reader is None:
            config = darling.metadata.ID03(self.h5file)
            reference_scan_id = scan_id[0] if isinstance(scan_id, list) else scan_id
            scan_params, sensors = config(reference_scan_id)
            if scan_params["motor_names"] is None:
                self.reader = darling.reader.Darks(self.h5file)
            elif len(scan_params["motor_names"]) == 1:
                self.reader = darling.reader.RockingScan(self.h5file)
            elif len(scan_params["motor_names"]) == 2:
                self.reader = darling.reader.MosaScan(self.h5file)
            else:
                raise ValueError("Could not find a reader for your h5 file")

        number_of_scans = len(scan_id) if isinstance(scan_id, list) else 1

        if number_of_scans == 1:
            self.data, self.motors = self.reader(scan_id, roi)
        else:
            scan_motor_values = np.zeros((len(scan_id),))
            with h5py.File(self.h5file) as h5file:
                for i, sid in enumerate(scan_id):
                    scan_motor_values[i] = h5file[sid][scan_motor][()]

            reference_data_block, reference_motors = self.reader(scan_id[0], roi)

            if reference_motors.ndim == 2:
                motor1 = reference_motors[0, :]
                motor2 = scan_motor_values
                motors = np.array(np.meshgrid(motor1, motor2, indexing="ij"))
            elif reference_motors.ndim == 3:
                motor1 = reference_motors[0, :, 0]
                motor2 = reference_motors[1, 0, :]
                motor3 = scan_motor_values
                motors = np.array(np.meshgrid(motor1, motor2, motor3, indexing="ij"))
            else:
                raise ValueError(
                    f"Each scan_id must hold a 1D or 2D scan but {reference_motors.ndim}D was found at scan_id={scan_id[0]}"
                )

            data = np.zeros(
                (*reference_data_block.data.shape, number_of_scans), np.uint16
            )
            data[..., 0] = reference_data_block[...]
            for i, sid in enumerate(scan_id[1:]):
                data_block, _ = self.reader(sid, roi)
                data[..., i + 1] = data_block[...]

            self.reader.scan_params["motor_names"].append(scan_motor)
            self.reader.scan_params["scan_shape"] = np.array(
                [*self.reader.scan_params["scan_shape"], number_of_scans]
            )
            self.reader.scan_params["integrated_motors"].append(False)
            self.reader.scan_params["scan_id"] = scan_id

            self.motors = motors
            self.data = data

    def subtract(self, background, dtype=np.uint16):
        """Subtract a fixed integer value from the data block.

        Warning: If dtype is not np.uint16, data will be cast to dtype, which will create a
            temporary copy of the data.

        Args:
            background (:obj:`int` or :obj:`numpy.ndarray`): fixed background value or background array to subtract.
                the array will be broadcast to the data block shape. I.e for a shape=(a,b,m,n) data block, the background
                could be of detector dimension shape=(a,b) or shape=(a,b,1,1).
            dtype (:obj:`numpy.dtype`): the data type of the output array. Defaults to np.uint16.
                in which case the data is clipped to the range [0, 2^16-1]. protects against uint16
                sign flips.

        """
        if isinstance(background, int):
            bg = np.full(self.data.shape[:2], background, dtype=dtype)
        elif isinstance(background, np.ndarray):
            if background.squeeze().shape != self.data.shape[:2]:
                raise ValueError(
                    f"First two dimensions of background shape must match detector dimension shape, but {background.squeeze().shape} != {self.data.shape[:2]}"
                )
            bg = background.copy().squeeze().astype(dtype)

        bg = bg[(...,) + (None,) * (self.data.ndim - bg.ndim)]

        if self.dtype != dtype:
            self.data = self.data.astype(dtype)

        if np.issubdtype(self.dtype, np.unsignedinteger):
            self.data.clip(bg, None, out=self.data)

        self.data -= bg


if __name__ == "__main__":
    pass
