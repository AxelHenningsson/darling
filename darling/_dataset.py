import time

import h5py
import meshio
import numpy as np
import scipy.ndimage

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

        self.mean, self.covariance = None, None
        self.mean_3d, self.covariance_3d = None, None
        self.kam = None
        self.kam_kernel_size = None

        self.data = None
        self.motors = None

        if scan_id is not None:
            self.load_scan(scan_id, roi=None)

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
            scan_params = config(reference_scan_id)
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
            print(scan_params)
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

    def subtract(self, value):
        """Subtract a fixed integer value form the data. Protects against uint16 sign flips.

        Args:
            value (:obj:`int`): value to subtract.

        """
        self.data.clip(value, None, out=self.data)
        self.data -= value

    def estimate_background(self):
        """Automatic background correction based on image statistics.

        a set of sample data is extracted from the data block. The median and standard deviations are iteratively
        fitted, rejecting outliers (which here is diffraction signal). Once the noise distirbution has been established
        the value corresponding to the 99.99% percentile is returned. I.e the far tail of the noise is returned.

        """
        sample_size = 40000
        index = np.random.permutation(sample_size)
        sample = self.data.flat[index]
        sample = np.sort(sample)
        noise = sample.copy()
        for i in range(20):
            mu = np.median(noise)
            std = np.std(noise)
            noise = noise[np.abs(noise) < mu + 2 * 3.891 * std]  # 99.99% confidence
        background = np.max(noise)
        return background

    def moments(self):
        """Compute first and second moments.

        The internal attributes self.mean and self.covariance are set when this function is run.

        Returns:
            (:obj:`tupe` of :obj:`numpy array`): mean and covariance maps of shapes (a,b,2) and (a,b,2,2)
                respectively with a=self.data.shape[0] and b=self.data.shape[1].
        """
        self.mean, self.covariance = darling.properties.moments(self.data, self.motors)
        return self.mean, self.covariance

    def kernel_average_misorientation(self, size=(5, 5)):
        """Compute the KAM (Kernel Average Misorientation) map.

        KAM is compute by sliding a kernel across the image and for each voxel computing
        the average misorientation between the central voxel and the surrounding voxels.

        NOTE: This is a projected KAM in the sense that the rotation the full rotation
        matrix of the voxels are unknown. I.e this is a computation of the misorientation
        between diffraction vectors Q and not orientation elements of SO(3).

        Args:
            size (:obj:`tuple`): The size of the kernel to use for the KAM computation.
                Defaults to (3, 3).

        Returns:
            :obj:`numpy array` : The KAM map of shape=(a, b). (same units as input.)
        """
        self.kam = darling.properties.kam(self.mean, size)
        self.kam_kernel_size = size
        return self.kam

    def integrate(self, axis=None, dtype=np.float32):
        """Return the summed data stack along the specified axes, avoiding data stack copying.

        If no axis is specified, the integration is performed over all dimensions
        except the first two, which are assumed to be the detector dimensions.

        Args:
            axis (:obj:`int` or :obj:`tuple`, optional): The axis or axes along which to integrate.
                If None, integrates over all axes except the first two.
            dtype (:obj:`numpy.dtype`, optional): The data type of the output array.
                Defaults to np.float32.

        Returns:
            :obj:`numpy.ndarray`: Integrated frames, a 2D numpy array of reduced
                shape and dtype `dtype`.
        """
        if axis is None:
            out = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=dtype)
            integrated_frames = np.sum(
                self.data, axis=tuple(range(2, self.data.ndim)), out=out
            )
        else:
            shape = list(self.data.shape)
            for ax in sorted(np.atleast_1d(axis), reverse=True):
                shape.pop(ax)
            out = np.zeros(shape, dtype=dtype)
            integrated_frames = np.sum(self.data, axis=axis, out=out)
        return integrated_frames

    def estimate_mask(
        self,
        threshold=200,
        erosion_iterations=3,
        dilation_iterations=25,
        fill_holes=True,
    ):
        """Segment the sample diffracting region based on summed intensity along motor dimensions.

        Args:
            threshold (:obj:`int`):  a summed count value above which the sample is defined.
            erosion_iterations (:obj:`int`): Number of times to erode the mask using a 2,2 structure.
            dilation_iterations (:obj:`int`): Number of times to dilate the mask using a 2,2 structure.
            fill_holes (:obj:`bool`):  Fill enclosed holes in the final mask.

        Returns:
            (:obj:`numpy array`): Returns: a binary 2D mask of the sample.

        """
        mask = self.integrate() > threshold
        mask = scipy.ndimage.binary_erosion(
            mask, structure=np.ones((2, 2)), iterations=erosion_iterations
        )
        mask = scipy.ndimage.binary_dilation(
            mask, structure=np.ones((2, 2)), iterations=dilation_iterations
        )
        if fill_holes:
            mask = scipy.ndimage.binary_fill_holes(mask)
        return mask

    def compile_layers(self, scan_ids, threshold=None, roi=None, verbose=False):
        """Sequentially load a series of scans and assemble the 3D moment maps.

        this loads the mosa data array with shape a,b,m,n,(o) where a,b are the detector dimension and
        m,n,(o) are the motor dimensions as ordered in the `self.motor_names`.

        NOTE: This function will load data sequentially and compute moments on the fly. While all
        moment maps are stored and concatenated, only one scan (the raw 4d or 5d data) is kept in
        memory at a time to enhance RAM performance.

        Args:
            scan_ids (:obj:`str`): scan ids to load, e.g 1.1, 2.1 etc...
            threshold (:obj:`int` or :obj:`str`): background subtraction value or string 'auto' in which
                case a default background estimation is performed and subtracted.
                Defaults to None, in which case no background is subtracted.
            roi (:obj:`tuple` or :obj:`int`): row_min row_max and column_min and column_max, defaults to None,
                in which case all data is loaded
            verbose (:obj:`bool`): Print loading progress or not.

        """
        mean_3d = []
        covariance_3d = []
        tot_time = 0
        for i, scan_id in enumerate(scan_ids):
            t1 = time.perf_counter()

            if verbose:
                print(
                    "\nREADING SCAN: "
                    + str(i + 1)
                    + " out of totally "
                    + str(len(scan_ids))
                    + " scans"
                )
            self.load_scan(scan_id, roi)

            if threshold is not None:
                if threshold == "auto":
                    if verbose:
                        print(
                            "    Subtracting estimated background for scan id "
                            + str(scan_id)
                            + " ..."
                        )
                    _threshold = self.estimate_background()
                    self.threshold(_threshold)
                else:
                    if verbose:
                        print(
                            "    Subtracting fixed background value="
                            + str(threshold)
                            + " for scan id "
                            + str(scan_id)
                            + " ..."
                        )
                    self.threshold(threshold)

            if verbose:
                print("    Computing moments for scan id " + str(scan_id) + " ...")

            mean, covariance = self.moments()

            if verbose:
                print(
                    "    Concatenating to 3D volume for scan id "
                    + str(scan_id)
                    + " ..."
                )
            mean_3d.append(mean)
            covariance_3d.append(covariance)

            t2 = time.perf_counter()
            tot_time += t2 - t1

            estimated_time_left = str((tot_time / (i + 1)) * (len(scan_ids) - i - 1))
            if verbose:
                print("    Estimated time left is : " + estimated_time_left + " s")

        self.mean_3d = np.array(mean_3d)
        self.covariance_3d = np.array(covariance_3d)

        if verbose:
            print("\ndone! Total time was : " + str(tot_time) + " s")

        return self.mean_3d, self.covariance_3d

    def to_paraview(self, file):
        """Write moment maps to paraview readable format for 3D visualisation.

        The written data array will have attributes as:

            cov_11, cov_12, (cov_13), cov_22, (cov_23, cov_33) : Elements of covariance matrix.
            mean_1, mean_2, (mean_3) : The first moments in each dimension.

        Here 1 signifies the self.motors[0] dimension while 2 is in self.motors[2], (and
        3 in self.motors[3], when the scan is 3D)

        NOTE: Requires that 3D moment maps have been compiled via compile_layers().

        Args:
            file (:obj:`string`): Absolute path ending with desired filename.

        """

        dim = np.array(self.mean_3d.shape)[-1]
        s, a, b = np.array(self.mean_3d.shape)[0:3]
        sg = np.linspace(0, s, s)
        ag = np.linspace(0, a, a)
        bg = np.linspace(0, b, b)
        mesh = np.meshgrid(sg, ag, bg, indexing="ij")
        points = np.array([x.flatten() for x in mesh])
        N = points.shape[1]
        cells = [("vertex", np.array([[i] for i in range(N)]))]

        if len(file.split(".")) == 1:
            filename = file + ".xdmf"
        else:
            filename = file

        point_data = {}
        for i in range(dim):
            point_data["mean_" + str(i + 1)] = self.mean_3d[:, :, :, i].flatten()
            for j in range(i, dim):
                point_data["cov_" + str(i + 1) + str(j + 1)] = self.covariance_3d[
                    :, :, :, i, j
                ].flatten()

        meshio.Mesh(
            points.T,
            cells,
            point_data=point_data,
        ).write(filename)


if __name__ == "__main__":
    pass
