import os

import h5py
import numpy as np


def _bin(a, binsize):
    """Bin an array by a given binsize.

    Args:
        a (numpy.ndarray): Array to bin. shape=(a,m,n) where a is the number of frames,
            m is the number of rows, and n is the number of columns. Must be of type uint16.
        binsize (int): Binsize to use.

    Returns:
        numpy.ndarray: Binned array.
    """
    assert a.dtype == np.uint16, "Array must be of type uint16"
    _, M, N = a.shape
    m = M // binsize
    n = N // binsize
    a = a.reshape(a.shape[0], m, binsize, n, binsize)
    a = a.mean(axis=(2, 4))
    a = np.round(a).astype(np.uint16)
    return a


def _check_inputs_copyh5(src, dst, detector_roi, detector_binning):
    if os.path.exists(dst):
        raise FileExistsError(f"Destination file {dst} already exists")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file {src} does not exist")

    r1, r2, c1, c2 = detector_roi
    dr = r2 - r1
    dc = c2 - c1

    if dr % detector_binning != 0:
        raise ValueError(
            f"Detector ROI row size {dr} is not divisible by detector binning {detector_binning}"
        )
    if dc % detector_binning != 0:
        raise ValueError(
            f"Detector ROI column size {dc} is not divisible by detector binning {detector_binning}"
        )


def copyh5(src, dst, detector_roi=(900, 1100, 900, 1100), detector_binning=1):
    """Recursive function to copy a h5 file from src to dst.

    This can be used to copy DFXM h5 data while downsampling the detector data,
    which is useful for creating smaller h5 files for testing and debugging in
    darling. This function is not meant to tbe used for general purpose h5 file copying
    or for darling end user data handling.

    Args:
        src (str): Path to the source h5 file.
        dst (str): Path to the destination h5 file.
        detector_roi (tuple): ROI of the detector to copy. Defaults to (900, 1100, 900, 1100),
            such that only the central 200x200 pixels are copied.
        detector_binning (int): Binning of the detector to copy. Defaults to 1,
            such that the entire detector is copied as is.
    """

    _check_inputs_copyh5(src, dst, detector_roi, detector_binning)
    r1, r2, c1, c2 = detector_roi

    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:

        def copy_h5_attrs(src, dst):
            for k, v in src.attrs.items():
                dst.attrs[k] = v

        def recur(src_group, dst_group):
            copy_h5_attrs(src_group, dst_group)
            for key in src_group:
                obj = src_group[key]
                if isinstance(obj, h5py.Group):
                    dg = dst_group.create_group(key)
                    recur(obj, dg)
                else:
                    if (
                        obj.ndim > 2 and obj.shape[1] > 1000 and obj.shape[2] > 1000
                    ):  # this is a heuristic to avoid copying large detector data
                        data = obj[:, r1:r2, c1:c2]
                        data = _bin(data, detector_binning)
                        new_shape = data.shape
                        chunks = obj.chunks
                        if chunks is not None:
                            chunks = tuple(min(c, s) for c, s in zip(chunks, new_shape))
                        d = dst_group.create_dataset(
                            key,
                            data=data,
                            dtype=obj.dtype,
                            compression=obj.compression,
                            compression_opts=obj.compression_opts,
                            shuffle=obj.shuffle,
                            fletcher32=obj.fletcher32,
                            scaleoffset=obj.scaleoffset,
                            chunks=chunks,
                        )
                    else:
                        d = dst_group.create_dataset(
                            key,
                            data=obj[...],
                            dtype=obj.dtype,
                            compression=obj.compression,
                            compression_opts=obj.compression_opts,
                            shuffle=obj.shuffle,
                            fletcher32=obj.fletcher32,
                            scaleoffset=obj.scaleoffset,
                            chunks=obj.chunks,
                        )
                    copy_h5_attrs(obj, d)

        copy_h5_attrs(f, g)
        recur(f, g)


if __name__ == "__main__":
    pass
