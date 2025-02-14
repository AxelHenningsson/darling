"""Module to load example data and phantoms."""

import os

import numpy as np

import darling

_root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
_asset_path = os.path.join(_root_path, "assets")


def path():
    return _asset_path


def motor_drift(scan_id="1.1"):
    data_path = os.path.join(
        _asset_path,
        "example_data",
        "motor_drift",
        "motor_drift.h5",
    )
    """load a (tiny) part of a 2d mosaicity scan collected at the ESRF id03.

    NOTE: This dataset features motor drift and has been inlcuded in the darling
    assets for unit testing.

    Args:
        scan_id (:obj:`str`): one of 1.1 or 2.1, specifying first or second layer scanned in the sample.

    Returns:
        data_path (:obj:`str`): absolute path to h5 file.
        data (:obj:`numpy array`):  Array of shape=(a, b, m, n) with intensity data. ``data[:,:,i,j]`` is a noisy
            detector image in type uint16 for phi and chi at index i and j respectively.
        coordinates (:obj:`numpy array`): array of shape=(2,m,n) continaning angle coordinates.

    """
    reader = darling.reader.MosaScan(data_path)
    dset = darling.DataSet(reader)
    dset.load_scan(scan_id)
    return data_path, dset.data, dset.motors


def mosaicity_scan(scan_id="1.1"):
    """Load a (tiny) part of a 2D mosaicity scan collected at the ESRF ID03.

    This is a central detector ROI for a 111 reflection in a 5% deformed Aluminium. Two layers
    are available with scan_id 1.1 and 2.1.

    Args:
        scan_id (:obj:`str`): One of 1.1 or 2.1, specifying first or second layer scanned in the sample.

    Returns:
        data_path (:obj:`str`): Absolute path to h5 file.
        data (:obj:`numpy array`): Array of shape (a, b, m, n) with intensity data.
        ``data[:,:,i,j]`` is a noisy detector image (uint16) for phi and chi at index ``i, j``.
        coordinates (:obj:`numpy array`): Array of shape (2, m, n) containing angle coordinates.
    """
    data_path = os.path.join(
        _asset_path, "example_data", "mosa_scan_id03", "mosa_scan.h5"
    )
    reader = darling.reader.MosaScan(data_path)
    dset = darling.DataSet(reader)
    dset.load_scan(scan_id)
    return data_path, dset.data, dset.motors


def energy_scan(scan_id="1.1"):
    """load a (tiny) part of a 2d energy-chi scan collected at the ESRF id03.

    The data was integrated over the rocking angle phi.

    This is a central detector ROI for a 111 reflection in a 5% deformed Aluminium. Two layers
    are available with scan_id 1.1 and 2.1. The data corresponds to that of mosaicity_scan().
    Energy scanning was achieved by pertubating the id03 upstreams monochromator.

    Args:
        scan_id (:obj:`str`): one of 1.1 or 2.1, specifying first or second layer scanned in the sample.

    Returns:
        data_path (:obj:`str`): absolute path to h5 file.
        data (:obj:`numpy array`): Array of shape=(n, m, a, b) with intensity data, ``data[:,:,i,j]`` is
        a noisy detector image in type uint16 for energy and chi at index i and j respectively.
        coordinates (:obj:`numpy array`): array of shape=(2,m,n) continaning energy and angle coordinates.

    """
    data_path = os.path.join(
        _asset_path,
        "example_data",
        "energy_scan_id03",
        "energy_scan_110_5pct_layer_" + str(int(scan_id[0])) + ".h5",
    )
    reader = darling.reader.EnergyScan(data_path)
    dset = darling.DataSet(reader)
    dset.load_scan(scan_id)
    return data_path, dset.data, dset.motors


def gaussian_blobs(N=32, m=9):
    """Phantom 2d scan of gaussian blobs with shifting means and covariance.

    Args:
        N,m (:obj:`int`): Desired data array size which is of shape=(N,N,m,m).

    Returns:
        data (:obj:`numpy array`): Array of shape=(N, N, m, m) with intensity data, ``data[:,:,i,j]``
        is a noisy detector image in type uint16 for motor x and y at index i and j respectively.
        coordinates (:obj:`numpy array`): array of shape=(2,m,m) continaning x and y coordinates.

    """
    x = y = np.linspace(-1, 1, m, dtype=np.float32)
    sigma0 = (x[1] - x[0]) / 3.0
    X, Y = np.meshgrid(x, y, indexing="ij")
    data = np.zeros((N, N, len(x), len(y)))
    S = np.eye(2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x0, y0 = sigma0 * i / N, sigma0 * j / N - 0.5 * sigma0 * i / N
            S[0, 0] = sigma0 + 0.5 * sigma0 * i / N
            S[1, 1] = sigma0 + 0.5 * sigma0 * j / N - 0.25 * sigma0 * i / N
            Si = 1.0 / np.diag(S)
            data[i, j] = (
                np.exp(-0.5 * (Si[0] * (X - x0) ** 2 + Si[1] * (Y - y0) ** 2)) * 64000
            )
    np.round(data, out=data)
    data = data.astype(np.uint16, copy=False)
    return data, np.array([X, Y], dtype=np.float32)


if __name__ == "__main__":
    pass
