"""Module to load example data and phantoms."""

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

import darling


@contextmanager
def _asset_path(*parts):
    ref = files("darling").joinpath("assets", *parts)
    with as_file(ref) as p:
        yield Path(p)


def _read_numpy(*parts):
    with _asset_path(*parts) as p:
        return np.load(p)


def _get_asset_abspath(*parts):
    return str(files("darling").joinpath("assets", *parts).resolve())


def path():
    return _get_asset_abspath()


def mosa_field():
    """Load a 2D numpy array of a mosaicity field.

    mosa[..., 0] is the chi angle.
    mosa[..., 1] is the mu angle.

    data originates from the 1 -1 1 reflection of a 5% deformed Al1050 sample.

    ESRF ID03, experiment hc6042.
    dataset : Al_1050_strengthening_strainmosa_stregnthening_pct5_large_range
    Scan number : 5.1.
    Analysis: background subtraction, and a first maxima gaussian fitting.

    mean omega angle was 17.8345 and mean chi angle was 0.60997 the crystal to
    sample grain orientation matrix was measured to be :

                [0.83550027,  0.33932153, -0.43220389]
                [0.01167249,  0.77541728,  0.63134127]
                [0.54936605, -0.53253069,  0.64390062]

    from 3DXRD. The sample is in tensile strain with the grain 1-11 aling the
    tensile axis wich is identified as the lab-z axis.

    Returns:
        mosa (:obj:`numpy array`): 2D array of shape=(m, n) with mosaicity values.
    """
    return _read_numpy(
        "example_data", "misc", "mosa_Al_1050_stregnthening_pct5_large_range.npy"
    )


def domains(scan_id="1.1"):
    """load a (tiny) part of a 2d mosaicity scan collected at the ESRF id03.

    in constrast to mosaicity_scan() this dataset features a domain structure
    partioned into cells.

    NOTE: due to the domain partitioning this dataset has been inlcuded in the darling
    assets for unit testing various domain related feature extarction methods such as
    gaussian fitting.

    Args:
        scan_id (:obj:`str`): one of 1.1 or 2.1, specifying first or second layer
            scanned in the sample.

    Returns:
        data_path (:obj:`str`): absolute path to h5 file.
        data (:obj:`numpy array`):  Array of shape=(a, b, m, n) with intensity data.
            ``data[:,:,i,j]`` is a noisy detector image in type uint16 for phi and
            chi at index i and j respectively.
        coordinates (:obj:`numpy array`): array of shape=(2,m,n) continaning
            angle coordinates.

    """
    data_path = _get_asset_abspath("example_data", "domains", "2D_domains.h5")
    reader = darling.reader.MosaScan(data_path)
    dset = darling.DataSet(reader)
    dset.load_scan(scan_id)
    return data_path, dset.data, dset.motors


def rocking_scan():
    """load a downsampled 1d rocking scan collected at the ESRF id03.

    Returns:
        data_path (:obj:`str`): absolute path to h5 file.
        data (:obj:`numpy array`):  Array of shape=(a, b, m, n) with intensity data. ``data[:,:,i,j]`` is a noisy
            detector image in type uint16 for phi and chi at index i and j respectively.
        coordinates (:obj:`numpy array`): array of shape=(2,m,n) continaning angle coordinates.

    """
    data_path = _get_asset_abspath("example_data", "rocking_scan_id03", "rocking.h5")
    reader = darling.reader.RockingScan(data_path)
    dset = darling.DataSet(reader)
    dset.load_scan(scan_id="1.1")
    return data_path, dset.data, dset.motors


def motor_drift(scan_id="1.1"):
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
    data_path = _get_asset_abspath("example_data", "motor_drift", "motor_drift.h5")
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
    data_path = _get_asset_abspath("example_data", "mosa_scan_id03", "mosa_scan.h5")
    reader = darling.reader.MosaScan(data_path)
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
