"""This is the extended io module for reading and writing data in Darling.

Beyond the core DataSet class, this module also contains the assets module for reading and writing data from
example data sets.

The detailed file-reader implementations and meta-data extraction is handled by this module and can be interfaced/overridden
to implement io for data formats that differs from the default Darling formats which implement support for the ID03 ESRF
beamline (time period: 2024 - 2026)
"""

from . import assets, metadata, reader
from ._dataset import DataSet

__all__ = ["DataSet", "assets", "metadata", "reader"]
