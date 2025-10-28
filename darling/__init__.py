from importlib.metadata import PackageNotFoundError, version

from . import (
    _color,
    assets,
    goniometer,
    metadata,
    peaksearcher,
    properties,
    reader,
    transforms,
    utils,
)
from ._dataset import DataSet

try:
    __version__ = version("darling-pypi")
except PackageNotFoundError:
    __version__ = "unknown"
