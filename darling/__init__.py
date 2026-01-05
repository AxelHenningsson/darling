from importlib.metadata import PackageNotFoundError, version

from . import crystal, diffraction, filters, geometry, io, properties, transforms
from .io import DataSet

try:
    __version__ = version("darling-pypi")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "DataSet",
    "__version__",
    "io",
    "filters",
    "properties",
    "transforms",
    "diffraction",
    "crystal",
    "geometry",
]
