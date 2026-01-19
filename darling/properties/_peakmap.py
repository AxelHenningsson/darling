import numpy as np

from ._constants import _FEATURE_MAPPING


class PeakMap:
    """Convenience class for fast manipulation of segmented peaks / domains.

    This class is a wrapper for the output of the darling.properties.peaks() function which defines a peak-searching strategy
    that segments peaks / domains and extracts features from a data-set.

    Args:
        feature_table (:obj: `dict` of `numpy.ndarray`): A dictionary of features. Each key in the dictionary is a feature name
            and correspoinds to a numpy array of shape (a,b,k) where a,b are the detector dimensions and k is the number of segmented
            peak / domains. See also darling.properties.peaks() for more details.

    Attributes:
        feature_table (:obj: `dict` of `numpy.ndarray`): A dictionary of features. Each key in the dictionary is a feature name
            and correspoinds to a numpy array of shape (a,b,k) where a,b are the detector dimensions and k is the number of segmented
            peak / domains. See also darling.properties.peaks() for more details.
        sorting (str): The key by which the feature table is sorted. Default is "sum_intensity", i.e upon instantiation the
            feature table is sorted by sum_intensity in descending order. Use the sort() function to sort by other features.
        ndim (int): The number of motor dimensions. I.e 1 for a rocking scan, 2 for a mosa-scan, etc. This is inferred from the feature_table
            key-set which is assumed to originate from the darling.properties.peaks() function.
        shape (tuple): The shape of the underlying data array. I.e (a,b,k), derived from the feature_table.
        k (int): The number of segmented peak / domains. Derived from the feature_table.

    """

    def __init__(self, feature_table):
        self.feature_table = feature_table
        self.sort(key="sum_intensity")
        self.ndim = self._infer_trailing_dims()
        self.shape = self.feature_table["sum_intensity"].shape
        self.k = self.feature_table["sum_intensity"].shape[-1]

    def _infer_trailing_dims(self):
        """get the number of trailing dimensions of the underlying data array"""
        for i, feature_mapping in enumerate(_FEATURE_MAPPING[1:]):
            if feature_mapping.keys() == self.feature_table.keys():
                return i + 1

    def sort(self, key, descending=True):
        """Sort the feature table by a given feature key.

        The default sorting key (from instantiation) is "sum_intensity". This function
        allows for sorting by other features, such as "number_of_pixels", "variance_axis_0", etc.

        Example:

        ... code-block:: python
            import darling
            _, data, motors = darling.io.assets.domains()
            peakmap = darling.properties.peaks(data, k=4, coordinates=motors)

            # now the peaks are sorted by sum_intensity in ascending order, each pixel has 4 peaks
            print( peakmap.sum_intensity[20, 20] )

            # we can re-sort by a different key
            peakmap.sort(key="number_of_pixels")

            # the 4 peaks at each pixel (i,j) are now sorted by number of pixels in ascending order
            print( peakmap.number_of_pixels[20, 20] )


        Args:
            key (str): the key to sort by, example: "sum_intensity", "number_of_pixels", "variance_axis_0", etc.
                see darling.properties.peaks() for a full list of supported feature keys.
            descending (bool): whether to sort in descending or ascending order. Default is True, i.e the largest
                numerical values are sorted first.

        """
        feature = self.feature_table[key]
        indices = np.argsort(feature, axis=-1, kind="quicksort")
        if descending:
            indices = indices[..., ::-1]

        for k in self.feature_table:
            feature = self.feature_table[k]
            self.feature_table[k] = np.take_along_axis(feature, indices, axis=-1)

        self.sorting = key

    def _getmap(self, key_pattern, k):
        map = np.zeros((self.shape[0], self.shape[1], self.ndim))
        for axis in range(self.ndim):
            key = key_pattern + str(axis)
            map[..., axis] = self.feature_table[key][..., k]
        dtype = self.feature_table[key].dtype
        return map.astype(dtype)

    def get_max(self, k):
        """Get the motor coordinates of the pixel with the highest intensity for the selected peak number k.

        Args:
            k (int): the peak number to get the maximum coordinates for.

        Returns:
            (:obj: `numpy.ndarray`): Motor coordinates of the pixel with the highest intensity for the currently set peaks (k=0). shape=(a,b,ndim).

        """
        return self._getmap("maxima_axis_", k)

    def get_mean(self, k):
        """Get the mean motor coordinates for the selected peak number k.

        Args:
            k (int): the peak number to get the mean coordinates for.

        Returns:
            (:obj: `numpy.ndarray`): Motor coordinates of the pixel with the highest intensity for the currently set peaks (k=0). shape=(a,b,ndim).

        """
        return self._getmap("mean_axis_", k)

    def get_variance(self, k):
        """Get the co-variance matrix-field for the selected peak number k.

        Args:
            k (int): the peak number to get the variance for.

        Returns:
            (:obj: `numpy.ndarray`): Co-variance matrix-field for the currently for peak number k. shape=(a,b,ndim,ndim) or (a,b) if ndim=1.

        """
        if self.ndim == 1:
            return self.feature_table["variance_axis_0"][..., k]
        var = np.zeros((self.shape[0], self.shape[1], self.ndim, self.ndim))
        for i in range(self.ndim):
            for j in range(self.ndim):
                if i < j:
                    key = f"variance_axis_{i}_axis_{j}"
                    var[..., i, j] = self.feature_table[key][..., k]
                    var[..., j, i] = var[..., i, j]
                if i == j:
                    var[..., i, j] = self.feature_table[f"variance_axis_{i}"][..., k]
        return var

    def get_dominance(self, k):
        """(:obj: `numpy.ndarray`): Dominance of selected peak number k. shape=(a,b). The dominnanace is defined as
        the ration between the integrated intensity of the selected peak number k and the sum of the integrated
        intensities of all peaks in the pixel.
        """
        return self.feature_table["sum_intensity"][..., k] / self.feature_table[
            "sum_intensity"
        ].sum(axis=-1, keepdims=True)

    @property
    def max(self):
        """(:obj: `numpy.ndarray`): Motor coordinates of the pixel with the highest intensity for the currently set peaks (k=0). shape=(a,b,ndim)."""
        return self.get_max(k=0)

    @property
    def mean(self):
        """(:obj: `numpy.ndarray`): Motor coordinates of the mean intensity for the currently set peaks (k=0). shape=(a,b,ndim)."""
        return self.get_mean(k=0)

    @property
    def variance(self):
        """(:obj: `numpy.ndarray`): Co-variance matrix-field for the currently set peaks (k=0). shape=(a,b,ndim,ndim) or (a,b) if ndim=1."""
        return self.get_variance(k=0)

    @property
    def dominance(self):
        """(:obj: `numpy.ndarray`): Dominance of the currently set peaks (k=0). shape=(a,b). The dominnanace is defined as
        the ration between the integrated intensity of the currently selected peak (k=0) and the sum of the integrated
        intensities of all peaks in the pixel.
        """
        return self.get_dominance(k=0)

    def __getattr__(self, name):
        """Just to expose the feature table keys as class attributes"""
        feature = self.__dict__.get("feature_table")
        if feature is not None and name in feature:
            return feature[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __getitem__(self, key):
        feature = self.__dict__.get("feature_table")
        if feature is None:
            raise KeyError(key)
        return feature[key]

    def __setitem__(self, key, value):
        feature = self.__dict__.get("feature_table")
        if feature is None:
            raise KeyError(key)
        if key not in feature:
            raise KeyError(key)
        feature[key] = value

    def __contains__(self, key):
        feature = self.__dict__.get("feature_table")
        return feature is not None and key in feature

    def keys(self):
        return self.feature_table.keys()

    def items(self):
        return self.feature_table.items()

    def values(self):
        return self.feature_table.values()

    def __iter__(self):
        return iter(self.feature_table)

    def __len__(self):
        return len(self.feature_table)


if __name__ == "__main__":
    pass
