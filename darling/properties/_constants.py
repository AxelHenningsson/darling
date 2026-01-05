# for darling.properties.peaks():
# Here we define the mapping between the features extracted from the segmented domains
# and the indices in the feature table. This is a static mapping which allows for the
# feature table computed in compiled code. The mapping is used to extract the features.
# see _extract_features_1D(), _extract_features_2D(), _extract_features_3D() for motivation.
_FEATURE_MAPPING = [
    None,  # <- dummy to start indexing from 1:
    {  # The 1D Case:
        "sum_intensity": 0,
        "max_intensity": 1,
        "number_of_pixels": 2,
        "maxima_axis_0": 3,
        "mean_axis_0": 6,
        "variance_axis_0": 9,
    },
    {  # The 2D Case:
        "sum_intensity": 0,
        "max_intensity": 1,
        "number_of_pixels": 2,
        "maxima_axis_0": 3,
        "maxima_axis_1": 4,
        "mean_axis_0": 6,
        "mean_axis_1": 7,
        "variance_axis_0": 9,
        "variance_axis_1": 10,
        "variance_axis_0_axis_1": 12,
    },
    {  # The 3D Case:
        "sum_intensity": 0,
        "max_intensity": 1,
        "number_of_pixels": 2,
        "maxima_axis_0": 3,
        "maxima_axis_1": 4,
        "maxima_axis_2": 5,
        "mean_axis_0": 6,
        "mean_axis_1": 7,
        "mean_axis_2": 8,
        "variance_axis_0": 9,
        "variance_axis_1": 10,
        "variance_axis_2": 11,
        "variance_axis_0_axis_1": 12,
        "variance_axis_0_axis_2": 13,
        "variance_axis_1_axis_2": 14,
    },
]
