import numpy as np
from bench import timeit

import darling


def bench_local_max_label(
    data_shape,
    warmup=True,
    verbose=False,
):
    number_of_pixels = np.prod(data_shape[0:2])
    pixel_shape = data_shape[2:]
    data = np.random.randint(0, 65000, data_shape, dtype=np.uint16)
    data[data < np.max(data) * 0.5] = 0  # make it a bit sparse

    total_time = timeit(
        darling.properties.local_max_label,
        warmup=warmup,
        data=data,
        loop_outer_dims=True,
    )

    time_per_pixel = total_time / number_of_pixels
    if verbose:
        print(
            f"Time per pixel: {time_per_pixel * 1000000:.1f} microseconds, for pixel shape={pixel_shape}"
        )
    return time_per_pixel, pixel_shape


if __name__ == "__main__":
    time_per_pixel, pixel_shape = bench_local_max_label(
        data_shape=(32, 32, 31, 31, 31), warmup=True, verbose=True
    )
    a, b = 2048, 2048
    inferred_time = time_per_pixel * a * b  # s
    print(f"Inferred time for 2048x2048 image: {inferred_time:.1f} s")
