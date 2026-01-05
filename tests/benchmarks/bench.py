import time


def timeit(
    func,
    warmup=True,
    **kwargs,
):
    if warmup:
        func(**kwargs)  # warmup / compile
    start_time = time.time()
    func(**kwargs)
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    pass
