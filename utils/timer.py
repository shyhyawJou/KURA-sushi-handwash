from time import perf_counter
from loguru import logger



class Timer:
    def __init__(self, name="Process", silent=False):
        self.name = name
        self.silent = silent
        self.elapsed = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = perf_counter() - self.start
        if not self.silent:
            logger.debug(f"{self.name}: {self.elapsed:.3f} (s)")