"""CPU and memory profiling utilities."""
import time
import psutil
import os
from contextlib import contextmanager
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def memory_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class CPUProfiler:
    """Context-manager based CPU/memory profiler."""

    def __init__(self, name: str = "block", log: bool = True):
        self.name = name
        self.log = log
        self.elapsed: float = 0.0
        self.mem_delta: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._m0 = memory_mb()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t0
        self.mem_delta = memory_mb() - self._m0
        if self.log:
            logger.info(
                f"[{self.name}] {self.elapsed:.3f}s | "
                f"ΔRAM={self.mem_delta:+.1f}MB | "
                f"RAM={memory_mb():.1f}MB"
            )


@contextmanager
def time_block(name: str = "block"):
    """Simple timing context manager."""
    t0 = time.perf_counter()
    yield
    logger.debug(f"[{name}] {time.perf_counter()-t0:.3f}s")
