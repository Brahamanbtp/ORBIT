import time
from collections import defaultdict

class TimingContext:
    def __init__(self, label: str):
        self.label = label
        self.elapsed_ms = None
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self._start) * 1000

class TimingAccumulator:
    def __init__(self):
        self._data = defaultdict(list)

    def add(self, label: str, elapsed_ms: float):
        self._data[label].append(elapsed_ms)

    def summary(self) -> dict:
        return {label: (sum(times) / len(times) if times else 0.0) for label, times in self._data.items()}
