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
        self._data = defaultdict(list)  # label -> list of elapsed_ms
        self._records = []  # list of dicts: {block_id, label, elapsed_ms}

    def add(self, label: str, elapsed_ms: float):
        self._data[label].append(elapsed_ms)

    def record(self, block_id: int, label: str, elapsed_ms: float) -> None:
        self._records.append({"block_id": block_id, "label": label, "elapsed_ms": elapsed_ms})
        self._data[label].append(elapsed_ms)

    def to_dataframe(self) -> list:
        return list(self._records)

    def summary(self) -> dict:
        return {label: (sum(times) / len(times) if times else 0.0) for label, times in self._data.items()}
