import numpy as np


def compute_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    values = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(values, minlength=256)
    probabilities = counts[counts > 0] / values.size

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(np.clip(entropy, 0.0, 8.0))
