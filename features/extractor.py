import numpy as np

from core.interfaces import Block, FeatureExtractor
from features.entropy import compute_entropy
from features.repetition import compute_repetition_score
from features.rle_proxy import compute_rle_ratio



class BlockFeatureExtractor(FeatureExtractor):
    _ALL_FEATURES = {
        "entropy": compute_entropy,
        "rle_ratio": compute_rle_ratio,
        "repetition": compute_repetition_score,
    }

    def __init__(self, enabled_features=None):
        if enabled_features is None:
            self._feature_names = ["entropy", "rle_ratio", "repetition"]
        else:
            for name in enabled_features:
                if name not in self._ALL_FEATURES:
                    raise ValueError(f"Unsupported feature: {name}")
            self._feature_names = list(enabled_features)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def feature_dim(self) -> int:
        return len(self._feature_names)

    def extract(self, block: Block) -> np.ndarray:
        values = [self._ALL_FEATURES[name](block.data) for name in self._feature_names]
        return np.array(values, dtype=float)
