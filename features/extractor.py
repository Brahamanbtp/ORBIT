import numpy as np

from core.interfaces import Block, FeatureExtractor
from features.entropy import compute_entropy
from features.repetition import compute_repetition_score
from features.rle_proxy import compute_rle_ratio


class BlockFeatureExtractor(FeatureExtractor):
    def extract(self, block: Block) -> np.ndarray:
        entropy = compute_entropy(block.data)
        rle_ratio = compute_rle_ratio(block.data)
        repetition_score = compute_repetition_score(block.data)
        return np.array([entropy, rle_ratio, repetition_score], dtype=float)
