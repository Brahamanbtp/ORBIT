from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.block import Block



class FeatureExtractor(ABC):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the feature vector dimension."""
        ...

    @abstractmethod
    def extract(self, block: Block) -> np.ndarray:
        ...


class BanditPolicy(ABC):
    @abstractmethod
    def select_action(self, features: Any) -> int:
        ...

    @abstractmethod
    def update(self, features: Any, action: int, reward: float, block_id=None) -> None:
        ...


class CodecAdapter(ABC):
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        ...

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        ...
