from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class ORBITConfig:
    block_size: int = 4096
    n_actions: int = 4
    alpha: float = 1.0
    feature_dim: int = 3

    @classmethod
    def load_yaml(cls, path: str) -> ORBITConfig:
        with open(path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return cls(**payload)
