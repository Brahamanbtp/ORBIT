from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Block:
    block_id: int
    data: bytes
    size: int
    offset: int
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_vector: Optional[np.ndarray] = None
