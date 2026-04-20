from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    block_id: int
    data: bytes
    size: int
    offset: int
    metadata: dict[str, Any] = field(default_factory=dict)
