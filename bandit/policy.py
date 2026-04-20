from __future__ import annotations

import json
from typing import Any

from core.interfaces import BanditPolicy


class PolicyLogger:
    def __init__(self, policy: BanditPolicy) -> None:
        self.policy = policy
        self.log: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def select_action(self, features) -> int:
        return self.policy.select_action(features)

    def update(self, features, action, reward) -> None:
        block_id = getattr(features, "block_id", None)
        self.log.append(
            {
                "block_id": block_id,
                "action": int(action),
                "reward": float(reward),
                "features": self._serialize_features(features),
            }
        )
        self.policy.update(features, action, reward)

    def dump_log(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.log:
                fh.write(json.dumps(entry) + "\n")

    @staticmethod
    def _serialize_features(features: Any) -> Any:
        if hasattr(features, "tolist"):
            return features.tolist()
        return features
