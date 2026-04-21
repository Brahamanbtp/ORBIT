from __future__ import annotations

import json
from typing import Any

from core.interfaces import BanditPolicy


class PolicyLogger:
        def compute_convergence_stats(self) -> dict:
            import math
            window = 50
            actions = []
            rewards = []
            action_entropy_over_time = []
            rolling_mean_reward = []
            from collections import Counter

            for i, entry in enumerate(self.log):
                actions.append(entry["action"])
                rewards.append(entry["reward"])
                # Action entropy up to this point
                counts = Counter(actions)
                total = len(actions)
                probs = [c / total for c in counts.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                action_entropy_over_time.append(entropy)
                # Rolling mean reward
                start = max(0, i - window + 1)
                mean = sum(rewards[start:i+1]) / (i - start + 1)
                rolling_mean_reward.append(mean)

            return {
                "action_entropy_over_time": action_entropy_over_time,
                "rolling_mean_reward": rolling_mean_reward,
            }
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
