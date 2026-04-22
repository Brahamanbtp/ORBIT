from __future__ import annotations

import json
from typing import Any

from core.interfaces import BanditPolicy


class PolicyLogger:
            def compute_normalized_regret(self, total_bytes: int) -> list[float]:
                """
                Divide each element of cumulative regret by the running byte sum up to that block.
                Returns a list of regret per byte, one per block.
                """
                cumulative = self.compute_cumulative_regret()
                running_bytes = 0
                result = []
                for i, entry in enumerate(self.log):
                    block_size = entry.get("original_size") or entry.get("features", {}).get("size") or 1
                    running_bytes += block_size
                    val = cumulative[i] / running_bytes if running_bytes > 0 else 0.0
                    result.append(val)
                assert len(result) == len(self.log), f"Normalized regret length {len(result)} does not match log length {len(self.log)}"
                return result
        self._weight_snapshots: dict[int, dict[int, list[float]]] = {}  # block_id -> {action: weight_vector}
        def log_weight_snapshot(self, block_id: int) -> None:
            """
            For each action, extract the weight vector as A_inv @ b from the wrapped LinUCB instance and store as a dict keyed by block_id.
            """
            # Only works if wrapped policy has A and b attributes (LinUCB)
            policy = self.policy
            if not (hasattr(policy, "A") and hasattr(policy, "b")):
                return
            import numpy as np
            snapshot = {}
            for action in range(getattr(policy, "n_actions", len(policy.A))):
                try:
                    A_inv = np.linalg.inv(policy.A[action])
                    b = policy.b[action]
                    theta = (A_inv @ b).tolist()
                    snapshot[action] = theta
                except Exception:
                    snapshot[action] = None
            self._weight_snapshots[block_id] = snapshot

        def dump_weight_snapshots(self, path: str) -> None:
            """
            Write all weight snapshots as JSONL, one per block_id.
            """
            with open(path, "w", encoding="utf-8") as fh:
                for block_id, weights in sorted(self._weight_snapshots.items()):
                    fh.write(json.dumps({"block_id": block_id, "weights": weights}) + "\n")
    def __init__(self, policy: BanditPolicy) -> None:
        self.policy = policy
        self.log: list[dict[str, Any]] = []
        self._oracle_actions: dict[int, int] = {}  # block_id -> oracle_action
        self._oracle_rewards: dict[int, float] = {}  # block_id -> oracle_reward

    def record_oracle_action(self, block_id: int, oracle_action: int, oracle_reward: float = None) -> None:
        """
        Record the oracle action (and optionally reward) for a given block_id.
        """
        self._oracle_actions[block_id] = oracle_action
        if oracle_reward is not None:
            self._oracle_rewards[block_id] = oracle_reward

    def compute_cumulative_regret(self) -> list[float]:
        """
        Compute cumulative regret: sum of (oracle_reward - actual_reward) for each block in log order.
        If oracle_reward is not available, regret is 0 for that block.
        Returns a list of cumulative regret values (one per block).
        """
        cumulative = []
        total = 0.0
        for entry in self.log:
            block_id = entry.get("block_id")
            actual_reward = entry.get("reward", 0.0)
            oracle_reward = self._oracle_rewards.get(block_id, None)
            regret = (oracle_reward - actual_reward) if oracle_reward is not None else 0.0
            total += regret
            cumulative.append(total)
        return cumulative
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
    # ...existing code...

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def select_action(self, features) -> int:
        # Call wrapped select_action
        action = self.policy.select_action(features)
        # Try to get block_id from features
        block_id = getattr(features, "block_id", None)
        # Log snapshot every 50 blocks (if block_id is int)
        if block_id is not None and isinstance(block_id, int) and block_id % 50 == 0:
            self.log_weight_snapshot(block_id)
        return action

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
