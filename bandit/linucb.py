import numpy as np

from core.interfaces import BanditPolicy





class LinUCB(BanditPolicy):
    def __init__(self, n_actions: int, feature_dim: int, alpha: float, random_seed=None, feature_extractor=None, burn_in_blocks: int = 20) -> None:
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.burn_in_blocks = burn_in_blocks
        self._call_count = 0
        if feature_extractor is not None:
            if hasattr(feature_extractor, "feature_dim"):
                if feature_extractor.feature_dim != feature_dim:
                    raise ValueError(f"FeatureExtractor feature_dim ({feature_extractor.feature_dim}) does not match LinUCB feature_dim ({feature_dim})")
        if random_seed is not None:
            np.random.seed(random_seed)
        self.A = np.array([np.eye(feature_dim, dtype=float) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, feature_dim), dtype=float)


    def select_action(self, features: np.ndarray) -> int:
        if self._call_count < self.burn_in_blocks:
            action = self._call_count % self.n_actions
            self._call_count += 1
            return action
        self._call_count += 1
        x = np.asarray(features, dtype=float).reshape(-1)
        scores = np.empty(self.n_actions, dtype=float)
        for action in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]
            exploration = np.sqrt(x @ A_inv @ x)
            scores[action] = x @ theta + self.alpha * exploration
        return int(np.argmax(scores))

    def update(self, features, action, reward) -> None:
        x = np.asarray(features, dtype=float).reshape(-1)
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + reward * x
