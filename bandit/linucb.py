import numpy as np

from core.interfaces import BanditPolicy



class LinUCB(BanditPolicy):
    def __init__(self, n_actions: int, feature_dim: int, alpha: float, random_seed=None) -> None:
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        if random_seed is not None:
            np.random.seed(random_seed)
        self.A = np.array([np.eye(feature_dim, dtype=float) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, feature_dim), dtype=float)

    def select_action(self, features: np.ndarray) -> int:
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
