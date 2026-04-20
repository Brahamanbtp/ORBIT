import numpy as np

from core.block import Block


class BlockRouter:
    def __init__(self, extractor, policy, action_space):
        self.extractor = extractor
        self.policy = policy
        self.action_space = action_space

    def route(self, block: Block) -> tuple[int, np.ndarray]:
        feature_vector = self.extractor.extract(block)
        action_id = self.policy.select_action(feature_vector)
        return action_id, feature_vector
