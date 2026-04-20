
def run_pipeline_check() -> bool:
	try:
		from codecs.raw_adapter import RawAdapter
		from features.extractor import BlockFeatureExtractor
		from bandit.linucb import LinUCB
		from bandit.action_space import ActionSpace
		from core.block import Block
		from configs.schema import ORBITConfig
		from pipeline.router import BlockRouter
		from bandit.reward import compute_reward

		config = ORBITConfig()
		extractor = BlockFeatureExtractor()
		policy = LinUCB(n_actions=config.n_actions, feature_dim=config.feature_dim, alpha=config.alpha)
		action_space = ActionSpace(["raw"])
		router = BlockRouter(extractor, policy, action_space)
		codec = RawAdapter()

		block = Block(block_id=0, data=b"A" * config.block_size, size=config.block_size, offset=0)
		action_id, features = router.route(block)
		compressed = codec.compress(block.data)
		reward = compute_reward(len(block.data), len(compressed), 0.0)
		policy.update(features, action_id, reward)
		return True
	except Exception:
		return False
